# Harness 架构总览（2026-11-06，供 checkpoint 使用）

本文按 2026-11-06 的本地代码（与 Hopper `skill_memory_agent_look/` 基本一致）整理。README 还停在中期版本（Qwen3-30B、颜色分割），以本文为准。

## 0. 一句话

一个在线机器人编程 Agent：看桌面（RGB-D）→ 认东西 → 推断每件东西该去哪 → 规划顺序 → 为每个子目标写 Python 程序在沙箱里控制 Panda → 用相机检查 → 失败就诊断、反思、重写 → 成功的程序拆成可组合的技能存进记忆，下一张桌子直接复用。仿真是 Robosuite/MuJoCo，模型可换（Qwen / Gemini / DeepSeek），抓取和运动规划可以调用 Contact-GraspNet 与 cuRobo。

## 1. 一张桌子的完整流程（`sma/agent/loop.py: run_episode`）

```
env.reset() ── RGB-D 俯视 + 侧视 ──► 感知（颜色 或 SAM2+深度+跨帧跟踪）
   │
   ├─①识别 identify（仅 neutral 桌面）：编号/描边图 + 放大图 + 深度几何 → LLM 说出每件东西的类别 → 角色（物品/容器/其他）
   ├─②长期记忆 user_memory.learn / retrieve：用户明说的偏好（带原话证据）+ 以前桌面上观察到的习惯
   ├─③场景分组 grouping.infer_homes（无 LLM）：同类东西放一起 → 每件散落物品的"家"
   ├─④细化 refine（LLM）：模糊指令 → 显式目标，每个目标必须标来源
   │     instruction（引原话）/ user_memory / scene_grouping / assumption；来源不成立的目标被丢弃
   ├─⑤规划 PLANNER（LLM）：有序子目标列表
   │
   └─⑥逐个子目标执行（每个目标最多 2 轮，总步数 ≤ 2×目标数+1）
         run_subgoal：最多 3 次尝试
           ├ 技能记忆检索 + 三级适用性检查（目标 / 状态 / 物理：cuRobo 或历史可达范围）
           │    适用且已验证 → 第 1 次尝试直接调用技能，不调 LLM
           │    否则 → CODER（LLM）写程序；提示里附：相关技能、helper、抓取经验、教训
           ├ Sandbox 执行（AST 白名单，禁止 import/try/class；每个 API 调用都记轨迹）
           ├ 相机检查 check_on_zone（只用相机几何，不用真值）
           └ 失败 → diagnose（从轨迹+相机算出的事实，不给建议）→ REFLECT（LLM 一句教训）→ 下一次尝试
         成功且是新代码 → EXTRACT（LLM）拆成 2–4 个 helper + 一个子目标函数 → 存入技能记忆
         每个子目标后 check_goals：检查全部目标
           失败 / 已完成的被碰乱 / 计划与剩余不符 → REPLANNER（LLM）重排剩余步骤
   │
   结束：go_home → private_evaluate（离线真值评分，只写进 episode.json）
   之后：新技能在 2 张新布局上无 LLM 验证 validate_skill，结果写进技能历史
```

## 2. 模块分层

| 层 | 文件 | 职责 |
|---|---|---|
| 仿真场景 | `sma/env/scene.py`、`assets.py`、`looks.py` | Robosuite/MuJoCo 桌面 + Franka Panda；物体（笔、杯、本子、纸团、书、积木）与容器（笔筒、杯垫、书堆、铁丝网废纸篓、托盘）；`looks.py` 只加显示几何做"真实外观"，碰撞与质量不变 |
| 任务生成 | `sma/env/tasks.py` | 桌面族与指令：`pen`/`pen_holder`、`realdesk`、`nbdesk`（A5 本子伸出桌边）、`cdesk`（每件一色、模型自己认）、`sdesk`（撞色桌：5 种颜色 9–10 件，真实外观，SAM2）、`groupdesk`；模糊指令 + 已整理好的"锚点"物品 |
| 机器人 API | `sma/env/robot_env.py`（`TabletopEnv`） | 各 API 层的实现；`observe` / `check_on_zone` / `check_goals`；私有 `private_evaluate` |
| 感知 | `perception.py`（颜色）、`perception_sam.py`（SAM2+深度）、`perception_service/server.py`（SAM2 服务） | 无类别分割 → 深度取舍 → 跨帧编号（匈牙利匹配、抓着跟手、松手后物体持久性）→ 尺寸/高度/空心/边沿等测量 |
| Agent | `sma/agent/loop.py`、`identify.py`、`grouping.py`、`refine.py`、`prompts.py`、`sandbox.py` | 上图①–⑥ |
| 记忆 | `sma/memory/store.py`（技能）、`user.py`（用户） | 见第 4 节 |
| LLM | `sma/agent/llm.py` | OpenAI 兼容客户端：vLLM（Qwen）/ Gemini / DeepSeek；全量调用日志；花费账本与硬上限；思考用尽输出额度时关思考重问 |
| 外部服务 | `aspire_qwen/provider_env` 里的 Contact-GraspNet、cuRobo | `grasp_candidates` 里的 GraspNet 候选；`carry_to` 的避障搬运（`SMA_MOTION=curobo`）、`move_gripper(path="curobo")`、`solve_ik(collision_check)`、技能适用性的物理检查 |
| 评估与运行 | `sma/eval/run_suite.py`、`report.py`、`replay.py`；`hpc/run_sma.pbs`、`submit.sh`；`scripts/*_gate.py` | 条件 `none/skill/user/both`；断点续跑；`SMA_MEMORY_INIT` 继承记忆；无模型的脚本闸门；带代码面板的回放视频 |

## 3. 动作空间：API 层（`SMA_API_LEVEL`）

| 层 | 模型能调用的 | 现状 |
|---|---|---|
| `task` | `pick` / `place` / `release` | 中期保底 |
| `mid` | 抓取候选、`grasp`、`free_spots`、`place_object`、`carry_to`、`rotate_by`、`lower_until_contact`… | 中期主线 |
| **`act`** | mid 去掉 `place_object`（不给"直接摆好物体"的黑盒），加 `gripper_pose`、`move_gripper(approach, finger_axis, path=line/joint/curobo)`、`move_by`、`solve_ik` | **当前主线**：手可以任意朝向，立笔、侧面夹本子都要模型自己组合 |
| `pose` / `aspire` | `move_to` + 夹爪 / ASPIRE CapApi | 对开源模型太低层（0 分），已放弃 |

`grasp_candidates` 返回几何候选 + GraspNet 候选 +（宽扁物体的）"夹边"侧抓候选，每个带 `fits_open_fingers`、`hand_clear`、`edge_free_below`、`reachable_checked` 等可通用的字段。API 文档只写通用语义、机器人几何事实和两个通用 few-shot 示例，不写任何针对测试桌的经验。

## 4. 两种记忆

**技能记忆 `SkillMemory`（memory.json）**

| 内容 | 来源 | 怎么用 |
|---|---|---|
| `skills`：子目标函数 `place_on_zone_vN(obj, zone)` + 适用条件 + 执行历史 | 相机确认成功的新程序，经 EXTRACT 参数化；硬编码场景坐标的会被拒 | 检索 → 三级检查：目标（同类物体进同类容器，最近 4 次成功率）、状态（夹爪空、容器空）、物理（cuRobo 无碰撞位姿 或 历史可达范围）→ applicable 直接执行 / uncertain 作为参考给 CODER |
| `helpers`：可组合子技能（选抓取、转向、搬运插入…） | EXTRACT 拆出 | 注入沙箱按名调用；2 次成功"已验证"，0 成功 3 失败"已拒绝" |
| `lessons`：失败后的一句教训 | REFLECT | 按物体类别给 CODER；看过它的后续尝试 ≥2 次都失败 → 标为"无效教训"并告诉模型别重复 |
| `grasp_log`：每类物体怎么抓、抓没抓住 | 轨迹 | 作为抓取经验给 CODER |

版本更替：存储技能在这里失败、新程序修复成功 → 新版本替换旧版本（`superseded`）；最近记录差的旧技能也会被新成功程序替换。

**用户记忆 `UserMemory`（user_memory.json）**

- `rules`：用户明说的长期放置偏好（必须有 always / from now on 等措辞，证据必须是原话）；
- `handling`：怎么放的要求（如"笔要和笔盒对齐"），原话转给 CODER；
- `habits`：每张桌上观察到的"某类东西放在某处"，跨桌累计，按类别相似度投票给出去处。

优先级：本次指令 > 用户明说偏好 > 观察到的习惯 / 场景分组 > 模型假设。

## 5. 信息边界

- Agent 看得到：俯视/侧视 RGB-D 的感知结果（物体名是颜色或 thingN，类别由模型自己认）、本体感觉、自己 API 调用的返回值、相机检查。
- 只给评分：仿真真值（`private_evaluate`、`private_thing_truth`）、接触诊断（`SMA_CONTACT_DEBUG`）。
- 记忆里的成败全部来自相机检查，不来自真值。

## 6. 每张桌的 LLM 调用点

`identify`（1 次，看图）→ `user_memory_write` → `refine`（可能多 1 轮一致性修正）→ `plan` → 每次尝试 `code`（直接复用技能时为 0）→ 失败时 `reflect` → 成功新程序 `extract` → 必要时 `replan`。所有调用写日志，按 run 和全项目两个账本计费。

## 7. 演进（为什么是现在这样）

1. **API 层**：task → mid（中期）→ act（11-04 起）。理由：评估"模型自己操作动作空间"时不能给解决该操作的黑盒原语。
2. **模型**：Qwen3-30B（中期）→ Gemini 3.8 Flash（11-04）→ DeepSeek-flash（11-05 起，预算原因）。项目目标是 harness，不做模型对比。
3. **感知**：颜色分割 → SAM2 无类别分割 + 深度 + 跨帧跟踪 + 模型 Set-of-Mark 识别；场景换成真实外观（11-05/06）。
4. **技能**：整段程序缓存 → helper 可组合子技能 + 版本替换 + 最近记录（11-04）。
5. **长程**：一次规划 → 每步后全局检查 + 重新规划（11-04）。
6. **去处推断**：只靠指令 → 场景分组（TidyBot 式，同类放一起）+ 观察习惯写入长期记忆（11-05）。
7. **约束**：所有改动必须可泛化，不针对测试桌调 harness/prompt（11-05/06 用户要求）。

## 8. 已有证据（截至 11-06）

| 实验 | 结果 | 位置 |
|---|---|---|
| 中期：笔进窄笔盒，技能记忆，3 组布局 × 10 | 7.3/10 vs 无记忆 2.7/10 | `results/main_experiment/` |
| 中期：用户记忆，10 任务含 6 条模糊指令 | 去处正确 100% vs 42% | 同上 |
| realdesk，Gemini act 层，CPU，20 桌 | 整桌 17/20 vs 6/20；笔 15/16 vs 3/16 | `docs/REALDESK_TASK_zh.md` |
| realdesk，开 GraspNet+cuRobo（GPU），10 桌 | 整桌 7/10 vs 2/10；笔 8/8 vs 1/8 | 同上 |
| sdesk 真实外观 + SAM2，DeepSeek 识别，8 桌 | 识别 72/74（标签）、71/74（描边）；角色 74/74 | 同上 |
| DeepSeek 全流程（识别→分组→规划→写码→技能/长期记忆，GraspNet+cuRobo），sdesk 17000/17001 | 两桌整桌真值成功，$0.48 | `results/ds_full2_sdesk_20261106T080723Z/` |

注意：最后一行只有 2 张桌，是"全流程跑通"的证据，不是成功率。

## 9. 当前开放问题

1. SAM2 碎块被当成独立物体去搬，重新规划会反复把它们加回来。
2. 同色物体放进同一容器后连成一块，相机检查认不出刚放进去的那个（真值成功，相机判失败，模型继续重试）。
3. 脚本闸门里笔仍有 2/5 失败：笔筒靠近高废纸篓时，可达的手方向都朝篓。
4. GraspNet 的斜向抓取被 `grasp()` 投影成竖直下降，模型从未按原始斜向姿态执行；cuRobo 主要由 harness 在搬运时调用，模型自选 `path="curobo"` 很少。
5. 11-04 之前的 Gemini act 运行都是 CPU 作业，没有 GraspNet/cuRobo（已更正，GPU 组单独报告）。
6. 代码与结果：v2 仓库最后一次提交是 10-18 中期版本，之后的全部改动（约 140 个文件）尚未提交。
