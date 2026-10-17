# 中间层 primitive 接口（`SMA_API_LEVEL=mid`）

代码：`sma/env/robot_env.py`（"mid-level API" 一节）、`sma/agent/prompts.py` 中的 `MID_API_DOC`（模型实际看到的英文说明）、
`sma/agent/sandbox.py`、`scripts/mid_gate.py`。提交 `c153dae`。task 级（`SMA_API_LEVEL=task`）不变，作为中期保底。

## 层级定位

| 层级 | 模型要决定的 | 系统负责的 |
|---|---|---|
| TidyBot | 物体放到哪个容器、放还是扔 | 其余全部（容器位置写死） |
| task 级（保底） | 抓取高度、手指方向 | 接近、抓取、搬运、放置、松手 |
| **mid 级（新）** | 选哪个抓取候选、是否调整、放在容器哪个位置和朝向、搬运高度、在哪松手、顺序、失败后怎么恢复 | 分割与几何、抓取候选生成、IK 与运动执行、相机检查 |
| ASPIRE 级 | 点云几何、位姿与四元数、规划请求 | 只提供工具 |

## 函数

| 函数 | 作用 | 返回 |
|---|---|---|
| `observe()` | 俯视 RGB-D 感知（与 task 级相同） | 物体：类别、中心、高度、外形长短边及方向；容器：中心、半尺寸；夹爪状态 |
| `grasp_candidates(object_name)` | 生成从上往下的抓取候选：Contact-GraspNet（只保留接近方向与竖直夹角 ≤30° 的），以及由外形几何算出的候选（横夹短边或长边，两个高度） | 列表，每项含 `x, y, height, yaw`，`squeeze_width`（从物体点云实测、两指之间的宽度），`fits_open_fingers`，`source`，`score`。**不按任务排序，由模型选** |
| `grasp(candidate)` | 张开，在安全高度移到上方，竖直下降，闭合，抬到桌面以上 0.18 m | `holding_something`、`finger_gap`、`expected_squeeze_width` |
| `free_spots(zone_name, object_name=None)` | 在容器里为物体外形找放得下、不碰其他物体的位置（长轴沿 x 或 y 两种朝向） | `occupants`（容器里已有的物体）、`spots`（每项含 `x, y`、`object_long_axis_yaw`、`gripper_yaw`、`clearance`）；放不下时 `spots` 为空 |
| `carry_to(x, y, height=0.18, yaw=None)` | 先竖直升到桌面以上 `height`，再水平移动到 (x, y) | 位置、是否仍拿着 |
| `lower_until_contact()` | 竖直下降，直到物体底部到达桌面高度，或中途被墙壁、其他物体挡住 | `stopped_by_contact`、`held_object_bottom_above_table` |
| `open_gripper()` / `close_gripper()` | 只动手指，不动手臂 | 手指宽度、是否拿着东西 |
| `retreat(dz=0.12)` | 竖直上移 | 位置 |
| `check(object_name, zone_name)` | 机械臂停到一边后，用相机检查物体是否在容器内、是否放稳 | `satisfied`、`inside_zone`、`resting`、`offset_from_zone_center` |
| `go_home()`、`wait(steps)` | 停到一边，或原地等待 | — |

给模型的通用顺序先验（和 pose 级一样是人工写的，每个任务都相同，不含任何针对具体物体的参数）：
`grasp_candidates → 选一个 → grasp → 检查 holding_something → free_spots → carry_to 到所选位置上方 → lower_until_contact → open_gripper → retreat → check`。

## 信息边界

- 模型和这些 primitive 都只用相机图像、深度、相机标定和机器人本体状态（关节、夹爪宽度、自己发出过的命令）。
- 仿真真实状态只用于离线评分（`private_evaluate`），不进入任何 primitive。
- `gripper_yaw` 由抓取时的手指方向与当时看到的物体长轴方向推算，不读仿真状态。

## 与上次设计的差异

- **运动没有用 cuRobo。** 搬运沿用现有的 IK 加关节控制：先升到安全高度，再水平移动，然后竖直下降。桌面场景只用从上往下的动作，这样就能避开物体和容器墙。这是为了赶中期，并且复用 task 级已验证的运动层。
- **Contact-GraspNet 只接了颜色分割模式。** 物体掩码来自颜色分割；在 Grounded-SAM2 模式下暂时只有几何候选。
- `push`、`rotate_held` 没有实现。旋转可以用 `carry_to` 的 `yaw` 完成。

## 计划中的 Hopper 运行（等你确认后提交）

- 队列：interactive，1 GPU，1.5 h。作业里会依次：
  1. 启动 Contact-GraspNet 服务。
  2. 跑脚本化门控（`scripts/mid_gate.py`，不调用模型）：积木、书、杯子各一个放置任务，外加一个容器被占用的清理任务；每个任务分别用几何候选和 GraspNet 候选各跑一次，共 8 次。
  3. 门控里至少 3 次真实状态评分成功，才启动模型。
- 模型：Qwen3-30B-A3B-Instruct-2507（bf16，vLLM，GPU 占用比例 0.70，给 GraspNet 留出显存），与 task 级 9/10 那次是同一个模型。
- 任务序列：和 task 级完全相同的 `session`（10 个任务）。条件：无记忆、只开技能记忆、只开用户记忆、两种都开，结果可以和 task 级的表逐项对比。
- 输出：`outputs/mid30b_<时间戳>/`，包括 `mid_gate/`、每个条件的 `summary.json` 和录像。
