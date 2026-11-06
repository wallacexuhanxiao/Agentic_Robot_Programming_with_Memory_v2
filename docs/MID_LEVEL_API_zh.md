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
| `grasp(candidate)` | 张开，在安全高度移到上方，竖直下降，闭合，抬到桌面以上 0.18 m；抬起后用相机测物体在手中的位姿 | `holding_something`、`finger_gap`、`expected_squeeze_width`、`object_in_hand` |
| `free_spots(zone_name, object_name=None)` | 在容器里为物体外形找放得下、不碰其他物体的位置（沿容器长轴或垂直两种朝向） | `occupants`（容器里已有的物体）、`spots`（每项含 `x, y`、`object_long_axis_yaw`、`clearance`）；放不下时 `spots` 为空 |
| `place_object(object_name, x, y, object_yaw=None)`（2026-11-04 新增） | 按物体位姿放置：物体中心到 (x, y)，长轴到 `object_yaw`（物体的朝向，不是夹爪朝向）；系统按手中位姿换算夹爪位姿，搬运、下降、松手、退让，再用相机测放好后的位姿 | `measured`（实测中心和长轴）、`position_error_m`、`yaw_error_deg`、`holding_something` |
| `carry_to(x, y, height=0.18, yaw=None)` | 先竖直升到桌面以上 `height`，再水平移动到 (x, y)；`yaw` 是夹爪的绝对朝向 | 位置、是否仍拿着 |
| `rotate_by(delta)` | 原地把夹爪和手里的物体绕竖直轴转 `delta` | 转前后的夹爪朝向、实际转角 |
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

## 2026-11-04 更新：GraspNet 坐标系修复与 `place_object`

### GraspNet 为什么之前一个候选都没给

- 已有运行里一共调用了 375 次 `grasp_candidates`，返回的候选中 GraspNet 来源的是 0 个，所有抓取都来自几何候选。
- 原因（读代码得出，门控作业在验证）：
  - 之前把**世界坐标系**下的点云直接交给 GraspNet 服务，桌面高度约 0.8 m。
  - 网络是在相机坐标系下训练的，默认相机在原点、沿 +z 方向看。在世界坐标里，这等于从地面往上看。
  - 第一次没有结果时，服务会在原点周围 ±0.25 m 内放随机虚拟相机重试，这些位置也都在桌面以下。
  - 所以返回的抓取都是从下往上接近，全被"与竖直方向夹角 ≤30°"的过滤条件去掉了。
- 修复（`_graspnet_raw`）：把物体点云和场景点云转换到真实俯视相机的坐标系（z 轴为观察方向）再发送，返回的抓取再转回世界坐标。旧调用方式保留为 `view="world"`，只用于门控对比。

### `place_object` 的设计

- **为什么加：** 笔任务里，模型主要错在把夹爪朝向当成笔的朝向。按物体位姿放置后，模型只需要给出"物体中心到哪、长轴朝哪"（`free_spots` 返回的 `x, y, object_long_axis_yaw` 可以直接用），从物体位姿到夹爪位姿的换算由系统完成。
- **手中位姿：**
  - `grasp` 抬起后，用相机测被抓物体：取高出桌面 4 cm 以上、颜色匹配的像素做最小外接矩形。夹爪挡住中间部分，但两端可见。
  - 由此得到物体中心相对抓取点的偏移（在夹爪坐标系里）和"物体长轴 − 夹爪朝向"。
  - 相机看不到时，退回抓取前的物体位姿估计。
  - 结果在 `grasp` 返回值的 `object_in_hand` 里。
- **换算：**
  - 夹爪朝向 = 目标物体朝向 − 手中角度差。手指两侧对称，所以取离初始朝向最近的等价角，避免手腕转过头。
  - 夹爪位置 = 目标中心 − 旋转后的手中偏移。
- **执行：** 复用已有的 `carry_to`（0.18 m 高度）、`lower_until_contact`、`open_gripper`、`retreat`，最后停到一边，用相机测物体实际位置，返回误差。
- **不变的：**
  - 原来的 `carry_to`、`rotate_by` 等低一级的调用仍然可用。
  - 给模型的通用顺序先验改为优先用 `place_object`，同时说明可以用低一级调用做更细的控制。
- **信息边界不变：** 只用相机和机器人本体状态，仿真真实状态只在门控里用来离线评估手中位姿测量的准确性。

### 门控（`scripts/pose_gate.py`，不调用模型）

- **任务：** 6 个笔任务（`pen_2000`–`2005`）+ 3 个单物体放置（积木、书、杯子，与整理会话的 place_1–3 相同）。
- **第 1 部分：** 同一份物体点云，分别用旧的世界坐标和新的相机坐标调用 GraspNet，统计返回数量、接近方向与竖直方向的夹角、能通过 30° 过滤的数量。
- **第 2 部分：** 每个任务分别用 GraspNet 候选和几何候选（各取第一个两指放得下的）抓取，然后对第一个空位调用 `place_object`。记录以下内容（仿真真值只用于离线评分）：
  - 是否抓住；
  - 手中位姿测量与仿真真值的误差（只对笔统计）；
  - 放置后的位置和朝向误差；
  - 相机检查结果；
  - 仿真真值评分。
- 每个任务在单独的进程里运行（以前长时间运行的模拟进程崩溃过），之后再跑一遍 `mid_gate.py` 作为回归检查。
- **作业：** `699728`（small 队列，经 `SMA_QUEUE=auto` 路由；先投的 interactive 作业 `699714` 排在另一个项目后面，已撤销），输出在 `outputs/posegate_20261104T061102Z/`。

## 2026-11-04 更新（二）：cuRobo 与 Gemini

### cuRobo（`SMA_CUROBO=1` 时在 mid 级启动 ASPIRE 的 cuRobo 服务）

- **物理可行性检查** `physical_feasibility(object, zone)`：技能记忆适用性判断的物理层。
  - 用 cuRobo 对三个夹爪朝下的位姿做考虑碰撞的 IK：物体上方 6 cm、容器第一个空位上方（桌面以上 0.18 m）、空位处物体底部离容器底 2 cm 的位置。
  - 障碍物是俯视深度点云（服务会扣掉机器人自身的点；待抓物体的点也事先去掉），桌面也在点云里。
  - 三个位姿都有解才算可行。不可行时技能判为 `not_applicable`，原因写明是哪个位姿没有解。
  - 服务不可用时，退回原来"落在以前成功过的范围内"的判断。两种结果都写进每次尝试的记录，方便算适用性判断的准确率。
- **运动后端**（`SMA_MOTION=curobo`）：
  - `carry_to` 的水平搬运改用 cuRobo 规划考虑碰撞的轨迹；手里的物体不算障碍物。
  - 规划失败时退回原来的直线运动，返回值里的 `planner` 字段写明实际用了哪种。
  - 竖直下降和抬起仍用直线（需要精确）。
  - 局限：cuRobo 不知道手里拿着物体，物体本身可能碰到东西。
- **门控** `scripts/curobo_gate.py`（作业 `699767`）：
  - A 部分：可行性判断与脚本化放置的实际结果对照。任务包括 3 个笔任务、3 个单物体放置、3 个"容器先被占用"的清理任务。
  - B 部分：6 个任务用 cuRobo 搬运跑一遍。

### Gemini（`SMA_LLM_PROVIDER=gemini`）

- 走 Gemini 的 OpenAI 兼容接口，默认模型 `gemini-3.8-flash`，思考深度 low。
- 密钥从 `SMA_API_KEY_FILE` 指向的文件读取，不写进日志。
- **花费账本**：
  - 每次运行一个 `spend.json`，按 token 用量和单价实时累计（默认每百万 token 输入 0.75、输出 3.75 美元，思考 token 按输出计）。
  - 达到 `SMA_SPEND_CAP_USD`（默认 1.5 美元）就停止整个运行，并在 `summary.json` 里注明。
  - 上限在每次调用前检查，所以最多超出一次调用的花费。
- 遇到限速或服务端错误时自动退避重试。
- `SMA_MODE=gemini_probe`：只测计算节点能否连上 Gemini，并做一次极小的调用（约 0.0001 美元）。
- **VLM 成功判定**：
  - 每次相机检查时，额外保存一张放大的俯视相机图（512 像素，裁到容器周围 15 cm），存在任务目录的 `check_views/` 下。
  - `scripts/vlm_check_eval.py` 把每个物体最后一张图交给 VLM 判断，与仿真真值、规则检查对比，并带花费上限。
