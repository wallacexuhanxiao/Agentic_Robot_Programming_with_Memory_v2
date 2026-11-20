# "整理好"的量化定义（2026-11-11）

老师的意见：报告里的 tidy 没有量化，写得不明确。这里给出一个可以直接写进报告的定义，并用已有运行重新算了一遍。不调用模拟器和模型。

- **重算**：`python3 scripts/tidy_metrics.py`
- **输出**：
  - `results/dev_stats/tidy.csv`：每桌一行；
  - `results/dev_stats/tidy_objects.csv`：每件物品一行；
  - `results/dev_stats/tidy_summary.json`：汇总。

## 1. 定义

一张桌子"整理好"，指桌上每件可移动的东西，都在它的归属位置里，并且是那个位置要求的姿态，同时机械臂手里没有拿东西。

| 符号 | 含义 |
|---|---|
| O | 桌上所有可移动的东西。开局不在位的记作 O_out（要整理的）；开局已在位的记作 O_in（不该被碰乱的）。 |
| h(o) | 东西 o 的归属位置。离线评分里取任务生成器给的目标映射：同类东西放在一起，例如笔进笔筒、本子上书堆。用户明确说过的偏好优先于这个默认映射。机器人看不到 h(o)，要自己从场景、指令和记忆里推断。 |
| Placed(o) | 用仿真真实位姿做的几何判定，标准见第 2 节。 |
| T(s) | 桌面状态 s 的整洁度：O 中满足 Placed 的比例。开局记作 T0（等于 \|O_in\| / \|O\|），结束记作 T1。 |
| G | 整洁度增益，即开局的乱有多少被整理掉，扣除碰乱的：G = (T1 − T0) / (1 − T0) = (放好的 O_out 件数 − 被碰乱的 O_in 件数) / \|O_out\|。全部整理好、没碰乱任何东西时 G = 1。 |
| P、K | P：O_out 放好的比例。K：O_in 保持在位的比例。 |
| S | 整桌成功：T1 = 1，且手里没有东西。 |

报告用的英文写法：

> A desk state *s* is **tidy** when every movable thing *o* is in its home *h(o)* in the pose that home requires, and the gripper is empty. *h(o)* is where things of its kind are kept on this desk, overridden by a placement the user has stated; the robot is never told *h(o)* and must infer it. Tidiness *T(s)* is the fraction of things that pass the geometric test *Placed(o, h(o))* (Table X), computed from the simulator pose for scoring only. We report the start and end tidiness *T0* and *T1*, the **tidiness gain** *G = (T1 − T0)/(1 − T0)* (the share of the initial mess removed, net of things knocked out of place), placement rate *P* on things out of place, keep rate *K* on things already in place, and whole-desk success *S* (*T1 = 1*).

## 2. Placed 的几何判定

代码：`sma/env/robot_env.py: private_evaluate`。表里的数字都直接取自这段评分代码；对比的两组运行用的是同一套阈值。

| 归属位置 | 东西 | 在里面 | 落稳 | 姿态 |
|---|---|---|---|---|
| 废纸篓（口径 17 cm） | 纸团 | 中心到篓中心的距离 ≤ 半径 − 5 mm | 由篓壁保证 | 不要求 |
| 杯垫（9 cm） | 杯子 | 中心距杯垫中心 ≤ 3 cm | 底面在杯垫上，高出不超过 1 cm | 竖直，倾斜 < 25.8° |
| 笔筒（6 cm） | 笔 | 下端在筒内；中心在筒口外扩 1 cm 的范围内 | 下端离筒底 ≤ 2.4 cm | 与竖直方向夹角 ≤ 40°（斜靠筒壁最多约 37°） |
| 书堆（25 × 18 cm） | 本子 | 中心离书堆边缘至少 2 cm | 平躺在书堆顶上，高度误差 < 6 mm | 平放，倾斜 < 8.1° |
| 托盘（以前的桌面） | 本子 | 四个角都在托盘里 | 不搭在 8 mm 的边沿上 | 平放 |

机器人自己用相机做检查，判定的含义和这张表相同，但只能用分割和深度来量，比真值严格。第 1.4 节（DEV_DATA_zh.md）给出了两者的一致率。

## 3. 用已有运行重算

### 3.1 SAM2 真实感知桌（11-06 及以前跑完的 11 张，正式结果）

| | 值 |
|---|---|
| 每桌东西数 | 5–6 件：3 件不在位，2–3 件已在位 |
| 开局整洁度 T0 | 0.47 |
| 结束整洁度 T1 | **0.88** |
| 整洁度增益 G | **0.76**（最低 0.67） |
| 放好率 P | 25/33（76%） |
| 保持率 K | 30/30（100%） |
| 整桌成功 S | 3/11 |

- 每张桌的 G 不是 1 就是 0.67：要么全部整理好，要么 3 件里漏了 1 件。没有一张桌出现"整理反而更乱"。
- 没放好的 8 件，多数是本子，具体如下。"差多少"是物体中心超出可接受区域的距离；0 表示中心在区域内，但姿态或落稳没满足。
  - 本子 5 件：差 0.3、2.4、4.1、6.3、28.9 cm（28.9 cm 那件平放着，但不在书堆上）。
  - 笔 3 件：中心都在笔筒附近，但没有竖直立在筒里，例如横搭在筒口，或者倾斜超过 40°。

### 3.2 技能记忆 vs 无记忆：只算两组都跑完的桌

| 设置 | 桌数 | 无记忆 T1 / G / S | 技能记忆 T1 / G / S |
|---|---|---|---|
| Gemini，颜色感知 | 20 + 20 | 0.81 / 0.75 / 6 | **0.97 / 0.96 / 17** |
| Gemini + GraspNet/cuRobo | 10 + 10 | 0.79 / 0.72 / 2 | **0.93 / 0.90 / 7** |
| DeepSeek + SAM2 | 3 + 3 | 0.89 / 0.78 / 1 | 0.82 / 0.67 / 0 |

- 6 组里保持率 K 都是 100%，没有碰乱原本已经放好的东西。
- SAM2 这组只有 3 张桌，技能记忆少放好 1 件；技能组本子失败的原因，后来已经修掉，但这个修复不进报告。

### 3.3 修复之后的 SAM2 运行（11-08，4 张桌，仅供参考）

T0 0.40 → T1 0.75，G 0.58，S 1/4，K 7/8。demo_17200 碰乱了一支原本在笔筒里的笔，这是所有运行里唯一一次 K < 1。

## 4. 这个定义不包括什么

- **摆放的美观**：东西之间的对齐和间距、杯柄朝向等，只要满足第 2 节就算整理好。
- **桌上没有归属位置的东西**：笔记本电脑、台灯、手机是固定摆设，不计入 O。
- **归属位置本身对不对**：T 按 h(o) 计分，机器人推断去处的准确率要单独报。SAM2 桌上，机器人推断的去处有 32/33 与 h(o) 一致（DEV_DATA_zh.md 第 1.2 节）。在用户记忆实验里，h(o) 换成这个用户自己的偏好。
