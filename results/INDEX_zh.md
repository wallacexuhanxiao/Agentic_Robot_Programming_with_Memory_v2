# 结果目录索引（期中之后）

每个目录是一次运行的输出（Hopper `outputs/<运行名>` 下载而来）。仓库里只放轻量文件：`summary.json`、`episode.json`、日志、闸门行记录。
这几类文件只留在本地和 Hopper，不进仓库：
- 每次观测的 PNG；
- 模型原始调用记录 `llm/`（含 base64 图像，每个约 2–3 MB）；
- 掩码 `.npz`；
- 大部分录像。
文档引用到的图片和录像，以及 `results/demo/` 下的演示录像，单独加入了仓库。

"文档引用"一栏指出哪篇文档的哪一节用到了这个目录；空着的是中间版本或诊断运行，结论以文档为准。

## Gemini 3.8 Flash，act 级 API（11-04 至 11-05）

| 目录 | 文档引用 |
|---|---|
| `actgemini_desk2_20261104T123650Z` |  |
| `actgemini_desk3_20261104T124912Z` |  |
| `actgemini_desk_20261104T090847Z` |  |
| `actgemini_gpu2_realdesk_none10_20261104T182458Z` |  |
| `actgemini_gpu3_realdesk_skill10_20261104T184856Z` |  |
| `actgemini_holder3_20261104T090704Z` |  |
| `actgemini_holder4_20261104T115318Z` |  |
| `actgemini_holder6000skill_20261104T094037Z` |  |
| `actgemini_holder_20261104T083948Z` |  |
| `actgemini_nb2_20261105T014714Z` |  |
| `actgemini_nbpile2_20261105T032223Z` | REALDESK_TASK_zh.md · 改进接口后重跑（`actgemini_nbpile2_20261105T032223Z`，只跑技能组，5 个桌面，$0.20） |
| `actgemini_nbpile_20261105T021138Z` |  |
| `actgemini_realdesk2_none_20261104T144951Z` |  |
| `actgemini_realdesk2_skill_20261104T144951Z` |  |
| `actgemini_realdesk2_skillrecent_20261104T152759Z` |  |
| `actgemini_realdesk3_none_20261104T155202Z` |  |
| `actgemini_realdesk3_skill10_20261104T155202Z` | REALDESK_TASK_zh.md · 技能记忆 vs 无记忆，真实视觉 4+4 桌（2026-11-06） |
| `actgemini_realdesk4_none10_20261104T164331Z` |  |
| `actgemini_realdesk4_skill10_20261104T164331Z` |  |
| `actgemini_realdesk_none_20261104T134601Z` |  |
| `actgemini_realdesk_skill_20261104T140236Z` |  |

## DeepSeek deepseek-flash，SAM2 真实视觉（11-06）

| 目录 | 文档引用 |
|---|---|
| `ds_cmp_none_20261106T115457Z` | REALDESK_TASK_zh.md · 技能记忆 vs 无记忆，真实视觉 4+4 桌（2026-11-06） |
| `ds_cmp_skill_20261106T115457Z` | REALDESK_TASK_zh.md · 技能记忆 vs 无记忆，真实视觉 4+4 桌（2026-11-06） |
| `ds_full2_sdesk_20261106T080723Z` | ARCHITECTURE_zh.md · 8. 已有证据（截至 11-06）；REALDESK_TASK_zh.md · DeepSeek 全流程 |
| `ds_full3_sdesk_20261106T104702Z` | REALDESK_TASK_zh.md · DeepSeek 第三次（`ds_full3_sdesk_20261106T104702Z`，条件 both，$0.60 到上限停止） |
| `ds_full_sdesk_20261106T063745Z` | REALDESK_TASK_zh.md · DeepSeek 全流程 |
| `ds_groupseq` |  |

## DeepSeek 演示运行，SAM2 真实视觉（11-08）

演示录像（原始、英文字幕、中文字幕三版）和每段的结果说明在 `results/demo/real_perception/`（见其中的 README.md）。

| 目录 | 文档引用 |
|---|---|
| `demo_real_20261108T083052Z` | results/demo/real_perception/README.md（demo_17200、demo_17201、demo_17202） |
| `varied_demo_20261108T123917Z` | results/demo/real_perception/README.md（varied_17308：多样布局桌） |

## SAM2 感知与跟踪闸门，无模型（11-05）

| 目录 | 文档引用 |
|---|---|
| `gate_expect_20261105T182218Z` |  |
| `gate_expect_20261105T190232Z` |  |
| `gate_final_auto_20261105T201603Z` |  |
| `gate_final_interactive_20261105T201603Z` |  |
| `gate_look_20261105T192837Z` |  |
| `gate_look_20261105T194617Z` |  |
| `gate_parts_auto_20261105T203529Z` |  |
| `gate_parts_interactive_20261105T203529Z` |  |
| `gate_persist_20261105T214750Z` | REALDESK_TASK_zh.md · 真实视觉：SAM2 + 深度（2026-11-05） |
| `gate_persist_auto_20261105T220532Z` | REALDESK_TASK_zh.md · 真实视觉：SAM2 + 深度（2026-11-05） |
| `gate_persist_interactive_20261105T220532Z` | REALDESK_TASK_zh.md · 真实视觉：SAM2 + 深度（2026-11-05） |

## 名字跟踪诊断，无模型（11-05）

| 目录 | 文档引用 |
|---|---|
| `track_both_auto_20261105T205203Z` |  |
| `track_both_auto_20261105T211058Z` |  |
| `track_both_auto_20261105T212945Z` |  |
| `track_both_interactive_20261105T205203Z` |  |
| `track_both_interactive_20261105T211057Z` |  |
| `track_both_interactive_20261105T212945Z` |  |
| `track_debug_20261105T180958Z` |  |
| `track_debug_20261105T185307Z` |  |
| `track_nb4_20261105T200408Z` |  |
| `track_persist_20261105T214750Z` | REALDESK_TASK_zh.md · 真实视觉：SAM2 + 深度（2026-11-05） |
| `track_sam5_20261105T200408Z` |  |
| `track_sam_20261105T191915Z` |  |
| `track_sam_20261105T192837Z` |  |

## SAM2 感知诊断与闸门，无模型

| 目录 | 文档引用 |
|---|---|
| `sam_debug2_20261106T092805Z` |  |
| `sam_debug3_20261106T100123Z` | REALDESK_TASK_zh.md · 感知碎块与跟踪（2026-11-06 晚） |
| `sam_debug_20261106T091449Z` | REALDESK_TASK_zh.md · 感知碎块与跟踪（2026-11-06 晚） |
| `sam_gate_20261105T110405Z` |  |

## 分割修复闸门，无模型（11-06）

| 目录 | 文档引用 |
|---|---|
| `seg_fix2_gate_20261106T102958Z` | REALDESK_TASK_zh.md · 感知碎块与跟踪（2026-11-06 晚） |
| `seg_fix_gate_20261106T092805Z` |  |

## 抓笔诊断与修复闸门，无模型（11-06）

| 目录 | 文档引用 |
|---|---|
| `pen_close_20261106T045010Z` | REALDESK_TASK_zh.md · 抓笔为什么全失败、怎么修的（闸门，参考程序，无模型） |
| `pen_contact_20261106T022750Z` | REALDESK_TASK_zh.md · 结果（`results/mesh_bin_20261106T014823Z`，作业 703177，同 8 张撞色桌，新布局） |
| `pen_fix2_20261106T062510Z` | REALDESK_TASK_zh.md · 抓笔为什么全失败、怎么修的（闸门，参考程序，无模型） |
| `pen_fix_20261106T050358Z` |  |

## 本子侧夹闸门与探测，无模型（11-05）

| 目录 | 文档引用 |
|---|---|
| `notebook_gate3_20261105T014150Z` |  |
| `notebook_gate_20261105T013326Z` |  |
| `notebook_pile_gate_20261105T020725Z` |  |
| `notebook_probe_20261105T012327Z` |  |

## 侧夹搬运闸门，无模型（11-06）

| 目录 | 文档引用 |
|---|---|
| `side_carry_dev_20261106T140810Z` |  |
| `side_carry_test2_20261106T143521Z` |  |
| `side_carry_test_20261106T140810Z` |  |

## 多样布局闸门，无模型（11-08）

`sdesk_varied` 桌（本子在任一侧桌边、笔在书堆上、收纳物位置分散）上的 SAM2 分割、初始检查和参考程序。第二次是在抓取高度改为从物体下方支撑面算起之后。

| 目录 | 文档引用 |
|---|---|
| `varied_gate_20261108T120741Z` | results/demo/real_perception/README.md |
| `varied_gate2_20261108T122424Z` | results/demo/real_perception/README.md |

## 视觉接入试验（11-05）

| 目录 | 文档引用 |
|---|---|
| `vision_ab_20261105T113647Z` |  |
| `vision_plain_20261105T180646Z` |  |
| `vision_plain_20261105T182323Z` | REALDESK_TASK_zh.md · 真实视觉：SAM2 + 深度（2026-11-05） |
| `vision_real_20261105T120258Z` | REALDESK_TASK_zh.md · 真实视觉：SAM2 + 深度（2026-11-05） |
| `vision_smoke` |  |

## 物品识别试验

| 目录 | 文档引用 |
|---|---|
| `identify35` |  |
| `identify35b` |  |
| `identifylite` |  |
| `identifytags` |  |
| `identifytags2` |  |

## 场景外观预览

| 目录 | 文档引用 |
|---|---|
| `look_mesh_20261106T014444Z` | REALDESK_TASK_zh.md · 结果（`results/mesh_bin_20261106T014823Z`，作业 703177，同 8 张撞色桌，新布局） |
| `look_preview` |  |
| `look_preview2` |  |

## 真实感桌面参考程序闸门

| 目录 | 文档引用 |
|---|---|
| `realdesk_gate3_20261104T133944Z` |  |
| `realdesk_gate_20261104T132730Z` |  |

## 其他试跑与早期运行

| 目录 | 文档引用 |
|---|---|
| `act_level` | REALDESK_TASK_zh.md · 动作空间的使用统计（`scripts/action_space_stats.py`，CPU 运行） |
| `cdeskcheck` |  |
| `coder_20261017T070859Z` |  |
| `cpuposegate_20261104T071235Z` |  |
| `curobogate_20261104T090343Z` |  |
| `final_20261017T082354Z` |  |
| `freelite2_cdesk` |  |
| `freelite_cdesk` |  |
| `grouping` | ARCHITECTURE_zh.md · 1. 一张桌子的完整流程（`sma/agent/loop.py: run_episode`）；ARCHITECTURE_zh.md · 2. 模块分层；REALDESK_TASK_zh.md · 场景分组：按"同类放一起"推断去处（类似 TidyBot，`sma/agent/grouping.py`） |
| `mesh_bin_20261106T014823Z` | REALDESK_TASK_zh.md · 结果（`results/mesh_bin_20261106T014823Z`，作业 703177，同 8 张撞色桌，新布局） |
| `ownsize_gate_20261106T101133Z` |  |
| `paid_full_cdesk` |  |
| `paid_lh_realdesk` |  |
| `pose30b_20261017T094212Z` |  |
| `q30b_20261017T080948Z` |  |
| `reflect_20261017T072611Z` |  |
| `replaypen_20261017T190053Z` |  |
| `standgate_20261104T073331Z` |  |
| `vlmtrial_20261104T063952Z` |  |
