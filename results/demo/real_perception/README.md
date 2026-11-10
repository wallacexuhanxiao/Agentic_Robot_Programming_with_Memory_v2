# Demo clips: realistic desk, real perception

All clips are from the realistic `sdesk` scene (MuJoCo/Robosuite Panda). The robot sees the desk only through its cameras: SAM2 class-agnostic segmentation and depth. DeepSeek identifies the things on the desk and writes the robot programs. Simulator ground truth is never given to the robot. It is used only for the offline score on each clip's closing card.

Each clip has three versions with the same file name:

- `raw/`: the episode's own `video.mp4`, byte-identical, with no captions.
- `en/`: English captions, for the talk.
- `zh/`: Chinese captions.

The captioned cuts are made with `scripts/make_demo_video.py EPISODE_DIR OUT.mp4 en|zh`. They add:

- a title card showing what SAM2 separated, what the model recognized, and the goals it inferred;
- one caption per attempt;
- a closing card with the robot's own camera checks and the offline ground-truth score.

## Results

| Clip | Memory | Out-of-place items (offline GT) | Already-tidy items kept | Whole desk | LLM calls | Source episode |
|---|---|---|---|---|---|---|
| `full2_17000` | skills | pen ✓, notebook ✓, paper ball ✓ | 3/3 | **success** | 29 | `results/ds_full2_sdesk_20261106T080723Z/both/sdesk_vague_anchors_17000` |
| `cmp_none_17102` | none | mug ✓, notebook ✓, paper ball ✓ | 2/2 | **success** | 42 | `results/ds_cmp_none_20261106T115457Z/none/sdesk_vague_anchors_17102` |
| `demo_17201` | skills | mug ✓, notebook ✓, paper ball ✓ | 2/2 | **success** | 16 | `results/demo_real_20261108T083052Z/skill/sdesk_vague_anchors_17201` |
| `cmp_skill_17101` | skills | pen ✓ (stored skill reused), notebook ✗, paper ball ✓ | 3/3 | partial | 18 | `results/ds_cmp_skill_20261106T115457Z/skill/sdesk_vague_anchors_17101` |
| `demo_17200` | skills | pen ✓, mug ✓, notebook ✗ | 1/2 | partial | 38 | `results/demo_real_20261108T083052Z/skill/sdesk_vague_anchors_17200` |
| `demo_17202` | skills | paper ball ✓, mug ✗, notebook ✗ | 2/2 | partial | 28 | `results/demo_real_20261108T083052Z/skill/sdesk_vague_anchors_17202` |
| `varied_17308` | skills | pen ✓ (on the pile at the start; stored skill reused), mug ✓, notebook ✗ (on the pile, 6 mm outside the tolerance) | 2/2 | partial | 24 | `results/varied_demo_20261108T123917Z/skill/sdesk_vague_anchors_varied_17308` |

These clips are examples, not a benchmark.

The robot's own camera check is stricter than the offline score. In three clips it rejected placements that ground truth counts as correct:

- `full2_17000`: camera 2 of 3, ground truth 3 of 3.
- `cmp_none_17102`: camera 2 of 5 (the plan also retargeted some items), ground truth 3 of 3.
- `demo_17200`: camera 1 of 3, ground truth 2 of 3.

In `varied_17308` the two disagree both ways. The camera rejected the pen, which ground truth counts as placed, and accepted the notebook, which ground truth does not.

A rejected placement makes the robot try again, and in `demo_17200` the retry did damage.

## Notes on the partial clips

- **`demo_17200`**:
  - The reused pen skill did put the pen into the holder on the first try (offline GT agrees). The camera check still said "not inside". The holder already held a second pen, the two leaned apart in a V, and the measured span (about 7.8 cm) was wider than the holder mouth (about 5.8 cm).
  - The retries that followed pushed the pen that was already there out of the holder.
  - The notebook never reached the pile: no clear top-down grasp, motion blocked, and a missed grasp.
- **`demo_17202`**:
  - The model recognized the cork coaster (tag 6 on the title card) as a "book". So it never saw a destination for the mug, which it left alone.
  - It instead tried to move the "book", which was really the coaster, onto the book pile. All six tries failed.
  - The notebook was lost from view after a failed carry.

## The varied desk (`varied_17308`)

On the standard `sdesk` desks the layout changes very little between seeds:

- the notebook always lies at the robot's right desk edge, 4-6 cm out over it;
- the receptacles stand in one row at the back.

The `sdesk_varied` suite (`sma/env/tasks.py`) changes the positions:

- the notebook lies at either side edge, drawn per seed, and sticks out 5.5-6.5 cm of its 14.8 cm width;
- the receptacles may stand anywhere in reach;
- the pen lies on top of the pile of books, where the notebook has to go.

In `varied_17308`:

- **Notebook**: at the left edge, the opposite side from all the other clips.
- **Pen**:
  - The skill stored on earlier desks, where pens lay on the table, was reused as is, with no LLM call, and stood the pen in the holder.
  - Picking a pen up from a pile of books needed one change to the robot API, made the same day: grasp heights are now measured from the surface under the object instead of from the table.
- **Notebook placement**:
  - The notebook ended flat on the pile, but its centre was 7.6 cm from the pile's centre across the pile.
  - The offline score allows 7.0 cm (2 cm in from the pile's edge), so it counts as not placed. The robot's camera check accepted it.

Perception limits seen on varied desks, in the no-LLM gate (`results/varied_gate2_20261108T122424Z`):

- On 2 of 6 desks, SAM2 merged a pen lying on the pile with the pile.
- On other desks the pile itself was not separated.
- The result also varied between two runs on the same desk.
