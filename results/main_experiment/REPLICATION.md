# Pen-case memory experiment: replication

Ten pen tasks per run (the teaching task is excluded). Mean ± sd (min–max) over runs.

## Sequence: teaching task + 10 pen tasks

| metric | none (n=3) | skill (n=3) | user (n=4) | both (n=4) |
|---|---|---|---|---|
| success | 2.7 ± 0.6 (2–3) | 7.3 ± 2.5 (5–10) | 4.2 ± 1.3 (3–6) | 6.0 ± 4.2 (0–10) |
| solved at first attempt | 2.3 ± 1.2 (1–3) | 7.0 ± 2.0 (5–9) | 3.2 ± 1.9 (2–6) | 6.0 ± 4.2 (0–10) |
| failed executions | 22.7 ± 2.9 (21–26) | 8.3 ± 7.0 (1–15) | 18.5 ± 4.4 (12–21) | 12.0 ± 12.7 (0–30) |
| attempts | 25.3 ± 2.3 (24–28) | 15.7 ± 4.5 (11–20) | 22.8 ± 3.2 (18–25) | 18.0 ± 8.5 (10–30) |
| LLM calls | 90.7 ± 8.1 (86–100) | 47.7 ± 19.6 (27–66) | 89.8 ± 11.9 (72–97) | 68.0 ± 36.6 (34–120) |
| tokens | 9964.3 ± 819.1 (9128–10765) | 7370.7 ± 2887.1 (4171–9781) | 10753.8 ± 1913.2 (7975–12328) | 8923.8 ± 5601.9 (4307–17082) |
| direct skill reuse | 0.0 ± 0.0 (0–0) | 6.0 ± 2.0 (4–8) | 0.0 ± 0.0 (0–0) | 4.0 ± 2.7 (0–6) |

## Sequence: 10 pen tasks (no teaching task)

| metric | none (n=1) | skill (n=2) |
|---|---|---|
| success | 0 | 1.0 ± 1.4 (0–2) |
| solved at first attempt | 0 | 0.5 ± 0.7 (0–1) |
| failed executions | 30 | 27.5 ± 3.5 (25–30) |
| attempts | 30 | 28.5 ± 2.1 (27–30) |
| LLM calls | 110 | 103.0 ± 9.9 (96–110) |
| tokens | 11697 | 12897.5 ± 204.4 (12753–13042) |
| direct skill reuse | 0 | 1.0 ± 1.4 (0–2) |

## Per run

| layout set | sequence | condition | success | first attempt | failed | LLM calls | skill reuse | dir |
|---|---|---|---|---|---|---|---|---|
| A | plain | none | 0/10 | 0 | 30 | 110 | 0 | `pen30b_20261017T152702Z/none` |
| A | plain | skill | 2/10 | 1 | 25 | 96 | 2 | `pen30b_skill_20261017T155808Z/skill` |
| A | plain | skill | 0/10 | 0 | 30 | 110 | 0 | `pen30b_20261017T152702Z/skill` |
| A | teach | none | 2/10 | 1 | 26 | 100 | 0 | `penrep_A_none_20261017T172651Z/none` |
| A | teach | skill | 7/10 | 7 | 9 | 50 | 6 | `penrep_A_skill_20261017T172651Z/skill` |
| A | teach | user | 4/10 | 2 | 21 | 97 | 0 | `penteach30b_user_20261017T162704Z/user` |
| A | teach | user | 4/10 | 2 | 20 | 94 | 0 | `penrep_A2_user_20261017T172651Z/user` |
| A | teach | both | 7/10 | 7 | 9 | 59 | 5 | `penteach30b_both_20261017T162704Z/both` |
| A | teach | both | 7/10 | 7 | 9 | 59 | 5 | `penrep_A2_both_20261017T172651Z/both` |
| B | teach | none | 3/10 | 3 | 21 | 86 | 0 | `penrep_B_none_20261017T172651Z/none` |
| B | teach | skill | 10/10 | 9 | 1 | 27 | 8 | `penrep_B_skill_20261017T172651Z/skill` |
| B | teach | user | 6/10 | 6 | 12 | 72 | 0 | `penrep_B_user_20261017T172651Z/user` |
| B | teach | both | 10/10 | 10 | 0 | 34 | 6 | `penrep_B_both_20261017T172651Z/both` |
| C | teach | none | 3/10 | 3 | 21 | 86 | 0 | `penrep_C_none_20261017T172651Z/none` |
| C | teach | skill | 5/10 | 5 | 15 | 66 | 4 | `penrep_C_skill_20261017T172651Z/skill` |
| C | teach | user | 3/10 | 3 | 21 | 96 | 0 | `penrep_C_user_20261017T172651Z/user` |
| C | teach | both | 0/10 | 0 | 30 | 120 | 0 | `penrep_C_both_20261017T172651Z/both` |

## Paired by layout set (teaching sequence): success of both − max(user, skill)

- set A: none 2, skill 7.0, user 4.0, both 7.0 (both − max(user, skill) = +0.0; both − (user + skill) = -4.0)
- set B: none 3, skill 10.0, user 6.0, both 10.0 (both − max(user, skill) = +0.0; both − (user + skill) = -6.0)
- set C: none 3, skill 5.0, user 3.0, both 0.0 (both − max(user, skill) = -5.0; both − (user + skill) = -8.0)
