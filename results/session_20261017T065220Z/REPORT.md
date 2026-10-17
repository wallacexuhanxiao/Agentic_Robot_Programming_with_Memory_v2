# Run session_20261017T065220Z

## session

| metric | none | skill | user | both |
|---|---|---|---|---|
| tasks | 10 | 10 | 10 | 10 |
| task_success_gt | 0/10 | 0/10 | 3/10 | 1/10 |
| subgoals_verified | 5/14 | 3/14 | 8/16 | 6/15 |
| code_generations | 37 | 37 | 38 | 35 |
| failed_executions | 32 | 34 | 30 | 29 |
| skill_calls | 0 | 0 | 0 | 0 |
| llm_calls | 57 | 59 | 68 | 66 |
| completion_tokens | 6927 | 9108 | 7654 | 6490 |
| control_steps | 7876 | 8580 | 7885 | 8752 |
| wall_seconds | 94 | 109 | 99 | 93 |

## Per task

| condition | task | GT success | subgoals ok | code gens | failed execs | skill calls |
|---|---|---|---|---|---|---|
| none | teach_4 | False | 1/2 | 4 | 3 | 0 |
| none | place_1 | False | 0/1 | 3 | 3 | 0 |
| none | place_2 | False | 0/1 | 3 | 3 | 0 |
| none | place_3 | False | 0/1 | 3 | 3 | 0 |
| none | tidy_1000 | False | 1/2 | 4 | 3 | 0 |
| none | tidy_1001 | False | 0/1 | 3 | 3 | 0 |
| none | tidy_1002 | False | 1/2 | 6 | 5 | 0 |
| none | tidy_1003 | False | 0/1 | 3 | 3 | 0 |
| none | tidy_1004 | False | 2/2 | 5 | 3 | 0 |
| none | tidy_1005 | False | 0/1 | 3 | 3 | 0 |
| skill | teach_4 | False | 0/2 | 6 | 6 | 0 |
| skill | place_1 | False | 0/1 | 3 | 3 | 0 |
| skill | place_2 | False | 0/1 | 3 | 3 | 0 |
| skill | place_3 | False | 0/1 | 3 | 3 | 0 |
| skill | tidy_1000 | False | 0/2 | 6 | 6 | 0 |
| skill | tidy_1001 | False | 0/1 | 3 | 3 | 0 |
| skill | tidy_1002 | False | 1/2 | 5 | 4 | 0 |
| skill | tidy_1003 | False | 0/1 | 3 | 3 | 0 |
| skill | tidy_1004 | False | 2/2 | 2 | 0 | 0 |
| skill | tidy_1005 | False | 0/1 | 3 | 3 | 0 |
| user | teach_4 | False | 0/2 | 6 | 6 | 0 |
| user | place_1 | True | 1/1 | 2 | 1 | 0 |
| user | place_2 | False | 0/1 | 3 | 3 | 0 |
| user | place_3 | False | 0/1 | 3 | 3 | 0 |
| user | tidy_1000 | False | 1/2 | 5 | 4 | 0 |
| user | tidy_1001 | False | 1/2 | 5 | 4 | 0 |
| user | tidy_1002 | False | 1/2 | 6 | 5 | 0 |
| user | tidy_1003 | False | 0/1 | 3 | 3 | 0 |
| user | tidy_1004 | True | 2/2 | 3 | 1 | 0 |
| user | tidy_1005 | True | 2/2 | 2 | 0 | 0 |
| both | teach_4 | False | 1/2 | 6 | 5 | 0 |
| both | place_1 | False | 0/1 | 3 | 3 | 0 |
| both | place_2 | False | 0/1 | 3 | 3 | 0 |
| both | place_3 | True | 1/1 | 1 | 0 | 0 |
| both | tidy_1000 | False | 1/2 | 4 | 3 | 0 |
| both | tidy_1001 | False | 1/2 | 4 | 3 | 0 |
| both | tidy_1002 | False | 1/2 | 4 | 3 | 0 |
| both | tidy_1003 | False | 0/1 | 3 | 3 | 0 |
| both | tidy_1004 | False | 1/2 | 4 | 3 | 0 |
| both | tidy_1005 | False | 0/1 | 3 | 3 | 0 |

## Skill memory at end of run (skill)

- place_on_zone_v1 (validated), from tidy_1002: cup 3 ok/0 fail
- place_on_zone_v2 (validated), from tidy_1004: cube 3 ok/0 fail

## Skill memory at end of run (both)

- place_on_zone_v1 (validated), from teach_4: cup 3 ok/0 fail
