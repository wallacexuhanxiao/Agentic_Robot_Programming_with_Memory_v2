# Run final_20261017T084229Z

## session

| metric | none | skill | user | both |
|---|---|---|---|---|
| tasks | 10 | 10 | 10 | 10 |
| task_success_gt | 4/10 | 5/10 | 9/10 | 9/10 |
| subgoals_verified | 11/17 | 11/16 | 15/18 | 15/18 |
| code_generations | 31 | 28 | 33 | 26 |
| failed_executions | 20 | 17 | 18 | 11 |
| skill_calls | 0 | 7 | 0 | 7 |
| llm_calls | 87 | 66 | 95 | 70 |
| completion_tokens | 10479 | 8850 | 9917 | 7205 |
| control_steps | 9740 | 8164 | 10886 | 8404 |
| wall_seconds | 132 | 107 | 139 | 111 |

## Per task

| condition | task | GT success | subgoals ok | code gens | failed execs | skill calls |
|---|---|---|---|---|---|---|
| none | teach_4 | False | 1/2 | 4 | 3 | 0 |
| none | place_1 | True | 1/1 | 1 | 0 | 0 |
| none | place_2 | True | 1/1 | 1 | 0 | 0 |
| none | place_3 | True | 1/2 | 4 | 3 | 0 |
| none | tidy_1000 | True | 2/2 | 3 | 1 | 0 |
| none | tidy_1001 | False | 1/1 | 1 | 0 | 0 |
| none | tidy_1002 | False | 1/3 | 8 | 7 | 0 |
| none | tidy_1003 | False | 1/2 | 4 | 3 | 0 |
| none | tidy_1004 | False | 1/2 | 4 | 3 | 0 |
| none | tidy_1005 | False | 1/1 | 1 | 0 | 0 |
| skill | teach_4 | True | 2/2 | 4 | 2 | 0 |
| skill | place_1 | True | 1/1 | 1 | 0 | 0 |
| skill | place_2 | True | 1/1 | 1 | 0 | 0 |
| skill | place_3 | True | 1/1 | 1 | 0 | 1 |
| skill | tidy_1000 | False | 1/2 | 4 | 3 | 1 |
| skill | tidy_1001 | False | 1/1 | 1 | 0 | 0 |
| skill | tidy_1002 | False | 0/3 | 9 | 9 | 1 |
| skill | tidy_1003 | False | 1/2 | 4 | 3 | 1 |
| skill | tidy_1004 | True | 2/2 | 2 | 0 | 2 |
| skill | tidy_1005 | False | 1/1 | 1 | 0 | 1 |
| user | teach_4 | True | 2/2 | 4 | 2 | 0 |
| user | place_1 | True | 1/1 | 1 | 0 | 0 |
| user | place_2 | True | 1/1 | 1 | 0 | 0 |
| user | place_3 | True | 1/1 | 2 | 1 | 0 |
| user | tidy_1000 | True | 2/2 | 3 | 1 | 0 |
| user | tidy_1001 | True | 2/2 | 4 | 2 | 0 |
| user | tidy_1002 | True | 2/3 | 6 | 4 | 0 |
| user | tidy_1003 | False | 0/2 | 6 | 6 | 0 |
| user | tidy_1004 | True | 2/2 | 3 | 1 | 0 |
| user | tidy_1005 | True | 2/2 | 3 | 1 | 0 |
| both | teach_4 | True | 2/2 | 2 | 0 | 0 |
| both | place_1 | True | 1/1 | 1 | 0 | 0 |
| both | place_2 | True | 1/1 | 1 | 0 | 0 |
| both | place_3 | True | 1/1 | 1 | 0 | 1 |
| both | tidy_1000 | True | 2/2 | 2 | 0 | 1 |
| both | tidy_1001 | True | 2/2 | 2 | 0 | 0 |
| both | tidy_1002 | True | 2/3 | 5 | 3 | 1 |
| both | tidy_1003 | False | 0/2 | 6 | 6 | 0 |
| both | tidy_1004 | True | 2/2 | 2 | 0 | 2 |
| both | tidy_1005 | True | 2/2 | 4 | 2 | 2 |

## Applicability checks

12/14 verdicts matched the outcome of the skill call (applicable -> success, uncertain/not_applicable -> failure).
- place_3: place_on_zone_v2 verdict=applicable outcome=ok
- tidy_1000: place_on_zone_v1 verdict=applicable outcome=ok
- tidy_1002: place_on_zone_v2 verdict=applicable outcome=fail
- tidy_1003: place_on_zone_v3 verdict=applicable outcome=ok
- tidy_1004: place_on_zone_v3 verdict=applicable outcome=ok
- tidy_1004: place_on_zone_v2 verdict=applicable outcome=ok
- tidy_1005: place_on_zone_v3 verdict=applicable outcome=ok
- place_3: place_on_zone_v1 verdict=applicable outcome=ok
- tidy_1000: place_on_zone_v2 verdict=applicable outcome=ok
- tidy_1002: place_on_zone_v1 verdict=applicable outcome=ok
- tidy_1004: place_on_zone_v3 verdict=applicable outcome=ok
- tidy_1004: place_on_zone_v1 verdict=applicable outcome=ok
- tidy_1005: place_on_zone_v1 verdict=applicable outcome=fail
- tidy_1005: place_on_zone_v3 verdict=applicable outcome=ok

## Skill memory at end of run (skill)

- place_on_zone_v1 (validated), from teach_4: book 4 ok/0 fail
- place_on_zone_v2 (validated), from teach_4: mug 5 ok/1 fail
- place_on_zone_v3 (validated), from place_1: block 6 ok/0 fail

## Skill memory at end of run (both)

- place_on_zone_v1 (validated), from teach_4: mug 6 ok/1 fail
- place_on_zone_v2 (validated), from teach_4: book 3 ok/1 fail
- place_on_zone_v3 (validated), from place_1: block 5 ok/0 fail
