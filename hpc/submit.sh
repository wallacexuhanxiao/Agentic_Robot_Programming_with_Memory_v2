#!/usr/bin/env bash
# Usage (on Hopper, in the project root): SMA_RUN_NAME=name SMA_MODE=dev|suite bash hpc/submit.sh
set -euo pipefail
ROOT=/scratch/e0985859/agentic_robot_programming/skill_memory_agent
PBS=/cm/shared/apps/pbspro/current/bin
VARS="SMA_RUN_NAME=${SMA_RUN_NAME:?},SMA_MODE=${SMA_MODE:-dev}"
[[ -n "${SMA_CONDITIONS:-}" ]] && VARS+=",SMA_CONDITIONS=$SMA_CONDITIONS"
[[ -n "${SMA_SUITES:-}" ]] && VARS+=",SMA_SUITES=$SMA_SUITES"
mkdir -p "$ROOT/outputs"
# Same pattern as the existing project: submit held, shrink the hook-expanded request, release.
JOB_ID=$($PBS/qsub -h -q interactive -v "$VARS" -o "$ROOT/outputs/$SMA_RUN_NAME.pbs.log" "$ROOT/hpc/run_sma.pbs")
$PBS/qalter -l select=1:ncpus=8:mem=48gb:ngpus=1 "$JOB_ID"
$PBS/qrls "$JOB_ID"
printf '%s\n' "$JOB_ID"
