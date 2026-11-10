#!/usr/bin/env bash
# Usage (on Hopper, in the project root): SMA_RUN_NAME=name SMA_MODE=dev|suite bash hpc/submit.sh
set -euo pipefail
ROOT=${SMA_ROOT:-/scratch/e0985859/agentic_robot_programming/skill_memory_agent}
PBS=/cm/shared/apps/pbspro/current/bin
VARS="SMA_RUN_NAME=${SMA_RUN_NAME:?},SMA_MODE=${SMA_MODE:-dev}"
[[ -n "${SMA_ROOT:-}" ]] && VARS+=",SMA_ROOT=$SMA_ROOT"
[[ -n "${SMA_CONDITIONS:-}" ]] && VARS+=",SMA_CONDITIONS=$SMA_CONDITIONS"
[[ -n "${SMA_SUITES:-}" ]] && VARS+=",SMA_SUITES=$SMA_SUITES"
[[ -n "${SMA_SETS:-}" ]] && VARS+=",SMA_SETS=${SMA_SETS// /:}"  # space-separated list; ":" on the wire
[[ -n "${SMA_CODER:-}" ]] && VARS+=",SMA_CODER=$SMA_CODER,SMA_GENERAL_GPU_UTIL=0.45"
[[ -n "${SMA_API_LEVEL:-}" ]] && VARS+=",SMA_API_LEVEL=$SMA_API_LEVEL"
for v in SMA_MAX_LEN SMA_MM_IMAGES SMA_CAMPAIGN_ROOT SMA_CAMPAIGN_ARMS SMA_DEADLINE_MIN SMA_RESUME SMA_CUROBO SMA_MOTION SMA_LLM_PROVIDER SMA_API_KEY_FILE SMA_API_KEY_TIER SMA_API_KEY_FILE_FALLBACK SMA_SUITE_SIZE SMA_MEMORY_INIT SMA_SPEND_CAP_USD SMA_REASONING_EFFORT SMA_PRICE_IN SMA_PRICE_OUT SMA_GATE_SUITE SMA_TIER_MODELS SMA_NEUTRAL SMA_GROUPING SMA_DEEPSEEK_CAP_USD SMA_LLM_URL SMA_DEEPSEEK_THINKING SMA_MEMORY_WARMUP SMA_NOTEBOOK_SUITE SMA_SEG SMA_MARKS SMA_SKIP_GATE SMA_DINO_SUITES SMA_SKIP_DINO SMA_DESKS SMA_TRACK_GATE SMA_CONTACT_DEBUG SMA_VIDEO_SIZE; do
  [[ -n "${!v:-}" ]] && VARS+=",$v=${!v}"
done
[[ -n "${SMA_VLLM_EXTRA:-}" ]] && VARS+=",SMA_VLLM_EXTRA=${SMA_VLLM_EXTRA// /@}"
[[ -n "${SMA_PERCEPTION:-}" ]] && VARS+=",SMA_PERCEPTION=$SMA_PERCEPTION"
[[ -n "${SMA_REPLAY_RUN:-}" ]] && VARS+=",SMA_REPLAY_RUN=$SMA_REPLAY_RUN,SMA_REPLAY_ITEMS=$SMA_REPLAY_ITEMS"
[[ -n "${SMA_GENERAL_MODEL:-}" ]] && VARS+=",SMA_GENERAL_MODEL=$SMA_GENERAL_MODEL,SMA_GENERAL_NAME=${SMA_GENERAL_NAME:-general}"
[[ -z "${SMA_GENERAL_MODEL:-}" && -n "${SMA_GENERAL_NAME:-}" ]] && VARS+=",SMA_GENERAL_NAME=$SMA_GENERAL_NAME"
[[ -n "${SMA_GENERAL_GPU_UTIL:-}" && -z "${SMA_CODER:-}" ]] && VARS+=",SMA_GENERAL_GPU_UTIL=$SMA_GENERAL_GPU_UTIL"
mkdir -p "$ROOT/outputs"
# Same pattern as the existing project: submit held, shrink the hook-expanded request, release.
# interactive (default) allows one running job per user; small allows three, for a parallel run.
JOB_ID=$($PBS/qsub -h -q "${SMA_QUEUE:-interactive}" -v "$VARS" -o "$ROOT/outputs/$SMA_RUN_NAME.pbs.log" "$ROOT/hpc/run_sma.pbs")
$PBS/qalter -l select=${SMA_SELECT:-1:ncpus=8:mem=48gb:ngpus=1} "$JOB_ID"  # SMA_SELECT: e.g. a CPU-only probe
[[ -n "${SMA_WALLTIME:-}" ]] && $PBS/qalter -l walltime="$SMA_WALLTIME" "$JOB_ID"
$PBS/qrls "$JOB_ID"
printf '%s\n' "$JOB_ID"
