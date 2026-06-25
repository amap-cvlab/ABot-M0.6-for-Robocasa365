#!/usr/bin/env bash
set -euo pipefail
unset PYTHONHOME PYTHONPATH

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

CONFIG="configs/eval/robocasa365_abot06.yaml"
CKPT_PATH="src/abot06/weights/abot06_step_120000.pt"
SERVER_PYTHON="/path/to/abotm06/bin/python"
CLIENT_ENV_PREFIX="/path/to/robocasa365"
EGL_VENDOR_LIBRARY_FILENAMES="${CLIENT_ENV_PREFIX}/etc/egl/nvidia_egl_icd.json"
MODEL_BASE_PATH="/path/to/pretrain_checkpoint"
OUTPUT_ROOT="${REPO_ROOT}/runs"
CUDA_VISIBLE_DEVICES="1,2,3"
EVAL_SPLIT="pretrain"
NUM_TRIALS=50
NUM_VIDEO_STEPS=10
NUM_ACTION_STEPS=10
EVAL_LEGACY_PROCESSING=true
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
BASE_ROOT="${OUTPUT_ROOT}/all_groups_${EVAL_SPLIT}_${RUN_TAG}"

export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/src"
export TOKENIZERS_PARALLELISM=false
export DIFFSYNTH_MODEL_BASE_PATH="${MODEL_BASE_PATH}"
export DIFFSYNTH_SKIP_DOWNLOAD=true
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export __EGL_VENDOR_LIBRARY_FILENAMES="${EGL_VENDOR_LIBRARY_FILENAMES}"

export LD_LIBRARY_PATH="${CLIENT_ENV_PREFIX}/lib:${CLIENT_ENV_PREFIX}/x86_64-conda-linux-gnu/sysroot/usr/lib64:/usr/local/nvidia/lib64:/usr/local/nvidia/lib"
export LIBGL_DRIVERS_PATH="${CLIENT_ENV_PREFIX}/lib/dri:${CLIENT_ENV_PREFIX}/x86_64-conda-linux-gnu/sysroot/usr/lib64/dri"

for path in "${CONFIG}" "${CKPT_PATH}" "${EGL_VENDOR_LIBRARY_FILENAMES}"; do
  if [ ! -f "${path}" ]; then
    echo "ERROR: missing required file: ${path}" >&2
    exit 1
  fi
done
if [ ! -x "${SERVER_PYTHON}" ]; then
  echo "ERROR: server python not executable: ${SERVER_PYTHON}" >&2
  exit 1
fi

run_group() {
  local group="$1"
  local task_list_file="$2"
  local run_root="${BASE_ROOT}/${group}"

  echo "[all-groups] start ${group}"
  "${SERVER_PYTHON}" \
    experiments/robocasa365/run_pool.py \
    --config "${CONFIG}" \
    --ckpt "${CKPT_PATH}" \
    --task-list-file "${task_list_file}" \
    --run-root "${run_root}" \
    --num-trials "${NUM_TRIALS}" \
    --num-video-inference-steps "${NUM_VIDEO_STEPS}" \
    --num-action-inference-steps "${NUM_ACTION_STEPS}" \
    --server-override "evaluation.use_legacy_eval_processing=${EVAL_LEGACY_PROCESSING}" \
    --client-override "evaluation.split=${EVAL_SPLIT}" \
    --client-override "evaluation.use_legacy_eval_processing=${EVAL_LEGACY_PROCESSING}"
}

echo "=========================================="
echo "Abot06 RoboCasa365 all groups"
echo "=========================================="
echo "groups             : atomic_seen composite_seen composite_unseen"
echo "config             : ${CONFIG}"
echo "ckpt               : ${CKPT_PATH}"
echo "server_python      : ${SERVER_PYTHON}"
echo "cuda devices       : ${CUDA_VISIBLE_DEVICES}"
echo "mujoco_gl          : ${MUJOCO_GL} vendor=${__EGL_VENDOR_LIBRARY_FILENAMES}"
echo "eval split         : ${EVAL_SPLIT}"
echo "num_trials         : ${NUM_TRIALS}"
echo "video/action steps : ${NUM_VIDEO_STEPS} / ${NUM_ACTION_STEPS}"
echo "legacy processing  : ${EVAL_LEGACY_PROCESSING}"
echo "base_root          : ${BASE_ROOT}"
echo "run_tag            : ${RUN_TAG}"
echo "=========================================="

run_group "atomic_seen" "experiments/robocasa365/tasks_all_groups_atomic_seen.txt"
run_group "composite_seen" "experiments/robocasa365/tasks_all_groups_composite_seen.txt"
run_group "composite_unseen" "experiments/robocasa365/tasks_all_groups_composite_unseen.txt"

echo "[all-groups] done"
echo "base_root: ${BASE_ROOT}"
