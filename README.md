# AbotM06 RoboCasa365 Evaluation

## 1. Install AbotM06

Create the server environment and install this repository:

```bash
conda create -n abotm06 python=3.10 -y
conda activate abotm06
pip install -U pip
pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --extra-index-url https://download.pytorch.org/whl/cu128
pip install -e .
```

## 2. Download Wan2.2 Weights

The eval script sets `DIFFSYNTH_SKIP_DOWNLOAD=true`, so Wan weights must be
downloaded before evaluation.

```bash
conda activate abotm06
export MODEL_BASE_PATH=/path/to/pretrain_checkpoint
mkdir -p "${MODEL_BASE_PATH}"

python - <<'PY'
from modelscope import snapshot_download

root = "/path/to/pretrain_checkpoint"

snapshot_download(
    "Wan-AI/Wan2.2-TI2V-5B",
    local_dir=f"{root}/Wan-AI/Wan2.2-TI2V-5B",
    allow_file_pattern="diffusion_pytorch_model*.safetensors",
)
snapshot_download(
    "Wan-AI/Wan2.1-T2V-1.3B",
    local_dir=f"{root}/Wan-AI/Wan2.1-T2V-1.3B",
    allow_file_pattern="google/umt5-xxl/*",
)
snapshot_download(
    "DiffSynth-Studio/Wan-Series-Converted-Safetensors",
    local_dir=f"{root}/DiffSynth-Studio/Wan-Series-Converted-Safetensors",
    allow_file_pattern=[
        "models_t5_umt5-xxl-enc-bf16.safetensors",
        "Wan2.2_VAE.safetensors",
    ],
)
PY
```

## 3. Fill Local Paths

Open `local_run_robocasa365_all_groups_abot06.sh` and fill the fixed local paths at the top:

```bash
CONFIG="configs/eval/robocasa365_abot06.yaml"
CKPT_PATH="src/abot06/weights/abot06_step_120000.pt"
SERVER_PYTHON="/path/to/abotm06/bin/python"
CLIENT_ENV_PREFIX="/path/to/robocasa365"
EGL_VENDOR_LIBRARY_FILENAMES="${CLIENT_ENV_PREFIX}/etc/egl/nvidia_egl_icd.json"
MODEL_BASE_PATH="/path/to/pretrain_checkpoint"
OUTPUT_ROOT="${REPO_ROOT}/runs"
CUDA_VISIBLE_DEVICES="1,2,3"
```

Meaning:

- `CKPT_PATH`: your trained AbotM06 checkpoint.
- `SERVER_PYTHON`: Python from the AbotM06 environment created above.
- `CLIENT_ENV_PREFIX`: existing RoboCasa365 environment root.
- `EGL_VENDOR_LIBRARY_FILENAMES`: EGL vendor JSON inside the RoboCasa365 environment.
- `MODEL_BASE_PATH`: Wan2.2 weight directory from step 2.
- `OUTPUT_ROOT`: where eval results, logs, and videos are written.
- `CUDA_VISIBLE_DEVICES`: GPUs used by the model servers.

## 4. Eval Parameters

The local script keeps these eval parameters explicit:

```bash
EVAL_SPLIT="pretrain"
NUM_TRIALS=50
NUM_VIDEO_STEPS=10
NUM_ACTION_STEPS=10
EVAL_LEGACY_PROCESSING=true
```

The script runs all three RoboCasa365 groups:

```text
atomic_seen
composite_seen
composite_unseen
```

## 5. Run Evaluation

Run all groups:

```bash
bash local_run_robocasa365_all_groups_abot06.sh
```

The script writes outputs under:

```bash
${OUTPUT_ROOT}/all_groups_${EVAL_SPLIT}_${RUN_TAG}/
```

Each group gets its own subdirectory under that run root.
