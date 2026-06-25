from __future__ import annotations

import os
from typing import Any

import torch

from abot06.models.wan22.model import Abot06


def mixed_precision_to_dtype(mp: str | None) -> torch.dtype:
    mp = str(mp or "bf16").strip().lower()
    if mp in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if mp in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def resolve_eval_device(gpu_id: int | str | None = None, device: str | None = None) -> str:
    if device:
        return str(device)
    if not torch.cuda.is_available():
        return "cpu"
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        return "cuda:0"
    if gpu_id is None:
        return "cuda:0"
    return f"cuda:{int(gpu_id)}"


def create_model_from_config(
    model_cfg: dict[str, Any],
    *,
    data_cfg: dict[str, Any] | None = None,
    device: str,
    model_dtype: torch.dtype,
    ckpt_path: str | os.PathLike[str] | None = None,
    config_source: str | os.PathLike[str] | None = None,
) -> Abot06:
    del ckpt_path, config_source
    model_cfg = dict(model_cfg)
    if data_cfg and "proprio_dim" not in model_cfg and "proprio_dim" in data_cfg:
        model_cfg["proprio_dim"] = data_cfg["proprio_dim"]
    variant = str(model_cfg.get("variant", "abot06")).strip().lower()
    if variant != "abot06":
        raise ValueError(f"Unsupported model variant: {variant}")

    video_scheduler = model_cfg.get("video_scheduler", {}) or {}
    action_scheduler = model_cfg.get("action_scheduler", {}) or {}
    memory_sizes = tuple(int(s) for s in model_cfg.get("memory_sizes", [16, 2, 1]))
    memory_kernel_long = tuple(int(s) for s in model_cfg.get("memory_kernel_long", [4, 8, 8]))
    memory_kernel_mid = tuple(int(s) for s in model_cfg.get("memory_kernel_mid", [2, 4, 4]))
    memory_kernel_current = tuple(int(s) for s in model_cfg.get("memory_kernel_current", [1, 2, 2]))

    model = Abot06.from_wan22_pretrained(
        device=device,
        torch_dtype=model_dtype,
        model_id=str(model_cfg["model_id"]),
        tokenizer_model_id=str(model_cfg.get("tokenizer_model_id", model_cfg["model_id"])),
        tokenizer_max_len=int(model_cfg.get("tokenizer_max_len", 128)),
        load_text_encoder=bool(model_cfg.get("load_text_encoder", False)),
        proprio_dim=int(model_cfg["proprio_dim"]) if "proprio_dim" in model_cfg else None,
        redirect_common_files=bool(model_cfg.get("redirect_common_files", True)),
        video_dit_config=dict(model_cfg.get("video_dit_config", {})),
        action_dit_config=dict(model_cfg.get("action_dit_config", {})),
        action_dit_pretrained_path=model_cfg.get("action_dit_pretrained_path"),
        skip_dit_load_from_pretrain=bool(model_cfg.get("skip_dit_load_from_pretrain", False)),
        mot_checkpoint_mixed_attn=bool(model_cfg.get("mot_checkpoint_mixed_attn", False)),
        video_train_shift=float(video_scheduler.get("train_shift", 5.0)),
        video_infer_shift=float(video_scheduler.get("infer_shift", 5.0)),
        video_num_train_timesteps=int(video_scheduler.get("num_train_timesteps", 1000)),
        action_train_shift=float(action_scheduler.get("train_shift", 5.0)),
        action_infer_shift=float(action_scheduler.get("infer_shift", 5.0)),
        action_num_train_timesteps=int(action_scheduler.get("num_train_timesteps", 1000)),
        memory=bool(model_cfg.get("memory", True)),
        memory_sizes=memory_sizes,
        memory_kernel_long=memory_kernel_long,
        memory_kernel_mid=memory_kernel_mid,
        memory_kernel_current=memory_kernel_current,
        memory_pred_size=int(model_cfg.get("memory_pred_size", 2)),
        memory_gain=bool(model_cfg.get("memory_gain", False)),
        memory_zero_time=bool(model_cfg.get("memory_zero_time", True)),
    )
    model.model_variant = "abot06"
    model.model_variant_source = "config"
    return model
