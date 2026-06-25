from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf

LEGACY_ROBOCASA_CAMERA_KEYS = [
    "video.roboto_eye_in_hand",
    "video.roboto_agentview_left",
    "video.roboto_agentview_right",
]
LEGACY_ROBOCASA_VIDEO_SIZE = [256, 768]
LEGACY_ROBOCASA_VIDEO_CONCAT_MODE = "horizontal"


def _default_cfg() -> dict[str, Any]:
    return {
        "server": {
            "host": "127.0.0.1",
            "port": 8001,
            "ready": False,
            "load_pretrained_backbones": False,
            "load_text_encoder": True,
            "mixed_precision": "bf16",
        },
        "remote": {
            "host": "127.0.0.1",
            "port": None,
            "base_port": 8001,
            "timeout_s": 600,
            "max_wait_ready_s": 600,
        },
        "evaluation": {
            "env_name": None,
            "group_name": "default",
            "env_num": 1,
            "num_trials": 50,
            "output_dir": "evaluate_results/robocasa365_debug",
            "split": "pretrain",
            "resume_trials": True,
            "base_seed": None,
            "max_steps": None,
            "replan_steps": None,
            "action_horizon": None,
            "use_action_ensembler": False,
            "visualize_future_video": False,
            "use_legacy_eval_processing": True,
            "input_processing_style": "legacy",
            "prompt_style": "legacy",
            "num_inference_steps": 20,
            "num_video_inference_steps": 10,
            "num_action_inference_steps": 10,
            "sigma_shift": None,
            "text_cfg_scale": 1.0,
            "negative_prompt": "",
            "rand_device": "cpu",
            "tiled": False,
            "camera_keys": None,
            "action_key_mapping": None,
            "action_postprocess": {
                "binarize_gripper": True,
                "binarize_ctrl_mode": True,
                "zero_base_d3": True,
                "base_motion_deadzone": 0.0,
                "gripper_threshold": 0.0,
                "ctrl_mode_threshold": 0.0,
            },
        },
        "model": {
            "variant": "abot06",
            "model_id": "Wan-AI/Wan2.2-TI2V-5B",
            "tokenizer_model_id": "Wan-AI/Wan2.1-T2V-1.3B",
            "tokenizer_max_len": 128,
            "load_text_encoder": False,
            "redirect_common_files": True,
            "action_dit_pretrained_path": None,
            "skip_dit_load_from_pretrain": False,
            "memory": True,
            "memory_sizes": [16, 2, 1],
            "memory_pred_size": 2,
            "memory_gain": False,
            "memory_zero_time": True,
            "mot_checkpoint_mixed_attn": False,
            "video_dit_config": {
                "has_image_input": False,
                "patch_size": [1, 2, 2],
                "in_dim": 48,
                "hidden_dim": 3072,
                "ffn_dim": 14336,
                "freq_dim": 256,
                "text_dim": 4096,
                "out_dim": 48,
                "num_heads": 24,
                "attn_head_dim": 128,
                "num_layers": 30,
                "eps": 1.0e-6,
                "seperated_timestep": True,
                "require_clip_embedding": False,
                "require_vae_embedding": False,
                "fuse_vae_embedding_in_latents": False,
                "use_gradient_checkpointing": False,
                "video_attention_mask_mode": "first_frame_causal",
                "action_conditioned": False,
                "action_dim": 12,
                "action_group_causal_mask_mode": "group_diagonal",
            },
            "action_dit_config": {
                "action_dim": 12,
                "hidden_dim": 1024,
                "ffn_dim": 4096,
                "num_heads": 24,
                "attn_head_dim": 128,
                "num_layers": 30,
                "text_dim": 4096,
                "freq_dim": 256,
                "eps": 1.0e-6,
                "use_gradient_checkpointing": False,
            },
            "video_scheduler": {
                "train_shift": 5.0,
                "infer_shift": 5.0,
                "num_train_timesteps": 1000,
            },
            "action_scheduler": {
                "train_shift": 5.0,
                "infer_shift": 5.0,
                "num_train_timesteps": 1000,
            },
        },
        "pool": {"max_tasks_per_gpu": 1, "server_ready_timeout_s": 3600},
        "seed": 42,
        "gpu_id": 0,
    }


def _deep_merge(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (update or {}).items():
        norm_key = key.lower() if key in {"SERVER", "REMOTE", "EVALUATION", "POOL"} else key
        if isinstance(value, dict) and isinstance(out.get(norm_key), dict):
            out[norm_key] = _deep_merge(out[norm_key], value)
        else:
            out[norm_key] = value
    return out


def _parse_value(raw: str) -> Any:
    try:
        return yaml.safe_load(raw)
    except Exception:
        return raw


def set_by_dotted_key(cfg: dict[str, Any], key: str, value: Any) -> None:
    parts = [p for p in key.split(".") if p]
    if not parts:
        return
    if parts[0] in {"SERVER", "REMOTE", "EVALUATION", "POOL"}:
        parts[0] = parts[0].lower()
    cur = cfg
    for part in parts[:-1]:
        if part not in cur or not isinstance(cur[part], dict):
            cur[part] = {}
        cur = cur[part]
    cur[parts[-1]] = value


def parse_overrides(items: list[str] | None) -> list[tuple[str, Any]]:
    parsed: list[tuple[str, Any]] = []
    for item in items or []:
        if not item or item == "--":
            continue
        if item.startswith("--override="):
            item = item.split("=", 1)[1]
        if item.startswith("--"):
            continue
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got {item!r}")
        key, raw_value = item.split("=", 1)
        parsed.append((key.strip(), _parse_value(raw_value)))
    return parsed


def _infer_robot_type(cfg: dict[str, Any]) -> str:
    data_cfg = cfg.get("data", {}) or {}
    return str(data_cfg.get("robot_type") or "robocasa365")


def _finalize(cfg: dict[str, Any]) -> dict[str, Any]:
    data_cfg = cfg.setdefault("data", {})
    robot_type = _infer_robot_type(cfg)
    data_cfg.setdefault("robot_type", robot_type)
    data_cfg.setdefault("camera_keys", list(LEGACY_ROBOCASA_CAMERA_KEYS))
    data_cfg.setdefault("video_size", list(LEGACY_ROBOCASA_VIDEO_SIZE))
    data_cfg.setdefault("video_concat_mode", LEGACY_ROBOCASA_VIDEO_CONCAT_MODE)
    default_action_dim = int(cfg.get("model", {}).get("action_dit_config", {}).get("action_dim", 12))
    data_cfg.setdefault("action_dim", default_action_dim)
    data_cfg.setdefault("proprio_dim", int(cfg.get("model", {}).get("proprio_dim", 16)))
    data_cfg.setdefault("action_video_freq_ratio", 4)
    data_cfg.setdefault("num_frames", 33)
    data_cfg.setdefault("zero_pad_len", int(sum(cfg.get("model", {}).get("memory_sizes", [16, 2, 1])[:-1])))

    eval_cfg = cfg.setdefault("evaluation", {})
    default_chunk = max(int(data_cfg.get("num_frames", 33)) - 1, 1)
    if eval_cfg.get("replan_steps") is None:
        eval_cfg["replan_steps"] = default_chunk
    if eval_cfg.get("action_horizon") is None:
        eval_cfg["action_horizon"] = int(eval_cfg["replan_steps"])
    eval_cfg["use_legacy_eval_processing"] = True
    eval_cfg["prompt_style"] = "legacy"
    eval_cfg["input_processing_style"] = "legacy"
    return cfg


def load_config(config_path: str | Path, overrides: list[str] | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    user_cfg = _load_yaml_with_bases(config_path)
    cfg = _deep_merge(_default_cfg(), user_cfg)
    for key, value in parse_overrides(overrides):
        set_by_dotted_key(cfg, key, value)
    return _finalize(cfg)


def _load_yaml_with_bases(config_path: Path, seen: set[Path] | None = None) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    seen = set() if seen is None else seen
    if config_path in seen:
        raise ValueError(f"recursive base_config include: {config_path}")
    seen.add(config_path)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    base_cfg: dict[str, Any] = {}
    bases = cfg.pop("base_config", None)
    if bases:
        if not isinstance(bases, list):
            bases = [bases]
        for base in bases:
            base_path = Path(str(base)).expanduser()
            if not base_path.is_absolute():
                base_path = config_path.parent / base_path
            base_cfg = _deep_merge(base_cfg, _load_yaml_with_bases(base_path, seen))
    return _deep_merge(base_cfg, cfg)


def build_memory_rollout_omegaconf(cfg: dict[str, Any]):
    data_train = dict(cfg.get("data", {}))
    data_train.setdefault("video_size", [256, 768])
    return OmegaConf.create({"model": cfg.get("model", {}), "data": {"train": data_train}})
