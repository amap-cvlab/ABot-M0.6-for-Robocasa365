from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from experiments.robocasa365.eval_statistics import (
    EPSILON as NORMALIZER_EPSILON,
    load_dataset_statistics,
    resolve_dataset_statistics_path,
)
from experiments.robocasa365.robocasa_utils import DEFAULT_OBS_CAM_KEYS, get_robocasa_image, get_robocasa_proprio

NORMALIZER_STD_REG = 1e-8
LEGACY_NORMALIZER_RANGE_TOL = 1e-4
LEGACY_DEFAULT_PROMPT = "A video recorded from a robot's point of view executing the following instruction: {task}"
LEGACY_VIDEO_SIZE = (256, 768)
LEGACY_VIDEO_CONCAT_MODE = "horizontal"


class RoboCasaEvalAdapter:
    def __init__(self, cfg: dict[str, Any], *, ckpt_path: str | None = None, dataset_statistics_path: str | None = None):
        self.cfg = cfg
        data_cfg = cfg.get("data", {}) or {}
        self.robot_type = str(data_cfg.get("robot_type", "robocasa365"))
        self.embodiment_tag = self.robot_type
        self.camera_keys = [str(x) for x in (data_cfg.get("camera_keys") or DEFAULT_OBS_CAM_KEYS)]
        self.action_dim = int(data_cfg.get("action_dim", 12))
        self.proprio_dim = int(data_cfg.get("proprio_dim", 16))
        self.use_legacy_eval_processing = True
        self.input_processing_style = "legacy"
        self.prompt_style = "legacy"
        self.video_size = tuple(int(x) for x in data_cfg.get("video_size", LEGACY_VIDEO_SIZE))
        self.video_concat_mode = str(data_cfg.get("video_concat_mode", LEGACY_VIDEO_CONCAT_MODE))

        stats_path = resolve_dataset_statistics_path(ckpt_path=ckpt_path, explicit_path=dataset_statistics_path, config=cfg)
        self.dataset_statistics_path = Path(stats_path)
        all_stats = load_dataset_statistics(stats_path)
        if self.embodiment_tag in all_stats:
            stats_key = self.embodiment_tag
        elif "robocasa365" in all_stats:
            stats_key = "robocasa365"
        elif len(all_stats) == 1:
            stats_key = next(iter(all_stats))
        else:
            raise KeyError(f"dataset_statistics.json has no tag {self.embodiment_tag!r}; keys={list(all_stats)}")
        self.stats = all_stats[stats_key]
        self.state_stats = self.stats["state"]
        self.action_stats = self.stats["action"]
        self.state_modes = list(self.state_stats.get("modes") or ["min_max"] * self.proprio_dim)
        self.action_modes = list(self.action_stats.get("modes") or ["min_max"] * self.action_dim)

    def make_prompt(self, task_description: str) -> str:
        task = str(task_description)
        eval_cfg = self.cfg.get("evaluation", {}) or {}
        template = str(eval_cfg.get("prompt_template") or LEGACY_DEFAULT_PROMPT)
        return template.format(task=task)

    @staticmethod
    def _center_crop_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
        pil_image = Image.fromarray(image)
        src_w, src_h = pil_image.size
        scale = max(width / src_w, height / src_h)
        resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
        rw, rh = resized.size
        left = max((rw - width) // 2, 0)
        top = max((rh - height) // 2, 0)
        cropped = resized.crop((left, top, left + width, top + height))
        return np.asarray(cropped, dtype=np.uint8)

    @staticmethod
    def _normalize_pixels(frame: np.ndarray, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        tensor = torch.tensor(frame).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=dtype)
        return tensor * (2.0 / 255.0) - 1.0

    def obs_to_image_tensor(
        self,
        obs: dict,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        camera_keys: list[str] | None = None,
    ) -> torch.Tensor:
        height, width = self.video_size
        keys = list(camera_keys or self.camera_keys)
        imgs = get_robocasa_image(obs, cam_keys=keys)
        image_order = ["wrist_image", "left_image", "right_image"][: len(keys)]
        per_cam_w = width // max(len(image_order), 1)
        per_camera: list[np.ndarray] = []
        for short_name in image_order:
            if short_name not in imgs:
                raise KeyError(f"Image key {short_name!r} missing; got keys={list(imgs.keys())}")
            image = np.asarray(imgs[short_name], dtype=np.uint8)
            per_camera.append(self._center_crop_resize(image, width=per_cam_w, height=height))
        frame = np.concatenate(per_camera, axis=1)
        if frame.shape[:2] != (height, width):
            raise ValueError(f"Image path produced frame shape={frame.shape[:2]}, expected {(height, width)}")
        return self._normalize_pixels(frame, device=device, dtype=dtype)

    @staticmethod
    def _linear_forward(value: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).copy()
        input_min = np.asarray(low, dtype=np.float32).copy()
        input_max = np.asarray(high, dtype=np.float32).copy()
        input_range = input_max - input_min
        ignore_dim = input_range < LEGACY_NORMALIZER_RANGE_TOL
        input_range[ignore_dim] = 2.0
        scale = 2.0 / input_range
        offset = -1.0 - scale * input_min
        offset[ignore_dim] = -input_min[ignore_dim]
        return np.clip(arr * scale + offset, -5.0, 5.0)

    @staticmethod
    def _linear_inverse(value: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).copy()
        input_min = np.asarray(low, dtype=np.float32).copy()
        input_max = np.asarray(high, dtype=np.float32).copy()
        input_range = input_max - input_min
        ignore_dim = input_range < LEGACY_NORMALIZER_RANGE_TOL
        input_range[ignore_dim] = 2.0
        scale = 2.0 / input_range
        offset = -1.0 - scale * input_min
        offset[ignore_dim] = -input_min[ignore_dim]
        return (arr - offset) / scale

    def _normalize_vector(self, value: np.ndarray, stats: dict[str, Any], modes: list[str]) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float32).reshape(-1).copy()
        low = np.asarray(stats["min"], dtype=np.float32)
        high = np.asarray(stats["max"], dtype=np.float32)
        q01 = np.asarray(stats.get("q01", low), dtype=np.float32)
        q99 = np.asarray(stats.get("q99", high), dtype=np.float32)
        mean = np.asarray(stats.get("mean", np.zeros_like(arr)), dtype=np.float32)
        std = np.asarray(stats.get("std", np.ones_like(arr)), dtype=np.float32)
        for i, mode in enumerate(modes[: arr.shape[0]]):
            if mode in {"min_max", "q99"}:
                bound_low, bound_high = (low, high) if mode == "min_max" else (q01, q99)
                arr[i] = self._linear_forward(arr[i : i + 1], bound_low[i : i + 1], bound_high[i : i + 1])[0]
            elif mode == "mean_std":
                arr[i] = (arr[i] - mean[i]) / (std[i] + NORMALIZER_STD_REG)
            elif mode not in {"identity", "none"}:
                arr[i] = (arr[i] - low[i]) / (high[i] - low[i] + NORMALIZER_EPSILON) * 2.0 - 1.0
        return arr

    def obs_to_proprio_tensor(self, obs: dict, *, device: torch.device | str, dtype: torch.dtype) -> torch.Tensor:
        proprio = get_robocasa_proprio(obs, expected_dim=self.proprio_dim)
        proprio = self._normalize_vector(proprio, self.state_stats, self.state_modes)
        return torch.from_numpy(proprio).unsqueeze(0).to(device=device, dtype=dtype)

    def obs_to_model_inputs(
        self,
        obs: dict,
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        camera_keys: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        image = self.obs_to_image_tensor(obs, device=device, dtype=dtype, camera_keys=camera_keys)
        proprio = self.obs_to_proprio_tensor(obs, device=device, dtype=dtype)
        return image, proprio

    def denormalize_action(self, action: Any) -> np.ndarray:
        arr = np.asarray(action.detach().cpu().numpy() if hasattr(action, "detach") else action, dtype=np.float32).copy()
        if arr.ndim == 1:
            arr = arr[None, :]
        if arr.ndim != 2:
            raise ValueError(f"Expected action [T,D], got {arr.shape}")
        if arr.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action dim {self.action_dim}, got shape {arr.shape}")
        low = np.asarray(self.action_stats["min"], dtype=np.float32)
        high = np.asarray(self.action_stats["max"], dtype=np.float32)
        q01 = np.asarray(self.action_stats.get("q01", low), dtype=np.float32)
        q99 = np.asarray(self.action_stats.get("q99", high), dtype=np.float32)
        mean = np.asarray(self.action_stats.get("mean", np.zeros_like(low)), dtype=np.float32)
        std = np.asarray(self.action_stats.get("std", np.ones_like(low)), dtype=np.float32)
        for i, mode in enumerate(self.action_modes[: arr.shape[-1]]):
            if mode in {"min_max", "q99"}:
                bound_low, bound_high = (low, high) if mode == "min_max" else (q01, q99)
                arr[..., i] = self._linear_inverse(arr[..., i], bound_low[i : i + 1], bound_high[i : i + 1])
            elif mode == "mean_std":
                arr[..., i] = arr[..., i] * (std[i] + NORMALIZER_STD_REG) + mean[i]
        return arr.astype(np.float32)

    def future_obs_to_pixels(
        self,
        future_obs_list: list[dict],
        *,
        device: torch.device | str,
        dtype: torch.dtype,
        camera_keys: list[str] | None = None,
    ) -> torch.Tensor:
        frames = [self.obs_to_image_tensor(obs, device=device, dtype=dtype, camera_keys=camera_keys).squeeze(0) for obs in future_obs_list]
        if not frames:
            raise ValueError("future_obs_list is empty")
        return torch.stack(frames, dim=1).unsqueeze(0)
