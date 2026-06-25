import os
import re
import time
from collections.abc import Mapping
from typing import Any, Iterable, Optional

import imageio
import numpy as np
from PIL import Image, ImageDraw


DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEFAULT_ACTION_KEY_MAPPING: dict[str, tuple[int, int]] = {
    "action.base_motion": (0, 4),
    "action.control_mode": (4, 5),
    "action.end_effector_position": (5, 8),
    "action.end_effector_rotation": (8, 11),
    "action.gripper_close": (11, 12),
}


DEFAULT_OBS_CAM_KEYS: list[str] = [
    "video.roboto_eye_in_hand",
    "video.roboto_agentview_left",
    "video.roboto_agentview_right",
]


ROBOCASA_CAMERA_NAMES: list[str] = ["wrist_image", "left_image", "right_image"]


ROBOCASA_CAMERA_KEY_TO_NAME: dict[str, str] = {
    "video.roboto_eye_in_hand": "wrist_image",
    "video.roboto_agentview_left": "left_image",
    "video.roboto_agentview_right": "right_image",
    "video.robot0_eye_in_hand": "wrist_image",
    "video.robot0_agentview_left": "left_image",
    "video.robot0_agentview_right": "right_image",
    "observation.images.roboto_eye_in_hand": "wrist_image",
    "observation.images.roboto_agentview_left": "left_image",
    "observation.images.roboto_agentview_right": "right_image",
    "observation.images.robot0_eye_in_hand": "wrist_image",
    "observation.images.robot0_agentview_left": "left_image",
    "observation.images.robot0_agentview_right": "right_image",
}


def _get_by_path(data: Any, path: str) -> Any:
    if isinstance(data, Mapping) and path in data:
        return data[path]
    current = data
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            return None
    return current


def _collect_arrays(obs: Any, prefix: str = "") -> list[tuple[str, np.ndarray]]:
    items: list[tuple[str, np.ndarray]] = []
    if isinstance(obs, Mapping):
        for k, v in obs.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            items.extend(_collect_arrays(v, key))
    elif isinstance(obs, np.ndarray):
        items.append((prefix, obs))
    return items


def _to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    img = np.asarray(image)
    if img.ndim != 3:
        raise ValueError(f"Expected HWC image, got shape={img.shape}")
    if img.shape[-1] == 4:
        img = img[..., :3]
    elif img.shape[-1] == 1:
        img = np.repeat(img, 3, axis=-1)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def flatten_obs_keys(obs: Any, prefix: str = "") -> list[dict[str, Any]]:

    keys: list[dict[str, Any]] = []
    if isinstance(obs, Mapping):
        for k, v in obs.items():
            new_prefix = f"{prefix}.{k}" if prefix else str(k)
            keys.extend(flatten_obs_keys(v, new_prefix))
    else:
        shape = tuple(obs.shape) if hasattr(obs, "shape") else ()
        dtype = str(obs.dtype) if hasattr(obs, "dtype") else type(obs).__name__
        keys.append({"key": prefix, "shape": shape, "dtype": dtype})
    return keys


def get_robocasa_env(env_name: str, split: str = "target", seed: Optional[int] = None):


    import gymnasium as gym
    import robocasa
    from robocasa.utils.dataset_registry_utils import get_task_horizon

    if split not in ("pretrain", "target", "all"):
        raise ValueError(f"split must be one of ['pretrain', 'target', 'all'], got {split!r}")
    clean_name = env_name.replace("robocasa/", "")
    try:
        task_horizon = int(get_task_horizon(clean_name))
    except ValueError:
        task_horizon = 500
        print(f"[warn] get_task_horizon has no entry for task {clean_name!r}; using fallback {task_horizon}.")
    make_kwargs: dict[str, Any] = {"split": split}
    if seed is not None:
        make_kwargs["seed"] = int(seed)
    env = gym.make(f"robocasa/{clean_name}", **make_kwargs)

    return env, task_horizon


def get_robocasa_image(obs: Mapping, cam_keys: Optional[list[str]] = None) -> dict[str, np.ndarray]:


    if cam_keys is None:
        cam_keys = list(DEFAULT_OBS_CAM_KEYS)

    out: dict[str, np.ndarray] = {}
    for lerobot_key in cam_keys:
        value = _get_by_path(obs, lerobot_key)
        if value is None:
            tail = lerobot_key.split(".")[-1]
            tail_aliases = [tail]
            if "robot0_" in tail:
                tail_aliases.append(tail.replace("robot0_", "roboto_"))
            if "roboto_" in tail:
                tail_aliases.append(tail.replace("roboto_", "robot0_"))
            for key_tail in tail_aliases:
                for prefix in ("video", "observation.images", "observation.video", "images"):
                    value = _get_by_path(obs, f"{prefix}.{key_tail}")
                    if value is not None:
                        break
                if value is not None:
                    break
        if value is None:
            raise KeyError(
                f"Camera key {lerobot_key!r} not found in RoboCasa observation. "
                f"Available keys: {[item['key'] for item in flatten_obs_keys(obs)]}"
            )
        short_name = ROBOCASA_CAMERA_KEY_TO_NAME.get(lerobot_key, lerobot_key.split(".")[-1])
        out[short_name] = _to_uint8_rgb(value)
    return out


def _resize_rgb(image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = int(size_hw[0]), int(size_hw[1])
    return np.asarray(Image.fromarray(_to_uint8_rgb(image)).resize((width, height), resample=Image.BILINEAR), dtype=np.uint8)


def _center_crop_resize_rgb(image: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    height, width = int(size_hw[0]), int(size_hw[1])
    pil_image = Image.fromarray(_to_uint8_rgb(image))
    src_w, src_h = pil_image.size
    scale = max(width / src_w, height / src_h)
    resized = pil_image.resize((round(src_w * scale), round(src_h * scale)), resample=Image.BILINEAR)
    rw, rh = resized.size
    left = max((rw - width) // 2, 0)
    top = max((rh - height) // 2, 0)
    return np.asarray(resized.crop((left, top, left + width, top + height)), dtype=np.uint8)


def render_robocasa_model_frame(
    obs: Mapping,
    *,
    cam_keys: Optional[list[str]] = None,
    video_size: tuple[int, int] | list[int] = (384, 256),
    video_concat_mode: str = "robocasa365_compact",
) -> np.ndarray:


    height, width = int(video_size[0]), int(video_size[1])
    images = get_robocasa_image(obs, cam_keys=cam_keys)
    keys = list(cam_keys or DEFAULT_OBS_CAM_KEYS)
    ordered_names = [ROBOCASA_CAMERA_KEY_TO_NAME.get(key, key.split(".")[-1]) for key in keys]
    ordered = [images[name] for name in ordered_names]
    mode = str(video_concat_mode or "horizontal")

    if mode == "robocasa365_compact":
        if len(ordered) != 3:
            raise ValueError(f"robocasa365_compact expects 3 cameras, got {len(ordered)}")
        wrist = _resize_rgb(ordered[0], (256, 256))
        left = _resize_rgb(ordered[1], (128, 128))
        right = _resize_rgb(ordered[2], (128, 128))
        frame = np.concatenate([wrist, np.concatenate([left, right], axis=1)], axis=0)
    elif mode == "horizontal":
        per_cam_w = width // max(len(ordered), 1)
        frame = np.concatenate([
            _center_crop_resize_rgb(image, (height, per_cam_w))
            for image in ordered
        ], axis=1)
    elif mode == "vertical":
        per_cam_h = height // max(len(ordered), 1)
        frame = np.concatenate([
            _center_crop_resize_rgb(image, (per_cam_h, width))
            for image in ordered
        ], axis=0)
    else:
        raise ValueError(f"Unsupported RoboCasa eval render mode: {mode!r}")

    if frame.shape[:2] != (height, width):
        raise ValueError(f"Rendered model frame shape={frame.shape[:2]}, expected {(height, width)}")
    return frame


def get_robocasa_proprio(obs: Mapping, expected_dim: Optional[int] = None) -> np.ndarray:


    for key in ("observation.state", "state"):
        value = _get_by_path(obs, key)
        if value is None:
            continue
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if expected_dim is not None and arr.shape[0] != expected_dim:
            raise ValueError(
                f"RoboCasa {key!r} has dim {arr.shape[0]}, expected {expected_dim}. "
                "Check `shape_meta.state` in the data config."
            )
        return arr

    split_keys = [
        "state.base_position",
        "state.base_rotation",
        "state.end_effector_position_relative",
        "state.end_effector_rotation_relative",
    ]
    split_vals = []
    for key in split_keys:
        value = _get_by_path(obs, key)
        if value is None:
            split_vals = []
            break
        split_vals.append(np.asarray(value, dtype=np.float32).reshape(-1))
    if split_vals:
        gripper = _get_by_path(obs, "state.gripper_gpos")
        if gripper is None:
            gripper = _get_by_path(obs, "state.gripper_qpos")
        if gripper is not None:
            split_vals.append(np.asarray(gripper, dtype=np.float32).reshape(-1))
            arr = np.concatenate(split_vals, axis=0)
            if expected_dim is not None and arr.shape[0] != expected_dim:
                raise ValueError(
                    f"RoboCasa split-state has dim {arr.shape[0]}, expected {expected_dim}. "
                    "Check state key mapping and `shape_meta.state`."
                )
            return arr

    available_keys = [item["key"] for item in flatten_obs_keys(obs)]
    raise KeyError(
        f"Cannot find `observation.state` / `state` in RoboCasa obs. Available keys: {available_keys}"
    )


def get_robocasa_task_description(obs: Mapping, fallback: Optional[str] = None) -> str:

    value = _get_by_path(obs, "annotation.human.task_description")
    if isinstance(value, str) and value.strip():
        return value.strip()
    if fallback is not None and str(fallback).strip():
        return str(fallback).strip()
    raise ValueError("RoboCasa obs has no 'annotation.human.task_description' and no fallback provided.")

def _resolve_action_key_mapping(
    action_key_mapping: Optional[Mapping[str, Any]],
    action_dim: int,
) -> dict[str, tuple[int, int]]:


    mapping: dict[str, tuple[int, int]] = dict(DEFAULT_ACTION_KEY_MAPPING)
    if action_key_mapping is None:
        return mapping

    for key, value in action_key_mapping.items():
        if key not in mapping:
            raise KeyError(f"Unsupported action mapping key {key!r}. Supported keys: {list(mapping.keys())}")
        if isinstance(value, (list, tuple)) and len(value) == 2:
            start, end = int(value[0]), int(value[1])
        else:
            raise ValueError(
                f"Action mapping value for {key!r} must be [start, end], got {value!r} (type={type(value).__name__})."
            )
        if not (0 <= start < end <= action_dim):
            raise ValueError(
                f"Invalid action mapping range for {key!r}: [{start}, {end}) with action_dim={action_dim}."
            )
        mapping[key] = (start, end)
    return mapping

def postprocess_robocasa_action(
    action_flat: np.ndarray,
    *,
    binarize_gripper: bool = True,
    binarize_ctrl_mode: bool = True,
    zero_base_d3: bool = True,
    base_motion_deadzone: float = 0.0,
    gripper_threshold: float = 0.0,
    ctrl_mode_threshold: float = 0.0,
) -> np.ndarray:


    a = np.asarray(action_flat, dtype=np.float32).reshape(-1).copy()
    if a.shape[0] != 12:
        raise ValueError(f"postprocess_robocasa_action expects 12-dim flat action, got {a.shape[0]}")

    if zero_base_d3:
        a[3] = 0.0

    if binarize_ctrl_mode:
        a[4] = 1.0 if a[4] > ctrl_mode_threshold else -1.0

    if binarize_gripper:
        a[11] = 1.0 if a[11] > gripper_threshold else -1.0

    if base_motion_deadzone > 0.0:
        mask = np.abs(a[0:3]) < float(base_motion_deadzone)
        a[0:3] = np.where(mask, 0.0, a[0:3])

    return a

def convert_action_to_robocasa_dict(
    action_flat: np.ndarray,
    action_key_mapping: Optional[Mapping[str, Any]] = None,
) -> dict[str, np.ndarray]:


    action = np.asarray(action_flat, dtype=np.float32).reshape(-1)
    if action.shape[0] != 12:
        raise ValueError(f"Expected flat action dim 12, got {action.shape[0]}")
    mapping = _resolve_action_key_mapping(action_key_mapping, action_dim=action.shape[0])
    return {key: action[start:end].copy() for key, (start, end) in mapping.items()}

def adapt_action_for_robocasa(
    action_step: np.ndarray,
    env,
    action_key_mapping: Optional[Mapping[str, Any]] = None,
    postprocess_kwargs: Optional[Mapping[str, Any]] = None,
) -> Any:


    import gymnasium as gym
    from gymnasium.spaces.utils import flatdim

    action_flat = np.asarray(action_step, dtype=np.float32).reshape(-1)
    if postprocess_kwargs:
        if action_flat.shape[0] != 12:
            raise ValueError(
                f"postprocess_kwargs is only supported for 12-dim RoboCasa actions, got dim={action_flat.shape[0]}"
            )
        action_flat = postprocess_robocasa_action(action_flat, **dict(postprocess_kwargs))

    space = env.action_space

    if isinstance(space, gym.spaces.Box):
        target_dim = int(np.prod(space.shape))
        if action_flat.shape[0] != target_dim:
            raise ValueError(
                f"action dim mismatch for Box action_space: model={action_flat.shape[0]}, env={target_dim}"
            )
        low = np.asarray(space.low, dtype=np.float32).reshape(-1)
        high = np.asarray(space.high, dtype=np.float32).reshape(-1)
        return np.clip(action_flat, low, high).reshape(space.shape)

    if isinstance(space, gym.spaces.Dict):
        total_dim = flatdim(space)
        if action_flat.shape[0] != total_dim:
            raise ValueError(
                f"action dim mismatch for Dict action_space: model={action_flat.shape[0]}, env flat dim={total_dim}"
            )
        return convert_action_to_robocasa_dict(action_flat, action_key_mapping=action_key_mapping)

    raise NotImplementedError(f"Unsupported action_space type: {type(space).__name__}")


def _to_even_frame(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    pad_h = h % 2
    pad_w = w % 2
    if pad_h == 0 and pad_w == 0:
        return frame
    return np.pad(frame, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")


def _concat_frames_by_longest_side(frames: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(frame) for frame in frames]
    if not arrays:
        raise ValueError("Cannot concatenate an empty frame list")
    height, width = arrays[0].shape[:2]
    axis = 1 if width >= height else 0
    return np.concatenate(arrays, axis=axis)


def _concat_frames_by_shared_long_edge(frames: Iterable[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(frame) for frame in frames]
    if not arrays:
        raise ValueError("Cannot concatenate an empty frame list")
    height, width = arrays[0].shape[:2]
    for frame in arrays[1:]:
        if frame.shape[:2] != (height, width):
            raise ValueError(
                f"Cannot stitch frames with different spatial shapes: "
                f"{frame.shape[:2]} vs {(height, width)}"
            )
    axis = 0 if width >= height else 1
    return np.concatenate(arrays, axis=axis)


def save_mp4(frames: Iterable, path: str, fps: int = 8) -> None:

    out_dir = os.path.dirname(path) or "."
    os.makedirs(out_dir, exist_ok=True)
    writer = imageio.get_writer(
        path, fps=max(int(fps), 1), codec="libx264", format="FFMPEG", pixelformat="yuv420p",
    )
    try:
        for frame in frames:
            arr = np.array(frame.convert("RGB")) if isinstance(frame, Image.Image) else np.asarray(frame)
            writer.append_data(_to_even_frame(arr))
    finally:
        writer.close()


def get_future_frame_capture_steps(replan_steps: int, action_video_freq_ratio: int) -> list[int]:

    num_future_frames = int(replan_steps) // int(action_video_freq_ratio)
    return [step_idx * int(action_video_freq_ratio) for step_idx in range(num_future_frames + 1)]


def frame_to_rgb_array(frame: Any) -> np.ndarray:

    if isinstance(frame, dict):
        images = []
        for value in frame.values():
            value_array = np.array(value) if isinstance(value, Image.Image) else np.array(value, copy=True)
            images.append(value_array)
        return np.concatenate(images, axis=1)
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"))
    return np.array(frame, copy=True)


def compute_clip_mean_psnr(gt_frames: list, pred_frames: list, eps: float = 1e-8) -> Optional[float]:

    if len(gt_frames) == 0 or len(pred_frames) == 0:
        return None
    if len(gt_frames) != len(pred_frames):
        raise AssertionError(f"GT/pred frame count mismatch: gt={len(gt_frames)} pred={len(pred_frames)}")

    frame_psnr_values = []
    for gt_frame, pred_frame in zip(gt_frames, pred_frames):
        gt_image = frame_to_rgb_array(gt_frame)
        pred_image = frame_to_rgb_array(pred_frame)
        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            raise AssertionError(
                f"GT/pred frame shape mismatch: gt={gt_image.shape[:2]} pred={pred_image.shape[:2]}. "
                "Render GT with the same model video layout before computing PSNR."
            )

        gt_f32 = gt_image.astype(np.float32)
        pred_f32 = pred_image.astype(np.float32)
        mse = float(np.mean((pred_f32 - gt_f32) ** 2))
        psnr = 10.0 * np.log10((255.0 * 255.0) / max(mse, eps))
        frame_psnr_values.append(float(psnr))
    return float(np.mean(frame_psnr_values)) if frame_psnr_values else None


def save_rollout_video(
    rollout_dir,
    rollout_images,
    idx,
    success,
    task_description,
    log_file=None,
    fps=20,
):

    processed_task_description = re.sub(r'[^\w\-]', '_', task_description.lower())
    filename = f"episode={idx}--success={success}--{processed_task_description}.mp4"
    if len(filename.encode("utf-8")) > 255:
        prefix = f"episode={idx}--success={success}--"
        suffix = ".mp4"
        max_desc_bytes = 255 - len(prefix.encode("utf-8")) - len(suffix.encode("utf-8"))
        processed_task_description = processed_task_description[-max_desc_bytes:]
        filename = f"{prefix}{processed_task_description}{suffix}"
    mp4_path = f"{rollout_dir}/{filename}"
    video_writer = imageio.get_writer(mp4_path, fps=fps)
    for img in rollout_images:
        if isinstance(img, dict):
            panels = []
            for key, value in img.items():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                pil_img = Image.fromarray(value_array)
                draw = ImageDraw.Draw(pil_img)
                draw.text((10, 10), f"{key}", fill=(255, 255, 255))
                panels.append(np.array(pil_img))
            frame = _concat_frames_by_longest_side(panels)
        elif isinstance(img, Image.Image):
            frame = np.array(img.convert("RGB"))
        else:
            frame = np.array(img)
        video_writer.append_data(frame)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def save_prediction_video(
    rollout_dir,
    gt_frames,
    pred_frames,
    idx,
    replan_idx,
    success,
    task_description,
    log_file=None,
    fps=8,
):

    num_frames = min(len(gt_frames), len(pred_frames))
    if num_frames <= 0:
        raise ValueError("Cannot save prediction video with empty GT/pred frame lists.")

    stitched = []
    for gt_frame, pred_frame in zip(gt_frames[:num_frames], pred_frames[:num_frames]):
        if isinstance(gt_frame, dict):
            panels = []
            for value in gt_frame.values():
                value_array = np.array(value) if isinstance(value, Image.Image) else value.copy()
                panels.append(value_array)
            gt_image = np.concatenate(panels, axis=1)
        elif isinstance(gt_frame, Image.Image):
            gt_image = np.array(gt_frame.convert("RGB"))
        else:
            gt_image = np.array(gt_frame)

        if isinstance(pred_frame, Image.Image):
            pred_image = np.array(pred_frame.convert("RGB"))
        else:
            pred_image = np.array(pred_frame)

        target_h, target_w = pred_image.shape[:2]
        if gt_image.shape[:2] != (target_h, target_w):
            raise ValueError(
                f"GT/pred frame shape mismatch: gt={gt_image.shape[:2]} pred={pred_image.shape[:2]}. "
                "Render GT with the same model video layout before saving prediction video."
            )

        gt_pil = Image.fromarray(gt_image)
        ImageDraw.Draw(gt_pil).text((10, 10), "gt", fill=(255, 255, 255))
        pred_pil = Image.fromarray(pred_image)
        ImageDraw.Draw(pred_pil).text((10, 10), "pred", fill=(255, 255, 255))
        stitched.append(Image.fromarray(_concat_frames_by_shared_long_edge([np.array(pred_pil), np.array(gt_pil)])))

    processed_task_description = re.sub(r'[^\w\-]', '_', task_description.lower())
    try:
        replan_tag = f"{int(replan_idx):04d}"
    except (TypeError, ValueError):
        replan_tag = str(replan_idx)
    filename = (
        f"episode={idx}--success={success}"
        f"--{processed_task_description}--replan={replan_tag}--gt-pred.mp4"
    )
    if len(filename.encode("utf-8")) > 255:
        prefix = f"episode={idx}--success={success}--"
        suffix = f"--replan={replan_tag}--gt-pred.mp4"
        max_desc_bytes = 255 - len(prefix.encode("utf-8")) - len(suffix.encode("utf-8"))
        processed_task_description = processed_task_description[-max_desc_bytes:]
        filename = f"{prefix}{processed_task_description}{suffix}"
    mp4_path = f"{rollout_dir}/{filename}"
    save_mp4(stitched, mp4_path, fps=fps)
    print(f"Saved predicted future comparison MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved predicted future comparison MP4 at path {mp4_path}\n")
    return mp4_path
