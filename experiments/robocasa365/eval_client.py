from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.robocasa365.robocasa_utils import (
    DEFAULT_ACTION_KEY_MAPPING,
    DEFAULT_OBS_CAM_KEYS,
    adapt_action_for_robocasa,
    compute_clip_mean_psnr,
    get_future_frame_capture_steps,
    get_robocasa_env,
    get_robocasa_image,
    get_robocasa_task_description,
    render_robocasa_model_frame,
    save_prediction_video,
    save_rollout_video,
)
from experiments.sim_remote import protocol
from experiments.sim_remote.client import WamRemoteClient
from experiments.robocasa365.runtime_config import load_config, set_by_dotted_key


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def _cfg_get(cfg: dict[str, Any], *keys: str, default=None):
    cur: Any = cfg
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_cam_keys(cfg: dict[str, Any]) -> list[str]:
    cam_keys = _cfg_get(cfg, "evaluation", "camera_keys")
    if cam_keys is None or (isinstance(cam_keys, str) and not cam_keys.strip()):
        data_keys = _cfg_get(cfg, "data", "camera_keys")
        return [str(k) for k in (data_keys or DEFAULT_OBS_CAM_KEYS)]
    if isinstance(cam_keys, str):
        return [x.strip() for x in cam_keys.split(",") if x.strip()]
    return [str(x) for x in cam_keys]


def _action_postprocess(cfg: dict[str, Any]) -> dict[str, Any] | None:
    raw = _cfg_get(cfg, "evaluation", "action_postprocess")
    if not raw:
        return None
    kwargs = dict(raw)
    active = (
        bool(kwargs.get("binarize_gripper", False))
        or bool(kwargs.get("binarize_ctrl_mode", False))
        or bool(kwargs.get("zero_base_d3", False))
        or float(kwargs.get("base_motion_deadzone", 0.0) or 0.0) > 0.0
    )
    return kwargs if active else None


def _expected_num_video_frames(cfg: dict[str, Any]) -> int:
    num_frames = int(cfg["data"].get("num_frames", 33))
    ratio = int(cfg["data"].get("action_video_freq_ratio", 4))
    if ratio <= 0:
        raise ValueError(f"action_video_freq_ratio must be positive, got {ratio}")
    return (num_frames - 1) // ratio + 1


def _as_bool_value(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() not in {"", "0", "false", "no", "off", "none", "null"}


def _validate_server_info(cfg: dict[str, Any], info: dict[str, Any]) -> None:
    eval_cfg = cfg.get("evaluation", {}) or {}
    strict = bool(eval_cfg.get("strict_server_config_match", True))
    problems: list[str] = []

    if bool(eval_cfg.get("visualize_future_video", False)):
        expected_video_frames = _expected_num_video_frames(cfg)
        got_video_frames = info.get("num_video_frames")
        if got_video_frames is None:
            problems.append("server /info is missing num_video_frames")
        elif int(got_video_frames) != expected_video_frames:
            problems.append(f"server num_video_frames={got_video_frames}, expected {expected_video_frames}")
        if "num_chunk_frames" not in info:
            problems.append("server /info is missing num_chunk_frames; this usually means the loaded server is stale")

    got_legacy = info.get("use_legacy_eval_processing")
    if got_legacy is None:
        problems.append("server /info is missing use_legacy_eval_processing; restart the server")
    elif not _as_bool_value(got_legacy):
        problems.append(f"server use_legacy_eval_processing={got_legacy!r}, expected True")

    expected_prompt_style = "legacy"
    expected_input_style = "legacy"

    if expected_prompt_style:
        got_prompt_style = info.get("prompt_style")
        if got_prompt_style is None:
            problems.append("server /info is missing prompt_style; restart the server to pick up prompt selection")
        elif str(got_prompt_style).strip().lower() != str(expected_prompt_style).strip().lower():
            problems.append(f"server prompt_style={got_prompt_style!r}, expected {expected_prompt_style!r}")

    if expected_input_style:
        got_input_style = info.get("input_processing_style")
        if got_input_style is None:
            problems.append("server /info is missing input_processing_style; restart the server to pick up input processing selection")
        elif str(got_input_style).strip().lower() != str(expected_input_style).strip().lower():
            problems.append(f"server input_processing_style={got_input_style!r}, expected {expected_input_style!r}")

    expected_video_size = cfg.get("data", {}).get("video_size")
    if expected_video_size:
        got_h = info.get("input_h")
        got_w = info.get("input_w")
        if got_h is None or got_w is None:
            problems.append("server /info is missing input_h/input_w")
        elif [int(got_h), int(got_w)] != [int(expected_video_size[0]), int(expected_video_size[1])]:
            problems.append(f"server input size={[got_h, got_w]}, expected {list(expected_video_size)}")

    expected_concat_mode = cfg.get("data", {}).get("video_concat_mode")
    if expected_concat_mode:
        got_concat_mode = info.get("video_concat_mode")
        if got_concat_mode is None:
            problems.append("server /info is missing video_concat_mode")
        elif str(got_concat_mode) != str(expected_concat_mode):
            problems.append(f"server video_concat_mode={got_concat_mode!r}, expected {expected_concat_mode!r}")

    if "model_class" not in info:
        problems.append("server /info is missing model_class; this usually means the loaded server is stale")

    if not problems:
        return
    message = "Server/config mismatch:\n  - " + "\n  - ".join(problems)
    if strict:
        raise RuntimeError(message)
    logging.warning(message)


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"", "0", "false", "no", "off"}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _flat_env_action(env_action: Any) -> np.ndarray:
    if isinstance(env_action, dict):
        flat = np.zeros((12,), dtype=np.float32)
        for key, (start, end) in DEFAULT_ACTION_KEY_MAPPING.items():
            if key not in env_action:
                continue
            value = np.asarray(env_action[key], dtype=np.float32).reshape(-1)
            flat[start:end] = value[: end - start]
        return flat
    return np.asarray(env_action, dtype=np.float32).reshape(-1)


def _log_flat_action_summary(label: str, value: Any) -> None:
    arr = np.asarray(value, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        logging.info("[action-debug] %s empty", label)
        return
    parts = [
        f"{label} dim={arr.shape[0]}",
        f"all mean={float(arr.mean()):.4f} std={float(arr.std()):.4f} min={float(arr.min()):.4f} max={float(arr.max()):.4f}",
    ]
    for key, (start, end) in DEFAULT_ACTION_KEY_MAPPING.items():
        if arr.shape[0] < end:
            continue
        chunk = arr[start:end]
        first = np.array2string(chunk, precision=4, separator=",", suppress_small=False)
        name = key.removeprefix("action.")
        parts.append(
            f"{name}[{start}:{end}] mean={float(chunk.mean()):.4f} std={float(chunk.std()):.4f} "
            f"min={float(chunk.min()):.4f} max={float(chunk.max()):.4f} value={first}"
        )
    logging.info("[action-debug] %s", " | ".join(parts))


def _episode_list(value: Any, total_trials: int) -> list[int]:
    out: list[int] = []
    if not isinstance(value, list):
        return out
    for item in value:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < total_trials and idx not in out:
            out.append(idx)
    return sorted(out)


def _load_json(path: Path | None) -> dict[str, Any]:
    if path is None or not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logging.warning("Failed to read resume file %s: %r", path, exc)
        return {}


def _merge_video_resume(results: dict[str, Any], video_dir: Path, env_name: str, total_trials: int) -> None:
    success_set = set(_episode_list(results.get("success_episodes"), total_trials))
    failure_set = set(_episode_list(results.get("failure_episodes"), total_trials))
    seen = success_set | failure_set
    pattern = re.compile(rf"^episode={re.escape(env_name)}_trial(?P<trial>\d+)--success=(?P<success>True|False)--.*\.mp4$")
    if not video_dir.is_dir():
        return
    for path in sorted(video_dir.glob(f"episode={env_name}_trial*--success=*.mp4")):
        match = pattern.match(path.name)
        if match is None:
            continue
        idx = int(match.group("trial"))
        if idx < 0 or idx >= total_trials or idx in seen:
            continue
        if match.group("success") == "True":
            success_set.add(idx)
        else:
            failure_set.add(idx)
        seen.add(idx)
    results["success_episodes"] = sorted(success_set)
    results["failure_episodes"] = sorted(failure_set)
    results["successes"] = len(success_set)


def _completed_trials(results: dict[str, Any], total_trials: int) -> set[int]:
    return set(_episode_list(results.get("success_episodes"), total_trials)) | set(_episode_list(results.get("failure_episodes"), total_trials))


def _write_progress(
    output_file: Path | None,
    base_results: dict[str, Any] | None,
    task_results: dict[str, Any],
    *,
    total_trials: int,
    start_time: float,
) -> None:
    if output_file is None:
        return
    output_file.parent.mkdir(parents=True, exist_ok=True)
    merged = dict(base_results or {})
    merged.update(task_results)
    merged["total_episodes"] = total_trials
    completed = len(_completed_trials(merged, total_trials))
    merged["completed_episodes"] = completed
    merged["success_rate"] = int(merged.get("successes", 0)) / max(1, total_trials)
    merged["completed_success_rate"] = int(merged.get("successes", 0)) / max(1, completed)
    merged["duration"] = time.time() - start_time
    tmp = output_file.with_name(output_file.name + ".tmp")
    tmp.write_text(json.dumps(merged, indent=4, cls=NumpyEncoder) + "\n", encoding="utf-8")
    tmp.replace(output_file)


def _find_existing_result_file(group_dir: Path, env_name: str) -> Path | None:
    if not group_dir.is_dir():
        return None
    files = sorted(group_dir.glob(f"gpu*_{env_name}_results.json"))
    return files[0] if files else None


def run_single_episode(
    *,
    env,
    seed: int,
    task_description: str,
    cfg: dict[str, Any],
    episode_idx: int,
    task_horizon: int,
    cam_keys: list[str],
    client: WamRemoteClient,
    session_id: str,
    action_key_mapping: dict[str, list[int]] | None,
    action_postprocess: dict[str, Any] | None,
    model_video_size: tuple[int, int],
    model_video_concat_mode: str,
) -> tuple[bool, list, list[dict[str, Any]], float | None, str]:
    eval_cfg = cfg["evaluation"]
    max_steps = int(task_horizon)
    replan_steps = int(eval_cfg.get("replan_steps") or max(int(cfg["data"].get("num_frames", 33)) - 1, 1))
    visualize_future_video = bool(eval_cfg.get("visualize_future_video", False))
    action_video_freq_ratio = int(cfg["data"].get("action_video_freq_ratio", 4))
    capture_steps = set(get_future_frame_capture_steps(replan_steps, action_video_freq_ratio)[1:])

    obs, _ = env.reset(seed=int(seed))
    episode_task_description = get_robocasa_task_description(obs, fallback=task_description)
    reset_reply = client.reset(
        session_id=session_id,
        task_description=episode_task_description,
        replan_steps=replan_steps,
        action_horizon=eval_cfg.get("action_horizon"),
        use_action_ensembler=bool(eval_cfg.get("use_action_ensembler", False)),
        visualize_future_video=visualize_future_video,
        num_inference_steps=eval_cfg.get("num_inference_steps"),
        num_video_inference_steps=eval_cfg.get("num_video_inference_steps"),
        num_action_inference_steps=eval_cfg.get("num_action_inference_steps"),
        sigma_shift=eval_cfg.get("sigma_shift"),
        text_cfg_scale=float(eval_cfg.get("text_cfg_scale", 1.0)),
        negative_prompt=str(eval_cfg.get("negative_prompt", "")),
        rand_device=str(eval_cfg.get("rand_device", "cpu")),
        tiled=bool(eval_cfg.get("tiled", False)),
        camera_keys=cam_keys,
        seed=(None if cfg.get("seed") is None else int(cfg["seed"])),
    )
    logging.info(
        "Reset session=%s task=%r prompt=%r",
        session_id,
        episode_task_description,
        reset_reply.get("prompt"),
    )

    replay_images: list[dict[str, Any]] = []
    predicted_future_video_clips: list[dict[str, Any]] = []
    episode_future_clip_psnr: list[float] = []
    current_predicted_future_clip: dict[str, Any] | None = None
    current_replan_step = 0
    current_replan_idx = -1
    has_memory = False
    memory_chunk_action_counter = 0
    memory_future_obs_buffer: list[dict] = []

    success_hit = False
    pbar = tqdm(total=max_steps, desc=f"Episode {episode_idx + 1}")
    for t in range(max_steps):
        pbar.update(1)
        response = client.act(session_id, obs)
        action = response["action"]
        is_replan_boundary = bool(response.get("is_replan_boundary", False))
        predicted_future_frames = response.get("predicted_future_frames")

        if is_replan_boundary:
            has_memory = bool(response.get("has_memory", False))
            memory_chunk_action_counter = 0
            memory_future_obs_buffer.clear()

        imgs = get_robocasa_image(obs, cam_keys=cam_keys)
        replay_images.append(imgs.copy())

        if is_replan_boundary and visualize_future_video and predicted_future_frames is not None:
            current_replan_idx += 1
            current_predicted_future_clip = {
                "replan_idx": current_replan_idx,
                "gt_frames": [
                    render_robocasa_model_frame(
                        obs,
                        cam_keys=cam_keys,
                        video_size=model_video_size,
                        video_concat_mode=model_video_concat_mode,
                    )
                ],
                "pred_frames": list(predicted_future_frames),
            }
            current_replan_step = 0
        elif is_replan_boundary:
            current_predicted_future_clip = None
            current_replan_step = 0

        env_action = adapt_action_for_robocasa(
            np.asarray(action),
            env,
            action_key_mapping=action_key_mapping,
            postprocess_kwargs=action_postprocess,
        )
        if _env_bool("ABOT06_LOG_ENV_ACTION", False) and t < _env_int("ABOT06_LOG_ENV_ACTION_N", 16):
            if t == 0:
                logging.info("[action-debug] env_action_space=%s", env.action_space)
                logging.info("[action-debug] action_postprocess=%s action_key_mapping=%s", action_postprocess, action_key_mapping)
            logging.info(
                "[action-debug] step=%d replan_boundary=%s env_action_type=%s",
                t,
                is_replan_boundary,
                type(env_action).__name__,
            )
            _log_flat_action_summary(f"step={t} server_action", action)
            _log_flat_action_summary(f"step={t} env_action", _flat_env_action(env_action))
        obs, _, terminated, truncated, info = env.step(env_action)
        success_hit = bool(info.get("success", False))
        should_stop = success_hit or bool(terminated) or bool(truncated)

        if has_memory:
            memory_chunk_action_counter += 1
            if memory_chunk_action_counter % action_video_freq_ratio == 0:
                memory_future_obs_buffer.append(obs)
            remaining = int(response.get("remaining_chunk", 0))
            if remaining == 0 or should_stop:
                if memory_future_obs_buffer:
                    reply = client.update_timeline(session_id, memory_future_obs_buffer)
                    if not reply.get("updated", False) and not should_stop:
                        logging.warning("Memory timeline was not updated: %s", reply)
                memory_future_obs_buffer.clear()
                memory_chunk_action_counter = 0

        if visualize_future_video and current_predicted_future_clip is not None:
            current_replan_step += 1
            if current_replan_step in capture_steps:
                current_predicted_future_clip["gt_frames"].append(
                    render_robocasa_model_frame(
                        obs,
                        cam_keys=cam_keys,
                        video_size=model_video_size,
                        video_concat_mode=model_video_concat_mode,
                    )
                )
            remaining = int(response.get("remaining_chunk", 0))
            if should_stop or remaining == 0:
                num_compare = min(len(current_predicted_future_clip["gt_frames"]), len(current_predicted_future_clip["pred_frames"]))
                if num_compare > 0:
                    current_predicted_future_clip["gt_frames"] = current_predicted_future_clip["gt_frames"][:num_compare]
                    current_predicted_future_clip["pred_frames"] = current_predicted_future_clip["pred_frames"][:num_compare]
                    clip_psnr = compute_clip_mean_psnr(current_predicted_future_clip["gt_frames"], current_predicted_future_clip["pred_frames"])
                    if clip_psnr is not None:
                        episode_future_clip_psnr.append(float(clip_psnr))
                predicted_future_video_clips.append(current_predicted_future_clip)
                current_predicted_future_clip = None

        if should_stop:
            break
    pbar.close()
    episode_mean_psnr = float(np.mean(episode_future_clip_psnr)) if episode_future_clip_psnr else None
    return bool(success_hit), replay_images, predicted_future_video_clips, episode_mean_psnr, episode_task_description


def run_single_task(
    env_name: str,
    cfg: dict[str, Any],
    video_dir: Path,
    predicted_video_dir: Path,
    predicted_video_all_dir: Path,
    *,
    cam_keys: list[str],
    client: WamRemoteClient,
    action_key_mapping: dict[str, list[int]] | None,
    action_postprocess: dict[str, Any] | None,
    model_video_size: tuple[int, int],
    model_video_concat_mode: str,
    base_results: dict[str, Any] | None = None,
    output_file: Path | None = None,
    resume_trials: bool = False,
    start_time: float | None = None,
) -> dict[str, Any]:
    eval_cfg = cfg["evaluation"]
    split = str(eval_cfg.get("split", "target"))
    base_seed_cfg = eval_cfg.get("base_seed")
    base_seed = int(base_seed_cfg) if base_seed_cfg is not None else int(cfg.get("seed") or 0)
    logging.info("Creating RoboCasa env env_name=%s split=%s seed=%s", env_name, split, base_seed)
    env, task_horizon_default = get_robocasa_env(env_name, split=split, seed=base_seed)
    logging.info("Created RoboCasa env env_name=%s horizon_default=%s", env_name, task_horizon_default)
    task_horizon = int(eval_cfg.get("max_steps") or task_horizon_default)
    logging.info("Task %s horizon=%s default=%s", env_name, task_horizon, task_horizon_default)

    total_trials = int(eval_cfg.get("num_trials", 1))
    visualize_future_video = bool(eval_cfg.get("visualize_future_video", False))
    results: dict[str, Any] = {
        "successes": 0,
        "failure_episodes": [],
        "success_episodes": [],
        "task_description": None,
        "task_horizon": task_horizon,
    }
    if visualize_future_video:
        results["episode_future_video_psnr"] = []
        results["future_video_psnr_mean"] = None
    if resume_trials:
        saved = _load_json(output_file)
        if saved:
            for key in ("success_episodes", "failure_episodes", "successes", "task_description", "task_horizon"):
                if key in saved:
                    results[key] = saved[key]
            if visualize_future_video:
                results["episode_future_video_psnr"] = list(saved.get("episode_future_video_psnr") or [])
                results["future_video_psnr_mean"] = saved.get("future_video_psnr_mean")
        _merge_video_resume(results, video_dir, env_name, total_trials)
        done = _completed_trials(results, total_trials)
        if done:
            logging.info("Resume enabled: skipping %d/%d completed trials for %s", len(done), total_trials, env_name)
            _write_progress(output_file, base_results, results, total_trials=total_trials, start_time=start_time or time.time())
    try:
        for trial_idx in range(total_trials):
            if resume_trials and trial_idx in _completed_trials(results, total_trials):
                logging.info("Skipping completed trial %s/%s for %s", trial_idx, total_trials, env_name)
                continue
            session_id = f"{env_name}-{trial_idx}-{uuid.uuid4().hex[:8]}"
            success, replay_images, predicted_clips, episode_mean_psnr, episode_task_description = run_single_episode(
                env=env,
                seed=base_seed + trial_idx,
                task_description=env_name,
                cfg=cfg,
                episode_idx=trial_idx,
                task_horizon=task_horizon,
                cam_keys=cam_keys,
                client=client,
                session_id=session_id,
                action_key_mapping=action_key_mapping,
                action_postprocess=action_postprocess,
                model_video_size=model_video_size,
                model_video_concat_mode=model_video_concat_mode,
            )
            if success:
                results["successes"] += 1
                results["success_episodes"].append(trial_idx)
            else:
                results["failure_episodes"].append(trial_idx)
            if results.get("task_description") is None:
                results["task_description"] = episode_task_description
            save_rollout_video(video_dir, replay_images, f"{env_name}_trial{trial_idx}", success=success, task_description=episode_task_description)
            if visualize_future_video:
                results["episode_future_video_psnr"].append(episode_mean_psnr)
                if not predicted_clips:
                    logging.warning("No predicted future frames for task %s trial %s.", env_name, trial_idx)
                else:
                    all_gt_frames, all_pred_frames = [], []
                    for clip in predicted_clips:
                        all_gt_frames.extend(clip["gt_frames"])
                        all_pred_frames.extend(clip["pred_frames"])
                        save_prediction_video(
                            predicted_video_dir,
                            clip["gt_frames"],
                            clip["pred_frames"],
                            f"{env_name}_trial{trial_idx}",
                            clip["replan_idx"],
                            success=success,
                            task_description=episode_task_description,
                        )
                    save_prediction_video(
                        predicted_video_all_dir,
                        all_gt_frames,
                        all_pred_frames,
                        f"{env_name}_trial{trial_idx}",
                        "all",
                        success=success,
                        task_description=episode_task_description,
                    )
            _write_progress(output_file, base_results, results, total_trials=total_trials, start_time=start_time or time.time())
    finally:
        env.close()
    if visualize_future_video:
        valid_psnr = [x for x in results["episode_future_video_psnr"] if x is not None]
        if valid_psnr:
            results["future_video_psnr_mean"] = float(np.mean(valid_psnr))
    _write_progress(output_file, base_results, results, total_trials=total_trials, start_time=start_time or time.time())
    return results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RoboCasa365 eval client")
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--group-name", default="default")
    parser.add_argument("--remote-host", default=protocol.DEFAULT_HOST)
    parser.add_argument("--remote-port", type=int, default=protocol.DEFAULT_PORT)
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--num-trials", type=int, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--override", action="append", default=[])
    args, unknown = parser.parse_known_args()
    args.override.extend(unknown)
    return args


def main() -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s client: %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    cfg = load_config(args.config, args.override)
    set_by_dotted_key(cfg, "evaluation.env_name", args.env_name)
    set_by_dotted_key(cfg, "evaluation.group_name", args.group_name)
    set_by_dotted_key(cfg, "remote.host", args.remote_host)
    set_by_dotted_key(cfg, "remote.port", args.remote_port)
    set_by_dotted_key(cfg, "gpu_id", args.gpu_id)
    if args.num_trials is not None:
        set_by_dotted_key(cfg, "evaluation.num_trials", int(args.num_trials))
    if args.output_dir:
        set_by_dotted_key(cfg, "evaluation.output_dir", args.output_dir)

    start_time = time.time()
    env_name = str(cfg["evaluation"].get("env_name") or "").replace("robocasa/", "")
    if not env_name:
        raise ValueError("--env-name is required")
    group_name = str(cfg["evaluation"].get("group_name") or "default")
    host = str(cfg["remote"].get("host") or protocol.DEFAULT_HOST)
    port = int(cfg["remote"].get("port") or protocol.DEFAULT_PORT)
    client = WamRemoteClient(host=host, port=port, timeout_s=float(cfg["remote"].get("timeout_s", protocol.DEFAULT_TIMEOUT_S)))
    info = client.wait_until_ready(max_wait_s=float(cfg["remote"].get("max_wait_ready_s", 600)), poll_interval_s=2.0)
    logging.info("Connected to server http://%s:%d info=%s", host, port, info)
    _validate_server_info(cfg, info)
    model_video_size = (int(info["input_h"]), int(info["input_w"]))
    model_video_concat_mode = str(info.get("video_concat_mode") or cfg.get("data", {}).get("video_concat_mode") or "horizontal")

    local_log_dir = Path(cfg["evaluation"].get("output_dir") or "evaluate_results")
    local_log_dir.mkdir(parents=True, exist_ok=True)
    results_root_dir = os.environ.get("RESULTS_ROOT_DIR", "").strip()
    if results_root_dir:
        video_dir = Path(results_root_dir) / env_name / "videos"
        predicted_video_dir = Path(results_root_dir) / env_name / "predicted_videos"
        predicted_video_all_dir = Path(results_root_dir) / env_name / "predicted_videos_all"
    else:
        video_dir = local_log_dir / group_name / "videos"
        predicted_video_dir = local_log_dir / group_name / "predicted_videos"
        predicted_video_all_dir = local_log_dir / group_name / "predicted_videos_all"
    video_dir.mkdir(parents=True, exist_ok=True)
    if bool(cfg["evaluation"].get("visualize_future_video", False)):
        predicted_video_dir.mkdir(parents=True, exist_ok=True)
        predicted_video_all_dir.mkdir(parents=True, exist_ok=True)

    cam_keys = _resolve_cam_keys(cfg)
    action_key_mapping = cfg["evaluation"].get("action_key_mapping")
    action_postprocess = _action_postprocess(cfg)
    results: dict[str, Any] = {
        "group_name": group_name,
        "env_name": env_name,
        "split": str(cfg["evaluation"].get("split", "target")),
        "task_description": None,
        "successes": 0,
        "total_episodes": int(cfg["evaluation"].get("num_trials", 1)),
        "gpu_id": int(cfg.get("gpu_id", 0)),
        "remote_host": host,
        "remote_port": port,
        "remote_info": info,
        "success_episodes": [],
        "failure_episodes": [],
        "start_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "duration": 0.0,
    }
    output_dir = Path(cfg["evaluation"].get("output_dir") or "evaluate_results") / group_name
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = _find_existing_result_file(output_dir, env_name) or (output_dir / f"gpu{int(cfg.get('gpu_id', 0))}_{env_name}_results.json")
    resume_trials = _as_bool_value(cfg["evaluation"].get("resume_trials"), default=False)
    task_results = run_single_task(
        env_name,
        cfg,
        video_dir,
        predicted_video_dir,
        predicted_video_all_dir,
        cam_keys=cam_keys,
        client=client,
        action_key_mapping=action_key_mapping,
        action_postprocess=action_postprocess,
        model_video_size=model_video_size,
        model_video_concat_mode=model_video_concat_mode,
        base_results=results,
        output_file=output_file,
        resume_trials=resume_trials,
        start_time=start_time,
    )
    results.update(task_results)
    results["success_rate"] = results["successes"] / max(1, int(cfg["evaluation"].get("num_trials", 1)))
    results["completed_episodes"] = len(_completed_trials(results, int(cfg["evaluation"].get("num_trials", 1))))
    results["completed_success_rate"] = results["successes"] / max(1, int(results["completed_episodes"]))
    results["duration"] = time.time() - start_time

    output_file.write_text(json.dumps(results, indent=4, cls=NumpyEncoder) + "\n", encoding="utf-8")

    success_rates_file = os.environ.get("SUCCESS_RATES_FILE", "").strip()
    if success_rates_file:
        from experiments.sim_remote.shared_results import update_success_rate
        update_success_rate(
            success_rates_file,
            env_name=env_name,
            group_name=group_name,
            successes=int(results["successes"]),
            total=int(cfg["evaluation"].get("num_trials", 1)),
            task_description=results.get("task_description") or env_name,
        )
    print(f"Task {env_name} completed: {results['successes']}/{cfg['evaluation'].get('num_trials', 1)} successes (success_rate={results['success_rate']:.4f})")
    return results


if __name__ == "__main__":
    main()
