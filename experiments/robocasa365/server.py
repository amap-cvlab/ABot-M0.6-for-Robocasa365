from __future__ import annotations

import argparse
import json
import logging
import os
import signal
import sys
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import numpy as np
import torch

project_root = Path(__file__).resolve().parents[2]
src_root = project_root / "src"
for _path in (project_root, src_root):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from experiments.robocasa365.action_ensembler import ActionEnsembler
from experiments.sim_remote import protocol
from experiments.robocasa365.eval_adapter import RoboCasaEvalAdapter
from experiments.robocasa365.runtime_config import build_memory_rollout_omegaconf, load_config
from abot06.inference.memory_state import MemoryConfig, MemoryState

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _cuda_memory_summary(device: str | torch.device) -> str:
    if not torch.cuda.is_available() or not str(device).startswith("cuda"):
        return "cuda=unavailable"
    dev = torch.device(device)
    allocated = torch.cuda.memory_allocated(dev) / (1024 ** 3)
    reserved = torch.cuda.memory_reserved(dev) / (1024 ** 3)
    max_reserved = torch.cuda.max_memory_reserved(dev) / (1024 ** 3)
    return f"allocated={allocated:.2f}GiB reserved={reserved:.2f}GiB max_reserved={max_reserved:.2f}GiB"


ACTION_SLICES: tuple[tuple[str, int, int], ...] = (
    ("base_motion", 0, 4),
    ("control_mode", 4, 5),
    ("end_effector_position", 5, 8),
    ("end_effector_rotation", 8, 11),
    ("gripper_close", 11, 12),
)


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


def _to_numpy_2d(value: Any) -> np.ndarray:
    arr = np.asarray(value.detach().cpu().numpy() if hasattr(value, "detach") else value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim > 2:
        arr = arr.reshape(-1, arr.shape[-1])
    return arr


def _log_action_summary(label: str, value: Any) -> None:
    arr = _to_numpy_2d(value)
    if arr.size == 0:
        logging.info("[action-debug] %s empty shape=%s", label, arr.shape)
        return
    parts = [
        f"{label} shape={tuple(arr.shape)}",
        f"all mean={float(arr.mean()):.4f} std={float(arr.std()):.4f} min={float(arr.min()):.4f} max={float(arr.max()):.4f}",
    ]
    for name, start, end in ACTION_SLICES:
        if arr.shape[-1] < end:
            continue
        chunk = arr[:, start:end]
        first = np.array2string(chunk[0], precision=4, separator=",", suppress_small=False)
        parts.append(
            f"{name}[{start}:{end}] mean={float(chunk.mean()):.4f} std={float(chunk.std()):.4f} "
            f"min={float(chunk.min()):.4f} max={float(chunk.max()):.4f} first={first}"
        )
    logging.info("[action-debug] %s", " | ".join(parts))


@dataclass
class SessionState:
    task_description: str
    prompt: str
    replan_steps: int
    action_horizon: int
    use_action_ensembler: bool
    visualize_future_video: bool
    num_inference_steps: int
    num_video_inference_steps: int | None
    num_action_inference_steps: int | None
    sigma_shift: float | None
    text_cfg_scale: float
    negative_prompt: str
    rand_device: str
    tiled: bool
    camera_keys: list[str]
    seed: int | None
    memory_state: Any | None
    prompt_context: torch.Tensor | None = None
    prompt_context_mask: torch.Tensor | None = None

    def __post_init__(self) -> None:
        self.pending_actions: list[np.ndarray] = []
        self.episode_step = 0
        self.replan_count = 0
        self.ensembler = ActionEnsembler() if self.use_action_ensembler else None
        if self.ensembler is not None:
            self.ensembler.reset()
        if self.memory_state is not None:
            self.memory_state.reset()


class WamModelService:
    def __init__(
        self,
        cfg: dict[str, Any],
        *,
        ckpt: str,
        gpu_id: int,
        device: str | None,
        mixed_precision: str | None,
        dataset_statistics_path: str | None,
        config_path: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.ckpt = str(ckpt)
        self.gpu_id = int(gpu_id)
        self.lock = threading.Lock()
        self.sessions: dict[str, SessionState] = {}
        self.ready = False

        from abot06.models.wan22.factory import (
            create_model_from_config,
            mixed_precision_to_dtype,
            resolve_eval_device,
        )
        from abot06.utils.pytorch_utils import set_global_seed

        if cfg.get("seed") is not None:
            set_global_seed(int(cfg["seed"]), get_worker_init_fn=False)

        self.device = resolve_eval_device(gpu_id=gpu_id, device=device)
        dtype = mixed_precision_to_dtype(
            mixed_precision or cfg.get("server", {}).get("mixed_precision", "bf16")
        )
        self.dtype = dtype
        model_cfg = dict(cfg["model"])
        server_cfg = cfg.get("server", {}) or {}
        if not bool(server_cfg.get("load_pretrained_backbones", False)):
            logging.info(
                "Skipping pretrained video/action backbones; expecting checkpoint to provide mot weights."
            )
            model_cfg["skip_dit_load_from_pretrain"] = True
            model_cfg["action_dit_pretrained_path"] = None
        if bool(server_cfg.get("load_text_encoder", True)):
            model_cfg["load_text_encoder"] = True
        logging.info("Loading Abot06 model on %s dtype=%s ckpt=%s", self.device, dtype, self.ckpt)
        t0 = time.perf_counter()
        self.model = create_model_from_config(
            model_cfg,
            data_cfg=cfg.get("data", {}),
            device=self.device,
            model_dtype=dtype,
            ckpt_path=self.ckpt,
            config_source=config_path,
        )
        logging.info(
            "Model constructed in %.2fs: %s variant=%s source=%s",
            time.perf_counter() - t0,
            type(self.model).__name__,
            getattr(self.model, "model_variant", "unknown"),
            getattr(self.model, "model_variant_source", "unknown"),
        )
        t0 = time.perf_counter()
        self.model.load_checkpoint(self.ckpt)
        self.model = self.model.to(self.device).eval()
        logging.info("Checkpoint loaded in %.2fs", time.perf_counter() - t0)
        logging.info("CUDA memory after checkpoint load: %s", _cuda_memory_summary(self.device))

        self.adapter = RoboCasaEvalAdapter(cfg, ckpt_path=self.ckpt, dataset_statistics_path=dataset_statistics_path)
        self.has_memory = bool(getattr(self.model, "memory", False))
        self.action_dim = int(getattr(self.model.action_expert, "action_dim", cfg.get("data", {}).get("action_dim", 12)))
        self.proprio_dim = int(cfg.get("data", {}).get("proprio_dim", 16))
        self.num_chunk_frames = int(cfg.get("data", {}).get("num_frames", 33))
        self.action_video_freq_ratio = int(cfg.get("data", {}).get("action_video_freq_ratio", 4))
        if self.action_video_freq_ratio <= 0:
            raise ValueError(f"action_video_freq_ratio must be positive, got {self.action_video_freq_ratio}")
        self.num_video_frames = (self.num_chunk_frames - 1) // self.action_video_freq_ratio + 1
        self.default_replan_steps = int(cfg.get("evaluation", {}).get("replan_steps") or max(self.num_chunk_frames - 1, 1))
        self.default_action_horizon = int(cfg.get("evaluation", {}).get("action_horizon") or self.default_replan_steps)
        self.input_h, self.input_w = self.adapter.video_size

        self.memory_cfg: Any | None = None
        if self.has_memory:
            memory_omega = build_memory_rollout_omegaconf(cfg)
            self.memory_cfg = MemoryConfig.from_cfg_and_model(memory_omega, self.model)
            MemoryState.validate_replan_steps(self.default_replan_steps, self.action_video_freq_ratio)
        self.ready = True
        logging.info(
            "Server ready: has_memory=%s action_dim=%d input=%dx%d frames=%d->%d/r%d "
            "legacy_eval_processing=%s prompt_style=%s input_processing_style=%s "
            "model_class=%s variant=%s source=%s",
            self.has_memory,
            self.action_dim,
            self.input_h,
            self.input_w,
            self.num_chunk_frames,
            self.num_video_frames,
            self.action_video_freq_ratio,
            self.adapter.use_legacy_eval_processing,
            self.adapter.prompt_style,
            self.adapter.input_processing_style,
            type(self.model).__name__,
            getattr(self.model, "model_variant", "unknown"),
            getattr(self.model, "model_variant_source", "unknown"),
        )

    def info(self) -> dict[str, Any]:
        return {
            "ok": True,
            "ready": self.ready,
            "ckpt": self.ckpt,
            "device": str(self.device),
            "gpu_id": self.gpu_id,
            "has_memory": self.has_memory,
            "memory": self.has_memory,
            "action_dim": self.action_dim,
            "proprio_dim": self.proprio_dim,
            "input_h": self.input_h,
            "input_w": self.input_w,
            "num_chunk_frames": self.num_chunk_frames,
            "num_video_frames": self.num_video_frames,
            "action_horizon": self.default_action_horizon,
            "replan_steps": self.default_replan_steps,
            "action_video_freq_ratio": self.action_video_freq_ratio,
            "use_legacy_eval_processing": bool(self.adapter.use_legacy_eval_processing),
            "prompt_style": str(self.adapter.prompt_style),
            "prompt_template_set": bool((self.cfg.get("evaluation", {}) or {}).get("prompt_template")),
            "input_processing_style": str(self.adapter.input_processing_style),
            "video_size": [self.input_h, self.input_w],
            "video_concat_mode": str(getattr(self.adapter, "video_concat_mode", "")),
            "dataset_statistics_path": str(self.adapter.dataset_statistics_path),
            "model_class": type(self.model).__name__,
            "model_variant": getattr(self.model, "model_variant", "unknown"),
            "model_variant_source": getattr(self.model, "model_variant_source", "unknown"),
        }

    def reset(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id") or "default")
        task_description = str(payload.get("task_description") or "")
        if not task_description:
            raise ValueError("reset requires task_description")
        prompt = self.adapter.make_prompt(task_description)
        replan_steps = int(payload.get("replan_steps") or self.default_replan_steps)
        action_horizon = int(payload.get("action_horizon") or replan_steps or self.default_action_horizon)
        if self.has_memory:
            if self.memory_cfg is None:
                raise RuntimeError("Memory model has no MemoryConfig")
            MemoryState.validate_replan_steps(replan_steps, self.action_video_freq_ratio)
            expected_actions = int(self.memory_cfg.num_actions)
            if replan_steps != expected_actions or action_horizon != expected_actions:
                raise ValueError(
                    f"Memory remote eval requires replan_steps=action_horizon=num_frames-1={expected_actions}; "
                    f"got replan_steps={replan_steps}, action_horizon={action_horizon}."
                )
            memory_state = MemoryState(self.model, self.memory_cfg, self.device, self.dtype)
        else:
            memory_state = None
        camera_keys = payload.get("camera_keys") or self.adapter.camera_keys
        state = SessionState(
            task_description=task_description,
            prompt=prompt,
            replan_steps=replan_steps,
            action_horizon=action_horizon,
            use_action_ensembler=bool(payload.get("use_action_ensembler", False)),
            visualize_future_video=bool(payload.get("visualize_future_video", False)),
            num_inference_steps=int(payload.get("num_inference_steps") or self.cfg.get("evaluation", {}).get("num_inference_steps") or 20),
            num_video_inference_steps=(None if payload.get("num_video_inference_steps") is None else int(payload.get("num_video_inference_steps"))),
            num_action_inference_steps=(None if payload.get("num_action_inference_steps") is None else int(payload.get("num_action_inference_steps"))),
            sigma_shift=(None if payload.get("sigma_shift") is None else float(payload.get("sigma_shift"))),
            text_cfg_scale=float(payload.get("text_cfg_scale", 1.0)),
            negative_prompt=str(payload.get("negative_prompt", "")),
            rand_device=str(payload.get("rand_device", "cpu")),
            tiled=bool(payload.get("tiled", False)),
            camera_keys=[str(x) for x in camera_keys],
            seed=(None if payload.get("seed") is None else int(payload.get("seed"))),
            memory_state=memory_state,
        )
        self.sessions[session_id] = state
        logging.info(
            "[reset] session=%s task=%r prompt=%r replan_steps=%d action_horizon=%d seed=%s camera_keys=%s",
            session_id,
            task_description,
            prompt,
            replan_steps,
            action_horizon,
            state.seed,
            state.camera_keys,
        )
        return {"ok": True, "session_id": session_id, "has_memory": self.has_memory, "prompt": prompt}

    def _session(self, session_id: str) -> SessionState:
        if session_id not in self.sessions:
            raise KeyError(f"Unknown session_id {session_id!r}; call /reset first")
        return self.sessions[session_id]

    def _cached_prompt_context(self, state: SessionState) -> tuple[torch.Tensor, torch.Tensor]:
        if state.prompt_context is None or state.prompt_context_mask is None:
            t0 = time.perf_counter()
            context, context_mask = self.model.encode_prompt(state.prompt)
            state.prompt_context = context.detach()
            state.prompt_context_mask = context_mask.detach()
            logging.info(
                "[prompt-cache] encoded task prompt once in %.3fs context=%s mask=%s",
                time.perf_counter() - t0,
                tuple(state.prompt_context.shape),
                tuple(state.prompt_context_mask.shape),
            )
        return state.prompt_context, state.prompt_context_mask

    @torch.no_grad()
    def _run_replan(self, state: SessionState, obs: dict[str, Any]) -> tuple[np.ndarray, list[Any] | None]:
        image, proprio = self.adapter.obs_to_model_inputs(obs, device=self.device, dtype=self.dtype, camera_keys=state.camera_keys)
        seed = state.seed
        context, context_mask = self._cached_prompt_context(state)
        infer_kwargs = dict(
            prompt=None,
            context=context,
            context_mask=context_mask,
            input_image=image,
            action_horizon=state.action_horizon,
            proprio=proprio,
            negative_prompt=state.negative_prompt,
            text_cfg_scale=state.text_cfg_scale,
            num_inference_steps=state.num_inference_steps,
            num_video_inference_steps=state.num_video_inference_steps,
            num_action_inference_steps=state.num_action_inference_steps,
            sigma_shift=state.sigma_shift,
            seed=seed,
            rand_device=state.rand_device,
            tiled=state.tiled,
        )
        if self.has_memory:
            assert state.memory_state is not None
            state.memory_state.set_observation_rgb(image)
            h_long, h_mid, h_short = state.memory_state.build_histories()
            infer_kwargs.update(history_long=h_long, history_mid=h_mid, history_short=h_short)
        if state.visualize_future_video:
            out = self.model.infer_joint(
                num_video_frames=self.num_video_frames,
                test_action_with_infer_action=False,
                **infer_kwargs,
            )
            predicted_frames = out.get("video")
        else:
            out = self.model.infer_action(**infer_kwargs)
            predicted_frames = None
        log_action_stats = _env_bool("ABOT06_LOG_ACTION_STATS", False)
        log_action_limit = _env_int("ABOT06_LOG_ACTION_STATS_N", 3)
        should_log_action = log_action_stats and state.replan_count < log_action_limit
        if should_log_action:
            _log_action_summary(f"replan={state.replan_count} raw_model_action", out["action"])
        action = self.adapter.denormalize_action(out["action"])
        if should_log_action:
            _log_action_summary(f"replan={state.replan_count} denorm_action", action)
        if action.shape[0] < state.replan_steps:
            pad = np.repeat(action[-1:], state.replan_steps - action.shape[0], axis=0)
            action = np.concatenate([action, pad], axis=0)
        return action[: state.replan_steps], predicted_frames

    def act(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id") or "default")
        obs = payload.get("obs")
        if obs is None:
            raise ValueError("act requires obs")
        state = self._session(session_id)
        predicted_frames = None
        is_replan_boundary = len(state.pending_actions) == 0
        with self.lock:
            if is_replan_boundary:
                action_chunk, predicted_frames = self._run_replan(state, obs)
                if state.ensembler is not None:
                    state.ensembler.add_actions(action_chunk, start_timestamp=state.episode_step)
                state.pending_actions = [a.astype(np.float32) for a in action_chunk]
                state.replan_count += 1
            if state.ensembler is not None:
                action = state.ensembler.get_action(state.episode_step).astype(np.float32)
                state.ensembler._cleanup(state.episode_step)
                if state.pending_actions:
                    state.pending_actions.pop(0)
            else:
                action = state.pending_actions.pop(0).astype(np.float32)
            remaining = len(state.pending_actions)
            state.episode_step += 1
        return {
            "ok": True,
            "session_id": session_id,
            "action": action,
            "is_replan_boundary": is_replan_boundary,
            "remaining_chunk": remaining,
            "has_memory": self.has_memory,
            "predicted_future_frames": predicted_frames,
        }

    def update_timeline(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload.get("session_id") or "default")
        if not self.has_memory:
            return {"ok": True, "updated": False, "skipped": True, "reason": "memory=false"}
        state = self._session(session_id)
        if state.memory_state is None or self.memory_cfg is None:
            return {"ok": True, "updated": False, "skipped": True, "reason": "no Memory state"}
        future_obs_list = list(payload.get("future_obs_list") or [])
        expected = int(self.memory_cfg.N_future)
        if not future_obs_list:
            return {
                "ok": True,
                "updated": False,
                "skipped": True,
                "reason": f"expected {expected} future obs, got 0",
            }
        if len(future_obs_list) < expected:
            future_obs_list.extend([future_obs_list[-1]] * (expected - len(future_obs_list)))
        elif len(future_obs_list) > expected:
            future_obs_list = future_obs_list[:expected]
        with self.lock:
            pixels = self.adapter.future_obs_to_pixels(future_obs_list, device=self.device, dtype=self.dtype, camera_keys=state.camera_keys)
            state.memory_state.after_chunk_executed(pixels)
        return {
            "ok": True,
            "updated": True,
            "num_future_obs": len(future_obs_list),
            "timeline_T": int(state.memory_state._timeline_lat.shape[2]),
            "chunk_count": int(state.memory_state._chunk_count),
        }


_SERVICE: WamModelService | None = None
_SHUTDOWN = False


class Handler(BaseHTTPRequestHandler):
    server_version = "Abot06RoboCasaRemote/1.0"

    def log_message(self, fmt: str, *args: Any) -> None:
        logging.info("%s - %s", self.address_string(), fmt % args)

    def _json(self, status: int, obj: dict[str, Any]) -> None:
        body = json.dumps(obj).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _pickle(self, status: int, obj: Any) -> None:
        body = protocol.dumps(obj)
        self.send_response(status)
        self.send_header("Content-Type", protocol.CONTENT_TYPE)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == protocol.ROUTE_HEALTH:
            self._json(200, {"ok": True, "ready": bool(_SERVICE and _SERVICE.ready)})
            return
        if self.path == protocol.ROUTE_INFO:
            if _SERVICE is None:
                self._json(503, {"ok": False, "ready": False})
            else:
                self._json(200, _SERVICE.info())
            return
        self._json(404, {"ok": False, "error": f"unknown route {self.path}"})

    def do_POST(self) -> None:
        global _SHUTDOWN
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = protocol.loads(self.rfile.read(length)) if length > 0 else {}
            if self.path == protocol.ROUTE_SHUTDOWN:
                _SHUTDOWN = True
                self._pickle(200, {"ok": True})
                threading.Thread(target=self.server.shutdown, daemon=True).start()
                return
            if _SERVICE is None or not _SERVICE.ready:
                raise RuntimeError("service is not ready")
            if self.path == protocol.ROUTE_RESET:
                self._pickle(200, _SERVICE.reset(payload))
            elif self.path == protocol.ROUTE_ACT:
                self._pickle(200, _SERVICE.act(payload))
            elif self.path == protocol.ROUTE_UPDATE_TIMELINE:
                self._pickle(200, _SERVICE.update_timeline(payload))
            else:
                self._pickle(404, {"ok": False, "error": f"unknown route {self.path}"})
        except Exception as exc:
            logging.exception("request failed: %s", self.path)
            self._pickle(500, {"ok": False, "type": type(exc).__name__, "error": str(exc)})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Abot06 RoboCasa365 server")
    parser.add_argument("--config", required=True, help="yaml config path, e.g. configs/robocasa365_test.yaml")
    parser.add_argument("--ckpt", required=True, help="checkpoint path")
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mixed-precision", default=None)
    parser.add_argument("--dataset-statistics-path", default=None)
    parser.add_argument("--override", action="append", default=[])
    args, unknown = parser.parse_known_args()
    args.override.extend(unknown)
    return args


def main() -> None:
    global _SERVICE
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s server: %(message)s", datefmt="%H:%M:%S")
    args = parse_args()
    cfg = load_config(args.config, args.override)
    host = args.host or str(cfg.get("server", {}).get("host", protocol.DEFAULT_HOST))
    port = int(args.port or cfg.get("server", {}).get("port", protocol.DEFAULT_PORT))
    httpd = ThreadingHTTPServer((host, port), Handler)
    logging.info("Bound http://%s:%d; loading model service...", host, port)
    try:
        _SERVICE = WamModelService(
            cfg,
            ckpt=args.ckpt,
            gpu_id=args.gpu_id,
            device=args.device,
            mixed_precision=args.mixed_precision,
            dataset_statistics_path=args.dataset_statistics_path,
            config_path=args.config,
        )
        logging.info("Listening on http://%s:%d", host, port)
        httpd.serve_forever(poll_interval=0.5)
    finally:
        httpd.server_close()
        logging.info("Server stopped")


if __name__ == "__main__":
    main()
