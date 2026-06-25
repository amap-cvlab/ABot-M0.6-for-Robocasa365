from __future__ import annotations

import json
import sys
import time
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any, Optional

import numpy as np


project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.sim_remote import protocol


class WamRemoteClient:


    def __init__(self, host: str = protocol.DEFAULT_HOST, port: int = protocol.DEFAULT_PORT,
                 timeout_s: float = float(protocol.DEFAULT_TIMEOUT_S)):
        self.host = str(host)
        self.port = int(port)
        self.timeout_s = float(timeout_s)

    def _url(self, route: str) -> str:
        return protocol.make_url(self.host, self.port, route)

    def _post_pickle(self, route: str, payload: dict[str, Any]) -> Any:
        body = protocol.dumps(payload)
        req = urllib.request.Request(
            self._url(route),
            data=body,
            method="POST",
            headers={"Content-Type": protocol.CONTENT_TYPE, "Content-Length": str(len(body))},
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                return protocol.loads(resp.read())
        except urllib.error.HTTPError as e:
            raw = e.read() if hasattr(e, "read") else b""
            try:
                err = protocol.loads(raw)
            except Exception:
                err = {"error": raw.decode("utf-8", errors="replace") or str(e), "type": "HTTPError"}
            raise RuntimeError(f"server error on {route}: {err}") from e

    def _get_json(self, route: str) -> dict[str, Any]:
        req = urllib.request.Request(self._url(route), method="GET")
        with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
            return json.loads(resp.read().decode("utf-8"))

    def health(self) -> dict[str, Any]:
        return self._get_json(protocol.ROUTE_HEALTH)

    def info(self) -> dict[str, Any]:
        return self._get_json(protocol.ROUTE_INFO)

    def wait_until_ready(self, *, max_wait_s: float = 600.0, poll_interval_s: float = 1.0) -> dict[str, Any]:
        deadline = time.time() + float(max_wait_s)
        last_err: Optional[Exception] = None
        while time.time() < deadline:
            try:
                health = self.health()
                if health.get("ready"):
                    return self.info()
            except Exception as e:
                last_err = e
            time.sleep(float(poll_interval_s))
        raise TimeoutError(
            f"Server at http://{self.host}:{self.port} not ready within {max_wait_s}s. "
            f"Last error: {last_err!r}"
        )

    def reset(self, session_id: str, *, task_description: str, replan_steps: int,
              action_horizon: Optional[int] = None, use_action_ensembler: bool = False,
              visualize_future_video: bool = False, num_inference_steps: Optional[int] = None,
              num_video_inference_steps: Optional[int] = None,
              num_action_inference_steps: Optional[int] = None,
              sigma_shift: Optional[float] = None, text_cfg_scale: float = 1.0,
              negative_prompt: str = "", rand_device: str = "cpu", tiled: bool = False,
              camera_keys: Optional[list[str]] = None, seed: Optional[int] = None) -> dict[str, Any]:
        return self._post_pickle(protocol.ROUTE_RESET, {
            "session_id": session_id,
            "task_description": task_description,
            "replan_steps": int(replan_steps),
            "action_horizon": action_horizon,
            "use_action_ensembler": bool(use_action_ensembler),
            "visualize_future_video": bool(visualize_future_video),
            "num_inference_steps": num_inference_steps,
            "num_video_inference_steps": num_video_inference_steps,
            "num_action_inference_steps": num_action_inference_steps,
            "sigma_shift": sigma_shift,
            "text_cfg_scale": float(text_cfg_scale),
            "negative_prompt": negative_prompt,
            "rand_device": rand_device,
            "tiled": bool(tiled),
            "camera_keys": camera_keys,
            "seed": seed,
        })

    def act(self, session_id: str, obs: dict, task_description: Optional[str] = None) -> dict[str, Any]:
        response = self._post_pickle(protocol.ROUTE_ACT, {
            "session_id": session_id,
            "obs": obs,
            "task_description": task_description,
        })
        action = response["action"]
        if not isinstance(action, np.ndarray):
            response["action"] = np.asarray(action, dtype=np.float32)
        return response

    def update_timeline(self, session_id: str, future_obs_list: list[dict]) -> dict[str, Any]:


        return self._post_pickle(protocol.ROUTE_UPDATE_TIMELINE, {
            "session_id": session_id,
            "future_obs_list": future_obs_list,
        })

    def shutdown(self) -> None:
        try:
            self._post_pickle(protocol.ROUTE_SHUTDOWN, {})
        except Exception:
            pass


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ping a Abot06 simulation eval server")
    parser.add_argument("--host", default=protocol.DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=protocol.DEFAULT_PORT)
    parser.add_argument("--timeout-s", type=float, default=10.0)
    args = parser.parse_args()

    client = WamRemoteClient(host=args.host, port=args.port, timeout_s=args.timeout_s)
    print("health:", client.health())
    print("info:", client.info())
