#!/usr/bin/env python3

from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from experiments.sim_remote import protocol

LOCAL_EGL_VENDOR_LIBRARY = "/path/to/robocasa365/etc/egl/nvidia_egl_icd.json"
LOCAL_SERVER_PYTHON = "/path/to/abotm06/bin/python"
LOCAL_CLIENT_PYTHON = "/path/to/robocasa365/bin/python"


_KEEPALIVE_PY = """
import os
import time

import torch

dev = torch.device("cuda:0")
matrix = int(os.environ.get("KEEPALIVE_MATRIX", "1024"))
burst_s = float(os.environ.get("KEEPALIVE_BURST_S", "0.5"))
sleep_s = float(os.environ.get("KEEPALIVE_SLEEP_S", "5.0"))
a = torch.randn(matrix, matrix, device=dev, dtype=torch.bfloat16)
b = torch.randn(matrix, matrix, device=dev, dtype=torch.bfloat16)
torch.cuda.synchronize()
t0 = time.time()
for _ in range(4):
    torch.matmul(a, b)
torch.cuda.synchronize()
per_op = max((time.time() - t0) / 4.0, 1e-4)
ops_per_burst = max(1, int(burst_s / per_op))
print(f"[keepalive] gpu busy: matrix={matrix} per_op={per_op*1000:.2f}ms ops_per_burst={ops_per_burst} sleep_s={sleep_s}", flush=True)
n = 0
while True:
    for _ in range(ops_per_burst):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    n += 1
    if n % 60 == 0:
        print(f"[keepalive] still alive bursts={n}", flush=True)
    time.sleep(sleep_s)
"""


def echo(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def load_task_lines(task_list_file: Path) -> list[str]:
    if not task_list_file.is_file():
        raise FileNotFoundError(f"task_list_file not found: {task_list_file}")
    lines: list[str] = []
    for raw in task_list_file.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "," not in line:
            raise ValueError(f"Task line must be group,env_name, got: {raw!r}")
        group, env_name = [part.strip() for part in line.split(",", 1)]
        if not group or not env_name:
            raise ValueError(f"Task line must be group,env_name, got: {raw!r}")
        lines.append(f"{group},{env_name}")
    if not lines:
        raise ValueError(f"task_list_file is empty: {task_list_file}")
    return lines


def resolve_project_path(raw: str | Path) -> Path:
    path = Path(raw).expanduser()
    if path.is_absolute():
        return path
    return project_root / path


class SharedTaskPool:
    def __init__(self, pool_dir: Path):
        self.pool_dir = pool_dir
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self.pending_file = self.pool_dir / "pending_tasks.txt"
        self.pending_lock = self.pool_dir / "pending_tasks.lock"
        self.completed_file = self.pool_dir / "completed_tasks.txt"
        self.completed_lock = self.pool_dir / "completed_tasks.lock"
        self.failed_file = self.pool_dir / "failed_tasks.txt"
        self.failed_lock = self.pool_dir / "failed_tasks.lock"

    @staticmethod
    def _read_plain_tasks(path: Path) -> list[str]:
        try:
            return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        except FileNotFoundError:
            return []

    def initialize(self, lines: list[str], *, resume: bool = False) -> None:
        with self.pending_lock.open("w") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            if resume and self.pending_file.exists():
                completed = set(self._read_plain_tasks(self.completed_file))
                pending = [line for line in self._read_plain_tasks(self.pending_file) if line not in completed]
                seen = set(pending)
                for line in lines:
                    if line in completed or line in seen:
                        continue
                    pending.append(line)
                    seen.add(line)
                self.pending_file.write_text(("\n".join(pending) + "\n") if pending else "", encoding="utf-8")
                self.completed_file.touch(exist_ok=True)
                self.failed_file.write_text("", encoding="utf-8")
                return
            self.pending_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
            self.completed_file.write_text("", encoding="utf-8")
            self.failed_file.write_text("", encoding="utf-8")

    def wait_until_ready(self, timeout_s: float = 900.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self.pending_file.is_file():
                return
            time.sleep(1)
        raise TimeoutError(f"Shared task pool not initialized at {self.pending_file}")

    def acquire_next_task(self) -> str | None:
        with self.pending_lock.open("w") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            try:
                lines = [line.strip() for line in self.pending_file.read_text(encoding="utf-8").splitlines() if line.strip()]
            except FileNotFoundError:
                return None
            if not lines:
                return None
            task = lines.pop(0)
            self.pending_file.write_text(("\n".join(lines) + "\n") if lines else "", encoding="utf-8")
            return task

    def mark_completed(self, task: str) -> None:
        with self.completed_lock.open("w") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            with self.completed_file.open("a", encoding="utf-8") as f:
                f.write(task.strip() + "\n")

    def mark_failed(self, task: str, rc: int, log_path: Path) -> None:
        with self.failed_lock.open("w") as lf:
            fcntl.flock(lf.fileno(), fcntl.LOCK_EX)
            with self.failed_file.open("a", encoding="utf-8") as f:
                f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')},{task},rc={rc},log={log_path}\n")

    def completed_count(self) -> int:
        try:
            return sum(1 for line in self.completed_file.read_text(encoding="utf-8").splitlines() if line.strip())
        except FileNotFoundError:
            return 0

    def failed_count(self) -> int:
        try:
            return sum(1 for line in self.failed_file.read_text(encoding="utf-8").splitlines() if line.strip())
        except FileNotFoundError:
            return 0

    def finished_count(self) -> int:
        return self.completed_count() + self.failed_count()


def wait_global_completion(
    pool: SharedTaskPool,
    *,
    total_tasks: int,
    rank: int,
    timeout_s: float,
    poll_s: float = 15.0,
    keepalive_procs: list[subprocess.Popen] | None = None,
) -> str:
    if total_tasks <= 0:
        return "no_tasks"
    keepalive_procs = keepalive_procs or []
    deadline = time.monotonic() + max(float(timeout_s), 60.0)
    last_log = 0.0
    while time.monotonic() < deadline:
        done = pool.finished_count()
        if done >= total_tasks:
            echo(f"rank{rank} global task pool finished: {done}/{total_tasks}")
            return "completed"
        dead = [idx for idx, proc in enumerate(keepalive_procs) if proc.poll() is not None]
        if dead:
            echo(f"rank{rank} keepalive processes exited early: {dead}")
        now = time.monotonic()
        if now - last_log >= 120.0:
            echo(f"rank{rank} waiting for global completion: {done}/{total_tasks}")
            last_log = now
        time.sleep(max(float(poll_s), 1.0))
    echo(f"rank{rank} global completion wait timed out: {pool.finished_count()}/{total_tasks}")
    return "timeout"


def tail_log(path: Path, lines: int = 80) -> str:
    if not path.is_file():
        return f"<missing log {path}>"
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return "\n".join(text[-lines:])


def spawn_gpu_keepalive(
    *,
    gpu_id: str,
    log_path: Path,
    python_bin: str,
    matrix_size: int,
    burst_s: float,
    sleep_s: float,
    env: dict[str, str],
) -> tuple[subprocess.Popen, Any]:
    keepalive_env = {
        **env,
        "CUDA_VISIBLE_DEVICES": str(gpu_id),
        "KEEPALIVE_MATRIX": str(int(matrix_size)),
        "KEEPALIVE_BURST_S": str(float(burst_s)),
        "KEEPALIVE_SLEEP_S": str(float(sleep_s)),
    }
    fp = log_path.open("w", encoding="utf-8")
    proc = subprocess.Popen(
        [python_bin, "-u", "-c", _KEEPALIVE_PY],
        env=keepalive_env,
        stdout=fp,
        stderr=subprocess.STDOUT,
    )
    return proc, fp


def wait_health_ready(host: str, port: int, timeout_s: float, proc: subprocess.Popen | None, log_path: Path | None) -> None:
    url = f"http://{host}:{port}{protocol.ROUTE_HEALTH}"
    deadline = time.monotonic() + float(timeout_s)
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited rc={proc.returncode} port={port} log={log_path}\n{tail_log(log_path) if log_path else ''}")
        try:
            with urllib.request.urlopen(url, timeout=5) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            if body.get("ok") and body.get("ready"):
                echo(f"health ok: {url}")
                return
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            echo(f"waiting {url}: {exc!r}")
        time.sleep(2)
    raise TimeoutError(f"server not ready within {timeout_s}s: {url}\n{tail_log(log_path) if log_path else ''}")


def parse_rank(args: argparse.Namespace) -> int:
    if args.worker_rank is not None:
        return int(args.worker_rank)
    for key in ("WORKER_RANK", "RANK"):
        value = os.environ.get(key)
        if value is not None and str(value).strip() != "":
            return int(value)
    return 0


def parse_worker_count(args: argparse.Namespace) -> int:
    if args.worker_count is not None:
        return int(args.worker_count)
    for key in ("WORKER_COUNT", "WORLD_SIZE"):
        value = os.environ.get(key)
        if value is not None and str(value).strip() != "":
            return int(value)
    return 1


def gpu_list(args: argparse.Namespace) -> list[str]:
    spec = args.cuda_visible_devices or os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if spec.strip():
        return [x.strip() for x in spec.split(",") if x.strip()]
    return [str(i) for i in range(max(int(args.num_gpus), 1))]


def normalize_split_override_args(argv: list[str] | None) -> list[str] | None:
    if argv is None:
        raw = sys.argv[1:]
        return None if not raw else normalize_split_override_args(list(raw))
    flags = {"--server-override", "--client-override"}
    out: list[str] = []
    i = 0
    while i < len(argv):
        item = argv[i]
        if (
            item in flags
            and i + 2 < len(argv)
            and "=" not in argv[i + 1]
            and not argv[i + 2].startswith("--")
        ):
            out.extend([item, f"{argv[i + 1]}={argv[i + 2]}"])
            i += 3
            continue
        out.append(item)
        i += 1
    return out


@dataclass
class RunningTask:
    proc: subprocess.Popen
    log_fp: Any
    log_path: Path
    task_line: str
    slot: int
    start_time: float


def start_servers(args: argparse.Namespace, *, root: Path, rank: int, gpus: list[str], logs_dir: Path, env: dict[str, str]) -> tuple[list[subprocess.Popen], list[Any], list[Path], int]:
    procs: list[subprocess.Popen] = []
    fps: list[Any] = []
    paths: list[Path] = []
    effective_base = int(args.base_port) + int(rank) * int(args.base_port_rank_stride)
    extra = list(args.server_override or [])
    for slot, gpu in enumerate(gpus):
        port = effective_base + slot
        log_path = logs_dir / f"server_slot{slot}_gpu{gpu}_p{port}.log"
        fp = log_path.open("w", encoding="utf-8")
        cmd = [
            args.server_python,
            str(root / "experiments/robocasa365/server.py"),
            "--config", args.config,
            "--ckpt", args.ckpt,
            "--gpu-id", str(gpu),
            "--host", args.host,
            "--port", str(port),
        ]
        if args.dataset_statistics_path:
            cmd.extend(["--dataset-statistics-path", args.dataset_statistics_path])
        for item in extra:
            cmd.extend(["--override", item])
        echo(f"start server slot={slot} gpu={gpu} port={port} log={log_path}")
        proc = subprocess.Popen(
            cmd,
            cwd=str(root),
            env={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
            stdout=fp,
            stderr=subprocess.STDOUT,
        )
        procs.append(proc)
        fps.append(fp)
        paths.append(log_path)
    try:
        for slot, proc in enumerate(procs):
            wait_health_ready(args.host, effective_base + slot, float(args.server_ready_timeout_s), proc, paths[slot])
    except Exception:
        terminate_all(procs)
        for fp in fps:
            try:
                fp.close()
            except Exception:
                pass
        raise
    return procs, fps, paths, effective_base


def launch_client(
    args: argparse.Namespace,
    *,
    root: Path,
    worker_out: Path,
    logs_dir: Path,
    results_root: Path,
    success_rates_file: Path,
    env: dict[str, str],
    task_line: str,
    slot: int,
    port: int,
) -> RunningTask:
    group, env_name = [x.strip() for x in task_line.split(",", 1)]
    safe = f"{group}_{env_name}".replace("/", "_")
    log_path = logs_dir / f"client_{safe}_slot{slot}.log"
    fp = log_path.open("w", encoding="utf-8")
    cmd = [
        args.client_python,
        str(root / "experiments/robocasa365/eval_client.py"),
        "--config", args.config,
        "--env-name", env_name,
        "--group-name", group,
        "--remote-host", args.host,
        "--remote-port", str(port),
        "--gpu-id", str(slot),
        "--num-trials", str(int(args.num_trials)),
        "--output-dir", str(results_root),
    ]
    for item in args.client_override or []:
        cmd.extend(["--override", item])
    client_env = {
        **env,
        "ROOT_DIR": str(root),
        "OUTPUT_DIR": str(results_root),
        "EXP_NAME": os.environ.get("EXP_NAME", ""),
        "MUJOCO_GL": env.get("MUJOCO_GL", "egl"),
        "PYOPENGL_PLATFORM": env.get("PYOPENGL_PLATFORM", "egl"),
        "__EGL_VENDOR_LIBRARY_FILENAMES": env.get("__EGL_VENDOR_LIBRARY_FILENAMES", LOCAL_EGL_VENDOR_LIBRARY),
        "RESULTS_ROOT_DIR": str(results_root),
        "SUCCESS_RATES_FILE": str(success_rates_file),
    }
    echo(f"launch task {task_line} on slot={slot} port={port} log={log_path}")
    proc = subprocess.Popen(cmd, cwd=str(root), env=client_env, stdout=fp, stderr=subprocess.STDOUT)
    return RunningTask(proc=proc, log_fp=fp, log_path=log_path, task_line=task_line, slot=slot, start_time=time.time())


def terminate_all(procs: list[subprocess.Popen]) -> None:
    for proc in procs:
        if proc.poll() is None:
            proc.send_signal(signal.SIGTERM)
    for proc in procs:
        if proc.poll() is None:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Unified RoboCasa365 eval pool")
    parser.add_argument("--config", required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--task-list-file", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--num-gpus", type=int, default=3)
    parser.add_argument("--cuda-visible-devices", default="")
    parser.add_argument("--worker-rank", type=int, default=None)
    parser.add_argument("--worker-count", type=int, default=None)
    parser.add_argument("--host", default=protocol.DEFAULT_HOST)
    parser.add_argument("--base-port", type=int, default=protocol.DEFAULT_PORT)
    parser.add_argument("--base-port-rank-stride", type=int, default=100)
    parser.add_argument("--num-trials", type=int, default=50)
    parser.add_argument("--num-video-inference-steps", type=int, default=10)
    parser.add_argument("--num-action-inference-steps", type=int, default=10)
    parser.add_argument("--max-tasks-per-gpu", type=int, default=1)
    parser.add_argument("--server-ready-timeout-s", type=float, default=3600.0)
    parser.add_argument("--global-completion-timeout-s", type=float, default=172800.0)
    parser.add_argument("--server-python", default=LOCAL_SERVER_PYTHON)
    parser.add_argument("--client-python", default=LOCAL_CLIENT_PYTHON)
    parser.add_argument("--keepalive-python", default="")
    parser.add_argument("--disable-keepalive", action="store_true")
    parser.add_argument("--keepalive-matrix-size", type=int, default=1024)
    parser.add_argument("--keepalive-burst-s", type=float, default=0.5)
    parser.add_argument("--keepalive-sleep-s", type=float, default=5.0)
    parser.add_argument("--dataset-statistics-path", default="")
    parser.add_argument("--server-override", action="append", default=[])
    parser.add_argument("--client-override", action="append", default=[])
    parser.add_argument("--resume-task-pool", action="store_true", default=True)
    parser.add_argument("--summarize", action="store_true", default=True)
    args = parser.parse_args(normalize_split_override_args(argv))

    if args.num_video_inference_steps is not None:
        item = f"evaluation.num_video_inference_steps={int(args.num_video_inference_steps)}"
        args.client_override.append(item)
        args.server_override.append(item)
    if args.num_action_inference_steps is not None:
        item = f"evaluation.num_action_inference_steps={int(args.num_action_inference_steps)}"
        args.client_override.append(item)
        args.server_override.append(item)

    root = project_root
    run_root = Path(args.run_root).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    rank = parse_rank(args)
    worker_count = parse_worker_count(args)
    gpus = gpu_list(args)
    total_slots = max(len(gpus) * int(args.max_tasks_per_gpu), 1)
    task_lines = load_task_lines(resolve_project_path(args.task_list_file))
    pool = SharedTaskPool(run_root / "task_pool")
    if rank == 0:
        pool.initialize(task_lines, resume=bool(args.resume_task_pool))
        echo(f"rank0 initialized task pool with {len(task_lines)} tasks")
    else:
        pool.wait_until_ready()
        echo(f"rank{rank} task pool ready")

    worker_out = run_root / "workers" / f"worker_{rank}"
    logs_dir = worker_out / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    results_root = run_root / "results"
    results_root.mkdir(parents=True, exist_ok=True)
    success_rates_file = run_root / "success_rates.json"

    env = os.environ.copy()


    env.pop("PYTHONHOME", None)
    env.pop("PYTHONPATH", None)
    env.setdefault("TOKENIZERS_PARALLELISM", "false")
    env.setdefault("MUJOCO_GL", "egl")
    env.setdefault("PYOPENGL_PLATFORM", "egl")
    env.setdefault("__EGL_VENDOR_LIBRARY_FILENAMES", LOCAL_EGL_VENDOR_LIBRARY)
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpus)
    server_procs: list[subprocess.Popen] = []
    server_fps: list[Any] = []
    keepalive_procs: list[subprocess.Popen] = []
    keepalive_fps: list[Any] = []
    keepalive_outcome: str | None = None
    running: list[RunningTask] = []
    tasks_done = 0
    tasks_failed = 0
    effective_base = int(args.base_port) + rank * int(args.base_port_rank_stride)
    try:
        server_procs, server_fps, _server_logs, effective_base = start_servers(args, root=root, rank=rank, gpus=gpus, logs_dir=logs_dir, env=env)
        slot_load = [0 for _ in range(len(gpus))]
        no_more_tasks = False
        while True:

            still_running: list[RunningTask] = []
            for item in running:
                rc = item.proc.poll()
                if rc is None:
                    still_running.append(item)
                    continue
                item.log_fp.close()
                slot_load[item.slot] = max(0, slot_load[item.slot] - 1)
                tasks_done += 1
                if rc != 0:
                    tasks_failed += 1
                    pool.mark_failed(item.task_line, rc, item.log_path)
                    echo(f"task failed rc={rc}: {item.task_line}\n{tail_log(item.log_path, 40)}")
                else:
                    pool.mark_completed(item.task_line)
                    echo(f"task completed: {item.task_line} elapsed={time.time() - item.start_time:.1f}s")
            running = still_running


            while len(running) < total_slots and not no_more_tasks:
                available = [i for i, load in enumerate(slot_load) if load < int(args.max_tasks_per_gpu)]
                if not available:
                    break
                task_line = pool.acquire_next_task()
                if task_line is None:
                    no_more_tasks = True
                    break
                slot = min(available, key=lambda i: slot_load[i])
                port = effective_base + slot
                slot_load[slot] += 1
                running.append(
                    launch_client(
                        args,
                        root=root,
                        worker_out=worker_out,
                        logs_dir=logs_dir,
                        results_root=results_root,
                        success_rates_file=success_rates_file,
                        env=env,
                        task_line=task_line,
                        slot=slot,
                        port=port,
                    )
                )

            if no_more_tasks and not running:
                break
            time.sleep(5)

        if not args.disable_keepalive and task_lines:
            keepalive_python = args.keepalive_python.strip() or args.server_python
            echo(
                f"rank{rank} starting GPU keepalive on {len(gpus)} slot(s): "
                f"matrix={args.keepalive_matrix_size} burst={args.keepalive_burst_s}s sleep={args.keepalive_sleep_s}s"
            )
            for gpu in gpus:
                keepalive_log = logs_dir / f"keepalive_gpu{gpu}.log"
                try:
                    proc, fp = spawn_gpu_keepalive(
                        gpu_id=gpu,
                        log_path=keepalive_log,
                        python_bin=keepalive_python,
                        matrix_size=int(args.keepalive_matrix_size),
                        burst_s=float(args.keepalive_burst_s),
                        sleep_s=float(args.keepalive_sleep_s),
                        env=env,
                    )
                    keepalive_procs.append(proc)
                    keepalive_fps.append(fp)
                except Exception as exc:
                    echo(f"rank{rank} failed to start keepalive gpu={gpu}: {exc!r}")

        keepalive_outcome = wait_global_completion(
            pool,
            total_tasks=len(task_lines),
            rank=rank,
            timeout_s=float(args.global_completion_timeout_s),
            keepalive_procs=keepalive_procs,
        )
    finally:
        terminate_all(keepalive_procs)
        for fp in keepalive_fps:
            try:
                fp.close()
            except Exception:
                pass
        for item in running:
            if item.proc.poll() is None:
                item.proc.terminate()
            try:
                item.log_fp.close()
            except Exception:
                pass
        terminate_all(server_procs)
        for fp in server_fps:
            try:
                fp.close()
            except Exception:
                pass

    summary = {
        "rank": rank,
        "worker_count": worker_count,
        "gpus": gpus,
        "tasks_done": tasks_done,
        "tasks_failed": tasks_failed,
        "run_root": str(run_root),
        "results_root": str(results_root),
        "success_rates_file": str(success_rates_file),
        "keepalive_enabled": not bool(args.disable_keepalive),
        "keepalive_outcome": keepalive_outcome,
    }
    (worker_out / "worker_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    echo(f"worker {rank} done: {summary}")
    if args.summarize and rank == 0:
        subprocess.run([args.client_python, str(root / "experiments/robocasa365/summarize_results.py"), "--output_dir", str(results_root)], cwd=str(root), check=False)
    return 1 if tasks_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
