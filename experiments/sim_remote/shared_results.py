from __future__ import annotations

import fcntl
import json
import time
from pathlib import Path
from typing import Any


def update_success_rate(
    json_path: Path | str,
    *,
    env_name: str,
    group_name: str,
    successes: int,
    total: int,
    task_description: str = "",
) -> None:


    json_path = Path(json_path)
    lock_path = json_path.with_suffix(".json.lock")
    json_path.parent.mkdir(parents=True, exist_ok=True)

    key = f"{group_name}/{env_name}"
    entry: dict[str, Any] = {
        "env_name": env_name,
        "group_name": group_name,
        "successes": int(successes),
        "total": int(total),
        "success_rate": round(int(successes) / max(int(total), 1), 4),
        "task_description": str(task_description),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    with lock_path.open("w") as lf:
        fcntl.flock(lf.fileno(), fcntl.LOCK_EX)


        data: dict[str, Any] = {}
        if json_path.is_file():
            try:
                raw = json_path.read_text(encoding="utf-8").strip()
                if raw:
                    data = json.loads(raw)
            except (json.JSONDecodeError, OSError):

                data = {}

        data[key] = entry


        tmp_path = json_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        tmp_path.rename(json_path)


def read_success_rates(json_path: Path | str) -> dict[str, Any]:


    json_path = Path(json_path)
    if not json_path.is_file():
        return {}
    try:
        raw = json_path.read_text(encoding="utf-8").strip()
        return json.loads(raw) if raw else {}
    except (json.JSONDecodeError, OSError):
        return {}
