from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_STATISTICS_FILENAME = "dataset_statistics.json"
EPSILON = 5e-4


def default_dataset_statistics_path_for_ckpt(ckpt_path: str | Path) -> Path:
    ckpt = Path(ckpt_path).expanduser().resolve()
    if len(ckpt.parents) >= 3:
        return ckpt.parents[2] / DATASET_STATISTICS_FILENAME
    return ckpt.parent / DATASET_STATISTICS_FILENAME


def resolve_dataset_statistics_path(
    ckpt_path: str | Path | None = None,
    explicit_path: str | Path | None = None,
    config: dict[str, Any] | None = None,
) -> Path:
    candidates: list[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path).expanduser())
    if ckpt_path:
        candidates.append(default_dataset_statistics_path_for_ckpt(ckpt_path))
    if config:
        eval_cfg = config.get("evaluation", {}) or {}
        data_cfg = config.get("data", {}) or {}
        cfg_path = eval_cfg.get("dataset_statistics_path") or data_cfg.get("dataset_statistics_path")
        if cfg_path:
            candidates.append(Path(cfg_path).expanduser())

    for path in candidates:
        if path.is_file():
            return path
    if candidates:
        tried = ", ".join(str(path) for path in candidates)
        raise FileNotFoundError(f"dataset_statistics.json not found. Tried: {tried}")
    raise FileNotFoundError(
        "dataset_statistics.json path could not be resolved. Pass "
        "--dataset-statistics-path or place it at the checkpoint run root."
    )


def load_dataset_statistics(path: str | Path) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload
