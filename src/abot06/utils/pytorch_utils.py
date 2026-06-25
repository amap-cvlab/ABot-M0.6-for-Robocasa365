from __future__ import annotations

import random
from typing import Callable

import numpy as np
import torch


def set_global_seed(seed: int, get_worker_init_fn: bool = False) -> Callable | None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))

    if not get_worker_init_fn:
        return None

    def _worker_init_fn(worker_id: int) -> None:
        worker_seed = int(seed) + int(worker_id)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _worker_init_fn
