from typing import Optional

import multiprocessing as mp
import os

import torch
from catalyst import utils as cata_ut


def setup_train_project(project_root: str, seed: int, subfolder_name: Optional[str] = None) -> str:
    checkpoints_dpath = os.path.join(project_root, "train_checkpoints")
    if subfolder_name is not None:
        checkpoints_dpath = os.path.join(checkpoints_dpath, subfolder_name)

    os.makedirs(checkpoints_dpath, exist_ok=True)

    # Setup checkpoints upload directory
    torch.hub.set_dir(checkpoints_dpath)

    cata_ut.set_global_seed(seed)
    cata_ut.torch.prepare_cudnn(deterministic=True, benchmark=True)

    return checkpoints_dpath


def get_n_workers(num_workers: int, logger) -> int:
    max_cpu_count = mp.cpu_count()
    if num_workers < 0:
        num_workers = max_cpu_count
        logger.info(f"Parameter `num_workers` is set to {num_workers}")

    num_workers = min(max_cpu_count, num_workers)

    return num_workers
