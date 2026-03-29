"""
training/distributed.py
DDP setup, teardown, and utilities for multi-process training.
Simulates multi-GPU on CPU when CUDA is unavailable.
"""

import os
import logging
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset, DistributedSampler
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def setup_ddp(rank: int, world_size: int, backend: str = "gloo"):
    """
    Initialize the distributed process group.
    Uses 'gloo' backend (CPU-compatible). Switch to 'nccl' for GPU.
    """
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"

    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )
    logger.info(f"[Rank {rank}/{world_size}] DDP initialized (backend={backend})")


def cleanup_ddp():
    """Tear down the distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


def wrap_model_ddp(model: torch.nn.Module, rank: int) -> torch.nn.Module:
    """Wrap model with DDP for gradient synchronization."""
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model = DDP(model, device_ids=[rank] if torch.cuda.is_available() else None)
    return model


def build_distributed_loaders(
    X_train: "np.ndarray",
    y_train: "np.ndarray",
    X_val: "np.ndarray",
    y_val: "np.ndarray",
    batch_size: int,
    rank: int,
    world_size: int,
) -> Tuple[DataLoader, DataLoader]:
    """
    Build DataLoaders with DistributedSampler for training.
    Validation uses a standard sampler (evaluated on rank-0 only).
    """
    import numpy as np

    X_train_t = torch.FloatTensor(X_train)
    y_train_t = torch.LongTensor(y_train)
    X_val_t   = torch.FloatTensor(X_val)
    y_val_t   = torch.LongTensor(y_val)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    val_dataset   = TensorDataset(X_val_t, y_val_t)

    # DistributedSampler partitions data across workers
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def is_main_process(rank: int) -> bool:
    return rank == 0


def all_reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """Average a tensor across all DDP workers."""
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()
