"""
training/trainer.py
Distributed trainer supporting DDP (multi-process) and single-process modes.
Handles: checkpointing, resuming, LR scheduling, early stopping.
"""

import os
import time
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
from pathlib import Path
from typing import Dict, Any, Optional, Callable

from .model import build_model, count_parameters
from .distributed import (
    setup_ddp, cleanup_ddp, wrap_model_ddp,
    build_distributed_loaders, is_main_process,
    all_reduce_tensor, barrier,
)

logger = logging.getLogger(__name__)


class Trainer:
    """
    Orchestrates distributed training with:
    - PyTorch DDP across N processes
    - Epoch-level checkpointing with resume
    - LR scheduler (cosine / step)
    - Early stopping
    - Callback hooks for experiment tracking
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.train_cfg = config["training"]
        self.checkpoint_dir = Path(self.train_cfg["checkpoint_dir"])
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------ #
    # Public entry point
    # ------------------------------------------------------------------ #

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        n_features: int,
        n_classes: int,
        on_epoch_end: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Launch training. Uses DDP if distributed=True, else single process.
        Returns final training history.
        """
        self.n_features = n_features
        self.n_classes = n_classes
        self.on_epoch_end = on_epoch_end

        if self.train_cfg.get("distributed", False):
            world_size = self.train_cfg.get("num_processes", 2)
            logger.info(f"Starting DDP training with {world_size} processes")
            history = self._run_distributed(X_train, y_train, X_val, y_val, world_size)
        else:
            logger.info("Starting single-process training")
            history = self._run_single(X_train, y_train, X_val, y_val)

        return history

    # ------------------------------------------------------------------ #
    # Single-process training
    # ------------------------------------------------------------------ #

    def _run_single(self, X_train, y_train, X_val, y_val) -> Dict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = build_model(self.config, self.n_features, self.n_classes).to(device)
        logger.info(f"Model parameters: {count_parameters(model):,}")

        start_epoch, model, optimizer_state = self._maybe_load_checkpoint(model)

        optimizer = self._build_optimizer(model)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        scheduler = self._build_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss()

        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.LongTensor(y_train)
        X_val_t   = torch.FloatTensor(X_val).to(device)
        y_val_t   = torch.LongTensor(y_val).to(device)

        from torch.utils.data import DataLoader, TensorDataset
        loader = DataLoader(
            TensorDataset(X_train_t, y_train_t),
            batch_size=self.train_cfg["batch_size"],
            shuffle=True,
        )

        history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
        best_val_loss = float("inf")
        patience_counter = 0
        patience = self.train_cfg.get("early_stopping_patience", 10)

        for epoch in range(start_epoch, self.train_cfg["epochs"]):
            model.train()
            epoch_loss = 0.0
            t0 = time.time()

            for X_batch, y_batch in loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_train_loss = epoch_loss / len(loader)

            # Validation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val_t)
                val_loss = criterion(val_logits, y_val_t).item()
                val_acc = (val_logits.argmax(1) == y_val_t).float().mean().item()

            lr_now = optimizer.param_groups[0]["lr"]
            history["train_loss"].append(avg_train_loss)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)
            history["lr"].append(lr_now)

            elapsed = time.time() - t0
            logger.info(
                f"Epoch [{epoch+1:3d}/{self.train_cfg['epochs']}] "
                f"train_loss={avg_train_loss:.4f} "
                f"val_loss={val_loss:.4f} "
                f"val_acc={val_acc:.4f} "
                f"lr={lr_now:.6f} "
                f"({elapsed:.1f}s)"
            )

            if self.on_epoch_end:
                self.on_epoch_end(epoch, avg_train_loss, val_loss, val_acc, lr_now)

            if scheduler:
                scheduler.step()

            # Checkpoint
            if (epoch + 1) % self.train_cfg["checkpoint_every"] == 0:
                self._save_checkpoint(model, optimizer, epoch, history, rank=0)

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self._save_checkpoint(model, optimizer, epoch, history, rank=0, tag="best")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

        self._save_checkpoint(model, optimizer, self.train_cfg["epochs"] - 1, history, rank=0, tag="final")
        return history

    # ------------------------------------------------------------------ #
    # Distributed (DDP) training
    # ------------------------------------------------------------------ #

    def _run_distributed(self, X_train, y_train, X_val, y_val, world_size) -> Dict:
        manager = mp.Manager()
        shared_history = manager.dict()

        mp.spawn(
            self._ddp_worker,
            args=(world_size, X_train, y_train, X_val, y_val, shared_history),
            nprocs=world_size,
            join=True,
        )

        return dict(shared_history)

    def _ddp_worker(self, rank, world_size, X_train, y_train, X_val, y_val, shared_history):
        setup_ddp(rank, world_size)

        device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

        base_model = build_model(self.config, self.n_features, self.n_classes)
        start_epoch, base_model, optimizer_state = self._maybe_load_checkpoint(base_model)
        model = wrap_model_ddp(base_model, rank)

        if is_main_process(rank):
            logger.info(f"[DDP] Model parameters: {count_parameters(base_model):,}")

        optimizer = self._build_optimizer(model)
        if optimizer_state:
            optimizer.load_state_dict(optimizer_state)
        scheduler = self._build_scheduler(optimizer)
        criterion = nn.CrossEntropyLoss().to(device)

        train_loader, val_loader = build_distributed_loaders(
            X_train, y_train, X_val, y_val,
            self.train_cfg["batch_size"], rank, world_size,
        )

        history = {"train_loss": [], "val_loss": [], "val_acc": [], "lr": []}
        best_val_loss = float("inf")
        patience_counter = 0
        patience = self.train_cfg.get("early_stopping_patience", 10)

        for epoch in range(start_epoch, self.train_cfg["epochs"]):
            train_loader.sampler.set_epoch(epoch)
            model.train()
            epoch_loss = torch.tensor(0.0, device=device)
            t0 = time.time()

            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.detach()

            # Sync loss across all workers
            avg_loss_tensor = epoch_loss / len(train_loader)
            all_reduce_tensor(avg_loss_tensor, world_size)
            avg_train_loss = avg_loss_tensor.item()

            # Validation on rank 0 only
            if is_main_process(rank):
                model.eval()
                total_val_loss, total_correct, total_samples = 0.0, 0, 0
                with torch.no_grad():
                    for X_b, y_b in val_loader:
                        X_b, y_b = X_b.to(device), y_b.to(device)
                        logits = model(X_b)
                        total_val_loss += criterion(logits, y_b).item()
                        total_correct += (logits.argmax(1) == y_b).sum().item()
                        total_samples += len(y_b)

                val_loss = total_val_loss / len(val_loader)
                val_acc = total_correct / total_samples
                lr_now = optimizer.param_groups[0]["lr"]
                elapsed = time.time() - t0

                logger.info(
                    f"[DDP Rank0] Epoch [{epoch+1:3d}/{self.train_cfg['epochs']}] "
                    f"train_loss={avg_train_loss:.4f} "
                    f"val_loss={val_loss:.4f} "
                    f"val_acc={val_acc:.4f} "
                    f"({elapsed:.1f}s)"
                )

                history["train_loss"].append(avg_train_loss)
                history["val_loss"].append(val_loss)
                history["val_acc"].append(val_acc)
                history["lr"].append(lr_now)

                if self.on_epoch_end:
                    self.on_epoch_end(epoch, avg_train_loss, val_loss, val_acc, lr_now)

                if (epoch + 1) % self.train_cfg["checkpoint_every"] == 0:
                    self._save_checkpoint(model, optimizer, epoch, history, rank=rank)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    self._save_checkpoint(model, optimizer, epoch, history, rank=rank, tag="best")
                else:
                    patience_counter += 1

            barrier()

            if scheduler:
                scheduler.step()

            # Broadcast early stopping signal
            stop = torch.tensor(1 if patience_counter >= patience else 0)
            torch.distributed.broadcast(stop, src=0)
            if stop.item():
                if is_main_process(rank):
                    logger.info(f"[DDP] Early stopping at epoch {epoch+1}")
                break

        if is_main_process(rank):
            self._save_checkpoint(model, optimizer, self.train_cfg["epochs"] - 1, history, rank=rank, tag="final")
            for k, v in history.items():
                shared_history[k] = v

        cleanup_ddp()

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _build_optimizer(self, model) -> optim.Optimizer:
        return optim.AdamW(
            model.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=self.train_cfg.get("weight_decay", 1e-4),
        )

    def _build_scheduler(self, optimizer):
        sched = self.train_cfg.get("scheduler", "cosine")
        epochs = self.train_cfg["epochs"]
        if sched == "cosine":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        elif sched == "step":
            return optim.lr_scheduler.StepLR(optimizer, step_size=epochs // 3, gamma=0.5)
        return None

    def _save_checkpoint(self, model, optimizer, epoch, history, rank, tag=None):
        if not is_main_process(rank):
            return
        raw_model = model.module if hasattr(model, "module") else model
        fname = f"checkpoint_epoch{epoch+1}.pt" if tag is None else f"checkpoint_{tag}.pt"
        path = self.checkpoint_dir / fname
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": raw_model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "history": history,
            "config": self.config,
        }, path)
        logger.info(f"Checkpoint saved: {path}")

    def _maybe_load_checkpoint(self, model):
        resume_path = self.train_cfg.get("resume_from")
        if not resume_path:
            # Try finding the latest checkpoint automatically
            ckpts = sorted(self.checkpoint_dir.glob("checkpoint_epoch*.pt"))
            if not ckpts:
                return 0, model, None
            resume_path = str(ckpts[-1])
            logger.info(f"Auto-resuming from: {resume_path}")

        ckpt = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt["epoch"]
        logger.info(f"Resumed from epoch {start_epoch}")
        return start_epoch, model, ckpt.get("optimizer_state_dict")
