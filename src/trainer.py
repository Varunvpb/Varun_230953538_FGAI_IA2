"""Reusable training loop with validation, clipping, and checkpointing."""

from __future__ import annotations

import time
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.metrics import evaluate_loader
from src.models import BaseLanguageModel
from src.utils import ensure_dir, set_seed


def build_criterion(pad_index: int, ignore_pad: bool = True) -> nn.Module:
    ignore_index = pad_index if ignore_pad else -100
    return nn.CrossEntropyLoss(ignore_index=ignore_index)


def save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict:
    return torch.load(path, map_location=map_location, weights_only=False)


def checkpoint_compatible(checkpoint: dict, architecture: str, vocab_size: int, sequence_length: int) -> bool:
    if checkpoint.get("architecture") != architecture:
        return False
    hyper = checkpoint.get("hyperparameters") or {}
    if int(hyper.get("vocab_size", -1)) != int(vocab_size):
        return False
    if int(checkpoint.get("sequence_length", -1)) != int(sequence_length):
        return False
    return True


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    clip_norm: float,
    log_every: int = 200,
) -> dict[str, float]:
    model.train()
    running_loss = 0.0
    seen = 0
    start = time.perf_counter()
    for step, (inputs, targets) in enumerate(loader, start=1):
        inputs = inputs.to(device)
        targets = targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits during training.")
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Non-finite loss at step {step}: {loss.item()}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()
        batch_size = targets.size(0)
        running_loss += float(loss.item()) * batch_size
        seen += batch_size
        if log_every and step % log_every == 0:
            print(f"    step {step}/{len(loader)}  loss={loss.item():.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    duration = time.perf_counter() - start
    return {"loss": running_loss / max(seen, 1), "duration_sec": duration, "num_targets": float(seen)}


def train_model(
    model: BaseLanguageModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    learning_rate: float,
    clip_norm: float,
    device: torch.device,
    checkpoint_path: Path,
    architecture: str,
    vocab_size: int,
    sequence_length: int,
    pad_index: int,
    random_seed: int,
    ignore_pad: bool = True,
    log_every: int = 200,
) -> dict:
    set_seed(random_seed)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = build_criterion(pad_index, ignore_pad=ignore_pad)
    history: list[dict] = []
    best_val_loss = float("inf")
    best_epoch = -1
    training_start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        print(f"  Epoch {epoch}/{epochs}", flush=True)
        train_stats = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
            device,
            clip_norm,
            log_every=log_every,
        )
        val_stats = evaluate_loader(model, val_loader, criterion, device)
        record = {
            "epoch": epoch,
            "train_loss": train_stats["loss"],
            "val_loss": val_stats["loss"],
            "val_perplexity": val_stats["perplexity"],
            "val_accuracy": val_stats["accuracy"],
            "epoch_duration_sec": train_stats["duration_sec"],
        }
        history.append(record)
        print(
            "    "
            f"train_loss={record['train_loss']:.4f}  "
            f"val_loss={record['val_loss']:.4f}  "
            f"val_ppl={record['val_perplexity']:.2f}  "
            f"val_acc={record['val_accuracy']:.4f}  "
            f"time={record['epoch_duration_sec']:.1f}s",
            flush=True,
        )
        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            best_epoch = epoch
            ensure_dir(checkpoint_path.parent)
            save_checkpoint(
                checkpoint_path,
                {
                    "architecture": architecture,
                    "model_state_dict": model.state_dict(),
                    "hyperparameters": model.hyperparameters(),
                    "vocab_size": vocab_size,
                    "sequence_length": sequence_length,
                    "pad_index": pad_index,
                    "best_val_loss": best_val_loss,
                    "epoch": epoch,
                    "random_seed": random_seed,
                    "learning_rate": learning_rate,
                    "clip_norm": clip_norm,
                    "history": history,
                },
            )

    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time.perf_counter() - training_start
    return {
        "architecture": architecture,
        "history": history,
        "best_val_loss": best_val_loss,
        "best_epoch": best_epoch,
        "total_training_time_sec": total_time,
        "epochs_completed": epochs,
        "device": str(device),
        "checkpoint_path": str(checkpoint_path),
        "parameter_count": model.parameter_count(),
    }
