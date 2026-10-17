"""Quantitative next-token metrics and generation-quality statistics."""

from __future__ import annotations

import math
from collections import Counter
from typing import Iterable, Sequence

import torch
from torch import nn


def safe_perplexity(mean_cross_entropy: float) -> float:
    """perplexity = exp(mean token-level cross-entropy), overflow-safe."""
    if mean_cross_entropy != mean_cross_entropy or math.isinf(mean_cross_entropy):
        return float("inf")
    clipped = min(max(mean_cross_entropy, -20.0), 20.0)
    return float(math.exp(clipped))


def next_word_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    predictions = logits.argmax(dim=-1)
    return (predictions == targets).float()


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_tokens = 0
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)
        logits = model(inputs)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits encountered during evaluation.")
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        total_loss += float(loss.item()) * batch_size
        total_correct += float(next_word_accuracy(logits, targets).sum().item())
        total_tokens += batch_size
    if total_tokens == 0:
        raise ValueError("Evaluation loader was empty.")
    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "perplexity": safe_perplexity(mean_loss),
        "accuracy": total_correct / total_tokens,
        "num_targets": float(total_tokens),
    }


def unique_token_ratio(tokens: Sequence[str]) -> float:
    if not tokens:
        return 0.0
    return len(set(tokens)) / len(tokens)


def distinct_n(tokens: Sequence[str], n: int) -> float:
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(ngrams)) / len(ngrams)


def repeated_ngram_ratio(tokens: Sequence[str], n: int = 2) -> float:
    """Fraction of n-grams that occur more than once (by instance count)."""
    if len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    repeated_instances = sum(count for count in counts.values() if count > 1)
    return repeated_instances / len(ngrams)


def generation_statistics(token_sequences: Iterable[Sequence[str]]) -> dict[str, float]:
    sequences = [list(seq) for seq in token_sequences]
    if not sequences:
        return {
            "unique_token_ratio": 0.0,
            "distinct_1": 0.0,
            "distinct_2": 0.0,
            "repeated_bigram_ratio": 0.0,
            "average_length": 0.0,
            "num_samples": 0.0,
        }
    unique_ratios = [unique_token_ratio(seq) for seq in sequences]
    d1 = [distinct_n(seq, 1) for seq in sequences]
    d2 = [distinct_n(seq, 2) for seq in sequences]
    rep = [repeated_ngram_ratio(seq, 2) for seq in sequences]
    lengths = [len(seq) for seq in sequences]
    return {
        "unique_token_ratio": sum(unique_ratios) / len(unique_ratios),
        "distinct_1": sum(d1) / len(d1),
        "distinct_2": sum(d2) / len(d2),
        "repeated_bigram_ratio": sum(rep) / len(rep),
        "average_length": sum(lengths) / len(lengths),
        "num_samples": float(len(sequences)),
    }
