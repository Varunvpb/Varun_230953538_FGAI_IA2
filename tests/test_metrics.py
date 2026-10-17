import math

import torch

from src.metrics import (
    distinct_n,
    evaluate_loader,
    generation_statistics,
    next_word_accuracy,
    repeated_ngram_ratio,
    safe_perplexity,
    unique_token_ratio,
)
from src.dataset import make_dataloader
from src.models import build_model
from src.trainer import build_criterion


def test_perplexity_is_exp_of_mean_ce():
    assert abs(safe_perplexity(0.0) - 1.0) < 1e-9
    assert abs(safe_perplexity(math.log(2)) - 2.0) < 1e-9
    assert safe_perplexity(float("nan")) == float("inf")


def test_next_word_accuracy():
    logits = torch.tensor([[0.1, 5.0, 0.2], [4.0, 0.1, 0.1]])
    targets = torch.tensor([1, 0])
    acc = next_word_accuracy(logits, targets).mean().item()
    assert acc == 1.0


def test_distinct_and_repetition_metrics():
    tokens = ["a", "b", "a", "b", "a"]
    assert unique_token_ratio(tokens) == 0.4
    assert distinct_n(tokens, 1) == 0.4
    assert distinct_n(tokens, 2) < 1.0
    assert repeated_ngram_ratio(tokens, 2) > 0
    assert unique_token_ratio([]) == 0.0
    assert distinct_n(["only"], 2) == 0.0
    assert repeated_ngram_ratio([], 2) == 0.0


def test_generation_statistics_empty():
    stats = generation_statistics([])
    assert stats["num_samples"] == 0.0
    assert stats["distinct_1"] == 0.0


def test_evaluate_loader_aggregates():
    ids = list(range(1, 20))
    loader = make_dataloader(ids, sequence_length=3, batch_size=4, stride=1)
    model = build_model("gru", 20, 8, 8, 0, dropout=0.0)
    criterion = build_criterion(0)
    stats = evaluate_loader(model, loader, criterion, torch.device("cpu"))
    assert stats["num_targets"] > 0
    assert stats["loss"] > 0
    assert stats["perplexity"] > 1.0
    assert 0.0 <= stats["accuracy"] <= 1.0
