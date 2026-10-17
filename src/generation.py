"""Temperature sampling for autoregressive next-token generation."""

from __future__ import annotations

import random
from typing import Iterable

import torch
from torch.nn import functional as F

from src.data_preprocessing import SPECIAL_TOKENS, Vocabulary, detokenize, tokenize
from src.models import BaseLanguageModel


def validate_temperature(temperature: float) -> float:
    value = float(temperature)
    if value <= 0.0 or value != value or value == float("inf"):
        raise ValueError(f"Temperature must be positive and finite, got {temperature}")
    return value


def sample_from_logits(logits: torch.Tensor, temperature: float) -> int:
    temperature = validate_temperature(temperature)
    scaled = logits.float() / temperature
    probabilities = F.softmax(scaled, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise RuntimeError("Non-finite probabilities after temperature softmax.")
    sampled = torch.multinomial(probabilities, num_samples=1)
    return int(sampled.item())


def encode_seed(seed_phrase: str, vocab: Vocabulary) -> list[int]:
    tokens = tokenize(seed_phrase)
    if not tokens:
        return [vocab.bos_index]
    return vocab.encode(tokens)


@torch.no_grad()
def generate_tokens(
    model: BaseLanguageModel,
    vocab: Vocabulary,
    seed_phrase: str,
    temperature: float,
    max_new_tokens: int,
    sequence_length: int,
    device: torch.device,
    sampling_seed: int | None = None,
) -> dict:
    model.eval()
    if sampling_seed is not None:
        random.seed(sampling_seed)
        torch.manual_seed(sampling_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(sampling_seed)

    context = encode_seed(seed_phrase, vocab)
    generated: list[int] = []
    stopped_on_eos = False
    for _ in range(int(max_new_tokens)):
        window = context[-sequence_length:]
        inputs = torch.tensor([window], dtype=torch.long, device=device)
        logits = model(inputs)[0]
        if not torch.isfinite(logits).all():
            raise RuntimeError("Non-finite logits during generation.")
        next_id = sample_from_logits(logits, temperature)
        if next_id == vocab.eos_index:
            stopped_on_eos = True
            break
        if next_id == vocab.pad_index:
            next_id = vocab.unk_index
        generated.append(next_id)
        context.append(next_id)

    generated_tokens = [vocab.decode_token(index) for index in generated]
    visible_tokens = [token for token in generated_tokens if token not in SPECIAL_TOKENS]
    seed_tokens = tokenize(seed_phrase)
    full_text = detokenize(seed_tokens + visible_tokens)
    return {
        "seed_phrase": seed_phrase,
        "temperature": float(temperature),
        "sampling_seed": sampling_seed,
        "generated_token_ids": generated,
        "generated_tokens": generated_tokens,
        "generated_token_count": len(generated_tokens),
        "stopped_on_eos": stopped_on_eos,
        "generated_text": detokenize(visible_tokens),
        "full_text": full_text,
    }


def generate_grid(
    model: BaseLanguageModel,
    vocab: Vocabulary,
    seeds: Iterable[str],
    temperatures: Iterable[float],
    max_new_tokens: int,
    sequence_length: int,
    device: torch.device,
    sampling_seed: int,
    model_name: str,
) -> list[dict]:
    samples = []
    for seed_index, seed in enumerate(seeds):
        for temp_index, temperature in enumerate(temperatures):
            local_seed = sampling_seed + 1000 * seed_index + temp_index
            sample = generate_tokens(
                model=model,
                vocab=vocab,
                seed_phrase=seed,
                temperature=temperature,
                max_new_tokens=max_new_tokens,
                sequence_length=sequence_length,
                device=device,
                sampling_seed=local_seed,
            )
            sample["model"] = model_name
            samples.append(sample)
    return samples
