import pytest
import torch

from src.data_preprocessing import build_vocabulary
from src.generation import encode_seed, generate_tokens, sample_from_logits, validate_temperature
from src.models import build_model


def test_temperature_must_be_positive_and_finite():
    assert validate_temperature(0.5) == 0.5
    with pytest.raises(ValueError):
        validate_temperature(0.0)
    with pytest.raises(ValueError):
        validate_temperature(-1.0)
    with pytest.raises(ValueError):
        validate_temperature(float("nan"))


def test_unknown_seed_tokens_map_to_unk():
    vocab = build_vocabulary(["ROMEO", ":", "KING"], max_vocab_size=16)
    ids = encode_seed("ROMEO: zebra", vocab)
    assert vocab.unk_index in ids
    assert vocab.encode_token("ROMEO") in ids


def test_reproducible_sampling():
    logits = torch.tensor([1.0, 2.0, 0.5, 0.1])
    torch.manual_seed(0)
    a = sample_from_logits(logits, 1.0)
    torch.manual_seed(0)
    b = sample_from_logits(logits, 1.0)
    assert a == b


def test_generation_length_and_eos_and_all_architectures():
    tokens = ["ROMEO", ":", "To", "be", "or", "not", "KING", "the", "king"]
    vocab = build_vocabulary(tokens, max_vocab_size=32)
    for name in ["simple_rnn", "bidirectional_rnn", "lstm", "gru"]:
        model = build_model(name, len(vocab), 8, 8, vocab.pad_index, dropout=0.0)
        sample = generate_tokens(
            model,
            vocab,
            seed_phrase="ROMEO:",
            temperature=1.0,
            max_new_tokens=12,
            sequence_length=4,
            device=torch.device("cpu"),
            sampling_seed=123,
        )
        assert sample["generated_token_count"] <= 12
        assert sample["generated_token_count"] >= 0
        assert isinstance(sample["generated_text"], str)


def test_eos_stops_generation():
    vocab = build_vocabulary(["hello"], max_vocab_size=8)

    class EosModel(torch.nn.Module):
        def eval(self):
            return self

        def __call__(self, token_ids):
            logits = torch.full((token_ids.size(0), len(vocab)), -20.0)
            logits[:, vocab.eos_index] = 10.0
            return logits

    sample = generate_tokens(
        EosModel(),
        vocab,
        "hello",
        temperature=1.0,
        max_new_tokens=20,
        sequence_length=4,
        device=torch.device("cpu"),
        sampling_seed=1,
    )
    assert sample["stopped_on_eos"] is True
    assert sample["generated_token_count"] == 0
