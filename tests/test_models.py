import torch

from src.models import MODEL_REGISTRY, build_model
from src.utils import count_parameters


def _forward_all(vocab_size=32, seq_len=6, batch=3, pad_index=0):
    x = torch.randint(1, vocab_size, (batch, seq_len))
    outputs = {}
    for name in MODEL_REGISTRY:
        model = build_model(name, vocab_size, 8, 10, pad_index, dropout=0.0)
        model.eval()
        logits = model(x)
        outputs[name] = (model, logits)
    return x, outputs


def test_forward_shapes_and_finite_logits():
    _, outputs = _forward_all()
    for name, (model, logits) in outputs.items():
        assert logits.shape == (3, 32), name
        assert torch.isfinite(logits).all(), name
        assert model.vocab_size == 32


def test_each_architecture_forward_pass():
    x = torch.randint(1, 20, (2, 5))
    for name in ["simple_rnn", "bidirectional_rnn", "lstm", "gru"]:
        model = build_model(name, 20, 8, 12, pad_index=0, dropout=0.0)
        logits = model(x)
        assert logits.shape[-1] == 20
        assert torch.isfinite(logits).all()


def test_parameter_count_positive_and_bidirectional_larger():
    simple = build_model("simple_rnn", 50, 8, 16, 0, dropout=0.0)
    bi = build_model("bidirectional_rnn", 50, 8, 16, 0, dropout=0.0)
    assert count_parameters(simple) > 0
    assert bi.parameter_count() > simple.parameter_count()


def test_output_dim_matches_vocab():
    for vocab_size in (12, 40):
        model = build_model("lstm", vocab_size, 8, 8, 0, dropout=0.0)
        logits = model(torch.randint(1, vocab_size, (1, 4)))
        assert logits.shape == (1, vocab_size)
