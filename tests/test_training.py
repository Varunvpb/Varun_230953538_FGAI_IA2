import copy

import torch

from src.dataset import make_dataloader
from src.models import build_model
from src.trainer import build_criterion, load_checkpoint, save_checkpoint, train_one_epoch


def _tiny_model(name="lstm"):
    return build_model(name, vocab_size=16, embedding_dim=8, hidden_dim=8, pad_index=0, dropout=0.0)


def test_one_training_step_each_model():
    ids = list(range(1, 16)) * 3
    loader = make_dataloader(ids, sequence_length=4, batch_size=4, stride=1, shuffle=False)
    criterion = build_criterion(pad_index=0)
    for name in ["simple_rnn", "bidirectional_rnn", "lstm", "gru"]:
        model = _tiny_model(name)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        before = [p.detach().clone() for p in model.parameters()]
        stats = train_one_epoch(model, loader, optimizer, criterion, torch.device("cpu"), clip_norm=1.0, log_every=0)
        assert stats["loss"] > 0
        changed = any(not torch.equal(a, b.detach()) for a, b in zip(before, model.parameters()))
        assert changed, name


def test_validation_does_not_update_parameters():
    model = _tiny_model("gru")
    snapshot = copy.deepcopy(model.state_dict())
    x = torch.randint(1, 16, (2, 4))
    y = torch.randint(1, 16, (2,))
    criterion = build_criterion(0)
    model.eval()
    with torch.no_grad():
        loss = criterion(model(x), y)
    assert torch.isfinite(loss)
    for key, value in model.state_dict().items():
        assert torch.equal(value, snapshot[key])


def test_gradient_clipping_runs():
    model = _tiny_model("simple_rnn")
    x = torch.randint(1, 16, (2, 4))
    y = torch.randint(1, 16, (2,))
    criterion = build_criterion(0)
    loss = criterion(model(x), y)
    loss.backward()
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    assert torch.isfinite(total)


def test_checkpoint_roundtrip(tmp_path):
    model = _tiny_model("lstm")
    path = tmp_path / "m.pt"
    payload = {
        "architecture": "lstm",
        "model_state_dict": model.state_dict(),
        "hyperparameters": model.hyperparameters(),
        "vocab_size": 16,
        "sequence_length": 4,
        "pad_index": 0,
        "best_val_loss": 1.23,
        "epoch": 2,
        "random_seed": 42,
    }
    save_checkpoint(path, payload)
    loaded = load_checkpoint(path)
    restored = _tiny_model("lstm")
    restored.load_state_dict(loaded["model_state_dict"])
    x = torch.randint(1, 16, (1, 4))
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.allclose(model(x), restored(x))
    assert loaded["architecture"] == "lstm"
    assert loaded["hyperparameters"]["vocab_size"] == 16
