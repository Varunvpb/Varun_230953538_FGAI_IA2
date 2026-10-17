from src.data_preprocessing import (
    Vocabulary,
    build_vocabulary,
    chronological_split,
    clean_text,
    detokenize,
    tokenize,
)
from src.dataset import NextTokenDataset


def test_clean_text_normalizes_newlines_and_control_chars():
    raw = "ROMEO:\r\nHello\x00 world"
    cleaned = clean_text(raw)
    assert "\r" not in cleaned
    assert "\x00" not in cleaned
    assert "ROMEO:" in cleaned


def test_tokenize_preserves_contractions_and_splits_punctuation():
    tokens = tokenize("ROMEO: I can't go. Hello, world!")
    assert "ROMEO" in tokens
    assert ":" in tokens
    assert "can't" in tokens
    assert "Hello" in tokens
    assert "," in tokens
    assert "world" in tokens
    assert "!" in tokens


def test_detokenize_roundtrip_readable():
    tokens = tokenize("To be, or not to be")
    text = detokenize(tokens)
    assert "To be" in text
    assert "," in text


def test_vocabulary_specials_and_unk():
    vocab = build_vocabulary(["the", "king", "the", "queen"], max_vocab_size=8)
    assert vocab.token_to_index["<PAD>"] == 0
    assert "<UNK>" in vocab.token_to_index
    assert vocab.encode_token("missing-word") == vocab.unk_index
    encoded = vocab.encode(["the", "unknown"])
    assert encoded[0] == vocab.token_to_index["the"]
    assert encoded[1] == vocab.unk_index


def test_split_is_chronological_and_complete():
    tokens = [f"t{i}" for i in range(10)]
    splits = chronological_split(tokens, 0.8, 0.1, 0.1)
    assert splits["train"] == tokens[:8]
    assert splits["val"] == tokens[8:9]
    assert splits["test"] == tokens[9:]
    assert splits["train"] + splits["val"] + splits["test"] == tokens


def test_input_target_alignment_known_example():
    # "To be or not to be" as six tokens; L=5 -> input first five, target last.
    tokens = ["To", "be", "or", "not", "to", "be"]
    vocab = build_vocabulary(tokens, max_vocab_size=16)
    ids = vocab.encode(tokens)
    dataset = NextTokenDataset(ids, sequence_length=5, stride=1)
    inputs, target = dataset[0]
    assert inputs.tolist() == vocab.encode(["To", "be", "or", "not", "to"])
    assert int(target.item()) == vocab.encode_token("be")


def test_sequence_length_and_no_boundary_overrun():
    ids = list(range(10))
    dataset = NextTokenDataset(ids, sequence_length=4, stride=1)
    last_inputs, last_target = dataset[-1]
    assert len(last_inputs) == 4
    assert int(last_target.item()) == ids[-1]
    assert last_inputs.tolist() + [int(last_target.item())] == ids[-5:]


def test_vocabulary_save_load(tmp_path):
    vocab = build_vocabulary(["a", "b", "a"], max_vocab_size=10)
    path = tmp_path / "vocab.json"
    vocab.save(path)
    loaded = Vocabulary.load(path)
    assert loaded.token_to_index == vocab.token_to_index
    assert len(loaded) == len(vocab)
