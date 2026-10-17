"""Text cleaning, word-level tokenization, vocabulary, and chronological splits."""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

from src.utils import PROJECT_ROOT, ensure_dir, load_config, resolve_path, save_json

TOKEN_PATTERN = re.compile(r"[A-Za-z]+(?:'[A-Za-z]+)?|\d+|[^\s\w]")
WORD_LIKE = re.compile(r"[A-Za-z0-9]")
CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")

SPECIAL_TOKENS = ("<PAD>", "<UNK>", "<BOS>", "<EOS>")


@dataclass
class Vocabulary:
    token_to_index: dict[str, int]
    index_to_token: list[str]
    max_vocab_size: int
    unk_token: str = "<UNK>"
    pad_token: str = "<PAD>"
    bos_token: str = "<BOS>"
    eos_token: str = "<EOS>"

    def __len__(self) -> int:
        return len(self.index_to_token)

    @property
    def pad_index(self) -> int:
        return self.token_to_index[self.pad_token]

    @property
    def unk_index(self) -> int:
        return self.token_to_index[self.unk_token]

    @property
    def bos_index(self) -> int:
        return self.token_to_index[self.bos_token]

    @property
    def eos_index(self) -> int:
        return self.token_to_index[self.eos_token]

    def encode_token(self, token: str) -> int:
        return self.token_to_index.get(token, self.unk_index)

    def encode(self, tokens: Iterable[str]) -> list[int]:
        return [self.encode_token(token) for token in tokens]

    def decode_token(self, index: int) -> str:
        if 0 <= index < len(self.index_to_token):
            return self.index_to_token[index]
        return self.unk_token

    def save(self, path: str | Path) -> None:
        target = resolve_path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "token_to_index": self.token_to_index,
            "index_to_token": self.index_to_token,
            "max_vocab_size": self.max_vocab_size,
            "unk_token": self.unk_token,
            "pad_token": self.pad_token,
            "bos_token": self.bos_token,
            "eos_token": self.eos_token,
        }
        with target.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "Vocabulary":
        with resolve_path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            token_to_index={str(k): int(v) for k, v in payload["token_to_index"].items()},
            index_to_token=list(payload["index_to_token"]),
            max_vocab_size=int(payload["max_vocab_size"]),
            unk_token=payload.get("unk_token", "<UNK>"),
            pad_token=payload.get("pad_token", "<PAD>"),
            bos_token=payload.get("bos_token", "<BOS>"),
            eos_token=payload.get("eos_token", "<EOS>"),
        )


def load_raw_text(path: str | Path, encoding: str = "utf-8") -> str:
    return resolve_path(path).read_text(encoding=encoding)


def clean_text(text: str) -> str:
    """Light cleaning that preserves Shakespearean dialogue and punctuation.

    Decisions:
    - Normalize CRLF/CR to LF.
    - Strip ASCII control characters other than tab/newline.
    - Convert newlines and tabs to spaces so word-level modelling is not dominated
      by line-break tokens, while keeping speaker names and punctuation.
    - Collapse repeated whitespace.
    - Do not lowercase (character names such as ROMEO remain identifiable).
    - Preserve apostrophes inside contractions (can't, 'tis).
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = CONTROL_CHARS.sub("", text)
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r" +", " ", text)
    return text.strip()


def tokenize(text: str) -> list[str]:
    """Word-level tokenizer: words, contractions, numbers, and isolated punctuation."""
    return TOKEN_PATTERN.findall(text)


def detokenize(tokens: Iterable[str]) -> str:
    """Reconstruct readable text from tokens.

    Spaces are inserted before word-like tokens. Punctuation is attached to the
    preceding token except for opening brackets/quotes.
    """
    pieces: list[str] = []
    opening = set("([{“\"'")
    closing_no_space_before = set(".,!?;:)]}%")
    for token in tokens:
        if token in SPECIAL_TOKENS:
            continue
        if not pieces:
            pieces.append(token)
            continue
        if token in closing_no_space_before or (
            all(not WORD_LIKE.search(ch) for ch in token) and token not in opening
        ):
            pieces.append(token)
        else:
            pieces.append(" " + token)
    return "".join(pieces).strip()


def build_vocabulary(train_tokens: list[str], max_vocab_size: int) -> Vocabulary:
    if max_vocab_size < len(SPECIAL_TOKENS):
        raise ValueError("max_vocab_size must be at least the number of special tokens.")
    counts = Counter(train_tokens)
    reserved = list(SPECIAL_TOKENS)
    remaining = max_vocab_size - len(reserved)
    most_common = [token for token, _ in counts.most_common() if token not in reserved]
    selected = reserved + most_common[:remaining]
    token_to_index = {token: index for index, token in enumerate(selected)}
    return Vocabulary(
        token_to_index=token_to_index,
        index_to_token=selected,
        max_vocab_size=max_vocab_size,
    )


def chronological_split(
    tokens: list[str],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> dict[str, list[str]]:
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1, got {total}")
    n = len(tokens)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    train = tokens[:train_end]
    val = tokens[train_end:val_end]
    test = tokens[val_end:]
    if not train or not val or not test:
        raise ValueError("Each split must contain at least one token.")
    return {"train": train, "val": val, "test": test}


def wrap_split(tokens: list[str]) -> list[str]:
    return ["<BOS>"] + list(tokens) + ["<EOS>"]


def encode_splits(splits: dict[str, list[str]], vocab: Vocabulary) -> dict[str, list[int]]:
    return {name: vocab.encode(wrap_split(tokens)) for name, tokens in splits.items()}


def preprocess(config: dict | None = None) -> dict:
    config = config or load_config()
    raw_path = config["paths"]["raw_text"]
    processed_dir = ensure_dir(config["paths"]["processed_dir"])
    encoding = config["dataset"].get("encoding", "utf-8")
    prep = config["preprocessing"]

    raw_text = load_raw_text(raw_path, encoding=encoding)
    cleaned = clean_text(raw_text)
    tokens = tokenize(cleaned)
    splits = chronological_split(
        tokens,
        train_ratio=float(prep["train_ratio"]),
        val_ratio=float(prep["val_ratio"]),
        test_ratio=float(prep["test_ratio"]),
    )
    vocab = build_vocabulary(splits["train"], int(prep["max_vocab_size"]))
    encoded = encode_splits(splits, vocab)

    vocab_path = processed_dir / "vocabulary.json"
    vocab.save(vocab_path)
    for name, ids in encoded.items():
        (processed_dir / f"{name}_ids.json").write_text(json.dumps(ids), encoding="utf-8")

    metadata = {
        "raw_characters": len(raw_text),
        "cleaned_characters": len(cleaned),
        "total_word_tokens": len(tokens),
        "unique_tokens_full_corpus": len(set(tokens)),
        "train_tokens": len(splits["train"]),
        "val_tokens": len(splits["val"]),
        "test_tokens": len(splits["test"]),
        "train_encoded_length": len(encoded["train"]),
        "val_encoded_length": len(encoded["val"]),
        "test_encoded_length": len(encoded["test"]),
        "vocabulary_size": len(vocab),
        "max_vocab_size": int(prep["max_vocab_size"]),
        "sequence_length": int(prep["sequence_length"]),
        "window_stride": int(prep.get("window_stride", 1)),
        "split_strategy": (
            "Chronological 80/10/10 split on the token stream before windowing. "
            "This evaluates next-word prediction on later held-out text and prevents "
            "overlapping windows from crossing split boundaries."
        ),
        "tokenization": (
            "Regex word-level tokenizer: alphabetic tokens with internal apostrophes, "
            "digit sequences, and isolated punctuation. Case is preserved."
        ),
        "paths": {
            "vocabulary": str(vocab_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            "train_ids": "data/processed/train_ids.json",
            "val_ids": "data/processed/val_ids.json",
            "test_ids": "data/processed/test_ids.json",
        },
    }
    save_json(processed_dir / "preprocess_metadata.json", metadata)
    return metadata


def load_encoded_ids(split: str, processed_dir: str | Path | None = None) -> list[int]:
    directory = resolve_path(processed_dir or "data/processed")
    path = directory / f"{split}_ids.json"
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
