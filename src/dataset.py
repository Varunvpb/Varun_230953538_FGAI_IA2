"""On-the-fly next-token windows that do not cross split boundaries."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Dataset


class NextTokenDataset(Dataset):
    """Fixed-length input windows with the immediately following token as the target.

    For tokens [t0, t1, ..., tn-1] and sequence_length L:
        input  = [t_i, ..., t_{i+L-1}]
        target = t_{i+L}

    Example: "To be or not to be" with L=5 yields
        input  = To be or not to
        target = be
    """

    def __init__(self, token_ids: list[int], sequence_length: int, stride: int = 1) -> None:
        if sequence_length < 1:
            raise ValueError("sequence_length must be >= 1")
        if stride < 1:
            raise ValueError("stride must be >= 1")
        if len(token_ids) <= sequence_length:
            raise ValueError("Need more tokens than sequence_length to form a next-token pair.")
        self.token_ids = token_ids
        self.sequence_length = sequence_length
        self.stride = stride
        last_start = len(token_ids) - sequence_length - 1
        self.starts = list(range(0, last_start + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        start = self.starts[index]
        end = start + self.sequence_length
        inputs = torch.tensor(self.token_ids[start:end], dtype=torch.long)
        target = torch.tensor(self.token_ids[end], dtype=torch.long)
        return inputs, target


def make_dataloader(
    token_ids: list[int],
    sequence_length: int,
    batch_size: int,
    stride: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    dataset = NextTokenDataset(token_ids, sequence_length=sequence_length, stride=stride)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
