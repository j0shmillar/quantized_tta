from __future__ import annotations

import json
import os
import random
from typing import Iterable, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def accuracy(logits, labels):
    return (logits.argmax(1) == labels).float().mean().item()


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0
    def update(self, value, n):
        self.sum += float(value) * n
        self.count += int(n)
    @property
    def avg(self) -> float:
        return self.sum / max(1, self.count)


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
