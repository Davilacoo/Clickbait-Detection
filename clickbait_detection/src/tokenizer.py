from __future__ import annotations

import re
from collections import Counter
from typing import Iterable, List

import numpy as np

from . import config

_TOKEN_RE = re.compile(r"[A-Za-z]+|\d+")


def tokenize(text: str) -> List[str]:
    if not isinstance(text, str):
        return []
    return _TOKEN_RE.findall(text.lower())


class WordTokenizer:

    def __init__(self,
                 vocab_size: int = config.VOCAB_SIZE,
                 max_len: int = config.MAX_LEN):
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []

    def fit(self, texts: Iterable[str]) -> "WordTokenizer":
        counter: Counter[str] = Counter()
        for t in texts:
            counter.update(tokenize(t))

        self.idx2word = ["<pad>", "<unk>"]
        for word, _count in counter.most_common(self.vocab_size - 2):
            self.idx2word.append(word)
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}
        return self

    def encode_one(self, text: str) -> np.ndarray:
        tokens = tokenize(text)[: self.max_len]
        ids = [self.word2idx.get(tok, config.UNK_IDX) for tok in tokens]
        if len(ids) < self.max_len:
            ids = ids + [config.PAD_IDX] * (self.max_len - len(ids))
        return np.asarray(ids, dtype=np.int64)

    def encode(self, texts: Iterable[str]) -> np.ndarray:
        return np.stack([self.encode_one(t) for t in texts], axis=0)

    def __len__(self) -> int:
        return len(self.idx2word)

    def __repr__(self) -> str:
        return f"WordTokenizer(vocab_size={len(self)}, max_len={self.max_len})"
