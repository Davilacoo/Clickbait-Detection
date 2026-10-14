from __future__ import annotations

import re
from typing import Iterable, List

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


CLICKBAIT_KEYWORDS = [
    "shocking", "shocked", "amazing", "unbelievable", "incredible",
    "you won't believe", "won't believe", "wont believe",
    "what happened next", "what happens next",
    "this is what", "this is why", "this is how",
    "secret", "secrets", "reasons", "things you",
    "you need to", "you should", "you have to",
    "trick", "hack", "hacks", "blew my mind",
    "mind-blowing", "mind blowing", "jaw-dropping", "jaw dropping",
    "the truth about", "the real reason",
    "won't stop", "will make you", "will change your",
    "here's what", "heres what", "here's why", "heres why",
    "you'll never", "youll never",
    "love it", "epic", "viral", "goals", "literally",
    "everyone is talking", "internet is freaking",
    "ranked", "quiz", "tweets", "perfectly sum",
]

PRONOUN_PATTERNS = re.compile(
    r"\b(you|your|yours|yourself|we|us|our|this|these)\b", re.IGNORECASE
)
DIGIT_RE = re.compile(r"\d")
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
STARTS_WITH_NUM_RE = re.compile(r"^\s*\d+\b")


def _count_caps_tokens(text: str) -> float:
    tokens = TOKEN_RE.findall(text)
    if not tokens:
        return 0.0
    long_tokens = [t for t in tokens if len(t) >= 2]
    if not long_tokens:
        return 0.0
    upper = sum(1 for t in long_tokens if t.isupper())
    return upper / len(long_tokens)


def _keyword_hits(text_lower: str) -> int:
    return sum(1 for kw in CLICKBAIT_KEYWORDS if kw in text_lower)


def extract_features(text: str) -> dict:
    if not isinstance(text, str):
        text = ""
    text_lower = text.lower()
    tokens = TOKEN_RE.findall(text)

    return {
        "char_len": float(len(text)),
        "n_words": float(len(tokens)),
        "n_exclaim": float(text.count("!")),
        "n_question": float(text.count("?")),
        "has_digit": float(bool(DIGIT_RE.search(text))),
        "starts_with_number": float(bool(STARTS_WITH_NUM_RE.match(text))),
        "n_pronouns": float(len(PRONOUN_PATTERNS.findall(text))),
        "frac_caps": _count_caps_tokens(text),
        "n_clickbait_kw": float(_keyword_hits(text_lower)),
        "has_clickbait_kw": float(_keyword_hits(text_lower) > 0),
        "avg_word_len": (
            float(np.mean([len(t) for t in tokens])) if tokens else 0.0
        ),
    }


def feature_names() -> List[str]:
    return [
        "char_len", "n_words", "n_exclaim", "n_question",
        "has_digit", "starts_with_number", "n_pronouns",
        "frac_caps", "n_clickbait_kw", "has_clickbait_kw", "avg_word_len",
    ]


def features_dataframe(texts: Iterable[str]) -> pd.DataFrame:
    rows = [extract_features(t) for t in texts]
    return pd.DataFrame(rows, columns=feature_names())


class ManualFeatureExtractor(BaseEstimator, TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = features_dataframe(X)
        return df.values.astype(np.float64)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(feature_names(), dtype=object)
