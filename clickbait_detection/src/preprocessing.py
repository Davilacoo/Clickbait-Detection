from __future__ import annotations

import re
from typing import Iterable, List

import pandas as pd

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")


def clean_for_tfidf(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def clean_series(texts: Iterable[str]) -> List[str]:
    return [clean_for_tfidf(t) for t in texts]


def add_cleaned_column(df: pd.DataFrame, source: str = "headline",
                      target: str = "headline_clean") -> pd.DataFrame:
    df = df.copy()
    df[target] = clean_series(df[source])
    return df
