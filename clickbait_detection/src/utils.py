from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from . import config


def set_seed(seed: int = config.RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _resolve_csv_path() -> Path:
    if config.DATA_PATH.exists():
        return config.DATA_PATH
    candidates = sorted(config.RAW_DIR.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(
            f"No CSV file found in {config.RAW_DIR}. "
            f"Place your dataset there or set config.DATA_PATH."
        )
    return candidates[0]


def _normalise_labels(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        return series.astype(int)
    cleaned = series.astype(str).str.strip().str.lower()
    pos_set = {str(v).lower() for v in config.POSITIVE_LABELS}
    return cleaned.isin(pos_set).astype(int)


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or _resolve_csv_path()
    df = pd.read_csv(csv_path)

    text_col = config.TEXT_COL if config.TEXT_COL in df.columns else df.columns[0]
    label_col = config.LABEL_COL if config.LABEL_COL in df.columns else df.columns[-1]

    df = df[[text_col, label_col]].rename(
        columns={text_col: "headline", label_col: "label"}
    )

    df = df.dropna(subset=["headline", "label"])
    df["headline"] = df["headline"].astype(str).str.strip()
    df = df[df["headline"].str.len() > 0]
    df["label"] = _normalise_labels(df["label"])
    df = df.drop_duplicates(subset=["headline"]).reset_index(drop=True)

    return df


def train_test_indices(
    df: pd.DataFrame, test_size: float = config.TEST_SIZE, seed: int = config.RANDOM_STATE
) -> Tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(df))
    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, stratify=df["label"], random_state=seed
    )
    return train_idx, test_idx
