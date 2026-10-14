from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import sparse
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MaxAbsScaler
from torch.utils.data import DataLoader, TensorDataset

from . import config
from .features import ManualFeatureExtractor
from .models import DeepMLP, ShallowMLP, TextCNN
from .preprocessing import clean_for_tfidf
from .tokenizer import WordTokenizer
from .utils import load_dataset, set_seed, train_test_indices


@dataclass
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None
    train_time_s: float

    def as_dict(self) -> Dict:
        return asdict(self)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _clean_list(texts):
    return [clean_for_tfidf(t) for t in texts]


def build_mlp_features(X_train_text, X_val_text, X_test_text):
    tfidf = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        min_df=config.TFIDF_MIN_DF,
        max_df=config.TFIDF_MAX_DF,
        sublinear_tf=config.TFIDF_SUBLINEAR_TF,
        strip_accents="unicode",
    )
    manual = ManualFeatureExtractor()
    scaler = MaxAbsScaler()

    X_train_clean = _clean_list(X_train_text)
    X_val_clean = _clean_list(X_val_text)
    X_test_clean = _clean_list(X_test_text)

    Xtr_tfidf = tfidf.fit_transform(X_train_clean)
    Xva_tfidf = tfidf.transform(X_val_clean)
    Xte_tfidf = tfidf.transform(X_test_clean)

    Xtr_manual = manual.transform(X_train_text)
    Xva_manual = manual.transform(X_val_text)
    Xte_manual = manual.transform(X_test_text)

    Xtr = sparse.hstack([Xtr_tfidf, sparse.csr_matrix(Xtr_manual)]).tocsr()
    Xva = sparse.hstack([Xva_tfidf, sparse.csr_matrix(Xva_manual)]).tocsr()
    Xte = sparse.hstack([Xte_tfidf, sparse.csr_matrix(Xte_manual)]).tocsr()

    Xtr = scaler.fit_transform(Xtr)
    Xva = scaler.transform(Xva)
    Xte = scaler.transform(Xte)

    Xtr = Xtr.toarray().astype(np.float32)
    Xva = Xva.toarray().astype(np.float32)
    Xte = Xte.toarray().astype(np.float32)

    artefacts = {
        "tfidf": tfidf,
        "manual": manual,
        "scaler": scaler,
        "input_dim": Xtr.shape[1],
    }
    return Xtr, Xva, Xte, artefacts


def build_cnn_features(X_train_text, X_val_text, X_test_text):
    tok = WordTokenizer()
    tok.fit(X_train_text)
    Xtr = tok.encode(X_train_text)
    Xva = tok.encode(X_val_text)
    Xte = tok.encode(X_test_text)
    artefacts = {"tokenizer": tok, "vocab_size": len(tok)}
    return Xtr, Xva, Xte, artefacts


def _make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool):
    if X.dtype == np.int64:
        X_t = torch.from_numpy(X)
    else:
        X_t = torch.from_numpy(X.astype(np.float32))
    y_t = torch.from_numpy(y.astype(np.float32))
    ds = TensorDataset(X_t, y_t)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


def train_torch_model(model: nn.Module,
                      X_train, y_train,
                      X_val, y_val,
                      device: torch.device,
                      name: str) -> nn.Module:
    model = model.to(device)
    optim = torch.optim.Adam(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    loss_fn = nn.BCEWithLogitsLoss()
    train_loader = _make_loader(X_train, y_train, config.BATCH_SIZE, shuffle=True)
    val_loader = _make_loader(X_val, y_val, config.BATCH_SIZE, shuffle=False)

    best_val_loss = float("inf")
    best_state: Dict | None = None
    epochs_no_improve = 0

    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        running_loss, n_seen = 0.0, 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optim.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
            optim.step()
            running_loss += loss.item() * xb.size(0)
            n_seen += xb.size(0)
        train_loss = running_loss / max(n_seen, 1)

        model.eval()
        val_loss_sum, val_n, correct = 0.0, 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                logits = model(xb)
                val_loss_sum += loss_fn(logits, yb).item() * xb.size(0)
                preds = (torch.sigmoid(logits) >= config.DEFAULT_THRESHOLD).float()
                correct += (preds == yb).sum().item()
                val_n += xb.size(0)
        val_loss = val_loss_sum / max(val_n, 1)
        val_acc = correct / max(val_n, 1)

        print(f"   [{name}] epoch {epoch:02d}  train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_acc={val_acc:.4f}")

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= config.PATIENCE:
                print(f"   [{name}] early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def predict_proba_torch(model: nn.Module, X: np.ndarray, device: torch.device) -> np.ndarray:
    model.eval()
    loader = _make_loader(X, np.zeros(len(X), dtype=np.float32),
                          config.BATCH_SIZE, shuffle=False)
    out = []
    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(device)
            logits = model(xb)
            out.append(torch.sigmoid(logits).cpu().numpy())
    return np.concatenate(out, axis=0)


def metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray,
            proba: np.ndarray | None, t_sec: float) -> ModelResult:
    return ModelResult(
        name=name,
        accuracy=accuracy_score(y_true, y_pred),
        precision=precision_score(y_true, y_pred, zero_division=0),
        recall=recall_score(y_true, y_pred, zero_division=0),
        f1=f1_score(y_true, y_pred, zero_division=0),
        roc_auc=(roc_auc_score(y_true, proba)
                 if proba is not None and len(set(y_true)) > 1 else None),
        train_time_s=t_sec,
    )


def train_all() -> Dict:
    set_seed()
    torch.manual_seed(config.RANDOM_STATE)

    device = get_device()
    print(f"Device: {device}")

    df = load_dataset()
    print(f"Loaded {len(df):,} headlines  |  "
          f"clickbait={int((df.label==1).sum()):,}  "
          f"non-clickbait={int((df.label==0).sum()):,}")

    train_idx, test_idx = train_test_indices(df)
    X_train_text = df["headline"].iloc[train_idx].tolist()
    X_test_text = df["headline"].iloc[test_idx].tolist()
    y_train = df["label"].iloc[train_idx].values
    y_test = df["label"].iloc[test_idx].values

    X_train_text, X_val_text, y_train, y_val = train_test_split(
        X_train_text, y_train,
        test_size=config.VAL_SIZE,
        stratify=y_train,
        random_state=config.RANDOM_STATE,
    )
    print(f"   train={len(X_train_text):,}  val={len(X_val_text):,}  test={len(X_test_text):,}")

    print("\nBuilding MLP features (TF-IDF + manual)...")
    Xtr_mlp, Xva_mlp, Xte_mlp, mlp_artefacts = build_mlp_features(
        X_train_text, X_val_text, X_test_text)
    print(f"   MLP input dim: {mlp_artefacts['input_dim']:,}")

    print("Building CNN features (tokenized + padded sequences)...")
    Xtr_cnn, Xva_cnn, Xte_cnn, cnn_artefacts = build_cnn_features(
        X_train_text, X_val_text, X_test_text)
    print(f"   CNN vocab size: {cnn_artefacts['vocab_size']:,}  max_len={config.MAX_LEN}")

    results: List[ModelResult] = []
    predictions: Dict[str, np.ndarray] = {}
    probabilities: Dict[str, np.ndarray] = {}
    fitted_models: Dict[str, object] = {}

    print("\n--- Training: Dummy (most-frequent baseline) ---")
    t0 = time.time()
    dummy = DummyClassifier(strategy="most_frequent", random_state=config.RANDOM_STATE)
    dummy.fit(np.zeros((len(X_train_text), 1)), y_train)
    elapsed = time.time() - t0
    y_pred = dummy.predict(np.zeros((len(X_test_text), 1)))
    proba = dummy.predict_proba(np.zeros((len(X_test_text), 1)))[:, 1] \
        if hasattr(dummy, "predict_proba") else None
    r = metrics("Dummy", y_test, y_pred, proba, elapsed)
    print(f"   Dummy  acc={r.accuracy:.4f}  f1={r.f1:.4f}  ({elapsed:.1f}s)")
    results.append(r)
    predictions["Dummy"] = y_pred
    probabilities["Dummy"] = proba
    fitted_models["Dummy"] = dummy

    print("\n--- Training: ShallowMLP (128-unit single hidden layer) ---")
    t0 = time.time()
    shallow = ShallowMLP(input_dim=mlp_artefacts["input_dim"])
    shallow = train_torch_model(shallow, Xtr_mlp, y_train, Xva_mlp, y_val,
                                 device, "ShallowMLP")
    elapsed = time.time() - t0
    proba = predict_proba_torch(shallow, Xte_mlp, device)
    y_pred = (proba >= config.DEFAULT_THRESHOLD).astype(int)
    r = metrics("ShallowMLP", y_test, y_pred, proba, elapsed)
    print(f"   ShallowMLP  acc={r.accuracy:.4f}  f1={r.f1:.4f}  ({elapsed:.1f}s)")
    results.append(r)
    predictions["ShallowMLP"] = y_pred
    probabilities["ShallowMLP"] = proba
    fitted_models["ShallowMLP"] = shallow

    print("\n--- Training: DeepMLP (256-128-64 with batch-norm) ---")
    t0 = time.time()
    deep = DeepMLP(input_dim=mlp_artefacts["input_dim"])
    deep = train_torch_model(deep, Xtr_mlp, y_train, Xva_mlp, y_val,
                              device, "DeepMLP")
    elapsed = time.time() - t0
    proba = predict_proba_torch(deep, Xte_mlp, device)
    y_pred = (proba >= config.DEFAULT_THRESHOLD).astype(int)
    r = metrics("DeepMLP", y_test, y_pred, proba, elapsed)
    print(f"   DeepMLP  acc={r.accuracy:.4f}  f1={r.f1:.4f}  ({elapsed:.1f}s)")
    results.append(r)
    predictions["DeepMLP"] = y_pred
    probabilities["DeepMLP"] = proba
    fitted_models["DeepMLP"] = deep

    print("\n--- Training: TextCNN (Kim 2014 style, kernels=3,4,5) ---")
    t0 = time.time()
    cnn = TextCNN(vocab_size=cnn_artefacts["vocab_size"])
    cnn = train_torch_model(cnn, Xtr_cnn, y_train, Xva_cnn, y_val,
                             device, "TextCNN")
    elapsed = time.time() - t0
    proba = predict_proba_torch(cnn, Xte_cnn, device)
    y_pred = (proba >= config.DEFAULT_THRESHOLD).astype(int)
    r = metrics("TextCNN", y_test, y_pred, proba, elapsed)
    print(f"   TextCNN  acc={r.accuracy:.4f}  f1={r.f1:.4f}  ({elapsed:.1f}s)")
    results.append(r)
    predictions["TextCNN"] = y_pred
    probabilities["TextCNN"] = proba
    fitted_models["TextCNN"] = cnn

    candidates = [r for r in results if r.name != "Dummy"]
    best = max(candidates, key=lambda r: r.f1)
    print(f"\nBest model: {best.name}  (F1 = {best.f1:.4f})")

    results_df = pd.DataFrame([r.as_dict() for r in results])
    results_df.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)

    best_model = fitted_models[best.name]
    save_payload = {
        "name": best.name,
        "threshold": config.DEFAULT_THRESHOLD,
    }
    if best.name in {"ShallowMLP", "DeepMLP"}:
        save_payload["kind"] = "mlp"
        save_payload["input_dim"] = mlp_artefacts["input_dim"]
        save_payload["state_dict"] = best_model.state_dict()
        save_payload["tfidf"] = mlp_artefacts["tfidf"]
        save_payload["manual"] = mlp_artefacts["manual"]
        save_payload["scaler"] = mlp_artefacts["scaler"]
        save_payload["arch"] = best.name
    elif best.name == "TextCNN":
        save_payload["kind"] = "cnn"
        save_payload["vocab_size"] = cnn_artefacts["vocab_size"]
        save_payload["state_dict"] = best_model.state_dict()
        save_payload["tokenizer"] = cnn_artefacts["tokenizer"]
        save_payload["arch"] = "TextCNN"
    else:
        save_payload["kind"] = "dummy"
        save_payload["model"] = best_model

    best_path = config.MODELS_DIR / "best_model.joblib"
    joblib.dump(save_payload, best_path)
    print(f"Saved best model bundle to {best_path}")

    meta = {
        "best_model": best.name,
        "best_f1": best.f1,
        "n_train": len(X_train_text),
        "n_val": len(X_val_text),
        "n_test": len(X_test_text),
        "device": str(device),
        "results": [r.as_dict() for r in results],
    }
    (config.MODELS_DIR / "metadata.json").write_text(json.dumps(meta, indent=2))

    return {
        "df": df,
        "X_train": X_train_text, "X_val": X_val_text, "X_test": X_test_text,
        "y_train": y_train, "y_val": y_val, "y_test": y_test,
        "results": results,
        "fitted_models": fitted_models,
        "predictions": predictions,
        "probabilities": probabilities,
        "best_name": best.name,
        "mlp_artefacts": mlp_artefacts,
        "cnn_artefacts": cnn_artefacts,
        "Xte_mlp": Xte_mlp,
        "Xte_cnn": Xte_cnn,
    }


if __name__ == "__main__":
    train_all()
