from __future__ import annotations

from collections import Counter
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, auc, confusion_matrix, roc_curve

from . import config

plt.rcParams.update({
    "figure.dpi": 110,
    "savefig.dpi": 140,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
})

COL_CB = "#d2503e"
COL_NCB = "#3a6ea5"


def plot_class_distribution(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 4))
    counts = df["label"].value_counts().sort_index()
    labels = ["non-clickbait", "clickbait"]
    ax.bar(labels, counts.values, color=[COL_NCB, COL_CB])
    for i, v in enumerate(counts.values):
        ax.text(i, v + max(counts) * 0.01, f"{v:,}", ha="center", fontsize=10)
    ax.set_ylabel("Number of headlines")
    ax.set_title("Class distribution")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "class_distribution.png")
    plt.close(fig)


def plot_length_distribution(df: pd.DataFrame) -> None:
    lens_cb = df.loc[df.label == 1, "headline"].str.len()
    lens_ncb = df.loc[df.label == 0, "headline"].str.len()
    fig, ax = plt.subplots(figsize=(7, 4))
    bins = np.linspace(0, max(lens_cb.max(), lens_ncb.max()), 50)
    ax.hist(lens_ncb, bins=bins, alpha=0.55, color=COL_NCB, label="non-clickbait")
    ax.hist(lens_cb, bins=bins, alpha=0.55, color=COL_CB, label="clickbait")
    ax.set_xlabel("Headline length (characters)")
    ax.set_ylabel("Frequency")
    ax.set_title("Headline length distribution by class")
    ax.legend()
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "length_distribution.png")
    plt.close(fig)


def plot_confusion(y_true, y_pred, model_name: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["non-clickbait", "clickbait"])
    ax.set_yticks([0, 1], labels=["non-clickbait", "clickbait"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {model_name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                    fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "confusion_matrix_best.png")
    plt.close(fig)


def plot_model_comparison(results: List) -> None:
    names = [r.name for r in results]
    f1s = [r.f1 for r in results]
    accs = [r.accuracy for r in results]
    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.bar(x - w / 2, accs, w, label="Accuracy", color=COL_NCB)
    ax.bar(x + w / 2, f1s, w, label="F1-score", color=COL_CB)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison")
    ax.legend()
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - w / 2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w / 2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "model_comparison.png")
    plt.close(fig)


def plot_roc(y_true, proba, model_name: str) -> None:
    if proba is None:
        return
    fpr, tpr, _ = roc_curve(y_true, proba)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr, tpr, color=COL_CB, lw=2, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], color="grey", lw=1, ls="--")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"ROC curve — {model_name}")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "roc_curve_best.png")
    plt.close(fig)


def error_examples(headlines, y_true, y_pred, proba, n_per_type: int = 5) -> pd.DataFrame:
    df = pd.DataFrame({
        "headline": headlines,
        "actual": y_true,
        "predicted": y_pred,
        "prob_clickbait": proba if proba is not None else np.nan,
    })
    fp = df[(df.actual == 0) & (df.predicted == 1)].copy()
    fn = df[(df.actual == 1) & (df.predicted == 0)].copy()
    if "prob_clickbait" in fp and fp["prob_clickbait"].notna().any():
        fp = fp.sort_values("prob_clickbait", ascending=False).head(n_per_type)
        fn = fn.sort_values("prob_clickbait", ascending=True).head(n_per_type)
    else:
        fp = fp.head(n_per_type)
        fn = fn.head(n_per_type)
    fp["error_type"] = "False positive (predicted clickbait, actually not)"
    fn["error_type"] = "False negative (predicted non-clickbait, actually clickbait)"
    return pd.concat([fp, fn], ignore_index=True)


def permutation_importance_manual(model: torch.nn.Module,
                                  X_test: np.ndarray,
                                  y_test: np.ndarray,
                                  manual_feature_names: List[str],
                                  manual_offset: int,
                                  device: torch.device,
                                  n_repeats: int = 5) -> pd.DataFrame:
    model.eval()
    rng = np.random.default_rng(config.RANDOM_STATE)

    def _predict(X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            logits = model(torch.from_numpy(X).to(device))
            probs = torch.sigmoid(logits).cpu().numpy()
        return (probs >= config.DEFAULT_THRESHOLD).astype(int)

    baseline_acc = accuracy_score(y_test, _predict(X_test))

    rows = []
    for i, fname in enumerate(manual_feature_names):
        col_idx = manual_offset + i
        drops = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            rng.shuffle(X_perm[:, col_idx])
            drops.append(baseline_acc - accuracy_score(y_test, _predict(X_perm)))
        rows.append({
            "feature": fname,
            "mean_drop": float(np.mean(drops)),
            "std_drop": float(np.std(drops)),
        })
    return pd.DataFrame(rows).sort_values("mean_drop", ascending=False).reset_index(drop=True)


def plot_permutation_importance(df_imp: pd.DataFrame, model_name: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    df_plot = df_imp.iloc[::-1]
    colors = [COL_CB if v >= 0 else COL_NCB for v in df_plot["mean_drop"]]
    ax.barh(df_plot["feature"], df_plot["mean_drop"], color=colors,
            xerr=df_plot["std_drop"])
    ax.axvline(0, color="grey", lw=0.5)
    ax.set_xlabel("Drop in accuracy when this feature is shuffled")
    ax.set_title(f"Permutation importance — manual features ({model_name})")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "permutation_importance.png")
    plt.close(fig)


def cnn_discriminative_tokens(cnn_model: torch.nn.Module,
                              tokenizer,
                              headlines: List[str],
                              device: torch.device,
                              top_n: int = 15,
                              cohort_size: int = 200) -> Dict[str, list]:
    from .tokenizer import tokenize

    cnn_model.eval()
    ids = tokenizer.encode(headlines)
    x_t = torch.from_numpy(ids).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(cnn_model(x_t)).cpu().numpy()

    order = np.argsort(probs)
    most_ncb_idx = order[:cohort_size]
    most_cb_idx = order[-cohort_size:]

    cb_counts: Counter[str] = Counter()
    ncb_counts: Counter[str] = Counter()
    for i in most_cb_idx:
        cb_counts.update(tokenize(headlines[i]))
    for i in most_ncb_idx:
        ncb_counts.update(tokenize(headlines[i]))

    def _disc(target, opposite):
        out = []
        for tok, c in target.most_common(400):
            if len(tok) <= 1:
                continue
            o = opposite.get(tok, 0)
            score = c - o
            if score > 0:
                out.append((tok, int(score)))
        return sorted(out, key=lambda x: -x[1])[:top_n]

    return {
        "clickbait_indicators": _disc(cb_counts, ncb_counts),
        "non_clickbait_indicators": _disc(ncb_counts, cb_counts),
    }


def plot_cnn_tokens(tokens: Dict[str, list]) -> None:
    pos = tokens["clickbait_indicators"][:12][::-1]
    neg = tokens["non_clickbait_indicators"][:12][::-1]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    axes[0].barh([t for t, _ in pos], [c for _, c in pos], color=COL_CB)
    axes[0].set_title("Tokens of headlines most confidently CLICKBAIT")
    axes[0].set_xlabel("Frequency excess vs. non-clickbait cohort")
    axes[1].barh([t for t, _ in neg], [c for _, c in neg], color=COL_NCB)
    axes[1].set_title("Tokens of headlines most confidently NON-CLICKBAIT")
    axes[1].set_xlabel("Frequency excess vs. clickbait cohort")
    fig.suptitle("Token-level interpretability — TextCNN")
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / "cnn_tokens.png")
    plt.close(fig)
