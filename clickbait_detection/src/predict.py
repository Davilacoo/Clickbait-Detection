from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import joblib
import numpy as np
import torch
from scipy import sparse

from . import config
from .features import ManualFeatureExtractor, extract_features
from .models import DeepMLP, ShallowMLP, TextCNN
from .preprocessing import clean_for_tfidf


_cache: Dict | None = None
_loaded_model: torch.nn.Module | None = None
_device: torch.device | None = None


def _device_lazy() -> torch.device:
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def _build_model(bundle: Dict) -> torch.nn.Module:
    if bundle["kind"] == "mlp":
        if bundle["arch"] == "ShallowMLP":
            model = ShallowMLP(input_dim=bundle["input_dim"])
        else:
            model = DeepMLP(input_dim=bundle["input_dim"])
    elif bundle["kind"] == "cnn":
        model = TextCNN(vocab_size=bundle["vocab_size"])
    else:
        raise ValueError(f"Unknown model kind: {bundle['kind']}")
    model.load_state_dict(bundle["state_dict"])
    model.eval()
    return model


def load_best_model(path: Path | None = None) -> Dict:
    global _cache, _loaded_model
    if _cache is not None:
        return _cache
    p = path or (config.MODELS_DIR / "best_model.joblib")
    if not p.exists():
        raise FileNotFoundError(
            f"No trained model at {p}. Run `python main.py` first."
        )
    bundle = joblib.load(p)
    if bundle["kind"] in {"mlp", "cnn"}:
        _loaded_model = _build_model(bundle).to(_device_lazy())
    else:
        _loaded_model = bundle.get("model")
    _cache = bundle
    return _cache


def _transform_mlp(headlines: List[str], bundle: Dict) -> np.ndarray:
    cleaned = [clean_for_tfidf(t) for t in headlines]
    tfidf_part = bundle["tfidf"].transform(cleaned)
    manual_part = bundle["manual"].transform(headlines)
    X = sparse.hstack([tfidf_part, sparse.csr_matrix(manual_part)]).tocsr()
    X = bundle["scaler"].transform(X)
    return X.toarray().astype(np.float32)


def _transform_cnn(headlines: List[str], bundle: Dict) -> np.ndarray:
    return bundle["tokenizer"].encode(headlines)


def _forward_proba(headlines: List[str], bundle: Dict) -> np.ndarray:
    if bundle["kind"] == "mlp":
        X = _transform_mlp(headlines, bundle)
        x_t = torch.from_numpy(X).to(_device_lazy())
    elif bundle["kind"] == "cnn":
        X = _transform_cnn(headlines, bundle)
        x_t = torch.from_numpy(X).to(_device_lazy())
    else:
        return bundle["model"].predict_proba([[0] * len(headlines)])[:, 1]

    with torch.no_grad():
        logits = _loaded_model(x_t)
        return torch.sigmoid(logits).cpu().numpy()


def _explain(headline: str, prob: float) -> List[str]:
    feats = extract_features(headline)
    reasons: List[str] = []
    if feats["starts_with_number"]:
        reasons.append("starts with a number (listicle pattern)")
    elif feats["has_digit"]:
        reasons.append("contains a digit")
    if feats["n_clickbait_kw"] > 0:
        reasons.append(f"contains {int(feats['n_clickbait_kw'])} clickbait keyword(s)")
    if feats["n_exclaim"] > 0:
        reasons.append(f"uses {int(feats['n_exclaim'])} exclamation mark(s)")
    if feats["n_question"] > 0:
        reasons.append(f"uses {int(feats['n_question'])} question mark(s)")
    if feats["n_pronouns"] >= 2:
        reasons.append("uses second-person/demonstrative pronouns (you, your, this)")
    if feats["frac_caps"] >= 0.3:
        reasons.append("uses many ALL-CAPS words")
    if not reasons:
        reasons.append("the vocabulary is similar to other clickbait headlines"
                       if prob >= 0.5
                       else "the wording is neutral and informational")
    return reasons


def predict_one(headline: str) -> Dict:
    return predict_many([headline])[0]


def predict_many(headlines: List[str]) -> List[Dict]:
    bundle = load_best_model()
    threshold = bundle.get("threshold", config.DEFAULT_THRESHOLD)
    probs = _forward_proba(headlines, bundle)
    return [
        {
            "headline": h,
            "prediction": "clickbait" if p >= threshold else "non-clickbait",
            "clickbait_probability": float(p),
            "explanation": _explain(h, float(p)),
            "model_name": bundle["name"],
            "threshold": threshold,
        }
        for h, p in zip(headlines, probs)
    ]


def _format(result: Dict) -> str:
    return "\n".join([
        f"Headline:               {result['headline']}",
        f"Prediction:             {result['prediction']}",
        f"Clickbait probability:  {result['clickbait_probability']:.3f}",
        f"Model:                  {result['model_name']}",
        f"Reasons:                {'; '.join(result['explanation'])}",
    ])


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python -m src.predict "your headline here"')
        sys.exit(1)
    for h in sys.argv[1:]:
        print(_format(predict_one(h)))
        print("-" * 60)
