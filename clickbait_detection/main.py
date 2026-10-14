from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import config, evaluate
from src.features import feature_names as manual_names
from src.train_models import train_all, get_device


def main() -> None:
    print("=" * 70)
    print("  CLICKBAIT DETECTION — neural-network pipeline")
    print("=" * 70)
    art = train_all()

    df = art["df"]
    results = art["results"]
    best_name = art["best_name"]
    y_test = art["y_test"]
    X_test = art["X_test"]
    fitted = art["fitted_models"]
    preds = art["predictions"]
    probas = art["probabilities"]

    print("\n--- Generating figures ---")
    evaluate.plot_class_distribution(df)
    evaluate.plot_length_distribution(df)
    evaluate.plot_model_comparison(results)
    evaluate.plot_confusion(y_test, preds[best_name], best_name)
    evaluate.plot_roc(y_test, probas[best_name], best_name)
    print(f"   Figures saved to {config.FIGURES_DIR.relative_to(config.PROJECT_ROOT)}/")

    print("\n--- Error analysis ---")
    errors = evaluate.error_examples(X_test, y_test, preds[best_name], probas[best_name])
    errors_path = config.REPORTS_DIR / "error_analysis.csv"
    errors.to_csv(errors_path, index=False)
    print(f"   Wrote {len(errors)} example errors to "
          f"{errors_path.relative_to(config.PROJECT_ROOT)}")

    print("\n--- Interpretability ---")
    device = get_device()

    if best_name in {"ShallowMLP", "DeepMLP"}:
        Xte_mlp = art["Xte_mlp"]
        mlp_art = art["mlp_artefacts"]
        manual_offset = mlp_art["input_dim"] - len(manual_names())
        imp_df = evaluate.permutation_importance_manual(
            model=fitted[best_name],
            X_test=Xte_mlp,
            y_test=y_test,
            manual_feature_names=manual_names(),
            manual_offset=manual_offset,
            device=device,
            n_repeats=3,
        )
        imp_df.to_csv(config.REPORTS_DIR / "permutation_importance.csv", index=False)
        evaluate.plot_permutation_importance(imp_df, best_name)
        print("   Saved permutation_importance.csv and permutation_importance.png")

    elif best_name == "TextCNN":
        tokens = evaluate.cnn_discriminative_tokens(
            cnn_model=fitted["TextCNN"],
            tokenizer=art["cnn_artefacts"]["tokenizer"],
            headlines=art["X_test"],
            device=device,
        )
        (config.REPORTS_DIR / "interpretability.json").write_text(
            json.dumps(tokens, indent=2)
        )
        evaluate.plot_cnn_tokens(tokens)
        print("   Saved interpretability.json and cnn_tokens.png")

    print("\n--- Final comparison table ---")
    df_results = pd.DataFrame([r.as_dict() for r in results])
    df_results["roc_auc"] = df_results["roc_auc"].apply(
        lambda x: f"{x:.4f}" if x is not None else "—"
    )
    df_results["train_time_s"] = df_results["train_time_s"].apply(lambda v: f"{v:.1f}s")
    print(df_results.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nBest model: {best_name}")
    print("Done.")


if __name__ == "__main__":
    main()
