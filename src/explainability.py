"""
explainability.py

Produces a SHAP-based explainability review for a trained TF-IDF + linear
model: which tokens actually drove each prediction, and whether the model
is keying on genuine stylistic signal or spurious dataset artefacts.

Usage:
    python src/explainability.py --model artifacts/logreg.joblib \
        --vectorizer artifacts/tfidf_vectorizer.joblib --data data/texts.csv
"""
import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap


def main():
    parser = argparse.ArgumentParser(description="Generate a SHAP explainability report.")
    parser.add_argument("--model", required=True, help="Path to a saved sklearn model (.joblib).")
    parser.add_argument("--vectorizer", required=True, help="Path to the saved TF-IDF vectorizer.")
    parser.add_argument("--data", required=True, help="CSV with a 'text' column to explain.")
    parser.add_argument("--n-samples", type=int, default=100, help="Number of rows to explain.")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    model = joblib.load(args.model)
    vectorizer = joblib.load(args.vectorizer)

    df = pd.read_csv(args.data).sample(n=min(args.n_samples, len(pd.read_csv(args.data))), random_state=42)
    X = vectorizer.transform(df["text"])
    feature_names = vectorizer.get_feature_names_out()

    # LinearExplainer is appropriate here because both Naive Bayes and
    # Logistic Regression are linear-in-the-features models over TF-IDF.
    explainer = shap.LinearExplainer(model, X, feature_names=feature_names)
    shap_values = explainer(X)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Global feature importance: which tokens most influence predictions overall.
    plt.figure()
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=150)
    plt.close()

    # Per-example explanation for the single most confidently-classified sample,
    # useful for spot-checking whether the model's reasoning looks sound.
    top_idx = model.predict_proba(X).max(axis=1).argmax()
    plt.figure()
    shap.plots.waterfall(shap_values[top_idx], show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_example_waterfall.png", dpi=150)
    plt.close()

    print(f"SHAP summary plot saved to {out_dir}/shap_summary.png")
    print(f"SHAP example waterfall saved to {out_dir}/shap_example_waterfall.png")
    print(
        "\nReview note: inspect the top contributing tokens in shap_summary.png. "
        "Tokens that reflect genuine stylistic markers (e.g. repetitive phrasing, "
        "generic transition words) support the model's validity; tokens tied to "
        "formatting artefacts (e.g. leftover markup) indicate the model may be "
        "exploiting a dataset leak rather than learning the intended signal."
    )


if __name__ == "__main__":
    main()
