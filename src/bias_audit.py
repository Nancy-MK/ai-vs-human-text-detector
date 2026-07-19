"""
bias_audit.py

Quantifies class imbalance in the AI-vs-human text dataset and produces a
stratified train/test split so that downstream models are not evaluated on
a naive-accuracy metric that hides the minority-class (AI-generated) risk.

Usage:
    python src/bias_audit.py --data data/texts.csv --test-size 0.2
"""
import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


def audit_class_balance(df: pd.DataFrame, label_col: str = "label") -> dict:
    """Return class counts, proportions, and the accuracy a trivial
    majority-class classifier would achieve on this dataset."""
    counts = df[label_col].value_counts().to_dict()
    total = len(df)
    proportions = {k: round(v / total, 4) for k, v in counts.items()}
    majority_class_accuracy = round(max(counts.values()) / total, 4)

    report = {
        "total_samples": total,
        "class_counts": counts,
        "class_proportions": proportions,
        "trivial_majority_classifier_accuracy": majority_class_accuracy,
        "risk_note": (
            "A classifier that always predicts the majority class would score "
            f"{majority_class_accuracy:.1%} accuracy without learning anything. "
            "Accuracy alone is not a safe metric for this dataset; report "
            "precision/recall/F1 per class instead."
        ),
    }
    return report


def stratified_split(df: pd.DataFrame, label_col: str, test_size: float, seed: int = 42):
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=seed,
    )
    return train_df, test_df


def main():
    parser = argparse.ArgumentParser(description="Audit class balance and create a stratified split.")
    parser.add_argument("--data", required=True, help="Path to CSV with 'text' and 'label' columns.")
    parser.add_argument("--label-col", default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    report = audit_class_balance(df, args.label_col)

    print(json.dumps(report, indent=2))

    train_df, test_df = stratified_split(df, args.label_col, args.test_size)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(out_dir / "train.csv", index=False)
    test_df.to_csv(out_dir / "test.csv", index=False)
    with open(out_dir / "bias_audit_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nStratified split written to {out_dir}/train.csv and {out_dir}/test.csv")


if __name__ == "__main__":
    main()
