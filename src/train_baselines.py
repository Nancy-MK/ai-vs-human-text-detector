"""
train_baselines.py

Trains and evaluates two interpretable baseline models (Naive Bayes and
Logistic Regression) on TF-IDF features for AI-vs-human text classification.
Both models are cheap to train, fast to run, and fully auditable, which
matters as much as raw accuracy when recommending a model for deployment.

Usage:
    python src/train_baselines.py --data data/texts.csv
"""
import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


def load_data(path: str):
    df = pd.read_csv(path)
    return df["text"], df["label"]


def build_and_evaluate(name, model, X_train_vec, X_test_vec, y_train, y_test):
    model.fit(X_train_vec, y_train)
    preds = model.predict(X_test_vec)

    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=["human", "ai_generated"]))
    print("Confusion matrix (rows=true, cols=pred):")
    print(confusion_matrix(y_test, preds))

    return model


def main():
    parser = argparse.ArgumentParser(description="Train TF-IDF baseline classifiers.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--out-dir", default="artifacts")
    args = parser.parse_args()

    X, y = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=42
    )

    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    nb = build_and_evaluate("Naive Bayes", MultinomialNB(), X_train_vec, X_test_vec, y_train, y_test)
    logreg = build_and_evaluate(
        "Logistic Regression",
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        X_train_vec,
        X_test_vec,
        y_train,
        y_test,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")
    joblib.dump(nb, out_dir / "naive_bayes.joblib")
    joblib.dump(logreg, out_dir / "logreg.joblib")

    print(f"\nModels and vectorizer saved to {out_dir}/")


if __name__ == "__main__":
    main()
