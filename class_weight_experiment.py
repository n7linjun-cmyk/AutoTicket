"""
class_weight_experiment.py
Tests whether class_weight='balanced' improves the Priority classifier,
which has very poor Medium/Low recall on the real dataset (see Review 2).

Same TF-IDF setup and train/test split (random_state=42) as train_model_v2.py /
compare_models.py, so results are directly comparable to the existing table.

Run: python class_weight_experiment.py
Input: data/support_tickets_real.csv
"""

import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

DATA_PATH = "data/support_tickets_real.csv"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    df["clean_text"] = df["ticket_text"].apply(clean_text)

    X_train_text, X_test_text, y_train_pri, y_test_pri = train_test_split(
        df["clean_text"], df["priority"],
        test_size=0.2, random_state=42, stratify=df["priority"]
    )
    print(f"Train: {len(X_train_text)} | Test: {len(X_test_text)}")
    print("\nTrain priority distribution:\n", y_train_pri.value_counts())

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train_text)
    X_test_vec = vectorizer.transform(X_test_text)

    results = []

    # ---------- Baseline: no class weighting (matches Review 2 result) ----------
    print("\n" + "=" * 70)
    print("BASELINE: LogisticRegression (default, no class_weight)")
    print("=" * 70)
    model_base = LogisticRegression(max_iter=1000)
    model_base.fit(X_train_vec, y_train_pri)
    preds_base = model_base.predict(X_test_vec)
    acc_base = accuracy_score(y_test_pri, preds_base)
    f1_base = f1_score(y_test_pri, preds_base, average="macro")
    print(classification_report(y_test_pri, preds_base, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred), labels order:", sorted(y_test_pri.unique()))
    print(confusion_matrix(y_test_pri, preds_base, labels=sorted(y_test_pri.unique())))
    results.append(["Baseline (no weighting)", acc_base, f1_base])

    # ---------- Experiment: class_weight='balanced' ----------
    print("\n" + "=" * 70)
    print("EXPERIMENT: LogisticRegression(class_weight='balanced')")
    print("=" * 70)
    model_bal = LogisticRegression(max_iter=1000, class_weight="balanced")
    model_bal.fit(X_train_vec, y_train_pri)
    preds_bal = model_bal.predict(X_test_vec)
    acc_bal = accuracy_score(y_test_pri, preds_bal)
    f1_bal = f1_score(y_test_pri, preds_bal, average="macro")
    print(classification_report(y_test_pri, preds_bal, zero_division=0))
    print("Confusion matrix (rows=true, cols=pred), labels order:", sorted(y_test_pri.unique()))
    print(confusion_matrix(y_test_pri, preds_bal, labels=sorted(y_test_pri.unique())))
    results.append(["class_weight='balanced'", acc_bal, f1_bal])

    # ---------- Summary ----------
    print("\n" + "=" * 70)
    print("SUMMARY: Priority classifier, before vs after class weighting")
    print("=" * 70)
    print(f"{'Setting':<30}{'Accuracy':<12}{'Macro-F1':<10}")
    for setting, acc, f1 in results:
        print(f"{setting:<30}{acc:<12.3f}{f1:<10.3f}")

    delta_acc = acc_bal - acc_base
    delta_f1 = f1_bal - f1_base
    print(f"\nDelta: accuracy {delta_acc:+.3f}, macro-F1 {delta_f1:+.3f}")


if __name__ == "__main__":
    main()
