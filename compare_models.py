"""
compare_models.py
Compares two feature extraction approaches for both classifiers
(priority and department) on the REAL dataset:

  Baseline : TF-IDF (unigrams + bigrams) + Logistic Regression
  Upgrade  : Sentence embeddings (all-MiniLM-L6-v2) + Logistic Regression

Prints a side-by-side accuracy / macro-F1 comparison table.

Requirements: pip install sentence-transformers
(First run will download the ~90MB MiniLM model - needs internet.)

Run: python compare_models.py
Input: data/support_tickets_real.csv (created by prepare_real_dataset.py)
"""

import re
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report

from sentence_transformers import SentenceTransformer

DATA_PATH = "data/support_tickets_real.csv"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def evaluate(X_train, X_test, y_train, y_test, label_name: str, method_name: str):
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    print(f"\n--- {method_name} | {label_name} ---")
    print(classification_report(y_test, preds, zero_division=0))
    return acc, f1


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows from {DATA_PATH}")
    df["clean_text"] = df["ticket_text"].apply(clean_text)

    X_train_text, X_test_text, y_train_pri, y_test_pri, y_train_dep, y_test_dep = train_test_split(
        df["clean_text"], df["priority"], df["department"],
        test_size=0.2, random_state=42, stratify=df["priority"]
    )

    results = []

    # ---------- Method 1: TF-IDF baseline ----------
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    acc, f1 = evaluate(X_train_tfidf, X_test_tfidf, y_train_pri, y_test_pri, "Priority", "TF-IDF")
    results.append(["TF-IDF", "Priority", acc, f1])

    acc, f1 = evaluate(X_train_tfidf, X_test_tfidf, y_train_dep, y_test_dep, "Department", "TF-IDF")
    results.append(["TF-IDF", "Department", acc, f1])

    # ---------- Method 2: Sentence embeddings ----------
    print("\nLoading sentence embedding model (all-MiniLM-L6-v2)... this needs internet on first run.")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    X_train_emb = embed_model.encode(X_train_text.tolist(), show_progress_bar=True)
    X_test_emb = embed_model.encode(X_test_text.tolist(), show_progress_bar=True)

    acc, f1 = evaluate(X_train_emb, X_test_emb, y_train_pri, y_test_pri, "Priority", "MiniLM Embeddings")
    results.append(["MiniLM Embeddings", "Priority", acc, f1])

    acc, f1 = evaluate(X_train_emb, X_test_emb, y_train_dep, y_test_dep, "Department", "MiniLM Embeddings")
    results.append(["MiniLM Embeddings", "Department", acc, f1])

    # ---------- Final comparison table ----------
    print("\n" + "=" * 60)
    print("FINAL COMPARISON: TF-IDF baseline vs Embedding upgrade")
    print("=" * 60)
    print(f"{'Method':<20}{'Task':<15}{'Accuracy':<12}{'Macro-F1':<10}")
    for method, task, acc, f1 in results:
        print(f"{method:<20}{task:<15}{acc:<12.3f}{f1:<10.3f}")


if __name__ == "__main__":
    main()
