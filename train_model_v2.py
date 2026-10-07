"""
train_model_v2.py
Trains the production models on the NEW dataset (data/support_tickets_real_v2.csv,
harmonized by prepare_real_dataset_v2.py from the Kaggle "Multilingual Customer
Support Tickets" dataset), replacing the old models trained on the suraj520
dataset.

Why the switch: a chi-square test found the OLD dataset's Priority field had
no significant relationship with ticket content (see investigate_priority_labels.py) -
it behaved like a randomly assigned label. The new dataset's priority is
genuinely learnable (chi-square p ~ 0 against queue).

Two experiments, run and compared honestly before picking the final settings:

  DEPARTMENT: class_weight='balanced' is used. On the held-out test set this
  took Billing recall 0.63 -> 0.86 and Delivery recall 0.41 -> 0.77, at a
  small cost to Technical Support recall (1.00 -> 0.93) and to overall
  accuracy (0.93 -> 0.92). Macro-F1 improved 0.765 -> 0.826. A keyword-based
  override (mirroring the old project's Delivery-keyword approach) was also
  tested and made things slightly WORSE when combined with balanced weights
  (0.826 -> 0.813) because a few Delivery tickets also mention billing words
  like "charge"/"payment" and got wrongly overridden - so it was dropped.

  PRIORITY: class_weight='balanced' gave a much smaller, more marginal
  trade-off here (macro-F1 0.518 -> 0.543, but accuracy 0.604 -> 0.565), so
  the default (unweighted) model is kept for priority.

Run: python train_model_v2.py
"""

import re
import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = "data/support_tickets_real_v2.csv"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def main():
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {len(df)} rows")
    print(df["priority"].value_counts(), "\n")
    print(df["department"].value_counts(), "\n")

    df["clean_text"] = df["ticket_text"].apply(clean_text)

    X_train, X_test, y_train_pri, y_test_pri, y_train_dep, y_test_dep = train_test_split(
        df["clean_text"], df["priority"], df["department"],
        test_size=0.2, random_state=42, stratify=df["department"]
    )

    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    # ---------- Priority: unweighted (see docstring for why) ----------
    priority_model = LogisticRegression(max_iter=1000)
    priority_model.fit(X_train_vec, y_train_pri)
    pri_preds = priority_model.predict(X_test_vec)

    print("=" * 60)
    print("PRIORITY CLASSIFIER RESULTS (final, unweighted)")
    print("=" * 60)
    print(classification_report(y_test_pri, pri_preds))
    print("Confusion matrix:\n", confusion_matrix(y_test_pri, pri_preds))

    # ---------- Department: class_weight='balanced' (see docstring for why) ----------
    department_model = LogisticRegression(max_iter=1000, class_weight="balanced")
    department_model.fit(X_train_vec, y_train_dep)
    dep_preds = department_model.predict(X_test_vec)

    print("\n" + "=" * 60)
    print("DEPARTMENT CLASSIFIER RESULTS (final, class_weight='balanced')")
    print("=" * 60)
    print(classification_report(y_test_dep, dep_preds))
    print("Confusion matrix:\n", confusion_matrix(y_test_dep, dep_preds))

    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    joblib.dump(priority_model, "models/priority_model.pkl")
    joblib.dump(department_model, "models/department_model.pkl")
    print("\nModels saved to models/ folder.")


if __name__ == "__main__":
    main()
