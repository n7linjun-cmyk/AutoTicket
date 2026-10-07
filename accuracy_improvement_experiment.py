"""
accuracy_improvement_experiment.py

Genuine attempt to beat the current production baseline (train_model_v2.py:
TF-IDF max_features=3000, ngram_range=(1,2), LogisticRegression). Same
methodology as every other experiment in this project: run it, report the
real number, keep it only if it's a real improvement.

Same train/test split as train_model_v2.py (test_size=0.2, random_state=42,
stratify=department) so every result here is directly comparable to the
numbers in the report.

Experiments:
  1. Baseline reproduction (sanity check against the report's numbers)
  2. More TF-IDF features (3000 -> 8000, 15000)
  3. Character n-grams (word-boundary char n-grams, catches typos/variants
     that word-level TF-IDF misses)
  4. LinearSVC instead of LogisticRegression (often stronger on sparse
     high-dimensional text features)
  5. Cross-validated C (regularization strength) search for LogisticRegression
  6. Priority-only: hand-engineered features (text length, exclamation
     count, urgency keywords) appended to TF-IDF -- priority is the weak
     task, so this targets it specifically rather than department, which
     is already fairly strong.

Run: python accuracy_improvement_experiment.py
"""

import re
import time
import numpy as np
import pandas as pd
from scipy.sparse import hstack, csr_matrix

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, f1_score

DATA_PATH = "data/support_tickets_real_v2.csv"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def score(name, y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro")
    print(f"{name:<55} accuracy={acc:.3f}  macro-F1={f1:.3f}")
    return acc, f1


def main():
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["ticket_text"].apply(clean_text)

    X_train_raw, X_test_raw, y_train_pri, y_test_pri, y_train_dep, y_test_dep, \
        train_raw_text, test_raw_text = train_test_split(
            df["clean_text"], df["priority"], df["department"], df["ticket_text"],
            test_size=0.2, random_state=42, stratify=df["department"]
        )

    results = []

    # =====================================================================
    # 1. Baseline reproduction
    # =====================================================================
    print("=" * 78)
    print("1. BASELINE (current production settings)")
    print("=" * 78)
    vec_base = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), stop_words="english")
    Xtr = vec_base.fit_transform(X_train_raw)
    Xte = vec_base.transform(X_test_raw)

    pri_base = LogisticRegression(max_iter=1000).fit(Xtr, y_train_pri)
    a, f = score("Priority (baseline, unweighted LR, 3000 feats)", y_test_pri, pri_base.predict(Xte))
    results.append(("Priority", "1. Baseline", a, f))

    dep_base = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, y_train_dep)
    a, f = score("Department (baseline, balanced LR, 3000 feats)", y_test_dep, dep_base.predict(Xte))
    results.append(("Department", "1. Baseline", a, f))

    # =====================================================================
    # 2. More TF-IDF features
    # =====================================================================
    print("\n" + "=" * 78)
    print("2. MORE TF-IDF FEATURES (8000, 15000)")
    print("=" * 78)
    for n_feat in (8000, 15000):
        vec = TfidfVectorizer(max_features=n_feat, ngram_range=(1, 2), stop_words="english")
        Xtr2 = vec.fit_transform(X_train_raw)
        Xte2 = vec.transform(X_test_raw)

        pri = LogisticRegression(max_iter=1000).fit(Xtr2, y_train_pri)
        a, f = score(f"Priority ({n_feat} feats)", y_test_pri, pri.predict(Xte2))
        results.append(("Priority", f"2. {n_feat} feats", a, f))

        dep = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr2, y_train_dep)
        a, f = score(f"Department ({n_feat} feats)", y_test_dep, dep.predict(Xte2))
        results.append(("Department", f"2. {n_feat} feats", a, f))

    # =====================================================================
    # 3. Character n-grams
    # =====================================================================
    print("\n" + "=" * 78)
    print("3. CHARACTER N-GRAMS (char_wb, 3-5, 5000 feats)")
    print("=" * 78)
    vec_char = TfidfVectorizer(max_features=5000, analyzer="char_wb", ngram_range=(3, 5))
    Xtr3 = vec_char.fit_transform(X_train_raw)
    Xte3 = vec_char.transform(X_test_raw)

    pri_char = LogisticRegression(max_iter=1000).fit(Xtr3, y_train_pri)
    a, f = score("Priority (char n-grams)", y_test_pri, pri_char.predict(Xte3))
    results.append(("Priority", "3. char n-grams", a, f))

    dep_char = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr3, y_train_dep)
    a, f = score("Department (char n-grams)", y_test_dep, dep_char.predict(Xte3))
    results.append(("Department", "3. char n-grams", a, f))

    # =====================================================================
    # 4. LinearSVC
    # =====================================================================
    print("\n" + "=" * 78)
    print("4. LINEARSVC (same 3000-feat TF-IDF as baseline)")
    print("=" * 78)
    pri_svc = LinearSVC(max_iter=5000).fit(Xtr, y_train_pri)
    a, f = score("Priority (LinearSVC, unweighted)", y_test_pri, pri_svc.predict(Xte))
    results.append(("Priority", "4. LinearSVC", a, f))

    dep_svc = LinearSVC(max_iter=5000, class_weight="balanced").fit(Xtr, y_train_dep)
    a, f = score("Department (LinearSVC, balanced)", y_test_dep, dep_svc.predict(Xte))
    results.append(("Department", "4. LinearSVC", a, f))

    # =====================================================================
    # 5. Cross-validated C search (LogisticRegression)
    # =====================================================================
    print("\n" + "=" * 78)
    print("5. CROSS-VALIDATED C SEARCH (LogisticRegression, 3-fold)")
    print("=" * 78)
    t0 = time.time()
    grid_pri = GridSearchCV(
        LogisticRegression(max_iter=1000), {"C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
        scoring="f1_macro", cv=3, n_jobs=-1,
    ).fit(Xtr, y_train_pri)
    print(f"Priority best C={grid_pri.best_params_['C']} (cv macro-F1={grid_pri.best_score_:.3f}), "
          f"{time.time()-t0:.1f}s")
    a, f = score("Priority (tuned C)", y_test_pri, grid_pri.best_estimator_.predict(Xte))
    results.append(("Priority", f"5. tuned C={grid_pri.best_params_['C']}", a, f))

    t0 = time.time()
    grid_dep = GridSearchCV(
        LogisticRegression(max_iter=1000, class_weight="balanced"),
        {"C": [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]},
        scoring="f1_macro", cv=3, n_jobs=-1,
    ).fit(Xtr, y_train_dep)
    print(f"Department best C={grid_dep.best_params_['C']} (cv macro-F1={grid_dep.best_score_:.3f}), "
          f"{time.time()-t0:.1f}s")
    a, f = score("Department (tuned C)", y_test_dep, grid_dep.best_estimator_.predict(Xte))
    results.append(("Department", f"5. tuned C={grid_dep.best_params_['C']}", a, f))

    # =====================================================================
    # 6. Priority-only: hand-engineered features + TF-IDF
    # =====================================================================
    print("\n" + "=" * 78)
    print("6. PRIORITY-ONLY: ENGINEERED FEATURES (length, urgency words) + TF-IDF")
    print("=" * 78)

    URGENCY_WORDS = re.compile(
        r"\b(urgent|asap|immediately|emergency|critical|now|please help|"
        r"cannot access|can't access|down|outage|broken|not working)\b"
    )

    def engineered_features(raw_texts):
        feats = []
        for t in raw_texts:
            t_lower = str(t).lower()
            length = len(t_lower)
            n_exclaim = t_lower.count("!")
            n_urgent_words = len(URGENCY_WORDS.findall(t_lower))
            all_caps_words = sum(1 for w in str(t).split() if w.isupper() and len(w) > 2)
            feats.append([length, n_exclaim, n_urgent_words, all_caps_words])
        return np.array(feats, dtype=float)

    train_eng = engineered_features(train_raw_text)
    test_eng = engineered_features(test_raw_text)
    # Scale engineered features to a comparable range to TF-IDF weights
    # (roughly 0-1) using train-set max, so a few huge raw counts don't
    # dominate the mostly-small TF-IDF values once concatenated.
    col_max = train_eng.max(axis=0)
    col_max[col_max == 0] = 1
    train_eng_scaled = train_eng / col_max
    test_eng_scaled = test_eng / col_max

    Xtr_combined = hstack([Xtr, csr_matrix(train_eng_scaled)])
    Xte_combined = hstack([Xte, csr_matrix(test_eng_scaled)])

    pri_eng = LogisticRegression(max_iter=1000).fit(Xtr_combined, y_train_pri)
    a, f = score("Priority (TF-IDF + engineered features)", y_test_pri, pri_eng.predict(Xte_combined))
    results.append(("Priority", "6. +engineered feats", a, f))

    # =====================================================================
    # Summary
    # =====================================================================
    print("\n" + "=" * 78)
    print("SUMMARY (sorted by macro-F1 within each task)")
    print("=" * 78)
    dfres = pd.DataFrame(results, columns=["task", "approach", "accuracy", "macro_f1"])
    for task in ["Priority", "Department"]:
        print(f"\n--- {task} ---")
        sub = dfres[dfres.task == task].sort_values("macro_f1", ascending=False)
        print(sub.to_string(index=False))


if __name__ == "__main__":
    main()
