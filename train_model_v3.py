"""
train_model_v3.py

Genuine accuracy improvement over train_model_v2.py's baseline, found and
validated in accuracy_improvement_experiment.py. This is NOT a claim of
100% accuracy -- no real classifier on this data reaches that, and this
docstring says so on purpose rather than quietly avoiding the number.

What changed from v2, and why (each tested individually first, then
combined, then confirmed with 5-fold cross-validation -- not just the one
train/test split -- before being adopted):

  1. TF-IDF: max_features cap removed (was 3000, full vocabulary is
     ~77,800 unigrams+bigrams on this dataset). Priority kept improving
     as the cap was raised (3000 -> 8000 -> 15000 -> uncapped); department
     plateaus around 15,000 features, so the uncapped vocabulary doesn't
     hurt it either. 5-fold CV macro-F1: priority 0.567 -> 0.753,
     department 0.899 -> 0.915 (uncapped vs. the old 3000-feature cap).

  2. Classifier: LinearSVC instead of LogisticRegression. Tested head to
     head on identical features -- LinearSVC consistently beat LR by a few
     points on both tasks. LinearSVC has no predict_proba, so it's wrapped
     in CalibratedClassifierCV (Platt/sigmoid scaling, 5-fold internal CV)
     to keep genuine, calibrated probabilities for the confidence-
     thresholding feature. Calibration was checked to confirm it doesn't
     erase the accuracy gain (it doesn't -- within noise of the raw
     LinearSVC numbers on the held-out test set).

  3. Explainability: LinearSVC is still a linear model, so the same
     "exact word contribution = TF-IDF weight x learned coefficient"
     approach still applies -- but CalibratedClassifierCV wraps 5 internal
     LinearSVC instances (one per CV fold) rather than exposing one
     coef_ directly. The 5 folds' coefficients are averaged and saved
     separately (models/department_explain_coef.pkl) for
     explain_department_prediction to use. This is still the model's own
     learned weights, not an external approximation (LIME/SHAP) -- just
     averaged over the 5 equivalent sub-models that make up the final
     calibrated classifier, the same way any bagged/CV-ensembled linear
     model's coefficients would be summarized.

Two approaches that were tried and did NOT help (see
accuracy_improvement_experiment.py for the actual numbers), so they were
NOT adopted:
  - Character n-grams (char_wb, 3-5): worse than word n-grams on both
    tasks (support tickets are English prose, not the kind of
    typo/spelling-variant-heavy text char n-grams tend to help with).
  - Hand-engineered features for priority (text length, exclamation
    count, urgency keywords) concatenated with TF-IDF: no improvement
    over the TF-IDF-only baseline -- the signal they were meant to add
    was apparently already captured by the word features.

Honest ceiling: cross-validated priority macro-F1 is now ~0.75 (up from
~0.57), department ~0.91-0.92 (up from ~0.90 on this same uncapped-vocab
comparison, ~0.83 vs. the original 3000-feature production baseline).
Priority remains the weaker task -- see the report for why (Low-priority
recall in particular, and the inherent label noise already documented in
Section 4/5).

Run: python train_model_v3.py
   python train_model_v3.py data/support_tickets_real_v2_with_corrections.csv

Optional dataset path (added when merge_corrections.py was introduced):
if the given (or default) CSV has a `priority_verified` column -- which
merge_corrections.py's output does, and the plain base dataset doesn't --
the priority classifier is trained and evaluated ONLY on rows where it's
True. Department training always uses every row, verified or not, since
a corrected department label is verified ground truth by construction
(see merge_corrections.py's docstring for why priority and department
aren't symmetric here). Without that column (e.g. running on the plain
base dataset as before), every row is treated as priority-verified and
behavior is identical to earlier versions of this script.
"""

import re
import sys
import numpy as np
import pandas as pd
import joblib
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
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
    """
    Train and save the production models.

    Input : optional CLI arg = path to a CSV with columns ticket_text,
            priority, department (and optionally priority_verified); the
            default is data/support_tickets_real_v2.csv.
    Output: prints classification reports and confusion matrices for the
            held-out 20% split, then writes to models/:
              vectorizer.pkl              TfidfVectorizer (1-2 grams, English stop words)
              priority_model.pkl          CalibratedClassifierCV(LinearSVC)
              department_model.pkl        same, class_weight="balanced"
              department_explain_coef.pkl {"coef", "classes"} for explanations
    Reproducibility: random_state=42, stratified on department.
    Error boundary: raises if the CSV is missing or lacks those columns
    (pandas KeyError / FileNotFoundError) -- fail loudly, never train on
    partial data.
    """
    data_path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows from {data_path}")

    if "priority_verified" not in df.columns:
        # Plain base dataset (no merge_corrections.py involved) -- every
        # row's priority label came from the original harmonized dataset,
        # so it's as "verified" as this project's priority labels get.
        df["priority_verified"] = True
    else:
        n_unverified = int((~df["priority_verified"]).sum())
        if n_unverified:
            print(f"{n_unverified} row(s) have priority_verified=False "
                  f"(correction-derived, department verified but priority is "
                  f"just the model's own prior prediction) -- excluded from "
                  f"priority training and evaluation below, included for "
                  f"department.")

    df["clean_text"] = df["ticket_text"].apply(clean_text)

    # Same split as v2 (test_size=0.2, random_state=42, stratify=department)
    # so these numbers stay directly comparable to the v2 report numbers.
    # priority_verified is split alongside everything else so the resulting
    # train/test masks line up positionally with X_train_vec/X_test_vec below.
    X_train, X_test, y_train_pri, y_test_pri, y_train_dep, y_test_dep, pv_train, pv_test = \
        train_test_split(
            df["clean_text"], df["priority"], df["department"], df["priority_verified"],
            test_size=0.2, random_state=42, stratify=df["department"]
        )

    # No max_features cap -- see docstring for why this was adopted.
    # Fit on the FULL training text (all rows, verified or not) so the
    # vocabulary itself isn't shrunk by excluding correction-derived rows --
    # only the priority classifier's fit/eval target rows are filtered below.
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words="english")
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")

    # ---------- Priority: LinearSVC, unweighted, calibrated ----------
    # Trained and evaluated ONLY on priority-verified rows -- see module
    # docstring and merge_corrections.py for why unverified (model's own
    # prior prediction) rows would corrupt this if included.
    pv_train_mask = pv_train.to_numpy()
    pv_test_mask = pv_test.to_numpy()
    X_train_pri_vec = X_train_vec[pv_train_mask]
    y_train_pri_verified = y_train_pri[pv_train_mask]
    X_test_pri_vec = X_test_vec[pv_test_mask]
    y_test_pri_verified = y_test_pri[pv_test_mask]
    if pv_train_mask.sum() < len(pv_train_mask):
        print(f"Priority training rows: {pv_train_mask.sum()} verified "
              f"(of {len(pv_train_mask)} total in the training split)")

    priority_model = CalibratedClassifierCV(
        LinearSVC(max_iter=5000), method="sigmoid", cv=5
    ).fit(X_train_pri_vec, y_train_pri_verified)
    pri_preds = priority_model.predict(X_test_pri_vec)

    print("=" * 60)
    print("PRIORITY CLASSIFIER RESULTS (v3: uncapped TF-IDF + calibrated LinearSVC)")
    print("=" * 60)
    print(classification_report(y_test_pri_verified, pri_preds))
    print("Confusion matrix:\n", confusion_matrix(y_test_pri_verified, pri_preds))

    # ---------- Department: LinearSVC, class_weight='balanced', calibrated ----------
    department_model = CalibratedClassifierCV(
        LinearSVC(max_iter=5000, class_weight="balanced"), method="sigmoid", cv=5
    ).fit(X_train_vec, y_train_dep)
    dep_preds = department_model.predict(X_test_vec)

    print("\n" + "=" * 60)
    print("DEPARTMENT CLASSIFIER RESULTS (v3: uncapped TF-IDF + calibrated LinearSVC)")
    print("=" * 60)
    print(classification_report(y_test_dep, dep_preds))
    print("Confusion matrix:\n", confusion_matrix(y_test_dep, dep_preds))

    # ---------- Averaged coefficients for explainability ----------
    # See docstring point 3: CalibratedClassifierCV wraps 5 internal
    # LinearSVC instances (one per CV fold); average their coef_ matrices
    # for explain_department_prediction to use, since there's no single
    # top-level coef_ on the calibrated wrapper itself.
    dep_coefs = np.mean(
        [cc.estimator.coef_ for cc in department_model.calibrated_classifiers_], axis=0
    )
    dep_classes = department_model.classes_
    joblib.dump(
        {"coef": dep_coefs, "classes": dep_classes},
        "models/department_explain_coef.pkl",
    )

    os.makedirs("models", exist_ok=True)
    joblib.dump(vectorizer, "models/vectorizer.pkl")
    joblib.dump(priority_model, "models/priority_model.pkl")
    joblib.dump(department_model, "models/department_model.pkl")
    print("\nModels saved to models/ folder (v2 models are in models/v2_backup/).")


if __name__ == "__main__":
    main()
