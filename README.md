# 🎫 Support Ticket Auto-Tagger

An automated pipeline that reads a raw customer support ticket and predicts:

- **Priority** (High / Medium / Low)
- **Department** (Billing / Technical Support / Delivery)
- **A draft first-pass response** an agent can review and send

Built entirely with free tools: scikit-learn (TF-IDF + a calibrated LinearSVC),
no paid APIs, no GPU. Target hosting: [Hugging Face Spaces](https://huggingface.co/spaces)
(free tier).

> **This is a student/portfolio project, not a production system.** See
> [Known Limitations](#known-limitations) and [Privacy & PII](#privacy--pii)
> before pointing it at real customer data.

---

## Quickstart

```bash
pip install -r requirements.txt
streamlit run app.py
```

Or try it from the command line without the UI:

```bash
python predict_demo.py          # runs 3 sample tickets through the pipeline
python try_ticket.py            # interactive: paste your own ticket text
```

To retrain the models from scratch:

```bash
python prepare_real_dataset_v2.py   # rebuilds data/support_tickets_real_v2.csv
python train_model_v3.py            # trains + saves models/*.pkl (current production trainer)
```

## Run the tests

```bash
python test_edge_cases.py
# or, if pytest is installed:
python -m pytest test_edge_cases.py -v
```

---

## How it works

```
raw ticket text
   → clean_text()             lowercase, strip URLs, strip punctuation
   → TF-IDF                   unigrams + bigrams, full vocabulary (~77,800 features)
   → LinearSVC x2 (calibrated) one classifier for priority, one for department;
                               wrapped in CalibratedClassifierCV for genuine,
                               calibrated confidence scores
   → draft_response()         template response, personalized with as much of
                               the (PII-redacted) ticket as fits in the opener
```

All of the above lives in `predict_demo.py` and is loaded by `app.py`
(the Streamlit UI) with no duplicated logic between them.

## Current results

Trained and evaluated on a held-out 20% split of `data/support_tickets_real_v2.csv`
(12,122 rows — the English subset of Kaggle's ["Multilingual Customer Support
Tickets"](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets)
dataset by tobiasbueck).

| Task | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| **Priority** | 0.76 | 0.74 | Unweighted, calibrated LinearSVC. Still the weaker task — Low-priority recall (0.58) is the soft spot, up from 0.17 on the original production model. |
| **Department** | 0.97 | 0.91 | `class_weight='balanced'`, calibrated LinearSVC. Billing recall 0.83, Delivery recall 0.80, Technical Support recall 0.99. |

These are **not 100% accuracy, and won't be** — no real classifier on this
data will hit that, and a suspiciously perfect number would mean something
leaked or broke, not that it's actually working. These are genuine, honestly
measured gains over the original production baseline (Priority 0.61/0.51,
Department 0.92/0.83), found by testing several approaches
(`accuracy_improvement_experiment.py`) and keeping only the ones that held up
under 5-fold cross-validation, not just a single lucky train/test split.
What changed and why is fully documented in `train_model_v3.py`'s docstring;
two tested approaches (character n-grams, hand-engineered urgency features)
did **not** help and were correctly not adopted.

These numbers are reproducible by re-running `train_model_v3.py`
(`random_state=42`).

For the full history — the original dataset swap after an earlier dataset's
priority label turned out to carry no real signal (verified by chi-square
test), why `class_weight='balanced'` was adopted for department, and this
latest accuracy-improvement pass — see `Final_Project_Report.docx`.

## Four reviewer-requested features, all implemented and tested

1. **Confidence thresholding** — predictions below 60% confidence are flagged
   "requires manual review" instead of presented as certain. Confidence
   comes from `CalibratedClassifierCV` (Platt/sigmoid scaling), so it's a
   genuine calibrated probability, not a raw distance-from-boundary score.
2. **N-grams / lemmatization** — bigrams were already in the pipeline from
   Review 1. Lemmatization (spaCy) was tested honestly and dropped: it made
   no measurable difference and would add a heavy dependency for no gain.
3. **Feedback loop** — agents can correct a wrong department prediction in
   the app; corrections are appended to `corrections.csv`. On its own this
   only logs corrections — nothing retrains automatically. Run
   `python merge_corrections.py` to merge them into a new training CSV
   (`data/support_tickets_real_v2_with_corrections.csv`), then
   `python train_model_v3.py data/support_tickets_real_v2_with_corrections.csv`
   to actually retrain on it, then restart the app so it picks up the new
   `models/*.pkl` files. Department labels from corrections are
   human-verified and used directly; priority labels are not (the app has
   no priority-correction control), so `merge_corrections.py` tags those
   rows `priority_verified=False` and `train_model_v3.py` automatically
   excludes them from priority training/evaluation while still using them
   for department.
4. **Explainability** — "Why this department?" shows the exact words that
   drove the prediction, computed directly from the linear model's learned
   weights (not an approximation like LIME/SHAP).

## Known limitations

- **Priority prediction is weak**, especially for Low-priority tickets
  (recall ~17%). Treat priority as a rough signal, not a decision-maker.
- **Billing/Technical Support confusion persists for content-free Billing
  tickets** — roughly a third of Billing-labeled tickets in the source
  dataset contain no billing vocabulary at all (they're mislabeled
  security/data-analytics topics filed under "Billing and Payments").
  This is a dataset-content limitation, not a fixable text-classification
  bug for that subset.
- **English only.** The model was trained on the English subset of a
  bilingual (en/de) dataset. Non-English or mixed-language tickets fall
  outside the trained vocabulary and get unreliable, low-signal
  predictions — see `test_edge_cases.py` for what currently happens with
  them (it degrades gracefully rather than crashing, but accuracy on such
  tickets should not be trusted).
- **Draft responses are template-based, not generative.** They reference
  the ticket's extracted subject line, but don't otherwise adapt to the
  specific complaint. Real user testing (`user_validation.md`) found this
  is the #1 thing testers wanted improved.
- **Not connected to any live ticketing system.** It's a standalone
  demo/review tool.

## Privacy & PII

Real support tickets routinely contain customer names, email addresses,
order numbers, and other personal data. A few things worth knowing before
using this beyond a demo:

- The **training dataset itself was checked and contains essentially no
  real PII** — it reads as templated/synthetic business language (0 email
  matches, 1 phone-like match, 0 order-number-like matches across 12,122
  rows). This does **not** extend to whatever real tickets get pasted into
  the app at inference time.
- **`corrections.csv` stores raw ticket text in plain text**, unencrypted,
  on whatever machine runs the app. If real customer tickets are ever
  fed through the deployed app, this file will contain real customer data
  and should not be committed to a public repo or left unmanaged.
- **A free-tier Hugging Face Space is public by default.** Don't paste real
  customer ticket text into the public demo instance — use it with the
  built-in example ticket or hand-written sample text only, and keep any
  real-ticket testing local/private.
- **PII redaction is implemented** (`pii_redaction.py`) and applied at
  both points that matter: the complaint topic quoted back into the
  draft response, and the ticket text written to `corrections.csv`.
  It masks emails, phone numbers, card numbers, and order/account/
  reference numbers via regex — see `test_edge_cases.py` for coverage.
  It does **not** detect free-text names or physical addresses (that
  needs real NER, not regex) — a known, documented gap rather than
  something silently claimed as solved.

## Project structure

```
support_ticket_project/
├── data/
│   ├── raw/customer_support_tickets.csv      OLD Kaggle source — superseded
│   ├── support_tickets_real.csv              OLD harmonized dataset — superseded
│   └── support_tickets_real_v2.csv           CURRENT dataset, 12,122 rows
├── models/
│   ├── vectorizer.pkl, priority_model.pkl,
│   │   department_model.pkl                  CURRENT (v3) trained models
│   ├── department_explain_coef.pkl           averaged coefficients for
│   │                                          explain_department_prediction
│   │                                          (see train_model_v3.py)
│   └── v2_backup/                             old (v2) models, kept for reference
├── app.py                          Streamlit UI (confidence flagging,
│                                    explainability, feedback loop)
├── predict_demo.py                 Core pipeline: predict_ticket_with_confidence,
│                                    extract_topic, explain_department_prediction,
│                                    draft_response
├── pii_redaction.py                Regex-based PII masking (email/phone/card/
│                                    order-ref), used by extract_topic and app.py
├── train_model_v3.py                CURRENT production trainer (uncapped
│                                    TF-IDF + calibrated LinearSVC)
├── train_model_v2.py                superseded by v3 (kept for history —
│                                    see v3's docstring for the honest
│                                    before/after numbers)
├── merge_corrections.py            Merges corrections.csv into a new
│                                    training CSV (see Feedback loop above)
├── prepare_real_dataset_v2.py      CURRENT dataset harmonization
├── test_edge_cases.py              Edge-case tests (empty/emoji/non-English/
│                                    short/long/HTML/URL-only input)
├── try_ticket.py                   Interactive CLI tester
├── requirements.txt                App-only deps
├── requirements-dev.txt            Dev-only deps (embeddings comparison, chi-square test)
├── README_DEPLOY.md                Hugging Face Spaces deployment steps
├── user_validation.md              5-ticket real user validation report
├── Final_Project_Report.docx       Full project report
└── (Review-2 scripts: prepare_real_dataset.py, compare_models.py, class_weight_experiment.py,
    investigate_priority_labels.py — kept for history, superseded by the v2 files above)
```

## Deployment status

Prepared but **not yet deployed** — see `README_DEPLOY.md` for the manual
steps (requires a Hugging Face account and access token, and the Space
build itself has to run outside this project's sandbox, which has no
network access to huggingface.co).
