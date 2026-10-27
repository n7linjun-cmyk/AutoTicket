# 🎫 Support Ticket Auto-Tagger

An automated pipeline that reads a raw customer support ticket and predicts:

- **Priority** (High / Medium / Low)
- **Department** (Billing / Technical Support / Delivery)
- A **draft first-pass response** an agent can review and send

Built entirely with free tools: scikit-learn (TF-IDF + a calibrated LinearSVC), no paid APIs, no GPU. Target hosting: Hugging Face Spaces (free tier).

> This is a student/portfolio project, not a production system. See [Known limitations](#known-limitations) and [Privacy & PII](#privacy--pii) before pointing it at real customer data.

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

### Run the tests

```bash
python test_edge_cases.py
python test_merge_corrections.py
# or, if pytest is installed (runs both files):
python -m pytest -v
```

Run from the project root (the tests load `models/*.pkl`). There are currently **50 tests** (40 in `test_edge_cases.py`, 10 in `test_merge_corrections.py`), all passing. See [Testing](#testing) for what they cover.

---

## How it works

```
raw ticket text
   → clean_text()              lowercase, strip URLs, strip punctuation
   → TF-IDF                    unigrams + bigrams, full vocabulary (~77,800 features)
   → LinearSVC x2 (calibrated) one classifier for priority, one for department;
                               wrapped in CalibratedClassifierCV for genuine,
                               calibrated confidence scores
   → draft_response()          template response, personalized with as much of
                               the (PII-redacted) ticket as fits in the opener
```

All of the above lives in `predict_demo.py` and is loaded by `app.py` (the Streamlit UI) with no duplicated logic between them.

---

## Current results

Trained and evaluated on a held-out 20% split of `data/support_tickets_real_v2.csv` (12,122 rows, the English subset of Kaggle's "Multilingual Customer Support Tickets" dataset by tobiasbueck).

| Task | Accuracy | Macro-F1 | Notes |
|---|---|---|---|
| Priority | 0.76 | 0.74 | Unweighted, calibrated LinearSVC. Still the weaker task. Low-priority recall is 0.58, up from 0.17 on the original production model. |
| Department | 0.97 | 0.91 | `class_weight='balanced'`, calibrated LinearSVC. Billing recall 0.83, Delivery recall 0.80, Technical Support recall 0.99. |

These are **not** 100% accuracy, and won't be. No real classifier on this data will hit that, and a suspiciously perfect number would mean something leaked or broke. These are genuine, honestly measured gains over the original production baseline (Priority 0.61 accuracy / 0.51 macro-F1, Department 0.92 / 0.83), found by testing several approaches (`accuracy_improvement_experiment.py`) and keeping only the ones that held up under 5-fold cross-validation, not just a single lucky train/test split. What changed and why is documented in the `train_model_v3.py` docstring. Two tested approaches (character n-grams, hand-engineered urgency features) did not help and were correctly not adopted.

These numbers are reproducible by re-running `train_model_v3.py` (`random_state=42`).

For the full history (the original dataset swap after an earlier dataset's priority label turned out to carry no real signal, verified by chi-square test; why `class_weight='balanced'` was adopted for department; and the latest accuracy-improvement pass) see `Final_Project_Report.docx`.

### Four reviewer-requested features, all implemented and tested

1. **Confidence thresholding.** Predictions below 60% confidence are flagged "requires manual review" instead of presented as certain. Confidence comes from `CalibratedClassifierCV` (Platt/sigmoid scaling), so it is a genuine calibrated probability, not a raw distance-from-boundary score.
2. **N-grams / lemmatization.** Bigrams were already in the pipeline from Review 1. Lemmatization (spaCy) was tested honestly and dropped: it made no measurable difference and would add a heavy dependency for no gain.
3. **Feedback loop.** Agents can correct a wrong department prediction in the app; corrections are appended to `corrections.csv`. On its own this only logs corrections, nothing retrains automatically. To actually retrain:
   1. `python merge_corrections.py` merges them into a new training CSV (`data/support_tickets_real_v2_with_corrections.csv`).
   2. `python train_model_v3.py data/support_tickets_real_v2_with_corrections.csv` retrains on it.
   3. Restart the app so it picks up the new `models/*.pkl` files.

   Department labels from corrections are human-verified and used directly. Priority labels are not (the app has no priority-correction control), so `merge_corrections.py` tags those rows `priority_verified=False` and `train_model_v3.py` automatically excludes them from priority training/evaluation while still using them for department.
4. **Explainability.** "Why this department?" shows the exact words that drove the prediction, computed directly from the linear model's learned weights (not an approximation like LIME/SHAP).

---

## Interfaces (no web API, no database)

This project is a Streamlit app (`app.py`) calling a Python module (`predict_demo.py`). There are **no REST endpoints and no database server**. The "API" is the set of functions below, and the "database" is a set of CSV and pickle files (see [Data schema](#data-schema)).

| Function (module) | Input | Output | Behaviour on bad input |
|---|---|---|---|
| `clean_text(text)` (predict_demo) | `str` | lower-cased `str`: URLs removed, only `a-z0-9` and spaces kept | Any `str` is safe; emoji, punctuation and non-Latin text become `""` |
| `has_signal(cleaned)` (predict_demo) | cleaned `str` | `bool`: True if at least one recognised alphabetic token | Returns False for `""`, digits-only, emoji-only |
| `predict_ticket(raw_text)` (predict_demo) | `str` | `(priority, department)` | No signal check; use the next function for anything user-facing |
| `predict_ticket_with_confidence(raw_text)` (predict_demo) | `str` | `dict`: `priority`, `priority_confidence`, `department`, `department_confidence`, `insufficient_text` | Never raises; no-signal text gives `insufficient_text=True` and confidences `0.0` |
| `explain_department_prediction(raw_text, top_n=6)` (predict_demo) | `str`, `int` | `list[(word, contribution)]` | Returns `[]` when no word contributed |
| `extract_topic(raw_text)` (predict_demo) | `str` | `str`, at most 150 chars (+ `...`) | Returns `""` for bare URLs, junk or empty text; strips HTML and `{placeholders}`; masks PII |
| `draft_response(priority, department, raw_text="")` (predict_demo) | labels + optional text | multi-line draft `str` | `KeyError` only if given a label the models can never emit |
| `redact_pii(text)` (pii_redaction) | `str` or `None` | `(masked_text, counts)`; `counts` has keys `email`, `phone`, `card`, `order_ref` | Never raises; names and addresses are not detected |
| `has_pii(text)` (pii_redaction) | `str` | `bool` | Same as above |
| `save_correction(...)` (app.py) | ticket text, predicted and corrected department, predicted priority | appends one row to `corrections.csv` | OS errors (permissions, read-only disk) propagate to the page |

Example:

```python
from predict_demo import predict_ticket_with_confidence, draft_response
r = predict_ticket_with_confidence("I was charged twice for my invoice")
# {'priority': 'High', 'priority_confidence': 0.61,
#  'department': 'Billing', 'department_confidence': 0.95,
#  'insufficient_text': False}
print(draft_response(r["priority"], r["department"], "I was charged twice for my invoice"))
```

Scripts and their command lines:

| Script | Purpose | Command |
|---|---|---|
| `app.py` | Web UI | `streamlit run app.py` |
| `train_model_v3.py` | Train and save models | `python train_model_v3.py [optional_csv]` |
| `merge_corrections.py` | Merge agent corrections into a new training CSV | `python merge_corrections.py [--corrections F] [--base F] [--output F]` |
| `try_ticket.py` | Interactive CLI tester | `python try_ticket.py` |

---

## Data schema

### Training data: `data/support_tickets_real_v2.csv` (12,122 rows)

| Column | Type | Values |
|---|---|---|
| `ticket_text` | text | Subject + `". "` + body (English only) |
| `priority` | category | High (5,561), Medium (4,782), Low (1,779) |
| `department` | category | Technical support (10,204), Billing (1,562), Delivery (356) |

### Feedback log: `corrections.csv`

Created by the app on first correction; git-ignored; plain text, PII-masked.

| Column | Type | Notes |
|---|---|---|
| `timestamp` | ISO-8601 string | Local time of submission |
| `ticket_text` | text | Emails, phones, cards and order/reference numbers masked |
| `predicted_priority` | category | Model output (never human-corrected) |
| `predicted_department` | category | Model output |
| `corrected_department` | category | Agent's choice; equals the prediction when confirmed |

### Merged training file: `data/support_tickets_real_v2_with_corrections.csv`

Written by `merge_corrections.py`. It has the three training columns plus `source` (`original` or `correction`) and `priority_verified` (True for original rows, False for correction rows, because priority is never checked by a human). `train_model_v3.py` trains the priority model only on `priority_verified=True` rows and the department model on all rows.

### Model artifacts: `models/`

| File | Contents |
|---|---|
| `vectorizer.pkl` | `TfidfVectorizer`, 1-2 grams, English stop words, about 77,800 features |
| `priority_model.pkl` | `CalibratedClassifierCV(LinearSVC)` |
| `department_model.pkl` | Same, with `class_weight="balanced"` |
| `department_explain_coef.pkl` | `{"coef": averaged coefficients, "classes": labels}` for the explanation panel |

---

## Error boundaries

Where each unusual input is caught, what the user sees, and which test protects it.

| Input condition | Caught in | Behaviour | Test(s) |
|---|---|---|---|
| Blank input in the UI | `app.py` | Warning shown; pipeline not run | (UI, manual) |
| Empty, whitespace, punctuation, digits-only, year-only or one-letter text | `has_signal` | `insufficient_text=True`, confidence 0.0, red banner in the UI | `test_empty_string_*`, `test_whitespace_only_*`, `test_punctuation_only_*`, `test_numbers_only_*`, `test_year_only_*`, `test_single_character_*` |
| Emoji-only text | `clean_text` then `has_signal` | Same as above | `test_emoji_only_*` |
| Emoji mixed with real words | `has_signal` | Not flagged; predicted normally | `test_emoji_mixed_with_real_text_*` |
| Non-English or mixed-language text | pipeline | No crash; partial signal from English words; documented as unreliable | `test_non_english_*`, `test_mixed_language_*` |
| Confidence below 0.60 | `app.py` (`CONFIDENCE_THRESHOLD`) | "Needs a second look" warning; number never shown | (UI, manual) |
| Very long text (about 5,000 words) | `extract_topic` | Topic capped at 150 chars with `...`; draft stays small | `test_very_long_text_*` |
| HTML or `<script>` in the ticket | `extract_topic` | Tags stripped before the text is quoted in the draft | `test_html_script_is_not_reflected_into_draft` |
| URL-only ticket | `extract_topic` | Topic rejected; generic opening used | `test_url_only_*` |
| PII in the ticket (email, phone, card, order/ref number) | `redact_pii`, called from `extract_topic` and `save_correction` | Masked as `[EMAIL]`, `[PHONE]`, `[CARD]`, `[REF#]` before it is quoted or stored | `test_redact_pii_*`, `test_draft_response_never_contains_raw_*` |
| Email containing a period at a split point | `extract_topic` (redacts before cutting) | Email fully masked | `test_extract_topic_does_not_truncate_email_at_its_own_period` |
| Corrections file with missing columns | `merge_corrections.load_corrections` | `ValueError` naming the columns | `test_corrections_file_missing_columns_raises_value_error` |
| Corrections rows with blank text, invalid labels or duplicates | `merge_corrections.main` | Rows dropped and counted in the printed summary | `test_blank_ticket_text_*`, `test_invalid_*`, `test_duplicate_ticket_*`, `test_correction_already_in_base_*` |
| Training CSV missing or malformed | `train_model_v3.main` | Raises; nothing is trained or saved | (manual) |

**Not handled (known gaps):** personal names and street addresses in free text; concurrent writes to `corrections.csv`; non-English tickets (low accuracy).

---

## Testing

`test_edge_cases.py` has 40 unit tests for pipeline robustness, and `test_merge_corrections.py` has 10 unit tests for the feedback-merge script (50 in total). They check that odd input never crashes the pipeline, that results always have the documented shape (valid labels, confidences between 0 and 1, a boolean `insufficient_text`), and that bugs found earlier stay fixed. They do **not** measure model accuracy; that is evaluated on a held-out split (see [Current results](#current-results)).

| Area | What is checked |
|---|---|
| Empty and no-signal text | Flagged `insufficient_text`, confidence forced to 0.0 (regression: used to report about 79% from class priors) |
| Emoji | No crash; flagged only when no real words remain |
| Non-English and mixed text | No crash; mixed text keeps partial signal |
| Short and long text | Single words and 5,000-word tickets work; topic and draft sizes are bounded |
| HTML and URLs | Markup never echoed into a draft; bare URLs rejected |
| PII redaction | Each category masked, unrelated numbers left alone, no PII in drafts, trailing punctuation preserved |

The merge tests use small temporary CSVs, so they never touch `data/` or the real `corrections.csv`, and they do not need the trained models. They check that a missing file or missing columns is handled, that blank text, invalid labels, duplicates and rows already in the base data are dropped, that merged rows get `priority_verified=False`, and that re-running does not double-count.

**How to add a test:** write a function named `test_<what>_<expected>` in `test_edge_cases.py` (it will be picked up by both runners), build the input, call the function under test, and assert on the documented output shape.

---

## Known limitations

- **Priority prediction is the weaker task.** Low-priority recall is 0.58 (up from 0.17 on the original model). Treat priority as a rough signal, not a decision-maker.
- **Billing/Technical Support confusion persists for content-free Billing tickets.** Roughly a third of Billing-labeled tickets in the source dataset contain no billing vocabulary at all (they are mislabeled security/data-analytics topics filed under "Billing and Payments"). This is a dataset-content limitation, not a fixable text-classification bug for that subset.
- **English only.** The model was trained on the English subset of a bilingual (en/de) dataset. Non-English or mixed-language tickets fall outside the trained vocabulary and get unreliable, low-signal predictions. See `test_edge_cases.py` for what currently happens with them: it degrades gracefully rather than crashing, but accuracy on such tickets should not be trusted.
- **Draft responses are template-based, not generative.** They reference the ticket's extracted subject line, but don't otherwise adapt to the specific complaint. Real user testing (`user_validation.md`) found this is the #1 thing testers wanted improved.
- **Not connected to any live ticketing system.** It's a standalone demo/review tool.

---

## Privacy & PII

Real support tickets routinely contain customer names, email addresses, order numbers, and other personal data. A few things worth knowing before using this beyond a demo:

- The training dataset itself was checked and contains essentially no real PII. It reads as templated/synthetic business language (0 email matches, 1 phone-like match, 0 order-number-like matches across 12,122 rows). This does not extend to whatever real tickets get pasted into the app at inference time.
- `corrections.csv` stores ticket text in plain text, unencrypted, on whatever machine runs the app. Emails, phone numbers, card numbers and order/reference numbers are masked, but names and addresses are not. If real customer tickets are ever fed through the deployed app, this file should not be committed to a public repo or left unmanaged.
- A free-tier Hugging Face Space is public by default. Don't paste real customer ticket text into the public demo instance. Use the built-in example ticket or hand-written sample text only, and keep any real-ticket testing local/private.
- PII redaction is implemented (`pii_redaction.py`) and applied at both points that matter: the complaint topic quoted back into the draft response, and the ticket text written to `corrections.csv`. It masks emails, phone numbers, card numbers, and order/account/reference numbers via regex. It does **not** detect free-text names or physical addresses (that needs real NER, not regex). This is a known, documented gap rather than something silently claimed as solved.

---

## Project structure

```
support_ticket_project/
├── data/
│   ├── raw/customer_support_tickets.csv      OLD Kaggle source (superseded)
│   ├── support_tickets_real.csv              OLD harmonized dataset (superseded)
│   └── support_tickets_real_v2.csv           CURRENT dataset, 12,122 rows
├── models/
│   ├── vectorizer.pkl, priority_model.pkl,
│   │   department_model.pkl                  CURRENT (v3) trained models
│   ├── department_explain_coef.pkl           averaged coefficients for
│   │                                         explain_department_prediction
│   └── v2_backup/                            old (v2) models, kept for reference
├── app.py                          Streamlit UI (confidence flagging,
│                                   explainability, feedback loop)
├── predict_demo.py                 Core pipeline: predict_ticket_with_confidence,
│                                   extract_topic, explain_department_prediction,
│                                   draft_response
├── pii_redaction.py                Regex-based PII masking (email/phone/card/
│                                   order-ref), used by extract_topic and app.py
├── train_model_v3.py               CURRENT production trainer (uncapped
│                                   TF-IDF + calibrated LinearSVC)
├── train_model_v2.py               superseded by v3 (kept for history)
├── merge_corrections.py            Merges corrections.csv into a new
│                                   training CSV (see Feedback loop above)
├── prepare_real_dataset_v2.py      CURRENT dataset harmonization
├── test_edge_cases.py              40 edge-case tests (empty/emoji/non-English/
│                                   short/long/HTML/URL-only/PII input)
├── test_merge_corrections.py       10 unit tests for merge_corrections.py
├── try_ticket.py                   Interactive CLI tester
├── requirements.txt                App-only deps
├── requirements-dev.txt            Dev-only deps (embeddings comparison, chi-square test)
├── README_DEPLOY.md                Hugging Face Spaces deployment steps
├── user_validation.md              5-ticket real user validation report
├── Final_Project_Report.docx       Full project report
└── (Review-2 scripts: prepare_real_dataset.py, compare_models.py,
    class_weight_experiment.py, investigate_priority_labels.py;
    kept for history, superseded by the v2 files above)
```

---

## Deployment status

Prepared but not yet deployed. See `README_DEPLOY.md` for the manual steps (requires a Hugging Face account and access token, and the Space build itself has to run outside this project's sandbox, which has no network access to huggingface.co).
