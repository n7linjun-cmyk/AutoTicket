"""
predict_demo.py
End-to-end demo: takes a raw ticket text, predicts priority + department,
and generates a draft response.

Run: python predict_demo.py
"""

import re
import joblib
from pii_redaction import redact_pii

vectorizer = joblib.load("models/vectorizer.pkl")
priority_model = joblib.load("models/priority_model.pkl")
department_model = joblib.load("models/department_model.pkl")
# Averaged linear coefficients for explainability (see train_model_v3.py):
# CalibratedClassifierCV wraps 5 internal LinearSVC instances rather than
# exposing one coef_ directly, so the 5 folds' coefficients were averaged
# and saved separately at training time.
_dep_explain = joblib.load("models/department_explain_coef.pkl")
_dep_explain_coef = _dep_explain["coef"]
_dep_explain_classes = list(_dep_explain["classes"])

SLA = {"High": "2 hours", "Medium": "24 hours", "Low": "3-5 business days"}

NEXT_STEPS = {
    "Billing": "our billing team will review the charge/invoice and follow up with details.",
    "Technical support": "our technical team will investigate the issue and provide troubleshooting steps or a fix.",
    "Delivery": "our logistics team will check the shipment status and update you on next steps.",
}


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_signal(cleaned_text: str) -> bool:
    """
    True if the vectorizer recognizes at least one ALPHABETIC token in the
    cleaned text (not just any nonzero match). Empty text, whitespace-only
    text, emoji-only text, and punctuation/number-only text all clean down
    to a string with zero vocabulary overlap, which produces an all-zero
    TF-IDF vector. On an all-zero vector, predict_proba doesn't fail or
    return a flat/uncertain distribution -- it returns the model's learned
    class-prior probabilities, which can look confidently wrong (observed:
    ~79% "confidence" on empty input). This check catches that case before
    it's mistaken for a real prediction.

    Checking for ANY nonzero match (nnz > 0) was the original approach, but
    testing after the vocabulary was uncapped (train_model_v3.py -- full
    ~77,800-token vocabulary, up from a 3000-feature cap) surfaced a real
    gap: a handful of purely-numeric tokens (placeholder-looking numbers
    and years, e.g. "1234567890", "2022") ended up in-vocabulary simply
    because they appear somewhere in the training tickets. A ticket that's
    just "1234567890" then produced a single nonzero match and sailed past
    this check with ~96% department confidence -- confidently wrong for
    the same underlying reason as the empty-text case, just via a
    different path (one coincidental digit-token match instead of a
    zero vector). Requiring the matched token(s) to include at least one
    letter closes that gap without losing legitimate short/unusual tickets
    that do contain real words.
    """
    if not cleaned_text:
        return False
    vec = vectorizer.transform([cleaned_text])
    if vec.nnz == 0:
        return False
    feature_names = vectorizer.get_feature_names_out()
    matched_tokens = feature_names[vec.nonzero()[1]]
    return any(re.search(r"[a-z]", t) for t in matched_tokens)


def predict_ticket(raw_text: str):
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    priority = priority_model.predict(vec)[0]
    department = department_model.predict(vec)[0]
    return priority, department


def predict_ticket_with_confidence(raw_text: str):
    """
    Same as predict_ticket, but also returns the model's own confidence
    (the winning class's predicted probability) for both priority and
    department. Suggestion #1: lets the UI flag low-confidence predictions
    for manual review instead of always presenting a guess as certain.

    Also returns "insufficient_text": True when the cleaned text has no
    real vocabulary signal (see has_signal -- at least one recognized
    ALPHABETIC token, not just any nonzero TF-IDF match). In that case
    priority/department/confidence are still returned (so callers that
    ignore the flag don't break), but confidence is forced to 0.0 rather
    than a misleadingly high probability the model would otherwise report,
    since there's nothing meaningful in the text driving the prediction.
    """
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    insufficient_text = not has_signal(cleaned)

    pri_probs = priority_model.predict_proba(vec)[0]
    pri_idx = pri_probs.argmax()
    priority = priority_model.classes_[pri_idx]
    pri_confidence = 0.0 if insufficient_text else float(pri_probs[pri_idx])

    dep_probs = department_model.predict_proba(vec)[0]
    dep_idx = dep_probs.argmax()
    department = department_model.classes_[dep_idx]
    dep_confidence = 0.0 if insufficient_text else float(dep_probs[dep_idx])

    return {
        "priority": priority, "priority_confidence": pri_confidence,
        "department": department, "department_confidence": dep_confidence,
        "insufficient_text": insufficient_text,
    }


def explain_department_prediction(raw_text: str, top_n: int = 6):
    """
    Suggestion #4 (explainability), implemented without adding a new
    dependency (eli5/LIME/SHAP): the department classifier is a linear
    model (LinearSVC, wrapped in CalibratedClassifierCV for probability
    calibration -- see train_model_v3.py), so for the predicted class we
    can directly read off, for each word actually present in this
    ticket, its (TF-IDF weight in this ticket) x (that word's learned
    coefficient for the predicted class). This is exact/honest rather
    than an approximation the way LIME/SHAP would be for a black-box
    model.

    The coefficient used here is the average of the 5 internal LinearSVC
    instances CalibratedClassifierCV fits (one per calibration fold),
    precomputed at training time and saved to
    models/department_explain_coef.pkl -- since the calibrated wrapper
    itself doesn't expose a single coef_. This is still the model's own
    learned weights, just summarized across its 5 equivalent
    sub-models, not an externally-approximated explanation.

    Returns a list of (word, contribution) tuples, sorted by contribution
    descending, for words that pushed the prediction TOWARD the winning
    class. Only returns words that literally appear in this ticket.
    """
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    pred_class = department_model.predict(vec)[0]
    class_idx = _dep_explain_classes.index(pred_class)

    feature_names = vectorizer.get_feature_names_out()
    tfidf_values = vec.toarray()[0]
    coefficients = _dep_explain_coef[class_idx]

    contributions = tfidf_values * coefficients
    nonzero_idx = contributions.nonzero()[0]

    word_contributions = [
        (feature_names[i], float(contributions[i]))
        for i in nonzero_idx if contributions[i] > 0
    ]
    word_contributions.sort(key=lambda x: x[1], reverse=True)
    return word_contributions[:top_n]


MAX_TOPIC_LEN = 150  # raised from 80: the topic used to stop at the first
# period (fine for the training dataset's clean "subject. body" tickets),
# which threw away everything after a customer's first short sentence.
# Now it's a length cap on how much of the ticket gets quoted, not a
# sentence-count cap -- see extract_topic for how the cut point is chosen.


def extract_topic(raw_text: str) -> str:
    """
    Pulls the complaint topic out of the raw ticket text, to quote back in
    the draft response.

    Originally this returned only the text before the FIRST period,
    modeled on how the training data was built (prepare_real_dataset.py):
        ticket_text = Ticket_Subject + ". " + Ticket_Description
    That worked well for that dataset's clean single-sentence subjects,
    but was too narrow for real, free-form tickets written as several
    short sentences -- e.g. "Order #55219. Email jane@example.com. Call
    555-123-4567." produced a topic of just "Order #55219", silently
    dropping everything after the first period even though none of it
    was PII, junk, or otherwise unusable. Real user feedback confirmed
    this made drafts feel too generic (Section 3/7 of the report).

    This now returns as much of the ticket as fits within MAX_TOPIC_LEN,
    preferring to cut at the last sentence-ending period within that
    window when one exists, and falling back to the last word boundary
    (plus an ellipsis) when it doesn't -- rather than always stopping at
    whichever period happens to come first, however early that is.

    Returns "" if no usable topic can be extracted (e.g. empty text, a
    bare URL, HTML/script content, or the text is just leftover
    placeholder junk), so the caller can fall back to the generic
    template. This return value is inserted verbatim into the draft
    response, so it is also sanitized here rather than left to the caller:
      - PII (emails, phone numbers, card numbers, order/account/
        reference numbers) is masked via pii_redaction.redact_pii FIRST,
        on the full raw ticket text, before anything else runs -- in
        particular, before any length-based or sentence-based cut point
        is chosen. This order is load-bearing, not cosmetic: it was
        fixed after testing surfaced a real leak where redacting only an
        already-split fragment let an email slip through untouched
        whenever the email itself contained the period the old code
        split on (see test_extract_topic_does_not_truncate_email_at_its_own_period
        for the full history). Redacting first means a matched PII span
        becomes a single token (e.g. "[EMAIL]", no periods in it) before
        any cut point is chosen, so a cut can no longer land inside a
        PII pattern.
      - HTML/script tags are stripped (a raw ticket containing
        "<script>...</script>" should never end up echoed into a draft
        an agent might paste into a richer-than-plain-text channel).
      - A topic that is only a URL is rejected outright, since
        otherwise "http://example.com/track?order=123" would either be
        echoed verbatim or truncated into a meaningless fragment.
    """
    # Redact PII across the WHOLE ticket before anything else -- see the
    # docstring note above on why this order matters.
    cleaned, _ = redact_pii(raw_text)
    # Strip HTML/script tags so no markup survives into the draft response.
    cleaned = re.sub(r"<[^>]*>", "", cleaned).strip()
    # Strip unresolved template placeholders like {product_purchased}
    cleaned = re.sub(r"\{[^}]*\}", "", cleaned).strip()
    # Collapse any whitespace left behind by the stripping above.
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Reject topics that are just a bare URL/URL fragment
    if re.match(r"^(https?://|www\.)\S*$", cleaned, re.IGNORECASE):
        return ""
    # Reject topics that are empty, too short, or still mostly punctuation/junk
    if len(cleaned) < 3 or not re.search(r"[a-zA-Z]{3,}", cleaned):
        return ""

    if len(cleaned) <= MAX_TOPIC_LEN:
        return cleaned

    window = cleaned[:MAX_TOPIC_LEN]
    # Prefer the last sentence-ending period within the window, so a
    # topic that happens to align with a real sentence boundary reads
    # naturally with no trailing ellipsis. Requiring it to be reasonably
    # far into the window (not just the first couple of characters)
    # avoids the old failure mode of stopping after a near-immediate,
    # not-very-informative period.
    last_period = window.rfind(".")
    if last_period >= 20:
        return window[: last_period + 1].strip()

    # No good sentence boundary in range -- cut at the last word boundary
    # instead of mid-word, and mark the truncation with an ellipsis.
    last_space = window.rfind(" ")
    if last_space >= 20:
        return window[:last_space].rstrip() + "..."
    return window.rstrip() + "..."


def draft_response(priority: str, department: str, raw_text: str = "") -> str:
    """
    raw_text is optional (defaults to "") so this stays backward-compatible
    with any existing call sites that only pass priority/department.
    When provided, the draft references the specific complaint instead of
    only the routing labels — the #1 complaint from user_validation.md.
    """
    topic = extract_topic(raw_text) if raw_text else ""

    if topic:
        opening = (
            f"Thanks for reaching out about \"{topic}\". We've reviewed your message and "
            f"routed it to our {department} team with {priority} priority."
        )
    else:
        opening = (
            f"Thanks for reaching out. We've reviewed your message and routed it to our "
            f"{department} team with {priority} priority."
        )

    return (
        f"Hi,\n\n"
        f"{opening}\n\n"
        f"{NEXT_STEPS[department].capitalize()} "
        f"You can expect a response within {SLA[priority]}.\n\n"
        f"Best regards,\nSupport Team"
    )


if __name__ == "__main__":
    sample_tickets = [
        "My order #482910 hasn't arrived and tracking hasn't updated in 5 days, I need it urgently!",
        "Hi, can you tell me how to switch my plan from monthly to yearly billing?",
        "The app crashes every time I open the settings page, please help.",
    ]

    for ticket in sample_tickets:
        priority, department = predict_ticket(ticket)
        response = draft_response(priority, department, raw_text=ticket)
        print("-" * 60)
        print("INPUT TICKET:\n", ticket)
        print(f"\nPREDICTED -> Priority: {priority} | Department: {department}")
        print("\nDRAFT RESPONSE:\n", response)
    print("-" * 60)
