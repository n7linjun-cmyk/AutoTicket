"""
pii_redaction.py

Lightweight, dependency-free PII redaction for ticket text.

Scope and intent
-----------------
This is deliberately regex-based rather than a full NER/PII model, in
keeping with the rest of this project (free, no extra heavy dependencies).
It targets the PII categories that are both common in support tickets and
reliably pattern-matchable: email addresses, phone numbers, payment-card
numbers, and order/account/reference numbers.

It does NOT attempt to redact free-text names or physical addresses --
those need real NER to catch reliably, and a regex attempt would either
miss most real names or false-positive on ordinary words. That's a known
limitation, documented in the README and the report, not something this
module claims to solve.

Where this is used
-------------------
Redaction is applied at two points, per the report's Privacy & PII
recommendation:
  1. The complaint topic extracted from a ticket and echoed into the
     draft response (predict_demo.extract_topic), so PII a customer
     typed doesn't get quoted back in generated text.
  2. The ticket text written to corrections.csv when an agent submits a
     feedback correction (app.py), so the local, unencrypted log doesn't
     accumulate raw customer PII.

It is intentionally NOT applied before the text is fed into the
TF-IDF vectorizer for prediction -- redacting first would strip real
signal (a phone number or order number carries no classification signal
either way, but running redaction on the multi-thousand-row training
data was out of scope here and untested, so this module only touches
the two points above, not the model's input path).
"""

import re

# Order matters: emails first (so an email's digits aren't later mistaken
# for a phone/card number), then phone, then card, then order/account refs.

EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}")

# US/international-ish phone numbers: optional country code, then a
# 3-part group (area code + two more groups), where the area code can
# be plain digits or wrapped in parentheses -- e.g. "555-123-4567",
# "1-800-799-0808", "(415) 555-2671".
PHONE_RE = re.compile(
    r"""
    (?<!\d)                                        # not preceded by a digit
    (?:\+?\d{1,3}[-.\s]?)?                          # optional leading country code
    (?:
        \(\d{2,4}\)\s?\d{3,4}[-.\s]?\d{3,4}          # (NNN) NNN-NNNN
        |
        \d{2,4}[-.\s]\d{3,4}[-.\s]\d{3,4}            # NNN-NNN-NNNN
    )
    (?!\d)                                         # not followed by a digit
    """,
    re.VERBOSE,
)

# Payment-card-like sequences: 13-19 digits, optionally grouped in 4s
# with spaces or dashes (covers "4111 1111 1111 1111" and "4111111111111111").
# Written as "digit, then (separator? digit) repeated" rather than
# "(digit, separator?) repeated" so a trailing space/dash after the last
# digit is never consumed as part of the match (that trailing-separator
# version was tested and found to eat the space before the next word,
# e.g. producing "[CARD]was charged" instead of "[CARD] was charged").
CARD_RE = re.compile(
    r"(?<!\d)\d(?:[ -]?\d){12,18}(?!\d)"
)

# Order / account / reference numbers: a '#' immediately followed by
# digits (e.g. "#482910"), or a keyword ("order", "account", "reference",
# "invoice", "ticket") followed by an optional separator and a run of
# 4+ digits.
ORDER_REF_RE = re.compile(
    r"""
    \#\d{4,}                                                  # "#482910"
    |
    \b(?:order|account|reference|invoice|ticket)\b            # keyword
    \s*(?:number|no\.?|\#|id)?                                # optional label
    \s*[:\#]?\s*
    (\d{4,})                                                  # the number itself
    """,
    re.IGNORECASE | re.VERBOSE,
)

REDACTION_LABELS = {
    "email": "[EMAIL]",
    "phone": "[PHONE]",
    "card": "[CARD]",
    "order_ref": "[REF#]",
}


def redact_pii(text: str):
    """
    Returns (redacted_text, counts) where counts is a dict like
    {"email": 1, "phone": 0, "card": 0, "order_ref": 2} recording how many
    matches of each category were masked. Safe on empty/None-ish input.
    """
    if not text:
        return text or "", {k: 0 for k in REDACTION_LABELS}

    counts = {k: 0 for k in REDACTION_LABELS}

    def _sub(pattern, label, s):
        def _replace(m):
            counts[label] += 1
            return REDACTION_LABELS[label]
        return pattern.sub(_replace, s)

    redacted = text
    redacted = _sub(EMAIL_RE, "email", redacted)

    # Card numbers before phone numbers: a 16-digit card formatted as
    # four space-separated groups of 4 (e.g. "4111 1111 1111 1111") would
    # otherwise have its first three groups matched by PHONE_RE (which
    # accepts 2-4 digit groups), mislabeling part of a card number as a
    # phone number and leaving the rest unmasked. Card numbers require at
    # least 13 total digits, well above any real phone number's digit
    # count, so matching them first avoids that overlap.
    redacted = _sub(CARD_RE, "card", redacted)

    # Order/reference numbers before phone: keyword-prefixed refs like
    # "invoice 998877" don't collide with phone formats (no separators
    # between digit groups), so order doesn't matter much here, but this
    # keeps all "labelled number" categories grouped before the more
    # permissive phone pattern.
    redacted = _sub(ORDER_REF_RE, "order_ref", redacted)
    redacted = _sub(PHONE_RE, "phone", redacted)

    return redacted, counts


def has_pii(text: str) -> bool:
    """Convenience check: True if redact_pii would mask anything."""
    _, counts = redact_pii(text)
    return any(counts.values())
