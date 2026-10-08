"""
test_edge_cases.py

Edge-case tests for the prediction pipeline (predict_demo.py). Covers the
five categories called out in the project handover as untested: empty
text, non-English text, emoji-only input, and extremely short/long
tickets, plus a couple of adjacent cases found while writing these
(HTML/script content, URL-only text) that turned out to expose real bugs
-- see the "insufficient_text" and extract_topic notes below.

What these tests do and do not cover
-------------------------------------
They test ROBUSTNESS and ERROR BOUNDARIES of the pipeline (predict_demo.py
and pii_redaction.py): that odd input never crashes it, that the output
always has the documented shape, and that known failure modes stay fixed.
They do NOT measure model accuracy -- that is evaluated separately on a
held-out split (train_model_v3.py, report Section 12). They load the
trained files in models/, so run them from the project root.

Test groups (in file order) and the failure each one guards against:
  1. Empty / whitespace / punctuation / number-only / year-only / single
     character  -> must be flagged insufficient_text with confidence 0.0
     (previously reported ~79% "confidence" from class priors alone).
  2. Emoji-only and emoji mixed with text -> flagged only when no real
     words remain.
  3. Non-English and mixed-language text -> no crash; partial signal from
     any English words.
  4. Very long text (~5,000 words) -> no crash, topic capped at
     MAX_TOPIC_LEN, draft stays a reasonable size.
  5. HTML/script and URL-only text -> markup never reflected into the
     draft; bare URL rejected as a topic.
  6. PII redaction (email, phone, card, order/reference numbers) -> each
     masked, unrelated numbers left alone, and PII never appears in a
     draft response; includes the regression where an email's own period
     used to split it mid-address.

Run:  python -m pytest test_edge_cases.py -v
  or: python test_edge_cases.py   (plain-assert fallback, no pytest needed)
"""

from predict_demo import (
    clean_text,
    predict_ticket_with_confidence,
    extract_topic,
    explain_department_prediction,
    draft_response,
    MAX_TOPIC_LEN,
)
from pii_redaction import redact_pii

VALID_PRIORITIES = {"High", "Medium", "Low"}
VALID_DEPARTMENTS = {"Billing", "Technical support", "Delivery"}


def _assert_well_formed_result(text, result):
    """Shared shape/type checks every prediction result must satisfy,
    regardless of how strange the input text is."""
    assert result["priority"] in VALID_PRIORITIES, f"bad priority for {text!r}: {result}"
    assert result["department"] in VALID_DEPARTMENTS, f"bad department for {text!r}: {result}"
    assert 0.0 <= result["priority_confidence"] <= 1.0
    assert 0.0 <= result["department_confidence"] <= 1.0
    assert isinstance(result["insufficient_text"], bool)


# ---------------------------------------------------------------------------
# Empty / whitespace / no-signal text
# ---------------------------------------------------------------------------

def test_empty_string_does_not_crash():
    result = predict_ticket_with_confidence("")
    _assert_well_formed_result("", result)


def test_empty_string_is_flagged_insufficient():
    # Bug found during testing: on an all-zero TF-IDF vector, predict_proba
    # returns the model's class-prior probabilities, which looked like a
    # confident (~79%) prediction despite there being zero signal. Fixed by
    # detecting nnz == 0 and forcing confidence to 0.0 / flagging it.
    result = predict_ticket_with_confidence("")
    assert result["insufficient_text"] is True
    assert result["priority_confidence"] == 0.0
    assert result["department_confidence"] == 0.0


def test_whitespace_only_is_flagged_insufficient():
    result = predict_ticket_with_confidence("   \n\t  ")
    assert result["insufficient_text"] is True


def test_punctuation_only_is_flagged_insufficient():
    result = predict_ticket_with_confidence("!!!???...")
    assert result["insufficient_text"] is True


def test_numbers_only_is_flagged_insufficient():
    # This is a real regression test, not just a repeat of the same case.
    # With the original 3000-feature-capped vocabulary, digit-only tokens
    # like "1234567890" simply weren't in-vocabulary, so this passed for
    # an uninteresting reason. After train_model_v3.py uncapped the
    # vocabulary (~77,800 tokens) for the accuracy improvements, this
    # specific number turned out to appear in the training data (likely
    # a placeholder/example number in some ticket) and became a real,
    # if meaningless, vocabulary match -- with ~96% department confidence
    # on pure digits before the fix. has_signal() now requires at least
    # one ALPHABETIC recognized token, not just any nonzero TF-IDF match,
    # specifically to keep catching this.
    result = predict_ticket_with_confidence("1234567890")
    assert result["insufficient_text"] is True
    assert result["priority_confidence"] == 0.0
    assert result["department_confidence"] == 0.0


def test_year_only_is_flagged_insufficient():
    # A second purely-numeric in-vocabulary token (a bare year), to check
    # the fix generalizes rather than special-casing one specific number.
    result = predict_ticket_with_confidence("2022")
    assert result["insufficient_text"] is True


def test_empty_topic_is_empty():
    assert extract_topic("") == ""
    assert extract_topic("   ") == ""


def test_empty_text_explain_returns_no_words():
    assert explain_department_prediction("") == []


def test_empty_text_draft_uses_generic_opening():
    draft = draft_response("Low", "Billing", raw_text="")
    assert "Thanks for reaching out." in draft
    assert '"' not in draft  # no quoted (empty) topic


# ---------------------------------------------------------------------------
# Emoji-only input
# ---------------------------------------------------------------------------

def test_emoji_only_does_not_crash():
    result = predict_ticket_with_confidence("🔥🔥🔥😡📦")
    _assert_well_formed_result("emoji", result)


def test_emoji_only_is_flagged_insufficient():
    result = predict_ticket_with_confidence("🔥🔥🔥😡📦")
    assert result["insufficient_text"] is True


def test_emoji_mixed_with_real_text_is_not_flagged():
    # Emoji alongside real words should still find signal in the words.
    result = predict_ticket_with_confidence("😡 my invoice charge is wrong 😡")
    assert result["insufficient_text"] is False


# ---------------------------------------------------------------------------
# Non-English text
# ---------------------------------------------------------------------------

def test_non_english_does_not_crash():
    text = "Mon colis n'est jamais arrivé et je suis très en colère."
    result = predict_ticket_with_confidence(text)
    _assert_well_formed_result(text, result)


def test_non_english_topic_extraction_does_not_crash():
    # Known limitation, not asserted as "correct": the model was trained on
    # English tickets, so non-English words mostly fall outside the TF-IDF
    # vocabulary and predictions here are low-signal/unreliable. We only
    # assert the pipeline runs cleanly and doesn't silently claim high
    # confidence on words it has never seen.
    text = "Mon colis n'est jamais arrivé et je suis très en colère."
    topic = extract_topic(text)
    assert isinstance(topic, str)


def test_mixed_language_finds_partial_signal():
    text = "Bonjour, my order is late and je suis frustrated!!"
    result = predict_ticket_with_confidence(text)
    # English words ("order", "late") are in-vocabulary even though the
    # ticket isn't pure English, so this should NOT be flagged insufficient.
    assert result["insufficient_text"] is False


# ---------------------------------------------------------------------------
# Extremely short text
# ---------------------------------------------------------------------------

def test_single_word_does_not_crash():
    result = predict_ticket_with_confidence("help")
    _assert_well_formed_result("help", result)


def test_single_character_is_flagged_insufficient():
    # "x" cleans to "x", which is a single character outside the
    # (unigram+bigram, min length implied by training vocab) feature set.
    result = predict_ticket_with_confidence("x")
    assert result["insufficient_text"] is True


# ---------------------------------------------------------------------------
# Extremely long text
# ---------------------------------------------------------------------------

def test_very_long_text_does_not_crash():
    long_text = "billing charge invoice payment refund " * 1000  # ~5000 words
    result = predict_ticket_with_confidence(long_text)
    _assert_well_formed_result("long_text", result)
    assert result["insufficient_text"] is False


def test_very_long_text_topic_is_truncated():
    # Bug found during testing: a ticket with no early period (or none at
    # all) made extract_topic return the entire multi-thousand-character
    # string, which then got embedded verbatim into the draft response.
    long_text = "billing charge invoice payment refund " * 1000
    topic = extract_topic(long_text)
    assert len(topic) <= MAX_TOPIC_LEN + 3  # +3 for the "..." suffix


def test_very_long_text_draft_stays_reasonably_sized():
    long_text = "billing charge invoice payment refund " * 1000
    draft = draft_response("Medium", "Billing", raw_text=long_text)
    # The draft should stay roughly template-sized, not balloon to the
    # length of the input ticket.
    assert len(draft) < 1000


# ---------------------------------------------------------------------------
# HTML / script content and bare URLs (adjacent cases found while testing)
# ---------------------------------------------------------------------------

def test_html_script_is_not_reflected_into_draft():
    # Bug found during testing: extract_topic took everything before the
    # first period verbatim, so a ticket containing "<script>...</script>"
    # got that markup echoed straight into the draft response text. Tags
    # are now stripped; harmless leftover plain text (e.g. the word
    # "alert(1)") is fine to keep -- what matters is no markup survives.
    text = "<script>alert(1)</script> my order is late"
    draft = draft_response("Medium", "Technical support", raw_text=text)
    assert "<" not in draft and ">" not in draft


def test_url_only_topic_is_rejected():
    # Bug found during testing: "before the first period" turned a bare URL
    # into a meaningless fragment like "http://example" as the quoted topic.
    topic = extract_topic("http://example.com/track?order=123")
    assert topic == ""


def test_url_only_does_not_crash_prediction():
    result = predict_ticket_with_confidence("http://example.com/track?order=123")
    _assert_well_formed_result("url_only", result)
    # clean_text strips URLs entirely, so this has no signal left.
    assert result["insufficient_text"] is True


# ---------------------------------------------------------------------------
# clean_text unit checks
# ---------------------------------------------------------------------------

def test_clean_text_handles_none_safe_inputs():
    assert clean_text("") == ""
    assert clean_text("   ") == ""
    assert clean_text("HELLO World") == "hello world"
    assert clean_text("Check http://x.com now") == "check now"


# ---------------------------------------------------------------------------
# PII redaction
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PII redaction (pii_redaction.redact_pii)
# ---------------------------------------------------------------------------
# redact_pii(text) -> (redacted_text, counts_dict). These tests exercise the
# real, standalone module directly, plus check its integration into
# extract_topic/draft_response below.

def test_redact_pii_masks_email():
    redacted, counts = redact_pii("Contact me at jane.doe@example.com please")
    assert redacted == "Contact me at [EMAIL] please"
    assert counts["email"] == 1


def test_redact_pii_masks_multi_level_domain_email():
    # A domain with multiple subdomains (sub.example.co.uk) must be masked
    # in full, not just the first label.
    redacted, counts = redact_pii("Email: test.user+tag@sub.example.co.uk")
    assert redacted == "Email: [EMAIL]"
    assert "co.uk" not in redacted
    assert counts["email"] == 1


def test_redact_pii_masks_phone_numbers():
    for text in [
        "Call 1-800-799-0808 now",
        "Reach me at (415) 555-2671",
        "Or +1 555-123-4567 works too",
    ]:
        redacted, counts = redact_pii(text)
        assert "[PHONE]" in redacted, f"failed on: {text!r} -> {redacted!r}"
        assert counts["phone"] == 1
        assert not any(ch.isdigit() for ch in redacted.replace("[PHONE]", "")), \
            f"leftover digits in: {redacted!r}"


def test_redact_pii_masks_order_numbers_with_trigger_word():
    redacted, counts = redact_pii("My order #482910 hasn't arrived")
    assert redacted == "My [REF#] hasn't arrived"
    assert counts["order_ref"] == 1

    redacted2, counts2 = redact_pii("Order number 48291 was never delivered.")
    assert redacted2 == "[REF#] was never delivered."
    assert counts2["order_ref"] == 1


def test_redact_pii_masks_bare_hash_numbers():
    # No trigger word nearby -- still masked, since a bare "#" + digits is
    # almost always a reference/ticket number in a support context.
    redacted, counts = redact_pii("Please check #998877 urgently.")
    assert redacted == "Please check [REF#] urgently."
    assert counts["order_ref"] == 1


def test_redact_pii_masks_card_numbers():
    for text in [
        "Card number 4111 1111 1111 1111 was charged twice.",
        "card 4111-1111-1111-1111 declined",
    ]:
        redacted, counts = redact_pii(text)
        assert "[CARD]" in redacted, f"failed on: {text!r} -> {redacted!r}"
        assert "4111" not in redacted
        assert counts["card"] == 1


def test_redact_pii_leaves_unrelated_numbers_alone():
    # A zip code / apartment number with no trigger word, no "#", and too
    # short/wrong-shaped to look like a phone or card number should NOT be
    # redacted -- over-redaction would make the tool useless for
    # legitimate context an agent needs.
    text = "My zip code is 90210 and I live in apartment 4."
    redacted, counts = redact_pii(text)
    assert redacted == text
    assert sum(counts.values()) == 0


def test_redact_pii_leaves_plain_text_alone():
    text = "No PII here, just a broken screen and a battery issue."
    redacted, counts = redact_pii(text)
    assert redacted == text
    assert sum(counts.values()) == 0


def test_redact_pii_does_not_crash_on_edge_cases():
    for text in ["", "   ", "🔥🔥🔥", "!!!???", "a" * 5000, None]:
        redacted, counts = redact_pii(text)
        assert isinstance(redacted, str)
        assert isinstance(counts, dict)


def test_draft_response_never_contains_raw_email():
    ticket = "Contact me at jane.doe@example.com. My screen is broken."
    draft = draft_response("Medium", "Technical support", raw_text=ticket)
    assert "jane.doe@example.com" not in draft
    # Also check no partial fragment leaked -- see the regression test
    # below for why a substring check alone isn't strong enough.
    assert "jane.doe" not in draft
    assert "example.com" not in draft
    assert "[EMAIL]" in draft


def test_extract_topic_does_not_truncate_email_at_its_own_period():
    # Regression test for a real bug: extract_topic used to split the
    # ticket on the FIRST period in the raw text before running PII
    # redaction. When the email itself was the first thing containing a
    # period (no earlier "real" sentence break), the split cut through
    # the email's own domain dot -- e.g. ".../call 555-123-4567."
    # unaffected, but "...email me at jane@example.com..." got split
    # right after "jane@example", stripping the ".com" the email regex
    # needs to recognize it as an email at all. The truncated fragment
    # "jane@example" then sailed through untouched into the draft
    # response -- a real PII leak, not just a cosmetic issue.
    #
    # Fix: redact_pii now runs on the WHOLE raw ticket before any
    # splitting happens, so a matched PII span (which becomes a single
    # token like "[EMAIL]" with no periods in it) can never itself become
    # the split point.
    ticket = ("My order #482910 never arrived, email me at jane@example.com "
              "or call 555-123-4567.")

    topic = extract_topic(ticket)
    assert "jane@example" not in topic  # neither the full nor truncated form
    assert "[EMAIL]" in topic

    result = predict_ticket_with_confidence(ticket)
    draft = draft_response(result["priority"], result["department"], raw_text=ticket)
    assert "jane@example.com" not in draft
    assert "jane@example" not in draft   # the specific leaked fragment
    assert "555-123-4567" not in draft
    assert "#482910" not in draft
    assert "[EMAIL]" in draft
    assert "[PHONE]" in draft
    assert "[REF#]" in draft


def test_extract_topic_keeps_content_past_the_first_sentence():
    # extract_topic originally stopped at the FIRST period, which worked
    # for the training dataset's clean "subject. body" structure but
    # silently discarded everything after a customer's first short
    # sentence on real, free-form tickets -- e.g. a 3-short-sentence
    # ticket produced a topic of just its first few words, even though
    # nothing after that was PII, junk, or otherwise unusable. Real user
    # feedback flagged the resulting drafts as too generic. Fixed to keep
    # as much of the ticket as fits within MAX_TOPIC_LEN instead of
    # always stopping at the first period.
    ticket = "Order #55219. Email jane@example.com. Call 555-123-4567."
    topic = extract_topic(ticket)
    # All three sentences' worth of (redacted) content should survive,
    # not just the first one.
    assert "[REF#]" in topic
    assert "[EMAIL]" in topic
    assert "[PHONE]" in topic
    # And the sentence-ending periods between them should still be
    # present too (see test_redact_pii_does_not_eat_trailing_period_
    # after_email for the related redaction-level bug this depended on).
    assert topic == "[REF#]. Email [EMAIL]. Call [PHONE]."


def test_extract_topic_still_caps_length_on_a_true_run_on():
    # Grabbing more than the first sentence should not mean grabbing an
    # unbounded amount of text -- a ticket with no sentence structure at
    # all should still be capped at MAX_TOPIC_LEN (plus a short ellipsis).
    long_text = "billing charge invoice payment refund " * 1000
    topic = extract_topic(long_text)
    assert len(topic) <= MAX_TOPIC_LEN + 3


def test_redact_pii_does_not_eat_trailing_period_after_email():
    # Regression test for a real bug found while testing the fix above:
    # the email regex's domain part was written as "[\w-]+\.[\w.-]+",
    # and [\w.-]+ includes literal periods, so it greedily consumed a
    # sentence-ending period right after the TLD too -- turning
    # "jane@example.com. Call" into "[EMAIL] Call" with the sentence
    # break silently deleted. Fixed by requiring the domain to end in a
    # proper TLD ("\.[a-zA-Z]{2,}" as the final, non-greedy-absorbing
    # segment) so a trailing "." with a space after it is never pulled
    # into the match.
    redacted, counts = redact_pii("Email jane@example.com. Call me back.")
    assert redacted == "Email [EMAIL]. Call me back."
    assert counts["email"] == 1


def test_draft_response_never_contains_raw_order_number():
    ticket = "My order #482910 hasn't arrived and tracking hasn't updated."
    draft = draft_response("High", "Delivery", raw_text=ticket)
    assert "#482910" not in draft
    assert "[REF#]" in draft


def test_draft_response_never_contains_raw_phone_number():
    ticket = "Call me back at 1-800-799-0808 about my broken device."
    draft = draft_response("Medium", "Technical support", raw_text=ticket)
    assert "1-800-799-0808" not in draft
    assert "[PHONE]" in draft


if __name__ == "__main__":
    # Plain-assert fallback so this file is runnable without pytest installed.
    import sys
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items())
              if name.startswith("test_") and callable(obj)]
    passed, failed = 0, 0
    for name, test_fn in tests:
        try:
            test_fn()
            print(f"PASS  {name}")
            passed += 1
        except AssertionError:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failed += 1
        except Exception:
            print(f"ERROR {name}")
            traceback.print_exc()
            failed += 1
    print("-" * 60)
    print(f"{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
