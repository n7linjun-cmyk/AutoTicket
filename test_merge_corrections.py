"""
test_merge_corrections.py

Unit tests for merge_corrections.py -- the script that turns the agent
feedback log (corrections.csv) into a retrainable dataset. These cover the
error boundaries listed in the README's "Error boundaries" table that had no
automated test: a corrections file with missing columns, and corrections
rows with blank text, invalid labels, duplicates or text already in the base
dataset.

They use small throw-away CSVs in a temporary directory, so they never touch
data/ or the real corrections.csv, and they do not need the trained models.

Run:  python -m pytest test_merge_corrections.py -v
  or: python test_merge_corrections.py   (plain-assert fallback, no pytest)
"""

import contextlib
import io
import os
import re
import sys
import tempfile

import pandas as pd

import merge_corrections

CORRECTION_COLS = ["timestamp", "ticket_text", "predicted_priority",
                   "predicted_department", "corrected_department"]

BASE_ROWS = [
    ("My invoice shows a double charge", "High", "Billing"),
    ("App crashes when I open settings", "Medium", "Technical support"),
]


def _run_merge(correction_rows, correction_cols=CORRECTION_COLS, write_corrections=True):
    """Build temp base + corrections CSVs, run merge_corrections.main() on
    them, and return (merged DataFrame, printed summary text)."""
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "base.csv")
        corr = os.path.join(tmp, "corrections.csv")
        out = os.path.join(tmp, "merged.csv")
        pd.DataFrame(BASE_ROWS, columns=["ticket_text", "priority", "department"]).to_csv(base, index=False)
        if write_corrections:
            pd.DataFrame(correction_rows, columns=correction_cols).to_csv(corr, index=False)

        old_argv = sys.argv
        sys.argv = ["merge_corrections.py", "--base", base,
                    "--corrections", corr, "--output", out]
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                merge_corrections.main()
        finally:
            sys.argv = old_argv
        return pd.read_csv(out), buf.getvalue()


def _dropped(summary, label):
    """Number printed on the "dropped (<label>)" line of the summary."""
    m = re.search(r"dropped \(" + re.escape(label) + r"\):\s*(\d+)", summary)
    assert m, f"no summary line for {label!r}"
    return int(m.group(1))


def _row(ts, text, pred_pri="High", pred_dep="Billing", corrected="Billing"):
    return (ts, text, pred_pri, pred_dep, corrected)


# ---- load_corrections -----------------------------------------------------

def test_missing_corrections_file_returns_empty_frame():
    df = merge_corrections.load_corrections("/nonexistent/corrections.csv")
    assert len(df) == 0
    assert list(df.columns) == CORRECTION_COLS


def test_corrections_file_missing_columns_raises_value_error():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bad.csv")
        pd.DataFrame({"timestamp": ["t"], "ticket_text": ["x"]}).to_csv(path, index=False)
        try:
            merge_corrections.load_corrections(path)
        except ValueError as e:
            assert "corrected_department" in str(e)
        else:
            raise AssertionError("expected ValueError for a file with missing columns")


# ---- main(): row-level cleaning ------------------------------------------

def test_no_corrections_writes_base_dataset_unchanged():
    merged, _ = _run_merge([], write_corrections=False)
    assert len(merged) == len(BASE_ROWS)
    assert set(merged["source"]) == {"original"}
    assert merged["priority_verified"].all()


def test_valid_correction_is_merged_and_priority_marked_unverified():
    merged, _ = _run_merge([_row("2026-01-01T10:00", "Parcel never arrived",
                                 "Low", "Technical support", "Delivery")])
    new = merged[merged["source"] == "correction"]
    assert len(new) == 1
    assert new.iloc[0]["department"] == "Delivery"      # the agent's label
    assert new.iloc[0]["priority"] == "Low"             # the model's own guess
    assert not bool(new.iloc[0]["priority_verified"])   # never human-checked
    assert merged[merged["source"] == "original"]["priority_verified"].all()


def test_blank_ticket_text_is_dropped():
    merged, out = _run_merge([_row("t1", "   "), _row("t2", "Real ticket text")])
    assert list(merged[merged["source"] == "correction"]["ticket_text"]) == ["Real ticket text"]
    assert _dropped(out, "blank text") == 1


def test_invalid_department_label_is_dropped():
    merged, out = _run_merge([_row("t1", "Some ticket", corrected="Sales")])
    assert (merged["source"] == "correction").sum() == 0
    assert _dropped(out, "invalid label") == 1


def test_invalid_predicted_priority_is_dropped():
    merged, _ = _run_merge([_row("t1", "Some ticket", pred_pri="Urgent")])
    assert (merged["source"] == "correction").sum() == 0


def test_duplicate_ticket_keeps_latest_correction():
    merged, out = _run_merge([
        _row("2026-01-01T10:00", "Same ticket", corrected="Billing"),
        _row("2026-01-02T10:00", "Same ticket", corrected="Delivery"),
    ])
    new = merged[merged["source"] == "correction"]
    assert len(new) == 1
    assert new.iloc[0]["department"] == "Delivery"      # last judgment wins
    assert _dropped(out, "dup ticket") == 1


def test_correction_already_in_base_is_skipped():
    merged, _ = _run_merge([_row("t1", BASE_ROWS[0][0])])
    assert (merged["source"] == "correction").sum() == 0
    assert len(merged) == len(BASE_ROWS)


def test_rerun_does_not_double_count():
    rows = [_row("t1", "Parcel never arrived", corrected="Delivery")]
    first, _ = _run_merge(rows)
    second, _ = _run_merge(rows)
    assert len(first) == len(second)   # output is rebuilt from base each time


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print("PASS ", name)
        except Exception as e:  # noqa: BLE001 - report any failure, keep going
            failed += 1
            print("FAIL ", name, "->", repr(e))
    print("-" * 60)
    print(f"{len(tests) - failed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
