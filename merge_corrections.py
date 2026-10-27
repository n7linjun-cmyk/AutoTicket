"""
merge_corrections.py

Closes the gap identified after the feedback loop (Section 6.3) was shipped:
corrections.csv was only ever logged, never actually merged into a
retrainable dataset. This script does that merge.

What this does NOT do: it does not retrain the model itself (run
train_model_v3.py separately, see the bottom of this file's output for the
exact command), and it does not touch data/support_tickets_real_v2.csv --
that file stays untouched as the canonical, reproducible base dataset,
consistent with how every earlier dataset version in this project was kept
rather than overwritten (see README's "superseded" files). This script
always builds a fresh combined file from (base dataset + current
corrections.csv), so re-running it after more corrections come in is safe
and idempotent -- it never appends to its own prior output, which would
double-count rows on a second run.

Important honesty note, not just an implementation detail: the agent
feedback UI only ever asks "was the DEPARTMENT routing correct?" -- there
is no equivalent priority-correction control in the app. So a row pulled
from corrections.csv has a human-verified department label, but its
priority value is still just the model's own (possibly wrong) prediction,
never checked by a person. Blending that into priority training as if it
were ground truth would quietly corrupt the priority label the same way
the old dataset's priority label turned out to be corrupted (Section 4) --
exactly the failure mode this project has gone out of its way to catch
elsewhere. So every merged row is tagged priority_verified=False, and
train_model_v3.py's priority classifier should only ever be trained on
priority_verified=True rows. The department classifier has no such
problem -- every corrected row's department label IS the verified one --
so all rows are usable for department training regardless of this flag.

Usage:
    python merge_corrections.py
    python merge_corrections.py --corrections path/to/other_corrections.csv
"""

import argparse
import os
import pandas as pd

BASE_DATASET = "data/support_tickets_real_v2.csv"
CORRECTIONS_FILE = "corrections.csv"
OUTPUT_PATH = "data/support_tickets_real_v2_with_corrections.csv"

VALID_DEPARTMENTS = {"Billing", "Technical support", "Delivery"}
VALID_PRIORITIES = {"High", "Medium", "Low"}


def load_corrections(path: str) -> pd.DataFrame:
    """
    Read corrections.csv (schema defined by app.save_correction):
    timestamp, ticket_text, predicted_priority, predicted_department,
    corrected_department.

    Error boundaries: a missing file returns an empty frame with those
    columns (the merge then just re-writes the base dataset); a file that
    lacks any expected column raises ValueError rather than silently
    mis-reading it. Row-level cleaning (blank text, invalid labels,
    duplicates) happens later in main().
    """
    if not os.path.exists(path):
        print(f"No corrections file found at {path} -- nothing to merge.")
        return pd.DataFrame(columns=[
            "timestamp", "ticket_text", "predicted_priority",
            "predicted_department", "corrected_department",
        ])
    df = pd.read_csv(path)
    expected_cols = {"timestamp", "ticket_text", "predicted_priority",
                      "predicted_department", "corrected_department"}
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing expected columns {missing} -- has this "
            f"file's format changed? (save_correction() in app.py defines "
            f"the schema this script expects.)"
        )
    return df


def main():
    """
    Build the merged training CSV from base dataset + corrections.csv.

    Input : command-line flags --base, --corrections, --output (all
            optional; defaults are the module constants above).
    Output: writes the merged CSV to --output with the columns
            ticket_text, priority, department, source ("original" or
            "correction") and priority_verified (True for base rows,
            False for correction rows), then prints a summary.
    Cleaning order (each step is counted in the printed summary):
            blank text -> invalid labels -> duplicate tickets (latest
            wins) -> text already present in the base dataset.
    Idempotent: the output is always rebuilt from the base file, never
    appended to a previous output, so re-running cannot double-count.
    Error boundaries: a missing corrections file just rewrites the base
    dataset; a corrections file with missing columns raises ValueError
    (see load_corrections); a missing --base file raises
    FileNotFoundError.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corrections", default=CORRECTIONS_FILE,
                         help=f"Path to corrections CSV (default: {CORRECTIONS_FILE})")
    parser.add_argument("--base", default=BASE_DATASET,
                         help=f"Path to base training dataset (default: {BASE_DATASET})")
    parser.add_argument("--output", default=OUTPUT_PATH,
                         help=f"Path to write the merged dataset (default: {OUTPUT_PATH})")
    args = parser.parse_args()

    base_df = pd.read_csv(args.base)
    base_df["source"] = "original"
    base_df["priority_verified"] = True

    corrections_df = load_corrections(args.corrections)

    if len(corrections_df) == 0:
        print("No corrections to merge. Writing out the base dataset unchanged "
              f"so {args.output} always exists and is safe to point a trainer at.")
        base_df.to_csv(args.output, index=False)
        return

    # Rows with blank/unreadable ticket text can't be used -- same
    # insufficient-text check used elsewhere in this project (predict_demo's
    # has_signal), applied here at the data-hygiene level instead of the
    # model-input level.
    before = len(corrections_df)
    corrections_df = corrections_df[
        corrections_df["ticket_text"].notna()
        & (corrections_df["ticket_text"].str.strip() != "")
    ]
    dropped_blank = before - len(corrections_df)

    # Validate labels rather than silently trusting the file -- a hand-edited
    # or corrupted corrections.csv shouldn't be able to inject a bad class
    # label into the training set.
    before = len(corrections_df)
    corrections_df = corrections_df[
        corrections_df["corrected_department"].isin(VALID_DEPARTMENTS)
        & corrections_df["predicted_priority"].isin(VALID_PRIORITIES)
    ]
    dropped_invalid = before - len(corrections_df)

    # If the same ticket was corrected more than once (an agent re-submitting,
    # or two agents disagreeing), keep only the most recent correction for
    # that exact ticket text -- last human judgment wins.
    before = len(corrections_df)
    corrections_df = corrections_df.sort_values("timestamp").drop_duplicates(
        subset="ticket_text", keep="last"
    )
    dropped_dupes = before - len(corrections_df)

    # A correction whose ticket text happens to exactly match a row already
    # in the base dataset isn't new signal -- skip it rather than creating a
    # duplicate row. (In practice this should rarely fire: corrections.csv
    # stores PII-redacted text, which generally won't exact-match the base
    # dataset's unredacted text -- but tickets with no PII in them could
    # still match, so it's checked for real rather than assumed away.)
    already_present = set(base_df["ticket_text"])
    before = len(corrections_df)
    corrections_df = corrections_df[~corrections_df["ticket_text"].isin(already_present)]
    dropped_already_present = before - len(corrections_df)

    merged_rows = pd.DataFrame({
        "ticket_text": corrections_df["ticket_text"],
        "priority": corrections_df["predicted_priority"],
        "department": corrections_df["corrected_department"],
        "source": "correction",
        "priority_verified": False,
    })

    combined = pd.concat([base_df, merged_rows], ignore_index=True)
    combined.to_csv(args.output, index=False)

    n_changed = (corrections_df["predicted_department"] != corrections_df["corrected_department"]).sum()
    n_confirmed = len(corrections_df) - n_changed

    print(f"Base dataset:              {len(base_df):>6} rows ({args.base})")
    print(f"Corrections file:          {len(load_corrections(args.corrections)):>6} rows read ({args.corrections})")
    print(f"  dropped (blank text):    {dropped_blank:>6}")
    print(f"  dropped (invalid label): {dropped_invalid:>6}")
    print(f"  dropped (dup ticket):    {dropped_dupes:>6}")
    print(f"  dropped (already in base):{dropped_already_present:>6}")
    print(f"  usable corrections:      {len(corrections_df):>6}  "
          f"({n_changed} actually re-routed, {n_confirmed} confirmed as already correct)")
    print(f"Merged dataset:            {len(combined):>6} rows -> {args.output}")
    print()
    print("All merged rows are tagged priority_verified=False (see this script's")
    print("docstring) -- only department training should use them unfiltered.")
    print()
    print("To retrain on this merged dataset:")
    print(f"    python train_model_v3.py {args.output}")


if __name__ == "__main__":
    main()
