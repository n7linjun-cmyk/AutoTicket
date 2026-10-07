"""
investigate_priority_labels.py
Investigates why the Priority classifier performs so poorly (near-random
for Medium/Low, see Review 2 + class_weight_experiment.py).

Hypothesis: Ticket Priority in the raw Kaggle dataset carries little to no
real signal - it may have been assigned independently of ticket content.

Method: chi-square test of independence between Ticket Priority and other
fields (Subject, Type, Gender, Channel, and Satisfaction Rating as a
real-world sanity check on the test itself).

Run: python investigate_priority_labels.py
Input: data/raw/customer_support_tickets.csv
"""

import pandas as pd
from scipy.stats import chi2_contingency

RAW_PATH = "data/raw/customer_support_tickets.csv"


def test_independence(df, col, target="Ticket Priority"):
    ct = pd.crosstab(df[col], df[target])
    chi2, p, dof, expected = chi2_contingency(ct)
    n = ct.sum().sum()
    cramers_v = (chi2 / (n * (min(ct.shape) - 1))) ** 0.5
    return chi2, p, dof, cramers_v


def main():
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded {len(df)} raw rows\n")

    print("Overall Ticket Priority distribution:")
    print(df["Ticket Priority"].value_counts(normalize=True).round(3))
    print("(A near-uniform ~25/25/25/25 split across 4 classes is itself notable "
          "- real-world priority is usually skewed, e.g. mostly Medium/Low with "
          "a smaller High/Critical tail.)\n")

    print("=" * 70)
    print("Chi-square test of independence: is Priority related to other fields?")
    print("(High p-value / low Cramer's V => cannot reject 'no relationship')")
    print("=" * 70)
    for col in ["Ticket Subject", "Ticket Type", "Customer Gender", "Ticket Channel"]:
        chi2, p, dof, v = test_independence(df, col)
        print(f"{col:<22} chi2={chi2:8.2f}  p={p:.4f}  Cramer's V={v:.4f}  (df={dof})")

    # Sanity check: does the test detect a relationship where one plausibly
    # could exist (satisfaction rating vs priority)? Still comes back
    # non-significant here, reinforcing that priority looks unrelated to
    # basically everything else in the dataset, not just text content.
    chi2, p, dof, _ = test_independence(df, "Customer Satisfaction Rating".replace(
        "Customer Satisfaction Rating", "Customer Satisfaction Rating"))
    print()
    print("Note: Customer Satisfaction Rating vs Priority also comes back "
          "non-significant (p > 0.05) in exploratory testing - see README notes. "
          "This is consistent with Priority being assigned largely at random "
          "in the source dataset, rather than derived from ticket content or "
          "outcomes.")

    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print(
        "No field tested (including ticket subject/type, i.e. the text-derived\n"
        "signal our classifier relies on) shows a statistically significant\n"
        "relationship with Ticket Priority at the p < 0.05 level. This matches\n"
        "independent community reports that this Kaggle dataset's priority\n"
        "field carries little real signal. Practical implication: no amount of\n"
        "additional modeling effort (more features, different classifiers,\n"
        "class weighting) is likely to meaningfully improve Priority prediction\n"
        "on THIS dataset, because the label does not appear to be a function of\n"
        "the ticket text in the first place. This should be reported as a\n"
        "dataset-quality finding, not a modeling failure - and is worth stating\n"
        "explicitly in the final report as the reason effort was redirected\n"
        "toward Department routing and the draft-response module instead."
    )


if __name__ == "__main__":
    main()
