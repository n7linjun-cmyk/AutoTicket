"""
prepare_real_dataset_v2.py
Converts the new "Multilingual Customer Support Tickets" dataset
(aa_dataset-tickets-multi-lang-5-2-50-version.csv, by tobiasbueck on Kaggle)
into our project's standard schema: ticket_text, priority, department.

Why this replaces the old dataset:
A chi-square test (see investigate_priority_labels.py results for the OLD
suraj520 dataset) found that dataset's Priority field had no statistically
significant relationship with ticket content - it behaved like a randomly
assigned label. A chi-square test on THIS dataset's priority-vs-queue
relationship comes back highly significant (p ~ 0, Cramer's V = 0.28),
with priority patterns that make real-world sense (e.g. "Service Outages
and Maintenance" tickets are 70% high priority; "General Inquiry" tickets
are mostly low priority). This is a genuine, learnable signal.

Mapping decisions (documented explicitly, same philosophy as the original
prepare_real_dataset.py):

  1. LANGUAGE: this dataset is multilingual (en/de). We keep English only
     for now, to stay comparable with the rest of the project. (German
     could be added later as a separate experiment.)

  2. PRIORITY: already low/medium/high -> title-cased to Low/Medium/High
     to match our existing scheme. No merging needed (this dataset has
     no separate "Critical" tier).

  3. DEPARTMENT: the raw data has a 'queue' column with 10 categories -
     more granular than our 3-class scheme (Billing / Technical support
     / Delivery). We handle this the same way as before: a documented,
     rule-based mapping rather than a silent reshape.

     a) DELIVERY override: applied first, same keyword approach as the
        original project (this dataset has no dedicated delivery/shipping
        queue either).
     b) Everything else maps by queue:
          Billing and Payments              -> Billing
          Technical Support                 -> Technical support
          IT Support                        -> Technical support
          Product Support                   -> Technical support
          Service Outages and Maintenance   -> Technical support
     c) Queues with NO reasonable mapping to our 3-class scheme are
        DROPPED (not force-mapped), because spot-checking their content
        showed they don't reliably belong to Billing/Technical/Delivery:
          Returns and Exchanges  (spot-checked: mostly general product
                                   questions, not actually about returns)
          Customer Service        (spot-checked: broad catch-all)
          Sales and Pre-Sales     (pre-purchase inquiries, out of scope)
          Human Resources         (internal HR tickets, not customer
                                    support at all)
          General Inquiry         (too ambiguous / catch-all)

Run:  python prepare_real_dataset_v2.py
Input:  /mnt/user-data/uploads/aa_dataset-tickets-multi-lang-5-2-50-version.csv
Output: data/support_tickets_real_v2.csv
"""

import pandas as pd
import re
import os

RAW_PATH = "/mnt/user-data/uploads/aa_dataset-tickets-multi-lang-5-2-50-version.csv"
OUT_PATH = "data/support_tickets_real_v2.csv"

# ---------- 1. Load raw data, English only ----------
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(f"Could not find {RAW_PATH}.")

df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df)} raw rows (all languages)")
before_lang = len(df)
df = df[df["language"] == "en"].copy()
print(f"Kept {len(df)} English rows (dropped {before_lang - len(df)} non-English)")

# ---------- 2. Build ticket_text from subject + body ----------
df["subject"] = df["subject"].fillna("")
df["body"] = df["body"].fillna("")
df["ticket_text"] = (df["subject"] + ". " + df["body"]).str.strip()

# ---------- 3. Map priority: low/medium/high -> Low/Medium/High ----------
df["priority"] = df["priority"].str.strip().str.title()

# ---------- 4. Map department ----------
DELIVERY_KEYWORDS = [
    "delivery", "deliver", "shipment", "shipping", "shipped", "tracking",
    "courier", "package", "parcel", "arrived", "delayed order", "order has not"
]

queue_to_department = {
    "Billing and Payments": "Billing",
    "Technical Support": "Technical support",
    "IT Support": "Technical support",
    "Product Support": "Technical support",
    "Service Outages and Maintenance": "Technical support",
}


def assign_department(row) -> str:
    text = str(row["ticket_text"]).lower()
    if any(kw in text for kw in DELIVERY_KEYWORDS):
        return "Delivery"
    return queue_to_department.get(row["queue"], None)


df["department"] = df.apply(assign_department, axis=1)

# ---------- 5. Clean up: drop rows with missing/unmapped labels ----------
before = len(df)
df = df.dropna(subset=["ticket_text", "priority", "department"])
df = df[df["ticket_text"].str.len() > 5]
after = len(df)
print(f"Dropped {before - after} rows with missing/unmapped text or labels "
      f"(mostly out-of-scope queues: Returns and Exchanges, Customer Service, "
      f"Sales and Pre-Sales, Human Resources, General Inquiry)")

# ---------- 6. Save in our standard schema ----------
final = df[["ticket_text", "priority", "department"]]
final.to_csv(OUT_PATH, index=False)

print(f"\nSaved {len(final)} cleaned rows -> {OUT_PATH}")
print("\nPriority distribution:\n", final["priority"].value_counts())
print("\nDepartment distribution:\n", final["department"].value_counts())
