"""
prepare_real_dataset.py
Converts the raw Kaggle "Customer Support Ticket Dataset" (by Suraj Gurav)
into our project's standard schema: ticket_text, priority, department.

Why this script exists:
The real dataset's columns don't match our labels 1:1, so we make and
document explicit mapping decisions instead of silently reshaping data:

  1. PRIORITY: raw data has 4 levels (Critical/High/Medium/Low).
     We merge Critical -> High to match our 3-level scheme.

  2. DEPARTMENT: raw data has a 'Ticket Type' column (Billing inquiry,
     Technical issue, Cancellation request, Product inquiry, Refund
     request) but NO dedicated "Delivery/Shipping" category.
     We handle this with a rule-based override:
       - If the ticket description contains delivery/shipping keywords
         (order, shipment, tracking, delivered, arrived, courier, package)
         -> label it Delivery, regardless of original Ticket Type.
       - Else map Ticket Type -> Billing / Technical support using a
         lookup table.
     This is a real, common data-harmonization step and is worth
     describing explicitly in your report as a documented design choice.

Run:  python prepare_real_dataset.py
Input:  data/raw/customer_support_tickets.csv   (download from Kaggle first)
Output: data/support_tickets_real.csv
"""

import pandas as pd
import re
import os

RAW_PATH = "data/raw/customer_support_tickets.csv"
OUT_PATH = "data/support_tickets_real.csv"

# ---------- 1. Load raw data ----------
if not os.path.exists(RAW_PATH):
    raise FileNotFoundError(
        f"Could not find {RAW_PATH}. Download the dataset from Kaggle "
        "(https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset) "
        "and place the CSV at this path first."
    )

df = pd.read_csv(RAW_PATH)
print(f"Loaded {len(df)} raw rows")
print("Columns found:", list(df.columns))

# ---------- 2. Build ticket_text from subject + description ----------
df["Ticket Subject"] = df["Ticket Subject"].fillna("")
df["Ticket Description"] = df["Ticket Description"].fillna("")
df["ticket_text"] = (df["Ticket Subject"] + ". " + df["Ticket Description"]).str.strip()

# ---------- 3. Map priority: Critical -> High ----------
priority_map = {
    "Critical": "High",
    "High": "High",
    "Medium": "Medium",
    "Low": "Low",
}
df["priority"] = df["Ticket Priority"].map(priority_map)

# ---------- 4. Map department ----------
DELIVERY_KEYWORDS = [
    "delivery", "deliver", "shipment", "shipping", "shipped", "tracking",
    "courier", "package", "parcel", "arrived", "delayed order", "order has not"
]

type_to_department = {
    "Billing inquiry": "Billing",
    "Refund request": "Billing",
    "Cancellation request": "Billing",
    "Technical issue": "Technical support",
    "Product inquiry": "Technical support",  # default fallback; overridden below if delivery-related
}


def assign_department(row) -> str:
    text = str(row["ticket_text"]).lower()
    if any(kw in text for kw in DELIVERY_KEYWORDS):
        return "Delivery"
    return type_to_department.get(row["Ticket Type"], None)


df["department"] = df.apply(assign_department, axis=1)

# ---------- 5. Clean up: drop rows with missing/unmapped labels ----------
before = len(df)
df = df.dropna(subset=["ticket_text", "priority", "department"])
df = df[df["ticket_text"].str.len() > 5]
after = len(df)
print(f"Dropped {before - after} rows with missing/unmapped text or labels")

# ---------- 6. Save in our standard schema ----------
final = df[["ticket_text", "priority", "department"]]
final.to_csv(OUT_PATH, index=False)

print(f"\nSaved {len(final)} cleaned rows -> {OUT_PATH}")
print("\nPriority distribution:\n", final["priority"].value_counts())
print("\nDepartment distribution:\n", final["department"].value_counts())
