# make_outcome.py
# Pulls the OUTCOME variable out of the master data into its own file.
# We keep the outcome separate from the criteria on purpose: the regression
# uses it as the thing being predicted, so it must never also be a criterion
# (that would be regressing something on itself). Keeping it in a separate
# file makes that impossible to do by accident.
#
# Current outcome: hs_grad_rate (4-year adjusted cohort graduation rate).
# If we get a better downstream outcome (e.g. Opportunity Atlas upward
# mobility), add it as a column to master_city_data.csv and list it here.
#
# run: python3 make_outcome.py master_city_data.csv outcome_data.csv

import sys
import pandas as pd

OUTCOME_COLS = ["hs_grad_rate"]   # add more outcome candidates here
ID_COLS = ["key", "city", "state", "fips"]

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "master_city_data.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "outcome_data.csv"

    df = pd.read_csv(src, dtype={"fips": str})
    out = df[ID_COLS].copy()
    for col in OUTCOME_COLS:
        x = pd.to_numeric(df[col], errors="coerce")
        out[col] = x                                   # raw value
        out[col + "_z"] = (x - x.mean()) / x.std(ddof=0)  # z-scored (what the model uses)

    out.to_csv(dst, index=False)
    n_missing = out[OUTCOME_COLS].isna().sum().sum()
    print(f"wrote {dst} ({len(out)} cities, outcomes: {OUTCOME_COLS}, {n_missing} missing values)")
