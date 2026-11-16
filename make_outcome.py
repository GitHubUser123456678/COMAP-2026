import sys
import pandas as pd
#no signifcant AI use in this file apart from minor debugging (included in ai report)
OUTCOME_COLS = ["hs_grad_rate"]
COLS = ["key", "city", "state", "fips"]

if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "master_city_data.csv"
    dst = sys.argv[2] if len(sys.argv) > 2 else "outcome_data.csv"

    df = pd.read_csv(src, dtype={"fips": str})
    out = df[COLS].copy()
    for col in OUTCOME_COLS:
        x = pd.to_numeric(df[col], errors="coerce")
        out[col] = x                                
        out[col + "_z"] = (x - x.mean()) / x.std(ddof=0) 

    out.to_csv(dst, index=False)
    n_missing = out[OUTCOME_COLS].isna().sum().sum()