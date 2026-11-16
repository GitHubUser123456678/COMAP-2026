import sys
import pandas as pd
import numpy as np
#no significant AI use in this file except for minor debugging (included in ai report)
CRITERIA = {
    #I got rid of all of the redundant features/ones that were unusable for this
    "seda_test_score_gcs":            "+",
    "ap_courses_per_offering_school": "+",
    "per_pupil_spending_usd":         "+",
    "violent_crime_per_100k":         "-",
    "child_poverty_rate":             "-",
    "pct_teen_disconnected":          "-",
    "teen_employment_rate":           "+",
    "alltransit_score":               "+",
    "pct_kids_10min_walk_park":       "+",
    "library_visits_per_capita":      "+",
    "pct_freq_mental_distress":       "-",
    "pct_u18_computer_broadband":     "+",
}

COLS = ["key", "city", "state", "fips"]


def zscore_normalize(file, criteria=CRITERIA):
    Z = file[COLS].copy()
    outliers = []
    for feature, effect in criteria.items():
        x = pd.to_numeric(file[feature], errors="coerce")
        std = x.std(ddof=0)
        z = (x - x.mean()) / std
        if effect == "-":
            z = -z
        Z[feature] = z
        for city, val in zip(file["city"], z):
            if np.isfinite(val) and abs(val) > 3:
                outliers.append((feature, city, round(float(val), 2)))

    return Z

if __name__ == "__main__":
    src = "master_city_data.csv"
    dst = "normalized_zscore.csv"
    df = pd.read_csv(src, dtype={"fips": str})
    Z, outliers = zscore_normalize(df)
    Z.to_csv(dst, index=False)