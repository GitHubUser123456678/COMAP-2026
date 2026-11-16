import os
import sys
import numpy as np
import pandas as pd
#AI WAS USED in the implementation of the MAIN FUNCTION
#model revision 1 so only hs graduation rate as our outcome to train on
CRITERIA = {
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
#lines 23, 24, and 25 were added by claude fable 5 (manual) to implement the main function
OUTCOME = "hs_grad_rate_z"   
BETA = 0.3
BETA_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
ALPHAS = np.logspace(-2, 3, 60)
MIN_STABILITY = 0.9        
W_PRIOR = {c: 1.0 for c in CRITERIA}
#code basis in a lab for a course (Andrew Ng ML Specialization) (NOTE TO SELF PLS SITE THE LAB and show the segments in the lab that are similar to the parts in the model!!!!)
def ridge_fit(X, y, alpha): 
    p = X.shape[1]
    return np.linalg.solve(X.T @ X + alpha * np.eye(p), X.T @ y)

def losspredictions(X, y, alpha):
    n = len(y)
    preds = np.zeros(n)
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        b = ridge_fit(X[mask], y[mask], alpha)
        preds[i] = X[i] @ b
    return preds


def alpha(X, y):
    bestalpha, berror = ALPHAS[0], 100000000
    for a in ALPHAS:
        error = np.sum((y - losspredictions(X, y, a)) ** 2)
        if error < berror:
            bestalpha, berror = a, error
    return bestalpha


def lossr2(X, y, alpha):
    predictions = losspredictions(X, y, alpha)
    return 1 - np.sum((y - predictions) ** 2) / np.sum((y - y.mean()) ** 2)


def stability(X, y, alpha):
    signs = np.sign(ridge_fit(X, y, alpha))
    a = np.zeros(X.shape[1])
    for i in range(X.shape[0]):
        mask = np.ones(X.shape[0], dtype=bool)
        mask[i] = False
        a += (np.sign(ridge_fit(X[mask], y[mask], alpha)) == signs)
    return a / X.shape[0]

def evidenceweights(coefficients, stability):
    w = np.clip(coefficients, 0, None)
    w[stability < MIN_STABILITY] = 0
    return w / w.sum()

def main():
    """the main function was vibecoded 
    I gave Claude Fable 5 (manual) a prompt which included lines 1-71 and asked it to implement 
    a main function based on the format I was using when 
    I wrote normalize further info is in the AI report 
    AI was NOT USED for creating any ideas/the model itself but was rather 
    utilized in implementing the model for all of the data"""
    src = "normalized_zscore.csv"
    outcome = "outcome_data.csv"
    outdir = sys.argv[3] if len(sys.argv) > 3 else "results"
    os.makedirs(outdir, exist_ok=True)

    Z = pd.read_csv(src, dtype={"fips": str})
    outcome = pd.read_csv(outcome, dtype={"fips": str})
    Z = Z.merge(outcome[["key", OUTCOME]], on="key", how="left")

    feats = [c for c in CRITERIA if c in Z.columns]
    overlap = set(feats) & set(outcome.columns) - set(COLS)
    assert not overlap, f"these columns are criteria AND outcomes (circular!): {overlap}"

    data = Z.dropna(subset=feats + [OUTCOME]).reset_index(drop=True)
    dropped = sorted(set(Z["city"]) - set(data["city"]))
   
    X = data[feats].to_numpy(float)
    y = data[OUTCOME].to_numpy(float)

    alpha = alpha(X, y)
    coefs = ridge_fit(X, y, alpha)
    stab = stability(X, y, alpha)
    r2 = lossr2(X, y, alpha)
    w_evd = evidenceweights(coefs, stab)

    w_pri = np.array([W_PRIOR[c] for c in feats], float)
    w_pri = w_pri / w_pri.sum()
    w_new = (1 - BETA) * w_pri + BETA * w_evd

    weights = pd.DataFrame({
        "criterion": feats,
        "w_prior": w_pri.round(4),
        "ridge_coef": coefs.round(4),
        "sign_stability": stab.round(2),
        "w_evidence": w_evd.round(4),
        "w_revised": w_new.round(4),
    }).sort_values("w_revised", ascending=False)

    print(f"outcome = {OUTCOME}, alpha = {alpha:.3g} (LOO-CV), "
          f"out-of-sample R^2 = {r2:.3f}, beta = {BETA}\n")
    print(weights.to_string(index=False))
    print()

    res = data[["city", "state"]].copy()
    res["score_prior"] = X @ w_pri
    res["score_revised"] = X @ w_new
    for c in ["score_prior", "score_revised"]:
        res["rank_" + c] = res[c].rank(ascending=False).astype(int)
    res = res.sort_values("rank_score_revised")

    show = ["city", "state", "rank_score_revised", "rank_score_prior", "score_revised"]

    sweep_rows = []
   
    for b in BETA_SWEEP:
        w = (1 - b) * w_pri + b * w_evd
        s = X @ w
        order = np.argsort(-s)
        top3 = data.loc[order[:3], "city"].tolist()
        sweep_rows.append({"beta": b, "no1": top3[0], "no2": top3[1], "no3": top3[2]})

    weights.to_csv(os.path.join(outdir, "weights.csv"), index=False)
    res.to_csv(os.path.join(outdir, "rankings.csv"), index=False)
    pd.DataFrame(sweep_rows).to_csv(os.path.join(outdir, "beta_sweep.csv"), index=False)
   
if __name__ == "__main__":
    main()
