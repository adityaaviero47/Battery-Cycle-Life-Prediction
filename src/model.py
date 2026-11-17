"""Predict battery cycle life from the first 100 cycles (Severson et al. 2019 LFP cells).

Data: per-cell early-life features extracted from the public MIT/Stanford/Toyota Research Institute dataset
(124 A123 APR18650M1A LFP/graphite cells, 1.1 Ah, 72 fast-charge policies, 30 C, 4C discharge).
Cycle life = cycles until capacity falls to 80% of nominal (0.88 Ah).
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNetCV, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import RepeatedKFold

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
FEATURES = {
    "variance": ["log10_dQ_var"],
    "discharge": ["log10_dQ_var", "log10_abs_dQ_min", "dQ_100_10_skew", "dQ_100_10_kurt",
                  "Qd_cycle2_Ah", "Qd_max_minus_cycle2_Ah"],
    "full": ["log10_dQ_var", "log10_abs_dQ_min", "fade_slope_2_100_Ah_per_cycle", "fade_intercept_2_100_Ah",
             "Qd_cycle2_Ah", "chargetime_mean_cycles2_6_min", "T_integral_2_100_degC_min",
             "IR_min_2_100_ohm", "IR_100_minus_2_ohm"],
    "capacity_only": ["Qd_cycle100_Ah", "fade_slope_2_100_Ah_per_cycle"],
}
SHORT_LIFE = 550          # cycles: "weak cell" flag for screening


def load():
    df = pd.read_csv(ROOT / "data" / "early_features.csv")
    df = df[df.paper_status.isin(["used", "used_merged"])].copy()
    df["life"] = np.where(df.paper_status == "used_merged", df.cycle_life_merged, df.cycle_life)
    df["T_integral_2_100_degC_min"] = df["T_integral_2_100_degC_min"].fillna(df["T_integral_2_100_degC_min"].median())
    # paper-style split: batches 1-2 interleaved into train / primary test; batch 3 (cycled ~1 year later) = secondary test
    b12 = df[df.batch.isin(["2017-05-12", "2017-06-30"])].sort_values(["batch", "cell_id"], key=lambda s: s.map(order) if s.name == "cell_id" else s)
    b12 = b12.reset_index(drop=True)
    df["split"] = "secondary_test"
    train_ids = set(b12.cell_id[b12.index % 2 == 1]) ; prim_ids = set(b12.cell_id[b12.index % 2 == 0])
    df.loc[df.cell_id.isin(train_ids), "split"] = "train"
    df.loc[df.cell_id.isin(prim_ids), "split"] = "primary_test"
    return df


def order(cell):
    return int(cell.split("c")[-1])


def cv_mape(model, tr, cols):
    """Model choice uses the 42 training cells only: 5-fold CV repeated 10 times, mean absolute % error."""
    from sklearn.base import clone
    errs = []
    for a, b in RepeatedKFold(n_splits=5, n_repeats=10, random_state=0).split(tr):
        A, B = tr.iloc[a], tr.iloc[b]
        lo, hi = A[cols].min(), A[cols].max()
        m = clone(model).fit(A[cols], np.log10(A.life))
        p = 10 ** m.predict(B[cols].clip(lo, hi, axis=1))
        errs.append(np.mean(np.abs(B.life - p) / B.life))
    return round(float(np.mean(errs) * 100), 1)


def scores(y, p):
    return {"rmse_cycles": round(float(np.sqrt(np.mean((y - p) ** 2))), 1),
            "mape_pct": round(float(np.mean(np.abs(y - p) / y) * 100), 1)}


def main():
    df = load()
    tr = df[df.split == "train"]
    out = {"cells": len(df), "split_sizes": df.split.value_counts().to_dict(),
           "life_range": [int(df.life.min()), int(df.life.max())], "life_mean": round(float(df.life.mean()), 1),
           "corr_logvar_loglife": round(float(np.corrcoef(df.log10_dQ_var, np.log10(df.life))[0, 1]), 3),
           "corr_capacity100_loglife": round(float(np.corrcoef(df.Qd_cycle100_Ah, np.log10(df.life))[0, 1]), 3),
           "models": {}}
    y_tr = np.log10(tr.life)
    preds = {}
    # naive baseline: predict the training mean life for every cell
    for name, cols in [("naive_mean", None)] + list(FEATURES.items()) + [("gbm_full", FEATURES["full"])]:
        if cols is None:
            f = lambda X: np.full(len(X), tr.life.mean())
        else:
            if name == "variance":
                m = LinearRegression()
            elif name == "gbm_full":
                m = GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0)
            else:
                m = make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.1, .5, .7, .9, .95, 1], cv=4, max_iter=50000, random_state=0))
            lo, hi = tr[cols].min(), tr[cols].max()          # clip inputs to the training range (guards bad sensor values)
            m.fit(tr[cols], y_tr)
            f = (lambda m, cols, lo, hi: (lambda X: 10 ** m.predict(X[cols].clip(lo, hi, axis=1))))(m, cols, lo, hi)
            out.setdefault("cv_train_mape_pct", {})[name] = cv_mape(m, tr, cols)
        res = {}
        for sp in ["train", "primary_test", "secondary_test"]:
            d = df[df.split == sp]
            p = f(d); res[sp] = scores(d.life.values, p)
            if sp != "train":
                preds.setdefault(name, {}).update(dict(zip(d.cell_id + "|" + d.batch, p)))
        out["models"][name] = res
    cv = out["cv_train_mape_pct"]
    out["chosen_by_train_cv"] = min(cv, key=cv.get)
    # screening: flag weak cells (life < 550) from the variance feature, threshold chosen on train
    thr_grid = np.quantile(tr.log10_dQ_var, np.linspace(0.05, 0.95, 91))
    def f1_at(t, d):
        flag = d.log10_dQ_var > t; weak = d.life < SHORT_LIFE
        tp = (flag & weak).sum(); prec = tp / max(flag.sum(), 1); rec = tp / max(weak.sum(), 1)
        return (2 * prec * rec / (prec + rec) if prec + rec else 0), prec, rec
    t = max(thr_grid, key=lambda t: f1_at(t, tr)[0])
    te = df[df.split != "train"]
    f1, prec, rec = f1_at(t, te)
    out["screening"] = {"weak_if_life_below": SHORT_LIFE, "test_cells": len(te), "weak_cells": int((te.life < SHORT_LIFE).sum()),
                        "precision": round(float(prec), 3), "recall": round(float(rec), 3), "f1": round(float(f1), 3),
                        "threshold_log10_var": round(float(t), 3)}
    out["test_time_saving_pct"] = round(100 * (1 - 100 / df.life.mean()), 1)
    json.dump(out, open(RES / "results.json", "w"), indent=2)
    rows = []
    for sp in ["primary_test", "secondary_test"]:
        for _, r in df[df.split == sp].iterrows():
            k = r.cell_id + "|" + r.batch
            rows.append({"split": sp, "batch": r.batch, "cell_id": r.cell_id, "policy": r.charge_policy_raw, "true_life": r.life,
                         **{f"pred_{n}": round(float(v[k]), 1) for n, v in preds.items()}})
    pd.DataFrame(rows).to_csv(RES / "test_predictions.csv", index=False)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
