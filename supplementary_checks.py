#!/usr/bin/env python3
"""
supplementary_checks.py
=======================
The two supplementary analyses reported in the paper, run on the same
data-generating processes, seeds and selected base-forest configuration
as run_simulation.py.

  trees       CRF refitted with B in {50, 100, 250, 500, 1000} trees.
              Reports the test slope, test RMSE and selected lambda, plus
              the out-of-bag and test slopes of the same forest at
              lambda = 0 (the uncorrected forest).

  downstream  The downstream association with the prediction used as a
              COVARIATE rather than as the response. With W = y + U as in
              run_simulation.py, reports
                  slope(W on yhat) - slope(W on y)
              for CRF and for the same grown forest at lambda = 0, next
              to the response-role distortion reported in Table I,
                  slope(yhat on W) - slope(y on W).

USAGE
    python supplementary_checks.py trees 20         # 20 replications
    python supplementary_checks.py downstream 100   # 100 replications

Output goes to supplementary_out/ beside this file: one CSV of
per-replication rows, and a mean (SD) summary printed to the console.
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_simulation as rs                       # noqa: E402

OUT_DIR = os.path.join(HERE, "supplementary_out")
TREE_GRID = (50, 100, 250, 500, 1000)


def _slope(x, y):
    """OLS slope of y on x."""
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def _replication_data(dgp, sim):
    """The data of replication `sim`, drawn exactly as run_one() draws it."""
    rng = np.random.RandomState(rs.SEED_BASE + sim)
    X_tr, y_tr = rs.gen_data(dgp, rs.N_TRAIN, rng)
    X_te, y_te = rs.gen_data(dgp, rs.N_TEST, rng)
    w_rng = np.random.RandomState(rs.W_SEED_BASE + sim)
    sd_tr = float(np.std(y_tr))
    rs.make_w(y_tr, w_rng, sd_reference=sd_tr)    # advance the W stream
    w_te = rs.make_w(y_te, w_rng, sd_reference=sd_tr)
    return X_tr, y_tr, X_te, y_te, w_te


def _one_trees(task):
    dgp, sim, forest_kw, n_trees = task
    X_tr, y_tr, X_te, y_te, _ = _replication_data(dgp, sim)
    f = rs.fit_crf("CRF", X_tr, y_tr, sim,
                   dict(forest_kw, n_estimators=n_trees))
    p = f.predict(X_te)
    row = dict(dgp=dgp, sim=sim, B=n_trees, lam=f.lam_used,
               eta=rs.eta_slope(y_te, p), rmse=rs.rmse(y_te, p),
               oob_eta_lam0=f.info.get("oob_eta_0"))
    f.set_lambda(0.0)
    row["test_eta_lam0"] = rs.eta_slope(y_te, f.predict(X_te))
    return [row]


def _one_downstream(task):
    dgp, sim, forest_kw = task
    X_tr, y_tr, X_te, y_te, w_te = _replication_data(dgp, sim)
    f = rs.fit_crf("CRF", X_tr, y_tr, sim, forest_kw)
    lam = f.lam_used
    rows = []
    for method, lam_at in (("CRF", lam), ("Uncorrected (lambda=0)", 0.0)):
        f.set_lambda(lam_at)
        p = f.predict(X_te)
        rows.append(dict(
            dgp=dgp, sim=sim, method=method, lam=lam_at,
            eta=rs.eta_slope(y_te, p),
            delta_response=_slope(w_te, p) - _slope(w_te, y_te),
            delta_covariate=_slope(p, w_te) - _slope(y_te, w_te)))
    return rows


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("trees", "downstream"):
        print(__doc__)
        sys.exit(1)
    mode = sys.argv[1]
    n_reps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    t0 = time.time()

    # The same base-forest selection the study uses (pilot data only).
    tuned = {d: rs.select_forest_params(d) for d in rs.DGPS}

    if mode == "trees":
        tasks = [(d, s, tuned[d], b) for d in rs.DGPS
                 for s in range(n_reps) for b in TREE_GRID]
        fn, keys = _one_trees, ["dgp", "B"]
        cols = ["eta", "rmse", "lam", "oob_eta_lam0", "test_eta_lam0"]
    else:
        tasks = [(d, s, tuned[d]) for d in rs.DGPS for s in range(n_reps)]
        fn, keys = _one_downstream, ["dgp", "method"]
        cols = ["eta", "delta_response", "delta_covariate"]

    rows = []
    with ProcessPoolExecutor(max_workers=rs.N_WORKERS) as ex:
        for res in ex.map(fn, tasks):
            rows += res

    os.makedirs(OUT_DIR, exist_ok=True)
    df = pd.DataFrame(rows)
    path = os.path.join(OUT_DIR, "supplementary_%s.csv" % mode)
    df.to_csv(path, index=False)
    print(df.groupby(keys)[cols].agg(["mean", "std"]).round(3).to_string())
    print("wrote %s (%.0f s)" % (path, time.time() - t0))


if __name__ == "__main__":
    main()
