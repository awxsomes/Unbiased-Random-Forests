#!/usr/bin/env python3
"""
run_padding.py
==============
Runs the POST-HOC "padding" correction -- the staged residual forest that
appends the running prediction to the design matrix as an extra ("padded")
feature -- over the same three DGPs used by the CRF study, and writes
output CSVs in exactly the schema make_figures.py consumes. The
data-generating processes below are the ones in run_simulation.py,
at the same seeds, so replication s of this study is fitted on the
same sample the CRF study saw and the two sets of columns are
directly comparable.

    Stage 0 :  f0 = RF(X -> y)
    Stage k :  X_k = [X, f_{k-1}] ;  f_k = f_{k-1} + RF(X_k -> y - f_{k-1})

Stages are no longer a fixed count. They run until the correction stops
moving -- a negligible stage increment, or no improvement in the training
residual, for STAGE_PATIENCE consecutive stages -- with MAX_STAGES as a
hard cap. Which of the four ended a fit is recorded per row as
stop_reason ('converged', 'stalled', 'diverged' or 'max_stages'), so a
run that hit the cap is never mistaken for one that settled. Replications
therefore differ in length: see the note in make_figures.py about what
that does to an averaged stage trajectory.

Two labelled variants are produced, so both appear as their own column in
the tables and their own panel in the figures:

    Padding      -- in-sample residual targets and in-sample padded feature
                    (faithful to the original ForestWithoutBiasSKLearn code)
    Padding-OOB  -- out-of-bag predictions supply both the padded feature and
                    the residual target at every stage, so the correction is
                    fit on honest residuals rather than in-bag ones

DIRECTORY LAYOUT
    <root>/                  written by run_simulation.py:
                             predictions_sim0.csv,
                             results_per_sim.csv, lambda_sweep.csv
        padding/             written by this script
        restacked_figures/   written by make_figures.py

USAGE
    python run_padding.py                    # writes ./sim_out/padding/
    python run_padding.py /path/to/sim_out   # writes <that>/padding/

OUTPUTS  (all in <root>/padding/)
    predictions_sim0.csv    per-observation predictions, replication 0
    results_per_sim.csv     per-replication metrics, both splits
    stage_metrics.csv       per-stage trajectory of every metric, plus
                            rel_step (the size of each stage's increment)
    summary_mean_sd.csv     mean and SD over replications
"""

import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# ==================================================================
# CONFIG
# ==================================================================
ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_out")
OUT_DIR = os.path.join(ROOT, "padding")

N_SIMS = 100
N_TRAIN = 500
N_TEST = 500
P = 200
SEED_BASE = 42          # replication s uses SEED_BASE + s
BETA_SEED = 20260724    # fixed linear coefficient vector, drawn once

# Padded-forest hyperparameters (from the original script)
N_TREES_BASE = 50
N_TREES_STAGE = 30
MAX_DEPTH = 6
MIN_SAMPLES_SPLIT = 2

# ---- stopping rule ----------------------------------------------
# Stages run until the correction stops moving, rather than for a fixed
# count. Two things can end a fit, whichever comes first:
#
#   converged  the stage increment is negligible --
#                  RMS(f_k - f_{k-1}) / sd(y_train) < STAGE_TOL
#              for STAGE_PATIENCE consecutive stages. One quiet stage is
#              not enough: a residual forest can add almost nothing at
#              one stage and a great deal at the next.
#   stalled    the training residual RMSE has not improved by at least
#              STAGE_MIN_DELTA (relative) for STAGE_PATIENCE stages.
#              This is what actually stops the OOB variant, whose
#              increments do not vanish -- out-of-bag residuals keep
#              carrying noise for the next stage to chase, so it wanders
#              rather than converging, and without this guard it would
#              run to MAX_STAGES every time.
#
# MAX_STAGES is the hard cap. Reaching it is reported, not silently
# accepted: a fit that stops there has not converged.
#
# COST. This replaces a fixed 10 stages with a data-dependent count that
# has run 25-45 in testing, so budget several times the old runtime for
# the same N_SIMS x DGPs x methods grid, and note that each fit is now a
# different length. Lower MAX_STAGES first if a run has to fit in a
# fixed window; STAGE_PATIENCE is the next knob.
MAX_STAGES = 2500
STAGE_TOL = 1e-4          # relative RMS of the stage increment
STAGE_MIN_DELTA = 1e-4    # relative train-RMSE improvement that counts
STAGE_PATIENCE = 25       # consecutive quiet stages before stopping

METHOD_ORDER = ("Padding", "Padding-OOB")
DGP_ORDER = ("linear_sparse", "friedman1", "friedman2")

# ==================================================================
# DGPs
# ==================================================================
# These are the DGPs of run_simulation.py, transcribed verbatim -- same
# functions, same fixed coefficient vector at the same BETA_SEED, same
# noise scale, and drawn from the same RandomState stream at the same
# replication seed. Replication s of this study therefore sees exactly
# the sample replication s of the CRF study saw, which is what makes the
# Padding columns comparable to the CRF column cell by cell. If the DGP
# block in run_simulation.py changes, copy it here again.

LINEAR_K = 10            # nonzero coefficients in the 'linear_sparse' DGP
R2_POP = 0.97            # population R^2, pins ||beta|| given SIGMA_E
SIGMA_E = 1.0


def make_beta(k=None, seed=BETA_SEED):
    """Fixed coefficient vector. k=None gives the dense design (every
    coefficient nonzero); k=int zeroes all but k of them. Either way the
    vector is rescaled so Var(X beta) / (Var(X beta) + SIGMA_E^2) equals
    R2_POP, so dense and sparse carry the SAME total signal -- only its
    concentration differs."""
    rng = np.random.RandomState(seed)
    beta = rng.randn(P)
    if k is not None:
        keep = rng.choice(P, int(k), replace=False)
        mask = np.zeros(P, dtype=bool)
        mask[keep] = True
        beta = np.where(mask, beta, 0.0)
    # X ~ MVN(0, I) => Var(X beta) = ||beta||^2; rescale to hit R2_POP
    target = R2_POP / (1.0 - R2_POP) * SIGMA_E ** 2
    return beta * np.sqrt(target / np.sum(beta ** 2))


BETA = make_beta()                     # dense
BETA_SPARSE = make_beta(k=LINEAR_K)    # LINEAR_K nonzero


def gen_data(dgp, n, rng):
    if dgp == "linear":
        X = rng.randn(n, P)
        f = X @ BETA
    elif dgp == "linear_sparse":
        X = rng.randn(n, P)
        f = X @ BETA_SPARSE
    elif dgp == "friedman1":
        X = rng.rand(n, P)
        f = (0.1 * np.exp(4.0 * X[:, 0])
             + 4.0 / (1.0 + np.exp(-20.0 * (X[:, 1] - 0.5)))
             + 3.0 * X[:, 2] + 2.0 * X[:, 3] + X[:, 4])
    elif dgp == "friedman2":
        X = rng.rand(n, P)
        f = (10.0 * np.sin(np.pi * X[:, 0] * X[:, 1])
             + 20.0 * (X[:, 2] - 0.5) ** 2
             + 10.0 * X[:, 3] + 5.0 * X[:, 4])
    else:
        raise ValueError(dgp)
    return X, f + SIGMA_E * rng.randn(n)


# ==================================================================
# METRICS  (identical definitions to run_simulation.py and
#           make_figures.py)
# ==================================================================
def eta_slope(y_true, y_pred):
    """Slope of the regression of yhat on y; reference value 1."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.cov(y_true, y_pred, bias=True)[0, 1] / np.var(y_true))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred, float)
                                  - np.asarray(y_true, float)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred, float)
                                - np.asarray(y_true, float))))


def group_residuals(y_true, y_pred):
    """Mean prediction error above and below the outcome mean."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    mu = float(np.mean(y_true))
    upper, lower = y_true >= mu, y_true < mu
    return (float(np.mean(y_pred[upper] - y_true[upper])),
            float(np.mean(y_pred[lower] - y_true[lower])))


def r2_pearson(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    sd = np.std(y_true) * np.std(y_pred)
    if sd < 1e-12:
        return np.nan
    return float(np.cov(y_true, y_pred, bias=True)[0, 1] / sd) ** 2


def r2_ols(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    if sst < 1e-12:
        return np.nan
    return 1.0 - float(np.sum((y_pred - y_true) ** 2)) / sst


def theta_slope(w, v):
    w, v = np.asarray(w, float), np.asarray(v, float)
    return float(np.cov(w, v, bias=True)[0, 1] / np.var(w))


def theta_delta(y_true, y_pred, w):
    """Downstream association slope, differenced against the oracle."""
    return theta_slope(w, y_pred) - theta_slope(w, y_true)


def compute_metrics(y_true, y_pred, w=None):
    r_u, r_l = group_residuals(y_true, y_pred)
    return dict(bias=eta_slope(y_true, y_pred),
                rmse=rmse(y_true, y_pred),
                r2_pearson=r2_pearson(y_true, y_pred),
                r2_ols=r2_ols(y_true, y_pred),
                mae=mae(y_true, y_pred),
                theta_delta=(np.nan if w is None
                             else theta_delta(y_true, y_pred, w)),
                r_U=r_u, r_L=r_l)


W_SEED_BASE = 42    # W draws, kept disjoint from SEED_BASE


def make_w(y, rng, sd_reference=None):
    """Noisy outcome proxy W = y + U(-a, a) with a = sqrt(3) sd(y), so
    Var(W - y) = Var(y) and the population slope of y on W is 0.5.

    sd_reference is passed the TRAINING sd for both splits, so the noise
    level is identical on train and test rather than following each
    split's own dispersion. Both the definition and the seeding follow
    run_simulation.py, so the two studies face the same realised W.
    """
    y = np.asarray(y, float)
    sd = float(np.std(y)) if sd_reference is None else float(sd_reference)
    a = np.sqrt(3.0) * sd
    return y + rng.uniform(-a, a, size=len(y))


# ==================================================================
# PADDED RESIDUAL FOREST
# ==================================================================
class PaddedForest:
    """Staged residual forest with the running prediction as an extra column.

    use_oob=True fits every stage against out-of-bag residuals and pads with
    out-of-bag predictions, so the residual target the stage forest chases is
    not the in-bag residual it can already interpolate.
    """

    def __init__(self, n_trees_base=N_TREES_BASE, n_trees_stage=N_TREES_STAGE,
                 max_depth=MAX_DEPTH, min_samples_split=MIN_SAMPLES_SPLIT,
                 max_stages=MAX_STAGES, tol=STAGE_TOL,
                 min_delta=STAGE_MIN_DELTA, patience=STAGE_PATIENCE,
                 use_oob=False, random_state=42):
        self.n_trees_base = n_trees_base
        self.n_trees_stage = n_trees_stage
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_stages = max_stages
        self.tol = tol
        self.min_delta = min_delta
        self.patience = patience
        self.use_oob = use_oob
        self.random_state = random_state
        self.models = []
        self.stage_trace = []   # one dict per stage, train and test metrics
        self.n_stages_ = None   # stages actually run, base stage included
        self.stop_reason_ = None

    def _forest(self, n_trees, seed):
        return RandomForestRegressor(
            n_estimators=n_trees,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features="sqrt",
            bootstrap=True,
            oob_score=self.use_oob,
            random_state=seed,
            n_jobs=-1,
        )

    def _in_sample(self, model, X_design):
        """Padded feature / residual source: OOB when requested, else in-bag."""
        if self.use_oob:
            oob = np.asarray(model.oob_prediction_, float)
            if np.isnan(oob).any():          # a few rows can miss OOB coverage
                fitted = model.predict(X_design)
                oob = np.where(np.isnan(oob), fitted, oob)
            return oob
        return model.predict(X_design)

    def fit(self, X, y, X_test=None, y_test=None):
        self.models = []
        self.stage_trace = []

        base = self._forest(self.n_trees_base, self.random_state)
        base.fit(X, y)
        self.models.append(base)

        train_pred = self._in_sample(base, X)
        residuals = y - train_pred
        test_pred = base.predict(X_test) if X_test is not None else None
        self._record_stage(0, y, train_pred, y_test, test_pred)

        sd_y = float(np.std(y)) or 1.0
        best_rmse = float(np.sqrt(np.mean(residuals ** 2)))
        quiet = 0                      # consecutive stages that moved nothing
        no_gain = 0                    # consecutive stages that gained nothing
        self.stop_reason_ = "max_stages"

        for stage in range(1, self.max_stages):
            X_padded = np.column_stack([X, train_pred])
            model = self._forest(self.n_trees_stage, self.random_state + stage)
            model.fit(X_padded, residuals)
            self.models.append(model)

            increment = self._in_sample(model, X_padded)
            train_pred = train_pred + increment
            residuals = y - train_pred
            if X_test is not None:
                test_pred = test_pred + model.predict(
                    np.column_stack([X_test, test_pred]))

            # How much this stage moved, and whether it bought anything.
            rel_step = float(np.sqrt(np.mean(increment ** 2))) / sd_y
            stage_rmse = float(np.sqrt(np.mean(residuals ** 2)))
            gained = (best_rmse - stage_rmse) / max(best_rmse, 1e-12)
            if gained > self.min_delta:
                best_rmse = stage_rmse
                no_gain = 0
            else:
                no_gain += 1
            quiet = quiet + 1 if rel_step < self.tol else 0

            self._record_stage(stage, y, train_pred, y_test, test_pred,
                               rel_step=rel_step)

            if not np.isfinite(stage_rmse):
                self.stop_reason_ = "diverged"
                break
            if quiet >= self.patience:
                self.stop_reason_ = "converged"
                break
            if no_gain >= self.patience:
                self.stop_reason_ = "stalled"
                break

        self.n_stages_ = len(self.models)
        self.train_pred_ = train_pred
        return self

    def _record_stage(self, stage, y, train_pred, y_test, test_pred,
                      rel_step=np.nan):
        row = {"stage": stage, "rel_step": rel_step}
        for key, value in compute_metrics(y, train_pred).items():
            row["train_" + key] = value
        if y_test is not None and test_pred is not None:
            for key, value in compute_metrics(y_test, test_pred).items():
                row["test_" + key] = value
        self.stage_trace.append(row)

    def predict(self, X):
        pred = self.models[0].predict(X)
        for model in self.models[1:]:
            pred = pred + model.predict(np.column_stack([X, pred]))
        return pred


# ==================================================================
# STUDY LOOP
# ==================================================================
def run_replication(dgp_name, sim, collect_predictions):
    """One replication: fit both padding variants, return metric rows."""
    rng = np.random.RandomState(SEED_BASE + sim)
    X_train, y_train = gen_data(dgp_name, N_TRAIN, rng)
    X_test, y_test = gen_data(dgp_name, N_TEST, rng)

    # W is drawn from its own stream, at the TRAINING sd on both splits.
    w_rng = np.random.RandomState(W_SEED_BASE + sim)
    sd_train = float(np.std(y_train))
    w_train = make_w(y_train, w_rng, sd_reference=sd_train)
    w_test = make_w(y_test, w_rng, sd_reference=sd_train)

    metric_rows, stage_rows, prediction_rows = [], [], []

    for method, use_oob in (("Padding", False), ("Padding-OOB", True)):
        model = PaddedForest(use_oob=use_oob,
                             random_state=SEED_BASE + sim)
        model.fit(X_train, y_train, X_test, y_test)
        if sim == 0:
            # One line per method on the first replication only: with
            # N_SIMS replications this would otherwise be hundreds of
            # lines. The per-DGP summary below reports the whole set.
            print("    %-12s %-14s stopped after %d stages (%s)"
                  % (method, dgp_name, model.n_stages_, model.stop_reason_),
                  flush=True)

        train_pred = model.train_pred_
        test_pred = model.predict(X_test)

        for split, y_obs, y_hat, w_obs in (("Train", y_train, train_pred, w_train),
                                           ("Test", y_test, test_pred, w_test)):
            row = {"dgp": dgp_name, "method": method, "split": split,
                   "sim": sim, "lam": np.nan,
                   "n_stages": model.n_stages_,
                   "stop_reason": model.stop_reason_}
            row.update(compute_metrics(y_obs, y_hat, w_obs))
            metric_rows.append(row)

            if collect_predictions:
                prediction_rows.append(pd.DataFrame({
                    "dgp": dgp_name, "method": method, "split": split,
                    "obs_index": np.arange(len(y_obs)),
                    "y_obs": y_obs, "y_pred": y_hat, "w_obs": w_obs,
                    "lam": np.nan}))

        for trace in model.stage_trace:
            stage_rows.append(dict(dgp=dgp_name, method=method, sim=sim,
                                   stop_reason=model.stop_reason_, **trace))

    return metric_rows, stage_rows, prediction_rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=" * 60)
    print("Padding (post-hoc) forest study")
    print("output directory: %s" % OUT_DIR)
    print("%d replications x %d DGPs x %d methods, p = %d"
          % (N_SIMS, len(DGP_ORDER), len(METHOD_ORDER), P))
    print("stages: run to convergence, tol=%g, patience=%d, cap=%d"
          % (STAGE_TOL, STAGE_PATIENCE, MAX_STAGES))
    print("=" * 60)

    all_metrics, all_stages, all_predictions = [], [], []

    for dgp_name in DGP_ORDER:
        started = time.time()
        for sim in range(N_SIMS):
            metric_rows, stage_rows, prediction_rows = run_replication(
                dgp_name, sim, collect_predictions=(sim == 0))
            all_metrics.extend(metric_rows)
            all_stages.extend(stage_rows)
            all_predictions.extend(prediction_rows)
            if (sim + 1) % 10 == 0 or sim == 0:
                print("  %-14s replication %3d/%d  (%.1fs elapsed)"
                      % (dgp_name, sim + 1, N_SIMS, time.time() - started))

        done = [r for r in all_metrics
                if r["dgp"] == dgp_name and r["split"] == "Test"]
        for method in METHOD_ORDER:
            rows = [r for r in done if r["method"] == method]
            slopes = [r["bias"] for r in rows]
            stages = [r["n_stages"] for r in rows]
            reasons = {}
            for r in rows:
                reasons[r["stop_reason"]] = reasons.get(r["stop_reason"], 0) + 1
            why = ", ".join("%s x%d" % (k, v)
                            for k, v in sorted(reasons.items()))
            print("  %-14s %-12s test slope mean %.3f (SD %.3f)   "
                  "stages mean %.1f, range %d-%d   [%s]"
                  % (dgp_name, method, np.mean(slopes), np.std(slopes),
                     np.mean(stages), min(stages), max(stages), why))

    results = pd.DataFrame(all_metrics)
    results.to_csv(os.path.join(OUT_DIR, "results_per_sim.csv"), index=False)

    stages = pd.DataFrame(all_stages)
    stages.to_csv(os.path.join(OUT_DIR, "stage_metrics.csv"), index=False)

    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(os.path.join(OUT_DIR, "predictions_sim0.csv"),
                       index=False)

    metric_cols = ["bias", "rmse", "r2_pearson", "r2_ols", "mae",
                   "theta_delta", "r_U", "r_L"]
    summary = (results.groupby(["dgp", "method", "split"])[metric_cols]
               .agg(["mean", "std"]))
    summary.columns = ["%s_%s" % (a, b) for a, b in summary.columns]
    summary.reset_index().to_csv(
        os.path.join(OUT_DIR, "summary_mean_sd.csv"), index=False)

    print("\nwrote results_per_sim.csv, stage_metrics.csv, "
          "predictions_sim0.csv, summary_mean_sd.csv")
    print("next: python make_figures.py %s" % ROOT)


if __name__ == "__main__":
    main()
