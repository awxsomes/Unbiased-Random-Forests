#!/usr/bin/env python3
"""
run_simulation.py
=================
Systematic-prediction-bias (SPB) correction for random forests by
lambda-scaled leaf adjustment, plus the three-DGP simulation study, in a
single self-contained file.

WHAT IS FITTED AND REPORTED
---------------------------
  * ONE constrained method is fitted and reported, 'CRF': the
    forest-level lambda correction with lambda selected so that the OOB
    calibration slope equals 1. This is the only correction implemented
    -- the per-tree variants that preceded it (an iterated per-tree
    update and a greedy per-leaf heuristic, plus their shared closed
    form) were dropped once the forest-level correction superseded
    them; see the CORRECTIONS section below.
  * Baselines are RF, NN and XGBoost, all at the hyperparameters used in
    the constrained-neural-network code (RF_PARAMS, NN_PARAMS,
    XGB_PARAMS below). Lasso remains implemented but is not fitted.
  * theta_delta (the downstream-association distortion) is computed and
    reported on ALL THREE DGPs, not only the linear one.
  * Outputs include a paper-shaped results table as CSV and as
    LaTeX, alongside the per-replication file, the summary file, the
    prediction files and the figures.
  * The CRF forest is grown with N_TREES = 500 trees.
  * The post-hoc PADDING comparator is fitted in the same study,
    as a further method ('Padding'). It is not a constrained forest
    and takes no lambda: it is the staged residual forest the paper
    contrasts the constraint with, run on the same replication's
    data so that its column compares with the CRF column cell by
    cell.

-------------------------------------------------------------------
THE CONSTRAINTS
-------------------------------------------------------------------
For predictions f(X_i), outcomes y_i, and the outcome mean ybar on a
calibration sample, the two group-mean (stratum) constraints are

    r_U = mean_{i in U} ( f(X_i) - y_i ) = 0 ,   U = { i : y_i >= ybar }
    r_L = mean_{i in L} ( f(X_i) - y_i ) = 0 ,   L = { i : y_i <  ybar }

A tree's prediction is LINEAR in its leaf values v, since
    r_U = sum_l w_l^U v_l - ybar_U ,   w_l^U = n_l^U / |U| ,
so both residuals are linear in v and every correction below is a
descent step on (1/2)(r_U^2 + r_L^2) in leaf space. All corrections act
on leaf values individually -- none of them is an affine map of the
predictions.

-------------------------------------------------------------------
CORRECTIONS  (config: CORRECTION)
-------------------------------------------------------------------
'forest_lambda' THE METHOD REPORTED. A lambda-scaled leaf update driven
                by the residuals of the AGGREGATED out-of-bag
                prediction rather than tree by tree. With
                    yhat_i^OOB = (1/|B_i|) sum_{b: i in O_b} f_b(X_i),
                    r_U = mean_{i in U} ( yhat_i^OOB - y_i ),
                the derivative of r_U with respect to leaf value v_{bl}
                is
                    omega_{bl}^U = (1/|U|) sum_{i in U, i in O_b,
                                    X_i in R_bl} 1 / |B_i| ,
                and the update over ALL leaves of ALL trees is
                    v_{bl} <- v_{bl}
                              - lam ( r_U omega_{bl}^U
                                    + r_L omega_{bl}^L ).
                Here lam IS a meaningful knob, because it is applied to
                the full projection step Omega^T (Omega Omega^T)^{-1} r
                rather than to a raw gradient:
                    lam = 0  no correction,
                    lam = 1  exact satisfaction of both constraints on
                             the aggregated OOB predictions,
                    lam > 1  deliberate overcorrection.
                Overcorrection is useful: OOB feasibility does not
                transfer intact to new data (the correction is fitted on
                OOB residuals, and test-set attenuation is larger), so
                the test-set optimum sits above 1.

                Why forest-level rather than per-tree: each tree has its
                own OOB sample, so per-tree feasibility does not compose
                into feasibility for the averaged forest. Correcting the
                aggregate addresses exactly the quantity that is
                evaluated. (An earlier per-tree version confirmed this
                empirically -- three independent derivations of a
                per-tree update, an iterated step, its greedy heuristic,
                and their shared closed form, all converged to the same
                leaves but did not compose into forest-level
                feasibility -- and was removed once the point was made.)

'none'          Plain forest, no correction.

-------------------------------------------------------------------
THE PADDING COMPARATOR  (config: PADDING_VARIANTS)
-------------------------------------------------------------------
A post-hoc alternative to constraining the forest: a staged
residual forest that appends the running prediction to the design
matrix as an extra ('padded') feature.

    Stage 0 :  f0 = RF(X -> y)
    Stage k :  X_k = [X, f_{k-1}] ;
               f_k = f_{k-1} + RF(X_k -> y - f_{k-1})

Stages are not a fixed count. They run until the correction stops
moving -- a negligible stage increment, or no improvement in the
training residual, for PAD_STAGE_PATIENCE consecutive stages --
with PAD_MAX_STAGES as a hard cap. Which of these ended a fit is
recorded per row as stop_reason ('converged', 'stalled',
'diverged' or 'max_stages'), so a fit that hit the cap is never
mistaken for one that settled. Replications therefore differ in
length, and the per-stage trajectory of every metric is written
to stage_metrics.csv.

'Padding'       THE METHOD REPORTED, and the only one implemented.
                In-sample residual targets and in-sample padded
                feature at every stage, faithful to the original
                code of Zhang and Lu (2012). (An out-of-bag variant,
                which fit every stage against honest OOB residuals
                instead, was tried and dropped: never reported, so
                removed from the file rather than carried as an
                unused branch.)

ALG 3 (stratum-balance constrained splits during growth) is available as
CONSTRAINED_SPLITS = True with BACKEND = 'custom'.

-------------------------------------------------------------------
CHOOSING LAMBDA  (config: LAMBDA_SELECT)
-------------------------------------------------------------------
'oob_slope'  THE SELECTION REPORTED. Pick lam so that the OOB
             calibration slope eta-hat equals 1. This uses training data
             only, so it is legitimate for reporting. Because the
             corrected predictions are affine in lam, so is the OOB
             slope, giving a closed form from two evaluations:
                 lam* = (1 - eta_0) / (eta_1 - eta_0).
             Selected values run about 1.1-1.4 on these DGPs.

             Note the honest caveat: OOB slope is optimistic relative to
             the test slope at the same lam, so lam* chosen this way
             lands the test slope below 1. Choosing a larger lam by
             looking at test slope would not be a legitimate procedure,
             and this file does not do it. lambda_sweep() reports both
             curves so the gap is visible.
'fixed'      use LAM as given. LAM = 1.0 is exact constraint
             satisfaction on OOB, with no tuning.
'oob_cv'     the same slope-matching target, with the step cross-fitted
             over K folds so the slope is read on points the step was
             not built from.

-------------------------------------------------------------------
BACKENDS  (config: BACKEND)
-------------------------------------------------------------------
'sklearn'  bagged sklearn DecisionTreeRegressor, with bootstrap indices
           drawn here so exact OOB sets are available. sklearn stores
           leaf predictions at tree_.value[leaf, 0, 0] and reads them at
           predict time, so in-place edits there ARE the leaf
           correction. About 7.6x faster than 'custom' at this problem
           size. DEFAULT.
'custom'   pure-NumPy trees written here: the readable reference
           implementation, and required for ALG 3.

There are no GPU backends. Every correction here rewrites individual
leaf values, and neither cuML nor XGBoost exposes writable leaves from
Python, so a device-side forest cannot be corrected this way at all.
Parallelism over replications (N_WORKERS) is the speedup that matters.

-------------------------------------------------------------------
SIMULATION STUDY
-------------------------------------------------------------------
Three DGPs, N_SIMS replications each, n_train = n_test = 500, p = 200:

  linear    : X ~ MVN(0, I_p); y = X beta + eps, eps ~ N(0, SIGMA_E^2).
              beta fixed across replications (drawn once at BETA_SEED),
              rescaled so the population R^2 equals R2_POP. Every
              coefficient is nonzero: the dense design. 'linear_sparse'
              carries the same total signal in LINEAR_K coefficients.
  friedman1 : X ~ U[0,1]^p;
              f(X) = 0.1 exp(4 X1) + 4 / (1 + exp(-20 (X2 - 0.5)))
                     + 3 X3 + 2 X4 + X5 + 0 * sum_{j=6}^{200} Xj
  friedman2 : X ~ U[0,1]^p;
              f(X) = 10 sin(pi X1 X2) + 20 (X3 - 0.5)^2
                     + 10 X4 + 5 X5 + 0 * sum_{j=6}^{200} Xj

Methods: CRF (this method) / Padding (the post-hoc comparator) / RF
(sklearn RandomForestRegressor) / NN (sklearn MLPRegressor) / XGBoost
(xgboost XGBRegressor).
Metrics: eta = OLS slope of yhat on y (target 1), RMSE, MAE, and
theta_hat - theta_oracle (target 0); mean and SD over replications, on
Train and Test rows.

-------------------------------------------------------------------
OUTPUTS  (all in OUT_DIR)
-------------------------------------------------------------------
    results_per_sim.csv     every replication x method x split x metric
    summary_mean_sd.csv     mean and SD of every metric, unformatted
    results_table.csv       paper-shaped table, 'mean (SD)' cells
    results_table.tex       the same table as LaTeX (booktabs)
    predictions_sim<k>.csv  the y / yhat / W vectors behind the figures
    all_predictions.csv     every replication's vectors (SAVE_ALL_PRED)
    lambda_sweep.csv        OOB and test curves against lambda
    stage_metrics.csv       per-stage trajectory of the padding
                            comparator, plus rel_step
    run_config.txt          the configuration and versions of this run
    <dgp>/*.png             the figures, one directory per DGP

-------------------------------------------------------------------
USAGE
-------------------------------------------------------------------
    python run_simulation.py                  # full study
    python run_simulation.py 10               # short run, 10 replications
    python run_simulation.py --figures        # redraw from saved results
    python run_simulation.py --tables         # rebuild tables from the CSV
    python run_simulation.py --sweep          # lambda sweep, all DGPs
    CRF_SMOKE=1 python run_simulation.py      # 2 reps, tiny budgets

    # Partial runs. Both flags restrict what is FITTED; the results are
    # merged into the existing files, so rows this run did not touch are
    # carried over rather than refitted. Adding one method to a finished
    # study therefore costs one method, not the whole study:
    python run_simulation.py --methods=XGBoost
    python run_simulation.py --dgps=linear_sparse
    python run_simulation.py --dgps=friedman1,friedman2 --methods=XGBoost

    Merging is only valid while the seeds and the DGP definitions are
    unchanged -- see merge_csv() for exactly what must hold.

Spawn-safe (Windows / macOS). Python <= 3.11.
"""

import os
import sys
import time
from dataclasses import dataclass
from typing import Optional

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

from sklearn.tree import DecisionTreeRegressor as SkTree
from sklearn.ensemble import RandomForestRegressor as SkRF
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler


# ==================================================================
# CONFIG
# ==================================================================

OUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         "sim_out")            # <- set to your own path

# ---- study -------------------------------------------------------
N_SIMS    = 100
N_TRAIN   = 500
N_TEST    = 500
P         = 200
SIGMA_E   = 1.0
R2_POP    = 0.97          # linear DGPs only: pins ||beta|| given SIGMA_E
BETA_SEED = 20260724      # linear beta drawn once, fixed across reps
LINEAR_K  = 10            # nonzero coefficients in the 'linear_sparse' DGP
SEED_BASE = 42            # replication s uses SEED_BASE + s

# 'linear' is the DENSE design: all 200 coefficients nonzero and
# comparable in size, so no single feature carries usable marginal
# signal and axis-aligned trees cannot approximate the surface. The base
# forest lands near RMSE/sd(y) = 0.95 with Pearson r2 about 0.26, and
# its uncorrected test slope sits around 0.1, so the correction is
# rescaling a nearly flat prediction: eta moves a long way toward 1
# while RMSE rises. That is the setting the constrained-neural-network
# study reports, and it is what this file now runs.
#
# 'linear_sparse' is the alternative: everything else identical, but the
# same total signal concentrated in LINEAR_K coefficients, which is a
# design trees can actually fit. Swap the name below to run it instead,
# or list both to report them side by side.
DGPS      = ("linear_sparse", "friedman1", "friedman2")

# Display names. results_per_sim.csv and every other machine-readable
# file keep the raw DGP key, so a merged run stays unambiguous; the
# figures, directories and tables use the name below. 'linear_sparse'
# is therefore reported as 'linear' -- which is accurate as long as the
# manuscript's DGP description says the coefficient vector is sparse.
DGP_DISPLAY = {"linear_sparse": "linear"}
N_WORKERS = 8

# ---- the constrained forest --------------------------------------
N_TREES    = 500              # trees in the CRF (was 100)
BACKEND    = "sklearn"        # 'sklearn' | 'custom'
CORRECTION = "forest_lambda"  # default for a single-variant run
LAMBDA_SELECT = "oob_slope"   # 'fixed' | 'oob_slope' | 'oob_cv'
LAM        = 1.0              # used when LAMBDA_SELECT == 'fixed'

FOREST_PARAMS = dict(n_estimators=N_TREES, max_depth=5, min_samples_leaf=5,
                     mtry="sqrt", bootstrap=True)

# ---- base-forest selection ---------------------------------------
TUNE_FOREST   = True        # select depth and mtry before the study
TUNE_SEED     = 20260815    # pilot data seed; keep OUTSIDE
                            # SEED_BASE .. SEED_BASE + N_SIMS
TUNE_VAL_FRAC = 0.2         # held-out share of the pilot sample
# The pilot forests are grown with TUNE_N_ESTIMATORS trees rather than
# N_TREES: the choice being made is depth and mtry, which does not need
# the full ensemble, and the grid would otherwise dominate the runtime.
# The selected depth and mtry are then applied at N_TREES.
TUNE_N_ESTIMATORS = 100
TUNE_GRID = dict(
    max_depth=[5, 8, 12, None],
    min_samples_leaf=[5],
    mtry_frac=[0.07, 0.3, 1.0],   # fraction of P; 0.07*200 = 14 ~ 'sqrt'
)

# ---- which corrections to run as separate methods -------------------
# Every entry becomes its own column in the tables and its own panel in
# the figures, fitted on the SAME data as every other entry in the same
# replication. The study reports ONE: the forest-level correction with
# lambda selected on OOB slope. Add an entry back to fit it alongside.
#
#   correction='forest_lambda', lambda_select='oob_slope'   <- reported
#   correction='forest_lambda', lambda_select='fixed'       lam fixed
#   correction='forest_lambda', lambda_select='oob_cv'      cross-fitted
CRF_VARIANTS = {
    "CRF": dict(correction="forest_lambda", lambda_select="oob_slope"),
}
# Dropped from earlier runs, all deliberately:
#   the per-tree correction (iterated update, its closed form, and a
#     greedy per-leaf heuristic) -- per-tree feasibility does not
#     compose into feasibility for the averaged forest; removed from
#     the file entirely once the forest-level correction superseded it.
#   "ALG4-lam1"   -- lam = 1 is exact OOB satisfaction, which leaves the
#                    test slope short of 1; superseded by lamOOB.
#   "ALG4-lamCV"  -- cross-fitted selection, materially the same lam.

# ---- the post-hoc padding comparator --------------------------------
# The staged residual forest of the docstring, fitted as a further
# method on the same replication's data. The PAD_ prefix keeps its
# hyperparameters clear of the constrained forest's own settings above.
PADDING_VARIANTS = {
    "Padding": dict(),
}
# Dropped: an out-of-bag variant ('Padding-OOB'), which fit every stage
# against honest OOB residuals instead of in-sample ones. It addressed
# a real circularity concern -- fitting a residual model on data it can
# already interpolate -- but was never reported, so it was removed from
# PaddedForest entirely rather than kept as an unused code path.
PAD_TREES_BASE  = 50      # trees in the stage-0 forest
PAD_TREES_STAGE = 30      # trees in each residual forest after it
PAD_MAX_DEPTH = 6
PAD_MIN_SAMPLES_SPLIT = 2

# The stopping rule. Stages run until the correction stops moving rather
# than for a fixed count, which has been about 50 stages in the study, so the
# padding columns cost several times what a fixed-10-stage version did.
# Lower PAD_MAX_STAGES first if a run has to fit a fixed window;
# PAD_STAGE_PATIENCE is the next knob.
PAD_MAX_STAGES = 2500
PAD_STAGE_TOL = 1e-4         # relative RMS of the stage increment
PAD_STAGE_MIN_DELTA = 1e-4   # relative train-RMSE improvement that counts
PAD_STAGE_PATIENCE = 25      # consecutive quiet stages before stopping
SAVE_STAGE_TRACE = True      # write stage_metrics.csv

BASELINES = ("RF", "NN", "XGBoost")

# Baseline hyperparameters, taken from the constrained-neural-network
# code so that the two studies report the same competitors.
#
# RF: n_jobs is 1 here rather than -1, because replications are already
# spread across processes by N_WORKERS and nesting the two oversubscribes
# the machine. Everything that affects the fitted model is unchanged.
RF_PARAMS     = dict(n_estimators=500, n_jobs=1)
# NN: full-batch Adam with the loss-plateau stop removed (tol = 0,
# n_iter_no_change = max_iter), so the baseline runs its full budget
# rather than a data-dependent one. batch_size is set at fit time to the
# number of training rows.
NN_PARAMS     = dict(hidden_layer_sizes=(512, 128), activation="relu",
                     solver="adam", learning_rate_init=1e-3, alpha=1e-4,
                     max_iter=5000, n_iter_no_change=5000, tol=0.0,
                     early_stopping=False)
# XGBoost: deliberately untuned and identical across the three DGPs,
# at the settings the constrained-neural-network study used. n_jobs is
# pinned to 1 for the same reason as RF -- replications are already
# spread across processes.
XGB_PARAMS    = dict(n_estimators=500, learning_rate=0.05, max_depth=6,
                     subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                     n_jobs=1)
# Lasso is still implemented; add "Lasso" back to BASELINES to fit it.
LASSO_PARAMS  = dict(cv=5, max_iter=100000, tol=1e-4, n_jobs=1)

# The RF baseline runs at RF_PARAMS, i.e. 500 unpruned trees, whatever
# the CRF's own depth and mtry turn out to be. Set this True to grow it
# instead at the CRF's settings, which makes it the uncorrected control
# for the same forest rather than an independent competitor.
RF_MATCH_CRF = False

LAM_MAX    = 6.0              # cap on a selected lambda
TOL        = 1e-8             # convergence tol for iterated updates
MAX_ITER   = 20000
CONSTRAINED_SPLITS = False    # ALG 3 (BACKEND='custom' only)
SWEEP_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)

# CRF 'Train' row source. 'insample' is directly comparable to the
# baselines but distorted: the correction is calibrated on OOB
# predictions, which are far more shrunk than in-sample ones, so the
# fitted step overshoots in-sample. 'oob' reports the aggregated
# out-of-bag predictions instead -- the forest-native analogue of a
# validation split, i.e. the data the correction actually saw. The
# baselines always report in-sample.
CRF_TRAIN_ROW = "oob"

MAKE_FIGURES = True       # draw every figure at the end of run_study()

# Figures are written to OUT_DIR/<dgp>/, with per-tree diagnostics in
# OUT_DIR/<dgp>/tree/. TREE_FIGS_N sets how many individual trees get an
# observed-vs-predicted figure: 0 none, -1 every tree in the forest.
# Each shows one tree BEFORE and AFTER correction on the same axes, so
# the shrinkage of a single tree and what the leaf update does to it are
# both visible. TREE_FIG_METHOD picks which fitted variant supplies them
# (must be a forest_lambda variant, since only that one retains the
# uncorrected leaf values).
TREE_FIGS_N     = 12
TREE_FIG_METHOD = "CRF"
RUN_SWEEP    = True       # also run lambda_sweep() so lambda_curve_*.png exist
SWEEP_REPS   = 5          # replications per DGP in that sweep

# ---- output files ------------------------------------------------
PER_SIM_CSV      = "results_per_sim.csv"
SUMMARY_CSV      = "summary_mean_sd.csv"
RESULTS_TABLE_CSV = "results_table.csv"
RESULTS_TABLE_TEX = "results_table.tex"
SWEEP_CSV        = "lambda_sweep.csv"
STAGE_CSV        = "stage_metrics.csv"
ALL_PRED_CSV     = "all_predictions.csv"
RUN_CONFIG_TXT   = "run_config.txt"

# Save the y / yhat / W vectors, not only the scalar metrics, so that
# any metric not anticipated here can be recomputed later without
# refitting anything.
SAVE_PREDICTIONS     = True    # the replication behind the figures
SAVE_ALL_PREDICTIONS = os.environ.get("CRF_SAVE_ALL", "0") == "1"

METHOD_ORDER = (tuple(CRF_VARIANTS) + tuple(PADDING_VARIANTS)
                + BASELINES)

# ---- smoke test --------------------------------------------------
# A dry run that exercises every code path in about a minute, for
# checking a change before committing an overnight run to it.
SMOKE_TEST = os.environ.get("CRF_SMOKE", "0") == "1"
if SMOKE_TEST:
    N_SIMS = 2
    N_TREES = 40
    FOREST_PARAMS = dict(FOREST_PARAMS, n_estimators=N_TREES)
    TUNE_GRID = dict(max_depth=[5, None], min_samples_leaf=[5],
                     mtry_frac=[0.07])
    TUNE_N_ESTIMATORS = 20
    RF_PARAMS = dict(RF_PARAMS, n_estimators=40)
    XGB_PARAMS = dict(XGB_PARAMS, n_estimators=40)
    NN_PARAMS = dict(NN_PARAMS, max_iter=100, n_iter_no_change=100)
    LASSO_PARAMS = dict(LASSO_PARAMS, cv=3)
    SWEEP_REPS = 2
    TREE_FIGS_N = 2
    PAD_TREES_BASE, PAD_TREES_STAGE = 20, 15
    PAD_MAX_STAGES = 40


# ==================================================================
# CUSTOM TREE  (reference implementation; required for ALG 3)
# ==================================================================

@dataclass
class Node:
    feature: Optional[int] = None
    threshold: Optional[float] = None
    left: Optional['Node'] = None
    right: Optional['Node'] = None
    value: Optional[float] = None
    node_id: Optional[int] = None

    def is_leaf(self):
        return self.value is not None


def _child_seeds(base, b, n=2):
    """n independent seeds derived from (base, b).

    Replaces `base + b`, which collides whenever two (sim, tree) pairs
    have the same sum. Returns plain ints in sklearn's accepted range.
    """
    ss = np.random.SeedSequence([int(base) & 0xFFFFFFFF, int(b)])
    return [int(x) % (2 ** 31 - 1)
            for x in ss.generate_state(n, dtype=np.uint32)]


def _mtry_int(frac, p):
    """Fraction of p as an integer count. Used instead of resolve_mtry
    so that mtry_frac = 1.0 gives p rather than 1."""
    return max(1, min(int(p), int(round(float(frac) * int(p)))))


def resolve_mtry(p, mtry="sqrt"):
    if mtry is None:
        return p
    if isinstance(mtry, str):
        return max(1, int(np.floor(np.sqrt(p)))) if mtry == "sqrt" \
               else max(1, int(np.floor(np.log2(p))))
    if 0 < mtry < 1:
        return max(1, int(np.ceil(mtry * p)))
    return max(1, min(p, int(mtry)))


class CustomTree:
    """CART regression tree with an optional constrained split rule (ALG 3)."""

    def __init__(self, max_depth=5, min_samples_split=2, min_samples_leaf=1,
                 mtry="sqrt", random_state=None, constrained=False,
                 feas_tol=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.mtry = mtry
        self.random_state = random_state
        self.constrained = constrained
        self.feas_tol = feas_tol
        self.root = None
        self.node_counter = 0
        self.leaf_values = {}

    def _scan_feature(self, x, y, in_U):
        """Best cut on one feature via prefix sums, O(n log n)."""
        n = len(y)
        order = np.argsort(x, kind="mergesort")
        xs, ys = x[order], y[order]

        valid = xs[1:] > xs[:-1]
        if self.min_samples_leaf > 1:
            k = np.arange(1, n)
            valid &= (k >= self.min_samples_leaf) & \
                     (n - k >= self.min_samples_leaf)
        if not valid.any():
            return np.inf, np.inf, None

        cs, cq = np.cumsum(ys), np.cumsum(ys * ys)
        tot, totq = cs[-1], cq[-1]
        k = np.arange(1, n, dtype=float)
        sse = (cq[:-1] - cs[:-1] ** 2 / k) + \
              ((totq - cq[:-1]) - (tot - cs[:-1]) ** 2 / (n - k))
        sse[~valid] = np.inf

        if not self.constrained:
            j = int(np.argmin(sse))
            return sse[j], 0.0, 0.5 * (xs[j] + xs[j + 1])

        u = in_U[order].astype(float)
        cu, cyu = np.cumsum(u), np.cumsum(ys * u)
        c1 = cs[:-1] / k
        c2 = (tot - cs[:-1]) / (n - k)
        nU_l, nU_r = cu[:-1], cu[-1] - cu[:-1]
        sU_l, sU_r = cyu[:-1], cyu[-1] - cyu[:-1]
        nL_l, nL_r = k - nU_l, (n - k) - nU_r
        sL_l, sL_r = cs[:-1] - sU_l, (tot - cs[:-1]) - sU_r
        viol = np.abs(nU_l * c1 - sU_l + nU_r * c2 - sU_r) + \
               np.abs(nL_l * c1 - sL_l + nL_r * c2 - sL_r)
        viol[~valid] = np.inf

        feas = viol <= self.feas_tol
        j = int(np.argmin(np.where(feas, sse, np.inf))) if feas.any() \
            else int(np.argmin(viol))
        return sse[j], viol[j], 0.5 * (xs[j] + xs[j + 1])

    def _best_split(self, X, y, in_U, rng):
        p = X.shape[1]
        feats = rng.choice(p, size=resolve_mtry(p, self.mtry), replace=False)
        best = (np.inf, np.inf, None, None)
        for f in feats:
            sse, viol, t = self._scan_feature(X[:, f], y, in_U)
            if t is None:
                continue
            if self.constrained:
                cand = (0, sse) if viol <= self.feas_tol else (1, viol)
                cur = (0, best[0]) if best[1] <= self.feas_tol else (1, best[1])
                if best[2] is None or cand < cur:
                    best = (sse, viol, f, t)
            elif sse < best[0]:
                best = (sse, viol, f, t)
        return best[2], best[3]

    def _grow(self, X, y, in_U, depth, rng):
        nid = self.node_counter
        self.node_counter += 1
        max_depth = np.inf if self.max_depth is None else self.max_depth
        if (depth >= max_depth or len(y) < self.min_samples_split
                or len(np.unique(y)) == 1):
            return Node(value=float(np.mean(y)), node_id=nid)
        f, t = self._best_split(X, y, in_U, rng)
        if f is None:
            return Node(value=float(np.mean(y)), node_id=nid)
        li = X[:, f] <= t
        if not li.any() or li.all():
            return Node(value=float(np.mean(y)), node_id=nid)
        return Node(feature=f, threshold=t,
                    left=self._grow(X[li], y[li], in_U[li], depth + 1, rng),
                    right=self._grow(X[~li], y[~li], in_U[~li], depth + 1, rng),
                    node_id=nid)

    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        if self.constrained and self.feas_tol is None:
            self.feas_tol = 1e-8 * max(1.0, float(np.std(y))) * len(y)
        in_U = y >= np.mean(y)
        rng = np.random.RandomState(self.random_state)
        self.node_counter = 0
        self.root = self._grow(X, y, in_U, 0, rng)
        self.leaf_values = self._collect(self.root)
        return self

    def _collect(self, node, out=None):
        if out is None:
            out = {}
        if node.is_leaf():
            out[node.node_id] = node.value
            return out
        self._collect(node.left, out)
        self._collect(node.right, out)
        return out

    def apply(self, X):
        X = np.asarray(X, float)
        out = np.empty(X.shape[0], dtype=np.int64)
        stack = [(self.root, np.arange(X.shape[0]))]
        while stack:
            node, idx = stack.pop()
            if node.is_leaf():
                out[idx] = node.node_id
                continue
            go_left = X[idx, node.feature] <= node.threshold
            stack.append((node.left, idx[go_left]))
            stack.append((node.right, idx[~go_left]))
        return out

    def predict(self, X):
        return np.array([self.leaf_values[i] for i in self.apply(X)],
                        dtype=float)

    # ---- uniform leaf interface used by every correction -----------
    def leaf_ids(self):
        return np.array(sorted(self.leaf_values.keys()), dtype=np.int64)

    def get_leaf_vector(self):
        return np.array([self.leaf_values[i] for i in self.leaf_ids()],
                        dtype=float)

    def set_leaf_vector(self, v):
        new = {int(i): float(x) for i, x in zip(self.leaf_ids(), v)}

        def rec(node):
            if node.is_leaf():
                if node.node_id in new:
                    node.value = new[node.node_id]
                return
            rec(node.left)
            rec(node.right)
        rec(self.root)
        self.leaf_values.update(new)


class SkTreeWrapper:
    """sklearn tree behind the same leaf interface.

    Leaf predictions live at tree_.value[node, 0, 0] and are read at
    predict time, so writing there IS the leaf correction.
    """

    def __init__(self, tree):
        self.tree = tree
        self._leaves = np.where(tree.tree_.children_left == -1)[0]

    def apply(self, X):
        return self.tree.apply(np.asarray(X, float))

    def predict(self, X):
        return self.tree.predict(np.asarray(X, float))

    def leaf_ids(self):
        return self._leaves

    def get_leaf_vector(self):
        return self.tree.tree_.value[self._leaves, 0, 0].astype(float).copy()

    def set_leaf_vector(self, v):
        self.tree.tree_.value[self._leaves, 0, 0] = np.asarray(v, float)


# ==================================================================
# FOREST
# ==================================================================

class UnbiasedForest:
    """Bagged forest with lambda-scaled leaf correction.

    After fit(), set_lambda(lam) re-applies the forest-level correction
    at a new lam in O(total leaves) without refitting anything -- that is
    what makes a lambda sweep cheap.
    """

    def __init__(self, backend="sklearn", correction="forest_lambda",
                 lambda_select="fixed", lam=1.0, lam_max=6.0, tol=1e-8,
                 max_iter=20000, constrained=False, random_state=123,
                 **forest_params):
        self.backend = backend
        self.correction = correction
        self.lambda_select = lambda_select
        self.lam = lam
        self.lam_max = lam_max
        self.tol = tol
        self.max_iter = max_iter
        self.constrained = constrained
        self.random_state = random_state
        self.params = dict(FOREST_PARAMS)
        self.params.update(forest_params)
        if constrained and backend != "custom":
            raise ValueError("constrained splits (ALG 3) require "
                             "BACKEND='custom'.")

        self.trees = []
        self.oob_indices = []
        self.oob_pred = None
        self.lam_used = None
        self.info = {}
        self._fl = None          # cached forest-level correction geometry

    # ---------------- fitting -------------------------------------
    def fit(self, X, y):
        X, y = np.asarray(X, float), np.asarray(y, float)
        n, p = X.shape
        self.trees, self.oob_indices = [], []

        for b in range(self.params["n_estimators"]):
            s_boot, s_tree = _child_seeds(self.random_state, b)
            rng = np.random.RandomState(s_boot)
            idx = rng.choice(n, n, replace=True) if self.params["bootstrap"] \
                else np.arange(n)
            oob = np.setdiff1d(np.arange(n), np.unique(idx))
            if self.backend == "custom":
                t = CustomTree(max_depth=self.params["max_depth"],
                               min_samples_leaf=self.params["min_samples_leaf"],
                               mtry=self.params["mtry"],
                               constrained=self.constrained,
                               random_state=s_tree)
                t.fit(X[idx], y[idx])
                self.trees.append(t)
            else:
                t = SkTree(max_depth=self.params["max_depth"],
                           min_samples_leaf=self.params["min_samples_leaf"],
                           max_features=resolve_mtry(p, self.params["mtry"]),
                           random_state=s_tree)
                t.fit(X[idx], y[idx])
                self.trees.append(SkTreeWrapper(t))
            self.oob_indices.append(oob)

        self._refresh_oob(X, y)
        self._ytr = y

        if self.correction == "none":
            return self
        if self.correction == "forest_lambda":
            self._apply_forest_lambda(X, y)
        else:
            raise ValueError("unknown correction: %r" % self.correction)
        self._refresh_oob(X, y)
        return self

    # ------- forest-level lambda correction ------------------------
    def _build_forest_geometry(self, X, y, use=None):
        """Precompute omega weights, the baseline leaf values, and the
        full projection step, so any lam can be applied instantly."""
        n = len(y)
        B = len(self.trees)
        cnt = np.zeros(n)
        for b in range(B):
            cnt[self.oob_indices[b]] += 1
        ok = cnt > 0
        sel = ok if use is None else (ok & use)
        mu = float(y[sel].mean())
        U = (y >= mu) & sel
        L = (y <  mu) & sel
        N_U, N_L = int(U.sum()), int(L.sum())

        cols, om_U, om_L = [], [], []
        for b in range(B):
            oob = self.oob_indices[b]
            ids = self.trees[b].leaf_ids()
            m = len(ids)
            pos = {int(i): j for j, i in enumerate(ids)}
            c = np.array([pos[int(i)] for i in self.trees[b].apply(X[oob])],
                         dtype=np.int64)
            cols.append((oob, c, m))
            wU, wL = np.zeros(m), np.zeros(m)
            inv = 1.0 / cnt[oob]
            iU, iL = U[oob], L[oob]
            np.add.at(wU, c[iU], inv[iU])
            np.add.at(wL, c[iL], inv[iL])
            om_U.append(wU / N_U if N_U else wU)
            om_L.append(wL / N_L if N_L else wL)

        V0 = [self.trees[b].get_leaf_vector().copy() for b in range(B)]
        Omega = np.vstack([np.concatenate(om_U), np.concatenate(om_L)])

        def oob_from(V):
            s = np.zeros(n)
            for b in range(B):
                oob, c, _ = cols[b]
                s[oob] += V[b][c]
            out = np.full(n, np.nan)
            out[ok] = s[ok] / cnt[ok]
            return out

        yh0 = oob_from(V0)
        r0 = np.array([float((yh0[U] - y[U]).mean()),
                       float((yh0[L] - y[L]).mean())])
        step = Omega.T @ np.linalg.lstsq(Omega @ Omega.T, r0, rcond=None)[0]
        self._fl = dict(V0=V0, step=step, oob_from=oob_from, ok=ok,
                        U=U, L=L, r0=r0)

    def oob_eta_at(self, lam, mask=None):
        fl = self._fl
        yh = fl["oob_from"](self._leaves_at(lam))
        m = fl["ok"] if mask is None else (fl["ok"] & mask)
        return eta_slope(self._ytr[m], yh[m])

    def _cv_lambda(self, X, y, K=5, seed=0):
        """lam with the step cross-fitted: each fold's slope is read on
        points the step was not built from. eta is affine in lam on
        every fold, so two evaluations per fold suffice."""
        fold = np.random.RandomState(seed).permutation(len(y)) % K
        e0 = e1 = 0.0
        for k in range(K):
            self._build_forest_geometry(X, y, use=(fold != k))
            e0 += self.oob_eta_at(0.0, fold == k)
            e1 += self.oob_eta_at(1.0, fold == k)
        e0, e1 = e0 / K, e1 / K
        self._build_forest_geometry(X, y)      # restore the all-data step
        return 1.0 if abs(e1 - e0) < 1e-12 else (1.0 - e0) / (e1 - e0)

    def _leaves_at(self, lam):
        """Leaf vectors at strength lam, without writing them."""
        fl, V, off = self._fl, [], 0
        for b in range(len(self.trees)):
            m = len(fl["V0"][b])
            V.append(fl["V0"][b] - lam * fl["step"][off:off + m])
            off += m
        return V

    def set_lambda(self, lam):
        """Apply the forest-level correction at strength lam (0 = none,
        1 = exact OOB satisfaction, >1 = overcorrection)."""
        if self._fl is None:
            raise RuntimeError("call fit() with correction='forest_lambda'")
        for b, v in enumerate(self._leaves_at(lam)):
            self.trees[b].set_leaf_vector(v)
        self.lam_used = float(lam)
        return self

    def _apply_forest_lambda(self, X, y):
        self._build_forest_geometry(X, y)
        if self.lambda_select == "fixed":
            lam = self.lam
        elif self.lambda_select == "oob_slope":
            # corrected predictions are affine in lam, so OOB eta is too:
            # eta(lam) = e0 + lam (e1 - e0)  =>  lam* = (1 - e0)/(e1 - e0)
            e0, e1 = self.oob_eta_at(0.0), self.oob_eta_at(1.0)
            lam = 1.0 if abs(e1 - e0) < 1e-12 else (1.0 - e0) / (e1 - e0)
            lam = float(np.clip(lam, 0.0, self.lam_max))
            self.info["oob_eta_0"] = e0
            self.info["oob_eta_1"] = e1
        elif self.lambda_select == "oob_cv":
            lam = float(np.clip(self._cv_lambda(X, y), 0.0, self.lam_max))
            self.info["lam_cv"] = lam
        else:
            raise ValueError("unknown lambda_select: %r" % self.lambda_select)
        self.set_lambda(lam)
        self.info["r0"] = self._fl["r0"].tolist()

    # ---------------- prediction ----------------------------------
    def predict(self, X):
        X = np.asarray(X, float)
        out = np.zeros(X.shape[0])
        for t in self.trees:
            out += t.predict(X)
        return out / len(self.trees)

    def _refresh_oob(self, X, y):
        n = X.shape[0]
        s = np.zeros(n)
        c = np.zeros(n, dtype=int)
        for b, tree in enumerate(self.trees):
            oob = self.oob_indices[b]
            if len(oob) == 0:
                continue
            s[oob] += tree.predict(X[oob])
            c[oob] += 1
        self.oob_pred = np.full(n, np.nan)
        ok = c > 0
        self.oob_pred[ok] = s[ok] / c[ok]


# ==================================================================
# METRICS
# ==================================================================
W_SEED_BASE = 42    # W draws, kept disjoint from SEED_BASE


def make_w(y, rng, sd_reference=None):
    """Noisy outcome proxy W = y + U(-a, a) with a = sqrt(3) sd(y), so
    Var(W - y) = Var(y) and the population slope of y on W is 0.5.

    sd_reference is passed the TRAINING sd for both splits, so the noise
    level is identical on train and test rather than following each
    split's own dispersion.
    """
    y = np.asarray(y, float)
    sd = float(np.std(y)) if sd_reference is None else float(sd_reference)
    a = np.sqrt(3.0) * sd
    return y + rng.uniform(-a, a, size=len(y))


def theta_slope(w, v):
    """OLS slope of v on w."""
    w, v = np.asarray(w, float), np.asarray(v, float)
    return float(np.cov(w, v, bias=True)[0, 1] / np.var(w))


def theta_delta(y_true, y_pred, w):
    """theta_hat - theta_oracle: the downstream association slope
    estimated with yhat in place of the outcome, minus the same slope
    estimated with the observed y. Target 0 -- a prediction that
    distorts the downstream regression no more than y itself does.
    Attenuation of yhat pulls theta_hat toward 0, so an uncorrected
    forest reports a negative delta."""
    return theta_slope(w, y_pred) - theta_slope(w, y_true)


def eta_slope(y_true, y_pred):
    """OLS slope of y_pred on y_true (the paper's eta-hat; target 1)."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.cov(y_true, y_pred, bias=True)[0, 1] / np.var(y_true))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred, float)
                                 - np.asarray(y_true, float)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred, float)
                                - np.asarray(y_true, float))))


#: Metrics reported everywhere, in table order. bias == eta == the OLS
#: slope of yhat on y, whose departure from 1 IS the systematic
#: prediction bias. This is exactly the set that appears in the paper's
#: tables; r^2 (both variants) and the two group residuals r_U, r_L
#: were computed in earlier runs but never reported, so they were
#: dropped along with the figures/columns that existed only to show
#: them.
METRIC_KEYS = ("bias", "rmse", "mae", "theta_delta")
SUMMARY_COLS = list(METRIC_KEYS)
METRIC_LABELS = {"bias": r"$\hat\eta$ (slope)", "rmse": "RMSE",
                 "mae": "MAE",
                 "theta_delta": r"$\hat\theta-\hat\theta_{\rm oracle}$"}
METRIC_REFERENCE = {"bias": 1.0, "rmse": None, "mae": None,
                    "theta_delta": 0.0}


def compute_metrics(y_true, y_pred, w=None):
    return dict(bias=eta_slope(y_true, y_pred),
                rmse=rmse(y_true, y_pred),
                mae=mae(y_true, y_pred),
                theta_delta=(np.nan if w is None
                             else theta_delta(y_true, y_pred, w)))


# ==================================================================
# THE PADDING COMPARATOR
# ==================================================================
# Post-hoc staging rather than a constraint: nothing here rewrites a
# leaf. Kept in this file, and fitted inside the same replication as
# everything else, so the comparison is on identical data.

class PaddedForest:
    """Staged residual forest with the running prediction as an extra
    column. Faithful to the original Zhang and Lu (2012) code: every
    stage is fit against the in-sample residual and padded with the
    in-sample prediction.
    """

    def __init__(self, n_trees_base=PAD_TREES_BASE,
                 n_trees_stage=PAD_TREES_STAGE, max_depth=PAD_MAX_DEPTH,
                 min_samples_split=PAD_MIN_SAMPLES_SPLIT,
                 max_stages=PAD_MAX_STAGES, tol=PAD_STAGE_TOL,
                 min_delta=PAD_STAGE_MIN_DELTA,
                 patience=PAD_STAGE_PATIENCE,
                 random_state=SEED_BASE):
        self.n_trees_base = n_trees_base
        self.n_trees_stage = n_trees_stage
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_stages = max_stages
        self.tol = tol
        self.min_delta = min_delta
        self.patience = patience
        self.random_state = random_state
        self.models = []
        self.stage_trace = []   # one dict per stage, train and test metrics
        self.n_stages_ = None   # stages actually run, base stage included
        self.stop_reason_ = None

    def _forest(self, n_trees, seed):
        # n_jobs is 1 for the same reason as RF_PARAMS: replications are
        # already spread across processes by N_WORKERS, and nesting the
        # two oversubscribes the machine.
        return SkRF(
            n_estimators=n_trees,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            max_features="sqrt",
            bootstrap=True,
            random_state=seed,
            n_jobs=1,
        )

    def fit(self, X, y, X_test=None, y_test=None):
        self.models = []
        self.stage_trace = []

        base = self._forest(self.n_trees_base, self.random_state)
        base.fit(X, y)
        self.models.append(base)

        train_pred = base.predict(X)
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
            model = self._forest(self.n_trees_stage,
                                 self.random_state + stage)
            model.fit(X_padded, residuals)
            self.models.append(model)

            increment = model.predict(X_padded)
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
# DATA-GENERATING PROCESSES
# ==================================================================

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


def rf_kwargs(forest_kw):
    """CRF forest kwargs -> sklearn RandomForestRegressor kwargs, so
    both models see identical settings. Used only when RF_MATCH_CRF."""
    return dict(n_estimators=forest_kw["n_estimators"],
                max_depth=forest_kw["max_depth"],
                min_samples_leaf=forest_kw["min_samples_leaf"],
                max_features=forest_kw["mtry"],
                bootstrap=forest_kw["bootstrap"],
                n_jobs=1)


def _fk_for(dgp, forest_kw):
    """Resolve forest kwargs for one DGP.

    forest_kw may be None (fall back to FOREST_PARAMS), a flat params
    dict, or a {dgp: params} map as produced by run_study.
    """
    if forest_kw is None:
        return dict(FOREST_PARAMS)
    if "n_estimators" in forest_kw:
        return dict(forest_kw)
    return dict(forest_kw.get(dgp, FOREST_PARAMS))


def select_forest_params(dgp, verbose=False):
    """Base-forest configuration for one DGP, chosen on pilot training
    data by the corrected slope on a held-out split.

    Only depth and mtry are selected. The pilot forests use
    TUNE_N_ESTIMATORS trees; the returned configuration is set back to
    N_TREES, which is what the study fits."""
    rng = np.random.RandomState(TUNE_SEED)
    X, y = gen_data(dgp, N_TRAIN, rng)
    n = len(y)
    perm = np.random.RandomState(TUNE_SEED).permutation(n)
    n_val = max(1, int(round(TUNE_VAL_FRAC * n)))
    i_val, i_fit = perm[:n_val], perm[n_val:]

    best = None
    for depth in TUNE_GRID["max_depth"]:
        for msl in TUNE_GRID["min_samples_leaf"]:
            for frac in TUNE_GRID["mtry_frac"]:
                kw = dict(FOREST_PARAMS,
                          n_estimators=TUNE_N_ESTIMATORS,
                          max_depth=depth,
                          min_samples_leaf=msl,
                          mtry=_mtry_int(frac, P))
                f = UnbiasedForest(backend=BACKEND,
                                   correction="forest_lambda",
                                   lambda_select="fixed", lam=1.0,
                                   random_state=TUNE_SEED,
                                   **kw).fit(X[i_fit], y[i_fit])
                eta_val = eta_slope(y[i_val], f.predict(X[i_val]))
                score = abs(eta_val - 1.0)
                if verbose:
                    print("    depth=%-5s mtry=%-4d  val eta=%.3f"
                          % (depth, kw["mtry"], eta_val), flush=True)
                if best is None or score < best[0]:
                    best = (score, kw, eta_val)
    chosen = dict(best[1], n_estimators=N_TREES)
    if verbose:
        print("    -> depth=%s mtry=%d (val eta %.3f), fitted with %d trees"
              % (chosen["max_depth"], chosen["mtry"], best[2], N_TREES),
              flush=True)
    return chosen


# ==================================================================
# ONE REPLICATION
# ==================================================================

def metric_rows(dgp, sim, method, y_tr, p_tr, y_te, p_te, lam=np.nan,
                w_tr=None, w_te=None):
    rows = []
    for split, y, pr, w in (("Train", y_tr, p_tr, w_tr),
                            ("Test", y_te, p_te, w_te)):
        row = dict(dgp=dgp, sim=sim, method=method, split=split, lam=lam)
        row.update(compute_metrics(y, pr, w))
        rows.append(row)
    return rows


def fit_crf(name, X_tr, y_tr, sim, forest_kw=None):
    """One CRF variant on one replication."""
    kw = dict(CRF_VARIANTS[name])
    fk = dict(FOREST_PARAMS if forest_kw is None else forest_kw)
    return UnbiasedForest(backend=BACKEND, lam_max=LAM_MAX, tol=TOL,
                          max_iter=MAX_ITER,
                          constrained=CONSTRAINED_SPLITS,
                          random_state=SEED_BASE + sim,
                          **dict(fk, **kw)).fit(X_tr, y_tr)


def fit_padding(name, X_tr, y_tr, X_te, y_te, sim):
    """One padding variant on one replication. The test sample is passed
    to fit() only so that the stage trace can carry test metrics; no
    stage is selected on it."""
    return PaddedForest(random_state=SEED_BASE + sim,
                        **PADDING_VARIANTS[name]).fit(X_tr, y_tr,
                                                      X_te, y_te)


def fit_quietly(estimator, X, y):
    """Fit while suppressing convergence chatter from the full-budget
    MLP, which by construction runs its whole budget rather than
    stopping on a plateau and so always warns."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        estimator.fit(X, y)
    return estimator


def make_xgb(seed):
    """xgboost.XGBRegressor at XGB_PARAMS.

    Imported lazily and by name: the XGBoost column is that model at
    those settings, and sklearn's HistGradientBoostingRegressor is a
    different model at different defaults, so no silent substitution is
    made. If the import fails the run stops here rather than filling the
    column with something else."""
    try:
        from xgboost import XGBRegressor
    except ImportError:
        raise ImportError(
            "xgboost is not installed, and the XGBoost column is "
            "xgboost.XGBRegressor at the settings in XGB_PARAMS. Install "
            "it with 'pip install xgboost', or drop \"XGBoost\" from "
            "BASELINES.")
    kwargs = dict(XGB_PARAMS, random_state=seed, tree_method="hist")
    try:
        return XGBRegressor(device="cpu", **kwargs)
    except TypeError:
        return XGBRegressor(**kwargs)      # xgboost < 2.0 has no `device`


def fit_baselines(X_tr, y_tr, sim, forest_kw=None, which=None):
    """The baselines, returning name -> predict callable.

    `which` restricts the set fitted, so a partial run can add one
    method to an existing results file without refitting the others.

    RF is grown at RF_PARAMS -- the settings the constrained-neural-
    network study used -- unless RF_MATCH_CRF, in which case it inherits
    the CRF's own depth and mtry and becomes the uncorrected control for
    the same forest.

    The NN and the Lasso are fitted on standardised X (and, for the NN,
    standardised y), with the transform inverted before scoring, so the
    reported metrics are in original y units for every method.
    """
    want = [m for m in BASELINES
            if which is None or m in which]
    out = {}
    if "RF" in want:
        rf_kw = rf_kwargs(forest_kw) if (RF_MATCH_CRF and forest_kw) \
            else dict(RF_PARAMS)
        out["RF"] = fit_quietly(SkRF(random_state=SEED_BASE + sim,
                                     **rf_kw), X_tr, y_tr).predict

    if "NN" in want:
        sx, sy = StandardScaler(), StandardScaler()
        Xs = sx.fit_transform(X_tr)
        ys = sy.fit_transform(y_tr.reshape(-1, 1)).ravel()
        nn = fit_quietly(MLPRegressor(random_state=SEED_BASE + sim,
                                      batch_size=len(ys), **NN_PARAMS),
                         Xs, ys)

        def nn_pred(X):
            return sy.inverse_transform(
                nn.predict(sx.transform(X)).reshape(-1, 1)).ravel()

        out["NN"] = nn_pred

    if "XGBoost" in want:
        out["XGBoost"] = fit_quietly(make_xgb(SEED_BASE + sim),
                                     X_tr, y_tr).predict

    if "Lasso" in want:
        lx = StandardScaler()
        Xl = lx.fit_transform(X_tr)
        las = fit_quietly(LassoCV(random_state=SEED_BASE + sim,
                                  **LASSO_PARAMS), Xl, y_tr)

        def lasso_pred(X):
            return las.predict(lx.transform(X))

        out["Lasso"] = lasso_pred
    return out


def _pred_frame(dgp, sim, method, split, y_obs, y_pred, w_obs, lam=np.nan):
    """One tidy block of prediction rows, for recomputing any metric
    later without refitting."""
    y_obs = np.asarray(y_obs, float).ravel()
    y_pred = np.asarray(y_pred, float).ravel()
    if w_obs is None:
        w_obs = np.full(y_obs.shape[0], np.nan)
    return pd.DataFrame({"dgp": dgp, "sim": sim, "method": method,
                         "split": split, "lam": lam,
                         "obs_index": np.arange(y_obs.shape[0]),
                         "y_obs": y_obs, "y_pred": y_pred,
                         "w_obs": np.asarray(w_obs, float).ravel()})


def run_one(task):
    """One replication of one DGP: fit every method, score it, and (when
    SAVE_ALL_PREDICTIONS) keep the prediction vectors as well."""
    dgp, sim = task[0], task[1]
    forest_kw = task[2] if len(task) > 2 else None
    methods = task[3] if len(task) > 3 else None
    rng = np.random.RandomState(SEED_BASE + sim)
    X_tr, y_tr = gen_data(dgp, N_TRAIN, rng)
    X_te, y_te = gen_data(dgp, N_TEST, rng)
    rows, frames, stages = [], [], []

    # W is drawn from its own stream, at the TRAINING sd on both splits.
    w_rng = np.random.RandomState(W_SEED_BASE + sim)
    sd_tr = float(np.std(y_tr))
    w_tr = make_w(y_tr, w_rng, sd_reference=sd_tr)
    w_te = make_w(y_te, w_rng, sd_reference=sd_tr)

    for name in CRF_VARIANTS:
        if methods is not None and name not in methods:
            continue
        f = fit_crf(name, X_tr, y_tr, sim, forest_kw)
        lam = f.lam_used if f.lam_used is not None else np.nan
        if CRF_TRAIN_ROW == "oob" and f.oob_pred is not None:
            ok = ~np.isnan(f.oob_pred)
            y_row, p_row, w_row = y_tr[ok], f.oob_pred[ok], w_tr[ok]
        else:
            y_row, p_row, w_row = y_tr, f.predict(X_tr), w_tr
        p_te = f.predict(X_te)
        rows += metric_rows(dgp, sim, name, y_row, p_row, y_te, p_te,
                            lam=lam, w_tr=w_row, w_te=w_te)
        if SAVE_ALL_PREDICTIONS:
            frames.append(_pred_frame(dgp, sim, name, "Train", y_row,
                                      p_row, w_row, lam))
            frames.append(_pred_frame(dgp, sim, name, "Test", y_te, p_te,
                                      w_te, lam))

    # The post-hoc comparator. Two extra columns per row, n_stages and
    # stop_reason, are NaN for every other method; nothing downstream
    # aggregates them, so the shared schema is unaffected.
    for name in PADDING_VARIANTS:
        if methods is not None and name not in methods:
            continue
        m = fit_padding(name, X_tr, y_tr, X_te, y_te, sim)
        p_tr, p_te = m.train_pred_, m.predict(X_te)
        pad_rows = metric_rows(dgp, sim, name, y_tr, p_tr, y_te, p_te,
                               w_tr=w_tr, w_te=w_te)
        for r in pad_rows:
            r["n_stages"], r["stop_reason"] = m.n_stages_, m.stop_reason_
        rows += pad_rows
        if SAVE_STAGE_TRACE:
            stages += [dict(dgp=dgp, method=name, sim=sim,
                            stop_reason=m.stop_reason_, **t)
                       for t in m.stage_trace]
        if SAVE_ALL_PREDICTIONS:
            frames.append(_pred_frame(dgp, sim, name, "Train", y_tr, p_tr,
                                      w_tr))
            frames.append(_pred_frame(dgp, sim, name, "Test", y_te, p_te,
                                      w_te))

    for name, pred in fit_baselines(X_tr, y_tr, sim, forest_kw,
                                    which=methods).items():
        p_tr, p_te = pred(X_tr), pred(X_te)
        rows += metric_rows(dgp, sim, name, y_tr, p_tr, y_te, p_te,
                            w_tr=w_tr, w_te=w_te)
        if SAVE_ALL_PREDICTIONS:
            frames.append(_pred_frame(dgp, sim, name, "Train", y_tr, p_tr,
                                      w_tr))
            frames.append(_pred_frame(dgp, sim, name, "Test", y_te, p_te,
                                      w_te))
    return (rows,
            pd.concat(frames, ignore_index=True) if frames else None,
            pd.DataFrame(stages) if stages else None)


# ==================================================================
# LAMBDA SWEEP
# ==================================================================

def lambda_sweep(n_reps=5, grid=SWEEP_GRID, dgps=DGPS, forest_kw=None):
    """OOB and test curves versus lam, on shared forests (each forest is
    fitted once and re-corrected at every lam)."""
    rows = []
    for dgp in dgps:
        fk = _fk_for(dgp, forest_kw)
        for s in range(n_reps):
            rng = np.random.RandomState(SEED_BASE + s)
            X_tr, y_tr = gen_data(dgp, N_TRAIN, rng)
            X_te, y_te = gen_data(dgp, N_TEST, rng)
            f = UnbiasedForest(backend=BACKEND,
                               correction="forest_lambda",
                               lambda_select="fixed", lam=0.0,
                               random_state=SEED_BASE + s,
                               **fk).fit(X_tr, y_tr)
            for lam in grid:
                f.set_lambda(lam)
                f._refresh_oob(X_tr, y_tr)
                ok = ~np.isnan(f.oob_pred)
                p = f.predict(X_te)
                rows.append(dict(dgp=dgp, sim=s, lam=lam,
                                 oob_eta=eta_slope(y_tr[ok],
                                                   f.oob_pred[ok]),
                                 test_eta=eta_slope(y_te, p),
                                 test_rmse=rmse(y_te, p)))
    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    merge_csv(df, os.path.join(OUT_DIR, SWEEP_CSV), ["dgp", "sim", "lam"])
    agg = df.groupby(["dgp", "lam"])[["oob_eta", "test_eta",
                                      "test_rmse"]].mean()
    print("\nLambda sweep (%d reps per DGP, means)" % n_reps)
    print(agg.round(3).to_string())
    print("\nlam selected by OOB slope = 1, and the test slope it gives:")
    for dgp in dgps:
        d = agg.loc[dgp]
        lam_star = float(np.interp(1.0, d["oob_eta"].values,
                                   d.index.values))
        print("  %-14s lam*=%.2f -> test_eta=%.3f  test_rmse=%.3f"
              % (dgp, lam_star,
                 np.interp(lam_star, d.index.values, d["test_eta"].values),
                 np.interp(lam_star, d.index.values,
                           d["test_rmse"].values)))
    return df


# ==================================================================
# FIGURES
# ==================================================================
# No figure-level titles are drawn: captions belong in LaTeX. Panels
# carry a short unbolded heading only where a grid would otherwise be
# unreadable, and axes carry plain labels.

def _lazy_plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


#: Baselines keep fixed colours; every CRF variant gets its own hue so
#: the same method reads the same across all figures.
_BASE_COLORS = {"RF": "#2E86C1", "NN": "#27AE60", "XGBoost": "#D68910",
                "Lasso": "#8E44AD", "RF-uncorr": "#7F8C8D"}
_VARIANT_CYCLE = ("#C0392B", "#16A085", "#2C3E50", "#8E44AD",
                  "#CB4335", "#7D3C98", "#1F618D")


def method_color(name):
    if name in _BASE_COLORS:
        return _BASE_COLORS[name]
    variants = [m for m in METHOD_ORDER if m not in _BASE_COLORS]
    i = variants.index(name) if name in variants else 0
    return _VARIANT_CYCLE[i % len(_VARIANT_CYCLE)]


METHOD_COLORS = {m: method_color(m) for m in METHOD_ORDER}


def _boxplot(ax, data, labels, widths=0.6):
    """boxplot with the tick-label kwarg matplotlib actually accepts
    (renamed from `labels` to `tick_labels` in 3.9)."""
    kw = dict(patch_artist=True, widths=widths,
              medianprops=dict(color="black", lw=1.4),
              flierprops=dict(marker=".", markersize=3, alpha=0.5))
    try:
        return ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kw)


def fig_metric_box(per_sim, dgp, split="Test", out_dir=None):
    """Distribution of every metric across replications, by method.

    This is the figure that shows the trade-off directly: CRF's bias
    box straddles a higher value while its RMSE box sits lower or
    higher depending on the DGP."""
    plt = _lazy_plt()
    out_dir = out_dir or OUT_DIR
    d = per_sim[(per_sim.dgp == dgp) & (per_sim.split == split)]
    keys = list(METRIC_KEYS)
    ncol = len(keys)
    wide = len(METHOD_ORDER) > 3
    fig, axes = plt.subplots(1, ncol,
                             figsize=(max(3.2, 0.62 * len(METHOD_ORDER) + 1.9)
                                      * ncol, 5.2 if wide else 3.8))
    for ax, met in zip(np.atleast_1d(axes), keys):
        data, labels, colors = [], [], []
        for meth in METHOD_ORDER:
            v = d[d.method == meth][met].dropna().values
            if len(v):
                data.append(v)
                labels.append(meth)
                colors.append(METHOD_COLORS.get(meth, "#7F8C8D"))
        bp = _boxplot(ax, data, labels, widths=0.6)
        for patch, c in zip(bp["boxes"], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.45)
            patch.set_edgecolor(c)
        ref = METRIC_REFERENCE.get(met)
        if ref is not None:
            ax.axhline(ref, color="black", ls="--", lw=1)
        ax.set_ylabel(METRIC_LABELS.get(met, met))
        ax.grid(True, axis="y", alpha=0.25)
        ax.tick_params(axis="both", which="both", labelsize=9)
        if len(labels) > 3:
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    fig.tight_layout()
    path = os.path.join(dgp_dir(dgp, out_dir), "metrics_box_%s_%s.png"
                        % (dgp_name(dgp), split.lower()))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def _shared_limits(rec, methods, splits=("Train", "Test")):
    vals = []
    for sp in splits:
        vals.append(rec[sp + "_y"])
        for m in methods:
            vals.append(rec[sp][m])
    lo = float(min(v.min() for v in vals))
    hi = float(max(v.max() for v in vals))
    pad = 0.15 * (hi - lo)
    return lo - pad, hi + pad


def _scatter_panel(ax, y, p, color, lo, hi, heading=None,
                   annotate=("bias",), lam=None):
    met = compute_metrics(y, p)
    b = met["bias"]
    a0 = float(np.mean(p) - b * np.mean(y))
    ax.scatter(y, p, s=11, alpha=0.45, color=color, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    xs = np.array([lo, hi])
    ax.plot(xs, b * xs + a0, color="#C0392B", lw=1.6)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed")
    ax.set_ylabel("Predicted")
    if heading:
        ax.set_title(heading, fontweight="normal", fontsize=11)
    lines = ["%s = %.3f" % (METRIC_LABELS.get(k, k), met[k])
             for k in annotate]
    if lam is not None and np.isfinite(lam):
        lines.append(r"$\lambda$ = %.2f" % lam)
    ax.text(0.04, 0.94, "\n".join(lines), transform=ax.transAxes,
            fontsize=9, va="top")
    ax.grid(True, alpha=0.22)
    ax.minorticks_on()
    return met


def fig_obs_pred(preds, dgp, out_dir=None, annotate=("bias",),
                 split=None):
    """Observed vs predicted for every method on one page.

    Columns are methods, rows are Train / Test (or a single row when
    `split` is given). All panels share square limits pooled over every
    observed and predicted value, padded 15%, with a dashed identity
    line and the fitted slope in red."""
    plt = _lazy_plt()
    out_dir = out_dir or OUT_DIR
    rec = preds[dgp]
    methods = methods_in(rec)
    splits = (split,) if split else ("Train", "Test")
    lo, hi = _shared_limits(rec, methods)

    fig, axes = plt.subplots(len(splits), len(methods),
                             figsize=(3.4 * len(methods), 3.5 * len(splits)),
                             squeeze=False)
    for i, sp in enumerate(splits):
        y = rec[sp + "_y"]
        for j, m in enumerate(methods):
            _scatter_panel(axes[i][j], y, rec[sp][m], METHOD_COLORS.get(m),
                           lo, hi, heading="%s (%s)" % (m, sp),
                           annotate=annotate,
                           lam=rec.get("lam", {}).get(m))
    fig.tight_layout()
    tag = "_%s" % split.lower() if split else ""
    path = os.path.join(dgp_dir(dgp, out_dir), "obs_pred_%s%s.png"
                        % (dgp_name(dgp), tag))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_obs_pred_method(preds, dgp, method, out_dir=None,
                        annotate=("bias", "rmse")):
    """Observed vs predicted for ONE method: Train and Test side by side,
    on the same shared limits used by the combined grid so panels stay
    comparable across files."""
    plt = _lazy_plt()
    out_dir = out_dir or OUT_DIR
    rec = preds[dgp]
    lo, hi = _shared_limits(rec, methods_in(rec))
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
    for ax, sp in zip(axes, ("Train", "Test")):
        _scatter_panel(ax, rec[sp + "_y"], rec[sp][method],
                       METHOD_COLORS.get(method), lo, hi, heading=sp,
                       annotate=annotate,
                       lam=rec.get("lam", {}).get(method))
    fig.tight_layout()
    safe = str(method).replace(" ", "-").replace("/", "-")
    path = os.path.join(dgp_dir(dgp, out_dir), "obs_pred_%s_%s.png"
                        % (dgp_name(dgp), safe))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_theta(preds, dgp, out_dir=None):
    """The downstream association, method by method: yhat regressed on
    the noisy proxy W, against the same regression run with the observed
    y. The oracle line is what a downstream analyst would obtain with
    the outcome in hand; the gap between the two slopes IS theta_delta,
    which is reported in the tables and boxplots."""
    plt = _lazy_plt()
    out_dir = out_dir or OUT_DIR
    rec = preds[dgp]
    if rec.get("Test_w") is None or not np.isfinite(rec["Test_w"]).any():
        return None
    methods = methods_in(rec)
    y, w = rec["Test_y"], rec["Test_w"]
    th_o = theta_slope(w, y)
    ws = np.array([w.min(), w.max()])

    fig, axes = plt.subplots(1, len(methods),
                             figsize=(3.5 * len(methods), 3.6),
                             squeeze=False)
    for ax, m in zip(axes[0], methods):
        p = rec["Test"][m]
        th = theta_slope(w, p)
        ax.scatter(w, p, s=11, alpha=0.40, color=METHOD_COLORS.get(m),
                   edgecolors="none")
        ax.plot(ws, th * (ws - w.mean()) + p.mean(),
                color=METHOD_COLORS.get(m), lw=1.8)
        ax.plot(ws, th_o * (ws - w.mean()) + y.mean(), color="black",
                ls="--", lw=1.3)
        ax.set_xlabel("$W$")
        ax.set_ylabel("Predicted")
        ax.set_title(m, fontweight="normal", fontsize=11)
        ax.text(0.04, 0.94,
                "$\\hat\\theta$ = %.3f\n$\\hat\\theta-\\hat\\theta_{\\rm "
                "oracle}$ = %.3f" % (th, th - th_o),
                transform=ax.transAxes, fontsize=9, va="top")
        ax.grid(True, alpha=0.22)
        ax.minorticks_on()
    fig.tight_layout()
    path = os.path.join(dgp_dir(dgp, out_dir), "association_%s.png" % dgp_name(dgp))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_lambda_curve(sweep, dgp, out_dir=None):
    """OOB and test slope, and test RMSE, against lam. Marks lam = 1
    (exact OOB constraint satisfaction) and the OOB-selected lam."""
    plt = _lazy_plt()
    out_dir = out_dir or OUT_DIR
    d = (sweep[sweep.dgp == dgp]
         .groupby("lam")[["oob_eta", "test_eta", "test_rmse"]].mean())
    lam = d.index.values
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.8))

    ax = axes[0]
    ax.plot(lam, d["oob_eta"].values, color="#2E86C1", lw=1.8,
            marker="o", ms=3.5, label="OOB")
    ax.plot(lam, d["test_eta"].values, color="#C0392B", lw=1.8,
            marker="s", ms=3.5, label="Test")
    ax.axhline(1.0, color="black", ls="--", lw=1)
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    lam_star = float(np.interp(1.0, d["oob_eta"].values, lam))
    ax.axvline(lam_star, color="#2E86C1", ls=":", lw=1.2)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel(r"$\hat\eta$ (slope)")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.minorticks_on()

    ax = axes[1]
    ax.plot(lam, d["test_rmse"].values, color="#C0392B", lw=1.8,
            marker="s", ms=3.5)
    ax.axvline(1.0, color="gray", ls=":", lw=1)
    ax.axvline(lam_star, color="#2E86C1", ls=":", lw=1.2)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Test RMSE")
    ax.grid(True, alpha=0.25)
    ax.minorticks_on()

    fig.tight_layout()
    path = os.path.join(dgp_dir(dgp, out_dir), "lambda_curve_%s.png" % dgp_name(dgp))
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def methods_in(rec):
    """Methods present in one DGP's prediction record, in table order."""
    return [m for m in METHOD_ORDER if m in rec["Test"]]


def dgp_name(dgp):
    """The name a DGP is shown under, per DGP_DISPLAY."""
    return DGP_DISPLAY.get(str(dgp), str(dgp))


def dgp_dir(dgp, out_dir=None, sub=None):
    """OUT_DIR/<display name>[/<sub>], created on demand."""
    p = os.path.join(out_dir or OUT_DIR, dgp_name(dgp))
    if sub:
        p = os.path.join(p, sub)
    os.makedirs(p, exist_ok=True)
    return p


def fig_tree_obs_pred(rec_tree, dgp, b, out_dir=None):
    """One individual tree, observed vs predicted, before and after the
    leaf correction.

    A single tree is a step function with very few distinct outputs, so
    the scatter forms horizontal bands -- one per leaf. That is the point
    of the figure: the correction moves those bands, and the fitted slope
    reports how far the tree still is from the identity.

    Read these as diagnostics, not as results. An INDIVIDUAL tree is
    heavily overcorrected by design: the forest-level step distributes
    one correction across every leaf of every tree in proportion to that
    leaf's share of OOB stratum mass, so a single tree's slope typically
    overshoots 1 by a wide margin and its RMSE rises. Only the AVERAGE
    over all B trees is calibrated -- which is exactly why per-tree
    feasibility was the wrong target. Compare against
    obs_pred_<dgp>.png, which shows the assembled forest."""
    plt = _lazy_plt()
    d = dgp_dir(dgp, out_dir, "tree")
    y = rec_tree["y"]
    lo = float(min(y.min(), rec_tree["before"].min(), rec_tree["after"].min()))
    hi = float(max(y.max(), rec_tree["before"].max(), rec_tree["after"].max()))
    pad = 0.15 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.9))
    for ax, key, head in ((axes[0], "before", "Uncorrected"),
                          (axes[1], "after", "Corrected")):
        _scatter_panel(ax, y, rec_tree[key], METHOD_COLORS.get(
            TREE_FIG_METHOD, "#C0392B"), lo, hi,
            heading="%s (tree %d)" % (head, b),
            annotate=("bias", "rmse"))
    fig.tight_layout()
    path = os.path.join(d, "tree_%03d.png" % b)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def collect_tree_predictions(dgp, sim=0, n_trees=None, forest_kw=None):
    """Per-tree predictions before and after correction, on the test set.

    Refits TREE_FIG_METHOD (a forest_lambda variant, whose cached
    geometry retains the uncorrected leaf values V0), then for each tree
    swaps its leaves back to V0 to get the 'before' curve and to the
    corrected values for 'after'."""
    kw = CRF_VARIANTS.get(TREE_FIG_METHOD)
    if kw is None or kw.get("correction") != "forest_lambda":
        raise ValueError("TREE_FIG_METHOD must name a forest_lambda "
                         "variant in CRF_VARIANTS; got %r"
                         % TREE_FIG_METHOD)
    rng = np.random.RandomState(SEED_BASE + sim)
    X_tr, y_tr = gen_data(dgp, N_TRAIN, rng)
    X_te, y_te = gen_data(dgp, N_TEST, rng)
    f = fit_crf(TREE_FIG_METHOD, X_tr, y_tr, sim, _fk_for(dgp, forest_kw))

    V0 = f._fl["V0"]
    n_trees = len(f.trees) if n_trees in (None, -1) \
        else min(n_trees, len(f.trees))
    out = []
    for b in range(n_trees):
        after = f.trees[b].get_leaf_vector().copy()
        f.trees[b].set_leaf_vector(V0[b])
        before = f.trees[b].predict(X_te)
        f.trees[b].set_leaf_vector(after)
        out.append(dict(y=y_te, before=before,
                        after=f.trees[b].predict(X_te)))
    return out


def collect_predictions(sim=0, dgps=DGPS, tuned=None):
    """Refit one replication per DGP, keeping every method's predictions,
    so the scatter figures can be drawn without storing every
    replication's vectors during the study."""
    preds = {}
    for dgp in dgps:
        fk = _fk_for(dgp, tuned)
        rng = np.random.RandomState(SEED_BASE + sim)
        X_tr, y_tr = gen_data(dgp, N_TRAIN, rng)
        X_te, y_te = gen_data(dgp, N_TEST, rng)
        w_rng = np.random.RandomState(W_SEED_BASE + sim)
        sd_tr = float(np.std(y_tr))
        w_tr = make_w(y_tr, w_rng, sd_reference=sd_tr)
        w_te = make_w(y_te, w_rng, sd_reference=sd_tr)
        rec = {"Train_y": y_tr, "Test_y": y_te, "Train_w": w_tr,
               "Test_w": w_te, "Train": {}, "Test": {}, "lam": {}}
        ok = None

        for name in CRF_VARIANTS:
            f = fit_crf(name, X_tr, y_tr, sim, fk)
            rec["lam"][name] = f.lam_used
            if CRF_TRAIN_ROW == "oob" and f.oob_pred is not None:
                ok = ~np.isnan(f.oob_pred)
                rec["Train"][name] = f.oob_pred[ok]
            else:
                rec["Train"][name] = f.predict(X_tr)
            rec["Test"][name] = f.predict(X_te)

        for name in PADDING_VARIANTS:
            m = fit_padding(name, X_tr, y_tr, X_te, y_te, sim)
            rec["Train"][name] = m.train_pred_
            rec["Test"][name] = m.predict(X_te)

        for name, pred in fit_baselines(X_tr, y_tr, sim, fk).items():
            rec["Train"][name] = pred(X_tr)
            rec["Test"][name] = pred(X_te)

        # The CRF's Train row is its OOB prediction, which covers only the
        # rows `ok`; every other method is cut to the same rows so the
        # panels of one figure are drawn on one sample.
        if CRF_TRAIN_ROW == "oob" and ok is not None:
            rec["Train_y"] = y_tr[ok]
            rec["Train_w"] = w_tr[ok]
            for name in list(rec["Train"]):
                if name not in CRF_VARIANTS:
                    rec["Train"][name] = rec["Train"][name][ok]
        preds[dgp] = rec
    return preds


def save_prediction_vectors(preds, sim=0, out_dir=None):
    """Write the y / yhat / W vectors behind the figures, so any metric
    not anticipated here can be recomputed later without refitting."""
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    frames = []
    for dgp, rec in preds.items():
        for sp in ("Train", "Test"):
            for m in methods_in(rec):
                frames.append(_pred_frame(dgp, sim, m, sp, rec[sp + "_y"],
                                          rec[sp][m], rec.get(sp + "_w"),
                                          rec.get("lam", {}).get(m, np.nan)))
    if not frames:
        return None
    path = os.path.join(out_dir, "predictions_sim%d.csv" % sim)
    merge_csv(pd.concat(frames, ignore_index=True), path,
              ["dgp", "sim", "method", "split", "obs_index"])
    return path


def make_all_figures(per_sim=None, sweep=None, preds=None, sim=0,
                     out_dir=None, sweep_from_disk=False, tuned=None,
                     dgps=None):
    """Every figure for every DGP. Reads results_per_sim.csv when the
    frame is not passed in. lambda_sweep.csv is read ONLY when
    sweep_from_disk=True, so a file left over from an earlier
    configuration cannot silently supply the lambda curves.

    `tuned` is the {dgp: forest params} map from run_study. Passing it
    matters: the scatter figures REFIT forests, and without it they
    would refit at FOREST_PARAMS rather than at the configuration the
    table was produced with."""
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    if per_sim is None:
        per_sim = pd.read_csv(os.path.join(out_dir, PER_SIM_CSV))
    if sweep is None and sweep_from_disk:
        p = os.path.join(out_dir, SWEEP_CSV)
        if os.path.exists(p):
            sweep = pd.read_csv(p)
            print("note: lambda curves drawn from the existing %s" % p)
    if sweep is None:
        print("note: no sweep data; skipping lambda_curve_*.png")
    dgps = tuple(per_sim.dgp.unique()) if dgps is None else tuple(dgps)
    if preds is None:
        # Every method is refitted for the scatter figures regardless of
        # which ones this run scored, so a partial run still redraws a
        # complete panel.
        preds = collect_predictions(sim=sim, dgps=dgps, tuned=tuned)
    made = []
    if SAVE_PREDICTIONS:
        p = save_prediction_vectors(preds, sim=sim, out_dir=out_dir)
        if p:
            made.append(p)
    for dgp in dgps:
        made.append(fig_metric_box(per_sim, dgp, "Test", out_dir))
        made.append(fig_metric_box(per_sim, dgp, "Train", out_dir))
        made.append(fig_obs_pred(preds, dgp, out_dir=out_dir))
        for m in methods_in(preds[dgp]):
            made.append(fig_obs_pred_method(preds, dgp, m, out_dir=out_dir))
        p = fig_theta(preds, dgp, out_dir)
        if p:
            made.append(p)
        if sweep is not None and (sweep.dgp == dgp).any():
            made.append(fig_lambda_curve(sweep, dgp, out_dir))
        if TREE_FIGS_N:
            trees = collect_tree_predictions(
                dgp, sim=sim, n_trees=TREE_FIGS_N,
                forest_kw=_fk_for(dgp, tuned))
            for b, rec_tree in enumerate(trees):
                made.append(fig_tree_obs_pred(rec_tree, dgp, b,
                                              out_dir=out_dir))
    for p in made:
        print("wrote", p)
    return made


# ==================================================================
# TABLES
# ==================================================================
# Formatting is applied last, to values aggregated at full precision, so
# results_table.csv, results_table.tex and summary_mean_sd.csv cannot
# disagree.

#: (metric key, CSV row label, LaTeX row label, format). Every metric is
#: reported for every DGP, including the association row, which earlier
#: versions restricted to the linear setting.
TABLE_ROWS = (
    ("bias",        "eta_hat",                 r"$\hat\eta$",      "%.3f"),
    ("rmse",        "RMSE",                    r"RMSE",            "%.3f"),
    ("mae",         "MAE",                     r"MAE",             "%.3f"),
    ("theta_delta", "theta_hat - theta_oracle",
     r"$\hat\theta-\hat\theta_{\mathrm{oracle}}$",                 "%.3f"),
)

#: Table labels, keyed by the raw DGP name. A DGP with no entry falls
#: back to its display name from DGP_DISPLAY.
DGP_LABELS = {"linear": "Linear", "linear_sparse": "Linear",
              "friedman1": "Friedman 1", "friedman2": "Friedman 2"}


def dgp_label(dgp):
    return DGP_LABELS.get(dgp, dgp_name(dgp).replace("_", " ").title())


def merge_csv(new, path, keys):
    """Fold `new` into the file at `path`, replacing any existing row
    with the same key and keeping every row the run did not touch.

    This is what makes a partial run legitimate: the data for a given
    (dgp, sim) is fixed by SEED_BASE, BETA_SEED and W_SEED_BASE, so rows
    carried over from an earlier run were fitted on exactly the samples
    this run would have generated. That holds only while those seeds,
    N_TRAIN, N_TEST, P, SIGMA_E and the DGP definitions are unchanged --
    change any of them and the old rows must be regenerated, not merged.
    """
    new = new.copy()
    if os.path.exists(path):
        old = pd.read_csv(path)
        if set(keys).issubset(old.columns) and set(keys).issubset(new.columns):
            tag = old.merge(new[keys].drop_duplicates(), on=keys,
                            how="left", indicator=True)["_merge"].values
            kept = old[tag == "left_only"]
            print("merging into %s: %d rows kept, %d replaced, %d added"
                  % (os.path.basename(path), len(kept),
                     len(old) - len(kept), len(new)))
            new = pd.concat([kept, new], ignore_index=True)
    new.to_csv(path, index=False)
    return new


def sort_per_sim(per_sim):
    """Stable ordering for the per-replication file: DGPs and methods in
    reporting order rather than in whatever order a merge produced."""
    d = per_sim.copy()
    d["_d"] = pd.Categorical(d.dgp, [x for x in DGPS
                                     if x in set(d.dgp)] +
                             sorted(set(d.dgp) - set(DGPS)), ordered=True)
    d["_m"] = pd.Categorical(d.method, [x for x in METHOD_ORDER
                                        if x in set(d.method)] +
                             sorted(set(d.method) - set(METHOD_ORDER)),
                             ordered=True)
    d = d.sort_values(["_d", "sim", "_m", "split"], kind="mergesort")
    return d.drop(columns=["_d", "_m"]).reset_index(drop=True)


def summarize(per_sim):
    s = (per_sim.groupby(["dgp", "split", "method"])[SUMMARY_COLS]
         .agg(["mean", "std", "median", "count"]))
    s.columns = ["%s_%s" % (m, a) for m, a in s.columns]
    return s.reset_index()


def format_cell(per_sim, dgp, split, method, metric, precision="%.3f"):
    """'mean (SD)' over replications, or '--' when the cell is empty."""
    v = per_sim[(per_sim.dgp == dgp) & (per_sim.split == split)
                & (per_sim.method == method)][metric].to_numpy(float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return "--"
    if v.size == 1:
        return precision % v[0]
    return ("%s (%s)" % (precision, precision)) % (np.mean(v),
                                                   np.std(v, ddof=1))


def write_results_table(per_sim, out_dir=None, filename=None):
    """The paper-shaped table: (DGP, split, metric) rows, method columns,
    cells 'mean (SD)' over replications."""
    out_dir = out_dir or OUT_DIR
    filename = filename or RESULTS_TABLE_CSV
    methods = [m for m in METHOD_ORDER if m in set(per_sim.method)]
    rows = []
    for dgp in [d for d in DGPS if d in set(per_sim.dgp)]:
        for split in ("Train", "Test"):
            for metric, label, _tex, prec in TABLE_ROWS:
                row = {"DGP": dgp_label(dgp), "Split": split,
                       "Metric": label}
                for m in methods:
                    row[m] = format_cell(per_sim, dgp, split, m, metric,
                                         prec)
                rows.append(row)
    table = pd.DataFrame(rows, columns=["DGP", "Split", "Metric"] + methods)
    path = os.path.join(out_dir, filename)
    table.to_csv(path, index=False)
    print("wrote", path)
    return table


def _tex_escape(s):
    return str(s).replace("_", r"\_").replace("&", r"\&")


def write_latex_table(per_sim, out_dir=None, filename=None):
    """The same table as LaTeX: one booktabs tabular per DGP.

    Requires \\usepackage{booktabs}. No caption text is invented here
    beyond a placeholder, and no \\label is guessed -- both are edited in
    the manuscript, where the surrounding text fixes what they should
    say."""
    out_dir = out_dir or OUT_DIR
    filename = filename or RESULTS_TABLE_TEX
    methods = [m for m in METHOD_ORDER if m in set(per_sim.method)]
    n_sims = per_sim["sim"].nunique()
    parts = ["% Generated by run_simulation.py -- do not edit by hand.",
             "% Requires \\usepackage{booktabs}.",
             "%% Cells are mean (SD) over %d replications." % n_sims, ""]

    for dgp in [d for d in DGPS if d in set(per_sim.dgp)]:
        lines = [r"\begin{table}[htbp]", r"\centering",
                 r"\caption{%s: mean (SD) over %d replications.}"
                 % (_tex_escape(dgp_label(dgp)), n_sims),
                 r"\label{tab:%s}" % dgp_name(dgp).replace("_", "-"),
                 r"\begin{tabular}{ll%s}" % ("r" * len(methods)),
                 r"\toprule",
                 "Split & Metric & %s \\\\"
                 % " & ".join(_tex_escape(m) for m in methods),
                 r"\midrule"]
        for k, split in enumerate(("Train", "Test")):
            if k:
                lines.append(r"\midrule")
            for j, (metric, _lab, tex, prec) in enumerate(TABLE_ROWS):
                cells = [format_cell(per_sim, dgp, split, m, metric, prec)
                         for m in methods]
                lines.append("%s & %s & %s \\\\"
                             % (split if j == 0 else "", tex,
                                " & ".join(cells)))
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
        parts.append("\n".join(lines))

    path = os.path.join(out_dir, filename)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(parts))
    print("wrote", path)
    return path


def write_run_config(out_dir=None, tuned=None, n_sims=None, seconds=None):
    """Everything that affects a reported number, next to the numbers."""
    import platform
    import sklearn
    out_dir = out_dir or OUT_DIR
    lines = ["run finished %s" % time.strftime("%Y-%m-%d %H:%M:%S"),
             "python %s on %s" % (platform.python_version(),
                                  platform.platform()),
             "numpy %s  pandas %s  scikit-learn %s"
             % (np.__version__, pd.__version__, sklearn.__version__),
             "",
             "n_sims=%s  n_train=%d  n_test=%d  p=%d  sigma_e=%.3g"
             % (n_sims, N_TRAIN, N_TEST, P, SIGMA_E),
             "dgps=%s" % (DGPS,),
             "seed_base=%d  beta_seed=%d  w_seed_base=%d"
             % (SEED_BASE, BETA_SEED, W_SEED_BASE),
             "backend=%s  n_trees=%d  crf_train_row=%s"
             % (BACKEND, N_TREES, CRF_TRAIN_ROW),
             "crf_variants=%s" % (CRF_VARIANTS,),
             "baselines=%s  rf_match_crf=%s" % (BASELINES, RF_MATCH_CRF),
             "dgp_display=%s" % (DGP_DISPLAY,),
             "forest_params=%s" % (FOREST_PARAMS,),
             "rf_params=%s" % (RF_PARAMS,),
             "nn_params=%s" % (NN_PARAMS,),
             "xgb_params=%s" % (XGB_PARAMS,),
             "tune_forest=%s  tune_seed=%d  tune_n_estimators=%d"
             % (TUNE_FOREST, TUNE_SEED, TUNE_N_ESTIMATORS),
             "tune_grid=%s" % (TUNE_GRID,)]
    if tuned:
        lines.append("")
        for dgp, kw in tuned.items():
            lines.append("selected forest [%s]: %s" % (dgp, kw))
    if seconds is not None:
        lines += ["", "elapsed %.1f s" % seconds]
    path = os.path.join(out_dir, RUN_CONFIG_TXT)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", path)
    return path


def print_table(per_sim, tuned=None):
    n = per_sim["sim"].nunique()
    methods = [m for m in METHOD_ORDER if m in set(per_sim.method)]
    print("\n=========== mean (SD) over %d simulations ===========" % n)
    print("backend=%s  n_train=%d  n_test=%d  p=%d  n_trees=%d"
          % (BACKEND, N_TRAIN, N_TEST, P, N_TREES))
    print("CRF variants actually fitted:")
    for name, kw in CRF_VARIANTS.items():
        print("  %-14s %s" % (name, ", ".join("%s=%s" % (k, kw[k])
                                              for k in sorted(kw))))
    print("baselines: %s" % ", ".join(BASELINES))
    print("base forest: %s"
          % ("tuned per DGP (see below)" if tuned
             else "fixed, %s" % dict(FOREST_PARAMS)))

    lam = per_sim[per_sim.split == "Test"]
    for dgp in DGPS:
        for meth in methods:
            d = lam[(lam.dgp == dgp) & (lam.method == meth)]["lam"].dropna()
            if len(d) and d.std(ddof=0) > 1e-9:
                print("  %-14s %-12s lam: mean %.3f (SD %.3f)"
                      % (dgp, meth, d.mean(), d.std()))

    for dgp in DGPS:
        if dgp not in set(per_sim.dgp):
            continue
        print("\n--- %s ---" % dgp)
        if tuned and dgp in tuned:
            print("  forest: %s" % tuned[dgp])
        for split in ("Train", "Test"):
            for met, label, _tex, prec in TABLE_ROWS:
                cells = ["%s %s" % (m, format_cell(per_sim, dgp, split, m,
                                                   met, prec))
                         for m in methods]
                print("  %-5s %-26s : %s" % (split, label, "  ".join(cells)))


# ==================================================================
# STUDY DRIVER
# ==================================================================

def run_study(n_sims=None, workers=None, dgps=None, methods=None,
              merge=False):
    """Fit and score every method on every DGP.

    `dgps` and `methods` restrict what is FITTED; `merge` folds the
    result into the existing results files instead of replacing them,
    so a method added later does not cost a refit of the ones already
    scored. Partial runs imply merging.
    """
    n_sims = N_SIMS if n_sims is None else n_sims
    workers = N_WORKERS if workers is None else workers
    dgps = DGPS if dgps is None else tuple(dgps)
    methods = METHOD_ORDER if methods is None else tuple(methods)
    partial = (set(dgps) != set(DGPS)) or (set(methods) != set(METHOD_ORDER))
    merge = merge or partial
    os.makedirs(OUT_DIR, exist_ok=True)
    t0 = time.time()
    if partial:
        print("partial run: dgps=%s methods=%s (results will be merged)"
              % (list(dgps), list(methods)), flush=True)

    tuned = {}
    for dgp in dgps:
        if TUNE_FOREST:
            print("tuning base forest on %s (pilot seed %d)"
                  % (dgp, TUNE_SEED), flush=True)
            tuned[dgp] = select_forest_params(dgp, verbose=True)
        else:
            tuned[dgp] = dict(FOREST_PARAMS)

    tasks = [(d, s, tuned[d], methods)
             for d in dgps for s in range(n_sims)]
    rows, frames, stage_frames = [], [], []

    def absorb(result):
        r, fr, st = result
        rows.extend(r)
        if fr is not None:
            frames.append(fr)
        if st is not None:
            stage_frames.append(st)

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for i, res in enumerate(pool.map(run_one, tasks, chunksize=1)):
                absorb(res)
                if (i + 1) % 10 == 0 or i + 1 == len(tasks):
                    el = time.time() - t0
                    print("[%4d/%d] %7.1fs (~%.1fs/rep)"
                          % (i + 1, len(tasks), el, el / (i + 1)),
                          flush=True)
    else:
        for i, task in enumerate(tasks):
            absorb(run_one(task))
            print("[%4d/%d] %s %7.1fs"
                  % (i + 1, len(tasks), task[:2], time.time() - t0),
                  flush=True)

    per_sim = pd.DataFrame(rows)
    per_sim_path = os.path.join(OUT_DIR, PER_SIM_CSV)
    per_sim = sort_per_sim(merge_csv(per_sim, per_sim_path,
                                     ["dgp", "sim", "method", "split"])
                           if merge else per_sim)
    per_sim.to_csv(per_sim_path, index=False)
    summary = summarize(per_sim)
    summary.to_csv(os.path.join(OUT_DIR, SUMMARY_CSV), index=False)
    if frames:
        ap = os.path.join(OUT_DIR, ALL_PRED_CSV)
        merge_csv(pd.concat(frames, ignore_index=True), ap,
                  ["dgp", "sim", "method", "split", "obs_index"])
        print("wrote", ap)
    if stage_frames:
        sp = os.path.join(OUT_DIR, STAGE_CSV)
        merge_csv(pd.concat(stage_frames, ignore_index=True), sp,
                  ["dgp", "method", "sim", "stage"])
        print("wrote", sp)

    print_table(per_sim, tuned)
    write_results_table(per_sim)
    write_latex_table(per_sim)

    sweep = lambda_sweep(n_reps=SWEEP_REPS, dgps=dgps, forest_kw=tuned) \
        if RUN_SWEEP else None
    if MAKE_FIGURES:
        make_all_figures(per_sim=per_sim, sweep=sweep, tuned=tuned,
                         dgps=dgps)
    write_run_config(tuned=tuned, n_sims=n_sims, seconds=time.time() - t0)
    print("\nDone in %.1fs. Outputs in %s" % (time.time() - t0, OUT_DIR))
    return per_sim, summary


# ==================================================================
# ENTRY POINT
# ==================================================================

if __name__ == "__main__":
    args = sys.argv[1:]
    rest = [a for a in args if not a.startswith("--")]

    def _list_opt(name):
        """--name=a,b  ->  ('a', 'b');  absent -> None."""
        for a in args:
            if a.startswith("--%s=" % name):
                return tuple(x.strip() for x in a.split("=", 1)[1].split(",")
                             if x.strip())
        return None

    _dgps, _methods = _list_opt("dgps"), _list_opt("methods")

    def _tuned_or_none():
        """Reproduce the study's forest configuration. Deterministic at
        TUNE_SEED, so this returns exactly what run_study selected."""
        if not TUNE_FOREST:
            return None
        return {d: select_forest_params(d) for d in DGPS}

    if "--figures" in args:
        make_all_figures(sim=int(rest[0]) if rest else 0,
                         sweep_from_disk=True, tuned=_tuned_or_none())
    elif "--tables" in args:
        _ps = pd.read_csv(os.path.join(OUT_DIR, PER_SIM_CSV))
        print_table(_ps)
        write_results_table(_ps)
        write_latex_table(_ps)
    elif "--sweep" in args:
        lambda_sweep(n_reps=int(rest[0]) if rest else 5,
                     forest_kw=_tuned_or_none())
    else:
        run_study(int(rest[0]) if rest else None, dgps=_dgps,
                  methods=_methods, merge="--merge" in args)