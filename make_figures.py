#!/usr/bin/env python3
"""
make_figures.py
===============
Reads the outputs of run_simulation.py (predictions_sim0.csv,
results_per_sim.csv, lambda_sweep.csv, stage_metrics.csv) and
redraws
the key figures with all three DGPs stacked vertically and substantially
increased font sizes for publication-quality readability.

The run directory is expected to hold one subfolder per study:

    <root>/            run_simulation.py output: CRF, Padding, RF, NN
                       and XGBoost, all from the one study
    <root>/v3/, <root>/padding/
                       alternative locations, kept so a run directory
                       written by the earlier two-script layout still
                       reads

Rows carrying the same dgp/method/split in more than one of them are
resolved in favour of the earlier entry in STUDY_DIRS.

USAGE
    python make_figures.py              # default: reads sim_out/
    python make_figures.py /path/to/sim_out

OUTPUTS  (all in <root>/restacked_figures/)
    stacked_obs_pred_full.png       observed vs predicted, Train and
                                    Test in one grid
    stacked_obs_pred_train.png      observed vs predicted, Train split
    stacked_obs_pred_test.png       observed vs predicted, Test split
    stacked_metrics_test.png        metric boxplots, Test split
    stacked_metrics_train.png       metric boxplots, Train split
    stacked_theta.png               downstream association slopes
    stacked_lambda_curves.png       lambda sweep curves
    stacked_stage_curves.png        padding stage trajectories
"""

import os
import sys
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# CONFIG  --  adjust these paths if your output directory differs
# ------------------------------------------------------------------
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_out")

# Where to look for the CSVs, in order; earlier entries win when rows
# collide. The study writes into the run directory itself, which is
# the "." entry; the two subfolder names are kept so that a directory
# written by the earlier layout, with the padding study in its own
# folder, still reads.
STUDY_DIRS = ("v3", "padding", ".")

FIG_DIR = os.path.join(OUT_DIR, "restacked_figures")
os.makedirs(FIG_DIR, exist_ok=True)

PRED_NAME = "predictions_sim0.csv"
RESULTS_NAME = "results_per_sim.csv"
SWEEP_NAME = "lambda_sweep.csv"
STAGE_NAME = "stage_metrics.csv"

# ------------------------------------------------------------------
# GLOBAL FONT RAMP-UP  --  "substantial" increases everywhere
# ------------------------------------------------------------------
plt.rcParams.update({
    "font.size": 18,
    "axes.titlesize": 22,
    "axes.labelsize": 20,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
    "figure.titlesize": 24,
    "axes.linewidth": 1.5,
    "xtick.major.width": 1.5,
    "ytick.major.width": 1.5,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
})

# ------------------------------------------------------------------
# METHOD ORDER & UPDATED PALETTE
# ------------------------------------------------------------------
_BASE_COLORS = {
    "CRF": "#2B5C8F",          # Deep Slate Blue
    "RF": "#2A9D8F",           # Teal
    "NN": "#E76F51",           # Coral / Burnt Orange
    "XGBoost": "#E9C46A",      # Warm Yellow
    "Lasso": "#6A4C93",        # Deep Violet
    "RF-uncorr": "#577590",    # Muted Blue-Gray
    "Padding": "#B5179E",      # Magenta
}

_VARIANT_CYCLE = (
    "#D90429", "#118AB2", "#06D6A0", "#FFD166",
    "#8338EC", "#3A86FF", "#FB5607"
)

# Column order used by every figure. All five are drawn side by side.
METHOD_ORDER = ("CRF", "Padding", "RF", "NN", "XGBoost")

# Methods produced by the post-hoc padding study (stage-trajectory figure).
PADDING_METHODS = ("Padding",)


def method_color(name):
    if name in _BASE_COLORS:
        return _BASE_COLORS[name]
    variants = [m for m in METHOD_ORDER if m not in _BASE_COLORS]
    i = variants.index(name) if name in variants else 0
    return _VARIANT_CYCLE[i % len(_VARIANT_CYCLE)]


METHOD_COLORS = {m: method_color(m) for m in METHOD_ORDER}

# Shared element colors
_FIT_LINE_COLOR = "#D90429"      # Crimson Red for slope fit lines
_OOB_CURVE_COLOR = "#2B5C8F"      # Slate Blue for OOB curves
_TEST_CURVE_COLOR = "#E76F51"     # Coral Orange for Test curves

# ------------------------------------------------------------------
# DGP LABELS
# ------------------------------------------------------------------
DGP_DISPLAY = {"linear_sparse": "linear",
               "linear": "linear",
               "friedman1": "friedman1",
               "friedman2": "friedman2"}

DGP_LABELS = {"linear_sparse": "Linear (sparse)",
              "linear": "Linear (dense)",
              "friedman1": "Friedman 1",
              "friedman2": "Friedman 2"}

DGP_ORDER = ["linear_sparse", "friedman1", "friedman2"]

# ------------------------------------------------------------------
# METRIC DEFINITIONS
#
# This is exactly the set reported in the paper's tables. r^2 (both
# variants) and the two group residuals r_U, r_L were computed in
# earlier runs but never reported, so they were dropped here along
# with the stacked_constraints_*.png figure that existed only to show
# them.
# ------------------------------------------------------------------
METRIC_KEYS = ("bias", "rmse", "mae", "theta_delta")
SUMMARY_COLS = list(METRIC_KEYS)
METRIC_LABELS = {
    "bias":        r"$\hat{\eta}$ (slope)",
    "rmse":        "RMSE",
    "mae":         "MAE",
    "theta_delta": r"$\hat{\theta}-\hat{\theta}_{\mathrm{oracle}}$",
}
METRIC_REFERENCE = {"bias": 1.0, "rmse": None, "mae": None,
                    "theta_delta": 0.0}


def eta_slope(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.cov(y_true, y_pred, bias=True)[0, 1] / np.var(y_true))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_pred, float)
                                 - np.asarray(y_true, float)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_pred, float)
                                - np.asarray(y_true, float))))


def theta_slope(w, v):
    w, v = np.asarray(w, float), np.asarray(v, float)
    return float(np.cov(w, v, bias=True)[0, 1] / np.var(w))


def theta_delta(y_true, y_pred, w):
    return theta_slope(w, y_pred) - theta_slope(w, y_true)


def compute_metrics(y_true, y_pred, w=None):
    return dict(bias=eta_slope(y_true, y_pred),
                rmse=rmse(y_true, y_pred),
                mae=mae(y_true, y_pred),
                theta_delta=(np.nan if w is None
                             else theta_delta(y_true, y_pred, w)))


# ------------------------------------------------------------------
# DATA LOADING HELPERS
# ------------------------------------------------------------------
def _study_paths(filename):
    """Existing <root>/<study>/<filename> paths, in STUDY_DIRS order."""
    found = []
    for study in STUDY_DIRS:
        path = os.path.join(OUT_DIR, study, filename)
        if os.path.exists(path):
            found.append((study, path))
    return found


def _load_studies(filename, dedupe_keys, description):
    """Concatenate one CSV across study subfolders; earlier studies win."""
    paths = _study_paths(filename)
    if not paths:
        print("WARNING: no %s found under %s. Skipping %s."
              % (filename, OUT_DIR, description))
        return None
    frames = []
    for study, path in paths:
        df = pd.read_csv(path)
        df["study"] = study
        frames.append(df)
        print("read %s (%d rows)" % (path, len(df)))
    merged = pd.concat(frames, ignore_index=True)
    keys = [k for k in dedupe_keys if k in merged.columns]
    if keys:
        merged = merged.drop_duplicates(subset=keys, keep="first")
    return merged


def load_predictions():
    """Return predictions DataFrame, or None if missing."""
    df = _load_studies(PRED_NAME,
                       ["dgp", "method", "split", "obs_index"],
                       "scatter / theta figures")
    if df is None:
        return None
    if "lam" not in df.columns:
        df["lam"] = np.nan
    return df.sort_values(["dgp", "method", "split", "obs_index"])


def load_results():
    """Return results_per_sim DataFrame, or None if missing."""
    return _load_studies(RESULTS_NAME,
                         ["dgp", "method", "split", "sim"],
                         "boxplot / constraint figures")


def load_sweep():
    """Return lambda_sweep DataFrame, or None if missing."""
    return _load_studies(SWEEP_NAME, ["dgp", "lam", "sim"], "lambda curves")


def load_stages():
    """Return the padding stage trajectory DataFrame, or None if missing."""
    return _load_studies(STAGE_NAME,
                         ["dgp", "method", "sim", "stage"],
                         "stage trajectory figure")


def get_pred_vectors(df, dgp, method, split):
    """Extract (y_obs, y_pred, w_obs) arrays for one DGP/method/split."""
    sub = df[(df.dgp == dgp) & (df.method == method) & (df.split == split)]
    if sub.empty:
        return None, None, None
    sub = sub.sort_values("obs_index")
    return (sub["y_obs"].to_numpy(float),
            sub["y_pred"].to_numpy(float),
            sub["w_obs"].to_numpy(float))


# ------------------------------------------------------------------
# SHARED PLOTTING UTILITIES
# ------------------------------------------------------------------
def _boxplot(ax, data, labels, widths=0.6):
    kw = dict(patch_artist=True, widths=widths,
              medianprops=dict(color="#1D3557", lw=2.2),
              flierprops=dict(marker="o", markersize=4, alpha=0.6))
    try:
        return ax.boxplot(data, tick_labels=labels, **kw)
    except TypeError:
        return ax.boxplot(data, labels=labels, **kw)


def _scatter_panel(ax, y, p, color, lo, hi, heading=None,
                   annotate=("bias",), lam=None):
    """Single observed-vs-predicted panel."""
    met = compute_metrics(y, p)
    b = met["bias"]
    a0 = float(np.mean(p) - b * np.mean(y))

    ax.scatter(y, p, s=45, alpha=0.55, color=color, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.8)
    xs = np.array([lo, hi])
    ax.plot(xs, b * xs + a0, color=_FIT_LINE_COLOR, lw=2.2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel("Observed", fontsize=18)
    ax.set_ylabel("Predicted", fontsize=18)
    if heading:
        ax.set_title(heading, fontweight="normal", fontsize=20, pad=10)
    lines = ["%s = %.3f" % (METRIC_LABELS.get(k, k), met[k])
             for k in annotate]
    if lam is not None and np.isfinite(lam):
        lines.append(r"$\lambda$ = %.2f" % lam)
    ax.text(0.04, 0.96, "\n".join(lines), transform=ax.transAxes,
            fontsize=14, va="top",
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white",
                      alpha=0.85,
                      edgecolor="gray"))
    ax.grid(True, alpha=0.25)
    ax.minorticks_on()
    return met


# ------------------------------------------------------------------
# FIGURE 1b:  OBSERVED vs PREDICTED, FULL GRID STYLE
# ------------------------------------------------------------------
def _scatter_panel_ref(ax, y, p, color="#2B5C8F", heading=None):
    """One observed-vs-predicted panel in reference style with customizable color."""
    b = eta_slope(y, p)
    a0 = float(np.mean(p) - b * np.mean(y))

    lo = float(min(y.min(), p.min()))
    hi = float(max(y.max(), p.max()))
    pad = 0.08 * (hi - lo)
    lo, hi = lo - pad, hi + pad

    ax.scatter(y, p, s=32, alpha=0.6, color=color, edgecolors="none")
    ax.plot([lo, hi], [lo, hi], "k--", lw=1.6)
    xs = np.array([lo, hi])
    ax.plot(xs, b * xs + a0, color=_FIT_LINE_COLOR, lw=2.2,
           label=r"$\hat{\eta}$ = %.3f" % b)
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    if heading:
        ax.set_title(heading, fontsize=22, pad=10)
    ax.legend(loc="upper left", frameon=True, fontsize=17,
             handlelength=1.4, handletextpad=0.5, borderpad=0.5)
    ax.tick_params(labelsize=13)


def make_full_obs_pred_grid(pred_df):
    """One figure: rows = (DGP x split), columns = methods."""
    methods = [m for m in METHOD_ORDER if m in pred_df.method.unique()]
    dgps = [d for d in DGP_ORDER if d in pred_df.dgp.unique()]
    splits = ("Train", "Test")
    row_specs = [(dgp, sp) for dgp in dgps for sp in splits]
    n_rows, n_cols = len(row_specs), len(methods)
    if n_rows == 0 or n_cols == 0:
        return

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4.6 * n_cols, 4.3 * n_rows),
                             squeeze=False)

    for r, (dgp, split) in enumerate(row_specs):
        for j, m in enumerate(methods):
            ax = axes[r][j]
            y, p, _ = get_pred_vectors(pred_df, dgp, m, split)
            if y is None:
                ax.set_visible(False)
                continue
            heading = m if r == 0 else None
            # Pass method-specific color for orderly visual distinction
            m_color = METHOD_COLORS.get(m, "#2B5C8F")
            _scatter_panel_ref(ax, y, p, color=m_color, heading=heading)
            if j == 0:
                ax.set_ylabel("%s\n%s\nPredicted $\\hat{y}$"
                              % (DGP_LABELS.get(dgp, dgp), split),
                              fontsize=19)
            else:
                ax.set_ylabel("")
            if r == n_rows - 1:
                ax.set_xlabel("Observed $y$", fontsize=19)

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stacked_obs_pred_full.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


# ------------------------------------------------------------------
# FIGURE 1:  OBSERVED vs PREDICTED  (stacked by DGP)
# ------------------------------------------------------------------
def make_stacked_obs_pred(pred_df, split="Test"):
    """One big figure: DGPs as rows, methods as columns."""
    methods = [m for m in METHOD_ORDER if m in pred_df.method.unique()]
    dgps = [d for d in DGP_ORDER if d in pred_df.dgp.unique()]
    n_dgp = len(dgps)
    n_meth = len(methods)

    fig, axes = plt.subplots(n_dgp, n_meth,
                             figsize=(4.2 * n_meth, 4.0 * n_dgp),
                             squeeze=False)

    for i, dgp in enumerate(dgps):
        # Axis limits are pooled within a DGP: the three outcome scales are
        # far apart, so a single global range would flatten two of the rows.
        vals = []
        for m in methods:
            y, p, _ = get_pred_vectors(pred_df, dgp, m, split)
            if y is not None:
                vals.extend([y, p])
        if not vals:
            for ax in axes[i]:
                ax.set_visible(False)
            continue
        lo = float(min(v.min() for v in vals))
        hi = float(max(v.max() for v in vals))
        pad = 0.15 * (hi - lo)
        lo, hi = lo - pad, hi + pad

        for j, m in enumerate(methods):
            ax = axes[i][j]
            y, p, _ = get_pred_vectors(pred_df, dgp, m, split)
            if y is None:
                ax.set_visible(False)
                continue
            lam_df = pred_df[(pred_df.dgp == dgp) & (pred_df.method == m)
                             & (pred_df.split == split)]["lam"]
            lam = lam_df.iloc[0] if len(lam_df) else np.nan
            heading = m if i == 0 else None
            _scatter_panel(ax, y, p, METHOD_COLORS.get(m, "#577590"),
                           lo, hi, heading=heading,
                           annotate=("bias", "rmse"), lam=lam)
            if j == 0:
                ax.set_ylabel(DGP_LABELS.get(dgp, dgp), fontsize=18)

    fig.suptitle("Observed vs Predicted  (%s split)" % split,
                 fontsize=24, fontweight="bold", y=1.02)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stacked_obs_pred_%s.png" % split.lower())
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


# ------------------------------------------------------------------
# FIGURE 2:  METRIC BOXPLOTS  (stacked by DGP)
# ------------------------------------------------------------------
def make_stacked_metrics(per_sim, split="Test"):
    """One figure: DGPs as rows, metrics as columns."""
    methods = [m for m in METHOD_ORDER if m in per_sim.method.unique()]
    dgps = [d for d in DGP_ORDER if d in per_sim.dgp.unique()]
    n_dgp = len(dgps)
    keys = list(METRIC_KEYS)
    n_met = len(keys)

    fig, axes = plt.subplots(n_dgp, n_met,
                             figsize=(max(3.5, 0.75 * len(methods) + 2.2) * n_met,
                                      5.0 * n_dgp),
                             squeeze=False,
                             sharex="col")

    for i, dgp in enumerate(dgps):
        d = per_sim[(per_sim.dgp == dgp) & (per_sim.split == split)]
        for j, met in enumerate(keys):
            ax = axes[i][j]
            data, labels, colors = [], [], []
            for meth in methods:
                v = d[d.method == meth][met].dropna().values
                if len(v):
                    data.append(v)
                    labels.append(meth)
                    colors.append(METHOD_COLORS.get(meth, "#577590"))
            if not data:
                ax.set_visible(False)
                continue
            bp = _boxplot(ax, data, labels, widths=0.55)
            for patch, c in zip(bp["boxes"], colors):
                patch.set_facecolor(c)
                patch.set_alpha(0.60)
                patch.set_edgecolor(c)
                patch.set_linewidth(1.5)
            ref = METRIC_REFERENCE.get(met)
            if ref is not None:
                ax.axhline(ref, color="black", ls="--", lw=1.8)
            ax.set_ylabel(METRIC_LABELS.get(met, met), fontsize=18)
            ax.grid(True, axis="y", alpha=0.25)
            ax.tick_params(axis="both", which="both", labelsize=14)
            if len(labels) > 3:
                ax.set_xticks(range(1, len(labels) + 1))
                ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=14)
            if i == 0:
                ax.set_title(METRIC_LABELS.get(met, met), fontsize=20, pad=10)
            if j == 0:
                ax.text(-0.35, 0.5, DGP_LABELS.get(dgp, dgp),
                        transform=ax.transAxes, fontsize=18,
                        rotation=90, va="center", ha="center")

    fig.suptitle("Metric Distributions  (%s split)" % split,
                 fontsize=24, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stacked_metrics_%s.png" % split.lower())
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


# ------------------------------------------------------------------
# FIGURE 3:  DOWNSTREAM ASSOCIATION (THETA)  (stacked by DGP)
# ------------------------------------------------------------------
def make_stacked_theta(pred_df):
    """One figure: DGPs as rows, methods as columns."""
    methods = [m for m in METHOD_ORDER if m in pred_df.method.unique()]
    dgps = [d for d in DGP_ORDER if d in pred_df.dgp.unique()]
    n_dgp = len(dgps)
    n_meth = len(methods)

    fig, axes = plt.subplots(n_dgp, n_meth,
                             figsize=(4.0 * n_meth, 4.0 * n_dgp),
                             squeeze=False,
                             sharex="row", sharey="row")

    for i, dgp in enumerate(dgps):
        for j, m in enumerate(methods):
            ax = axes[i][j]
            # Each method carries its own W and its own observed y: the two
            # studies draw their replications independently, so the oracle
            # line has to be recomputed within the panel rather than shared.
            y, p, w = get_pred_vectors(pred_df, dgp, m, "Test")
            if (p is None or y is None or w is None
                    or not np.isfinite(w).any()):
                ax.set_visible(False)
                continue
            th_o = theta_slope(w, y)
            ws = np.array([w.min(), w.max()])
            th = theta_slope(w, p)
            ax.scatter(w, p, s=45, alpha=0.50, color=METHOD_COLORS.get(m),
                       edgecolors="none")
            ax.plot(ws, th * (ws - w.mean()) + p.mean(),
                    color=METHOD_COLORS.get(m), lw=2.5)
            ax.plot(ws, th_o * (ws - w.mean()) + y.mean(), color="black",
                    ls="--", lw=2.0)
            ax.set_xlabel(r"$W$", fontsize=18)
            ax.set_ylabel("Predicted", fontsize=18)
            if i == 0:
                ax.set_title(m, fontsize=20, pad=10)
            txt = (r"$\hat{\theta}$ = %.3f" "\n"
                   r"$\hat{\theta}-\hat{\theta}_{\mathrm{oracle}}$ = %.3f"
                   % (th, th - th_o))
            ax.text(0.04, 0.96, txt,
                    transform=ax.transAxes, fontsize=14, va="top",
                    bbox=dict(boxstyle="round,pad=0.3",
                              facecolor="white", alpha=0.85,
                              edgecolor="gray"))
            ax.grid(True, alpha=0.25)
            ax.minorticks_on()
            if j == 0:
                ax.set_ylabel(DGP_LABELS.get(dgp, dgp) + "\nPredicted",
                              fontsize=18)

    fig.suptitle("Downstream Association Slopes  (Test split)",
                 fontsize=24, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stacked_theta.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


# ------------------------------------------------------------------
# FIGURE 4:  LAMBDA SWEEP CURVES  (stacked by DGP)
# ------------------------------------------------------------------
def make_stacked_lambda_curves(sweep_df):
    """One figure: DGPs as rows, (slope, RMSE) as columns."""
    dgps = [d for d in DGP_ORDER if d in sweep_df.dgp.unique()]
    n_dgp = len(dgps)

    fig, axes = plt.subplots(n_dgp, 2,
                             figsize=(11.0, 5.0 * n_dgp),
                             squeeze=False)

    for i, dgp in enumerate(dgps):
        d = (sweep_df[sweep_df.dgp == dgp]
             .groupby("lam")[["oob_eta", "test_eta", "test_rmse"]]
             .mean())
        lam = d.index.values

        # --- left: slopes ---
        ax = axes[i][0]
        ax.plot(lam, d["oob_eta"].values, color=_OOB_CURVE_COLOR, lw=2.5,
                marker="o", ms=7, label="OOB")
        ax.plot(lam, d["test_eta"].values, color=_TEST_CURVE_COLOR, lw=2.5,
                marker="s", ms=7, label="Test")
        ax.axhline(1.0, color="black", ls="--", lw=1.8)
        ax.axvline(1.0, color="gray", ls=":", lw=1.5)
        lam_star = float(np.interp(1.0, d["oob_eta"].values, lam))
        ax.axvline(lam_star, color=_OOB_CURVE_COLOR, ls=":", lw=1.8)
        ax.set_xlabel(r"$\lambda$", fontsize=18)
        ax.set_ylabel(r"$\hat{\eta}$ (slope)", fontsize=18)
        ax.legend(frameon=False, fontsize=16)
        ax.grid(True, alpha=0.25)
        ax.minorticks_on()
        if i == 0:
            ax.set_title("Calibration Slope", fontsize=20, pad=10)
        ax.text(-0.18, 0.5, DGP_LABELS.get(dgp, dgp),
                transform=ax.transAxes, fontsize=18,
                rotation=90, va="center", ha="center")

        # --- right: RMSE ---
        ax = axes[i][1]
        ax.plot(lam, d["test_rmse"].values, color=_TEST_CURVE_COLOR, lw=2.5,
                marker="s", ms=7)
        ax.axvline(1.0, color="gray", ls=":", lw=1.5)
        ax.axvline(lam_star, color=_OOB_CURVE_COLOR, ls=":", lw=1.8)
        ax.set_xlabel(r"$\lambda$", fontsize=18)
        ax.set_ylabel("Test RMSE", fontsize=18)
        ax.grid(True, alpha=0.25)
        ax.minorticks_on()
        if i == 0:
            ax.set_title("Test RMSE", fontsize=20, pad=10)

    fig.suptitle("Lambda Sweep Curves", fontsize=24, fontweight="bold", y=1.01)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stacked_lambda_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


# ------------------------------------------------------------------
# FIGURE 5:  PADDING STAGE TRAJECTORIES  (stacked by DGP)
# ------------------------------------------------------------------
def make_stacked_stage_curves(stage_df):
    """One figure: DGPs as rows, (slope, RMSE) as columns.

    Solid lines are the Test split, dashed the Train split, so the gap
    between them shows how much of each stage's slope gain is in-sample.
    Values are averaged over replications.
    """
    dgps = [d for d in DGP_ORDER if d in stage_df.dgp.unique()]
    methods = [m for m in PADDING_METHODS if m in stage_df.method.unique()]
    if not dgps or not methods:
        return
    n_dgp = len(dgps)

    fig, axes = plt.subplots(n_dgp, 2,
                             figsize=(12.0, 5.0 * n_dgp),
                             squeeze=False)

    for i, dgp in enumerate(dgps):
        for col, (metric, ylabel) in enumerate(
                (("bias", METRIC_LABELS["bias"]), ("rmse", "RMSE"))):
            ax = axes[i][col]
            for m in methods:
                d = (stage_df[(stage_df.dgp == dgp) & (stage_df.method == m)]
                     .groupby("stage")
                     .mean(numeric_only=True))
                stages = d.index.values
                color = METHOD_COLORS.get(m, "#577590")
                test_col, train_col = "test_" + metric, "train_" + metric
                if test_col in d:
                    ax.plot(stages, d[test_col].values, color=color, lw=2.5,
                            marker="s", ms=7, label="%s (Test)" % m)
                if train_col in d:
                    ax.plot(stages, d[train_col].values, color=color, lw=2.2,
                            ls="--", marker="o", ms=6, alpha=0.75,
                            label="%s (Train)" % m)
            ref = METRIC_REFERENCE.get(metric)
            if ref is not None:
                ax.axhline(ref, color="black", ls="--", lw=1.8)
            ax.set_xlabel("Stage", fontsize=18)
            ax.set_ylabel(ylabel, fontsize=18)
            ax.grid(True, alpha=0.25)
            ax.minorticks_on()
            if i == 0 and col == 0:
                ax.legend(frameon=False, fontsize=14)
            if col == 0:
                ax.text(-0.26, 0.5, DGP_LABELS.get(dgp, dgp),
                        transform=ax.transAxes, fontsize=18,
                        rotation=90, va="center", ha="center")

    fig.tight_layout()
    path = os.path.join(FIG_DIR, "stacked_stage_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote", path)
    return path


# ------------------------------------------------------------------
# MAIN
# ------------------------------------------------------------------
def main():
    print("=" * 60)
    print("Restacking figures from: %s" % OUT_DIR)
    print("Studies: %s" % ", ".join(STUDY_DIRS))
    print("Output directory: %s" % FIG_DIR)
    print("=" * 60)

    pred_df = load_predictions()
    res_df = load_results()
    sweep_df = load_sweep()
    stage_df = load_stages()

    if pred_df is not None:
        make_full_obs_pred_grid(pred_df)
        make_stacked_obs_pred(pred_df, split="Train")
        make_stacked_obs_pred(pred_df, split="Test")
        make_stacked_theta(pred_df)

    if res_df is not None:
        make_stacked_metrics(res_df, split="Train")
        make_stacked_metrics(res_df, split="Test")

    if sweep_df is not None:
        make_stacked_lambda_curves(sweep_df)

    if stage_df is not None:
        make_stacked_stage_curves(stage_df)

    print("\nDone. All restacked figures written to:")
    print("  %s" % FIG_DIR)


if __name__ == "__main__":
    main()