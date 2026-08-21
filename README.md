# Unbiased Predictions for Continuous Outcomes Using Constrained Random Forests

This package contains the complete source code for the experiments reported in
the paper: the constrained random forest (CRF), the leaf-correction algorithms
it is assembled from, the post-hoc padding comparator, the three comparison
estimators, the three data-generating processes, the evaluation metrics and the
figure code. Everything runs from scratch with no external inputs -- the
experiments are synthetic, and the two studies are three commands end to end.

---

## 1. Layout

```
run_simulation.py   the CRF study: three DGPs x 100 replications, with the
                    RF, NN and XGBoost baselines
run_padding.py      the post-hoc padding comparator on the same three DGPs,
                    at the same seeds
make_figures.py     redraws the key figures with the three DGPs stacked
                    vertically at publication font sizes
requirements.txt    dependencies
```

`run_simulation.py` is self-contained: the trees, the corrections, the forest,
the metrics, the DGPs, the baselines, the study loop, the result files and the
figures all live in one file, sectioned in the order a reader follows one
replication, with a banner at each section. The configuration block sits at
the top, under `CONFIG`, and nothing that affects a reported number is set on
the command line.

| Banner | Contents |
| --- | --- |
| `CONFIG` | every setting of the run |
| `CUSTOM TREE` | `CustomTree`, the pure-NumPy reference implementation, and `SkTreeWrapper`, the bagged `DecisionTreeRegressor` of the default backend |
| `PER-TREE CORRECTIONS` | `correct_tree_lambda` (ALG 1), `correct_greedy` (ALG 2), `correct_exact` (the closed form) |
| `FOREST` | `UnbiasedForest`: bagging, exact out-of-bag sets, the forest-level correction, and lambda selection |
| `METRICS` | eta-hat, RMSE, MAE, Pearson and OLS r-squared, the two group residuals, the downstream association slope |
| `DATA-GENERATING PROCESSES` | the three DGPs, plus base-forest selection |
| `ONE REPLICATION` | `run_one`, the unit the process pool distributes |
| `LAMBDA SWEEP`, `CORRECTION COMPARISON` | the two analyses reachable from `--sweep` and `--compare` |
| `FIGURES`, `TABLES`, `STUDY DRIVER`, `ENTRY POINT` | everything written to disk, and the command line |

`run_padding.py` is a separate study rather than another method inside the
first, because its stopping rule makes each fit a different length. It carries
its own copy of the DGP block, the metrics and the proxy, transcribed from
`run_simulation.py` (Section 5), and writes into a `padding/` subdirectory of
the same run directory in the schema `make_figures.py` reads.

---

## 2. Requirements

```
numpy
pandas
scikit-learn
matplotlib
xgboost
```

```
pip install -r requirements.txt
```

`xgboost` is required. The gradient-boosting column of the results table is
produced by `xgboost.XGBRegressor` at the hyperparameters in `XGB_PARAMS`;
scikit-learn's `HistGradientBoostingRegressor` is a different model at
different settings, so there is no silent fallback -- a missing `xgboost` stops
the run with a message rather than filling the column with another estimator.
Dropping `"XGBoost"` from `BASELINES` is the explicit opt-out.

Both drivers are spawn-safe (Windows and macOS) and target Python 3.11 or
earlier. No data files are needed: the three DGPs are generated at the seeds in
the config blocks.

---

## 3. Running

```bash
# the CRF study: three DGPs x 100 replications      -> sim_out/
python run_simulation.py

# the padding comparator on the same samples        -> sim_out/padding/
python run_padding.py sim_out

# the stacked figures                               -> sim_out/restacked_figures/
python make_figures.py sim_out
```

`run_simulation.py` takes these arguments:

| Command | Effect |
| --- | --- |
| `python run_simulation.py` | the full study at the configured `N_SIMS` |
| `python run_simulation.py 10` | a short run, 10 replications |
| `python run_simulation.py --figures` | redraw the figures from saved results |
| `python run_simulation.py --tables` | rebuild `results_table.csv` / `.tex` from `results_per_sim.csv` |
| `python run_simulation.py --sweep` | the lambda sweep, all DGPs |
| `python run_simulation.py --compare` | the corrections head to head |
| `python run_simulation.py --dgps=friedman1,friedman2` | fit only these DGPs |
| `python run_simulation.py --methods=XGBoost` | fit only these methods |
| `python run_simulation.py --merge` | merge this run's rows into the existing result files |

`--dgps` and `--methods` restrict what is **fitted**. The rows they did not
touch are carried over from the existing files rather than refitted, so adding
one method to a finished study costs one method and not the whole study.
Merging is valid only while the seeds and the DGP definitions are unchanged;
`merge_csv()` states exactly what must hold.

`run_padding.py` takes the run directory as its only argument, defaulting to
`sim_out/` beside the script, and writes into `<root>/padding/`. It has no
subcommands: it fits both padding variants on every DGP and writes its four
files. Its cost is the stopping rule -- stages run until the correction stops
moving rather than for a fixed count, which has been 25 to 45 stages in
testing, with `MAX_STAGES` as a hard cap. Lower `MAX_STAGES` first if a run has
to fit a fixed window; `STAGE_PATIENCE` is the next knob.

Two environment variables exist because worker processes are spawned and
re-import the file, so these must reach them without a mutated module global:

| Variable | Effect |
| --- | --- |
| `CRF_SMOKE=1` | 2 replications, 40 trees, short budgets -- a dry run that exercises every code path in about a minute. Not the reported configuration |
| `CRF_SAVE_ALL=1` | also write every replication's prediction vectors, not only the one behind the figures |

Replications are independent and run in a process pool of `N_WORKERS` workers.
Wall-clock time is printed at the end of the run, and the configuration and
library versions that produced it are written to `run_config.txt`.

---

## 4. The method

**`UnbiasedForest`** is the constrained random forest. It grows a bagged
forest, keeps the exact out-of-bag set of every tree, and then rewrites leaf
values so that the aggregated out-of-bag prediction satisfies two group-mean
constraints.

For predictions `f(X_i)`, outcomes `y_i` and the outcome mean `ybar` on the
calibration sample, the constraints are

```
r_U = mean_{i in U} ( f(X_i) - y_i ) = 0 ,   U = { i : y_i >= ybar }
r_L = mean_{i in L} ( f(X_i) - y_i ) = 0 ,   L = { i : y_i <  ybar }
```

A tree's prediction is linear in its leaf values, so both residuals are linear
in the leaves and every correction in the file is a descent step on
`(1/2)(r_U^2 + r_L^2)` in leaf space. All of them act on leaf values
individually; none is an affine map of the predictions.

| `CORRECTION` | What it does |
| --- | --- |
| `forest_lambda` | **the method reported.** One lambda-scaled update over all leaves of all trees, driven by the residuals of the aggregated out-of-bag prediction rather than tree by tree. `lam = 0` is no correction, `lam = 1` is exact satisfaction on the out-of-bag sample, `lam > 1` is deliberate overcorrection |
| `tree_lambda` | ALG 1: the same step per tree, on that tree's own out-of-bag sample, iterated to convergence. Here `lam` is a step size, not a regularizer -- every `lam` that converges lands on the same leaves |
| `greedy` | ALG 2, the priority-order heuristic: sort leaves by dominant-stratum coverage, absorb the leaf-local residual, broadcast. Retained for comparison |
| `exact` | the closed form of `tree_lambda` at convergence |
| `none` | the plain forest |

Correcting the aggregate rather than each tree is deliberate: every tree has
its own out-of-bag sample, so per-tree feasibility does not compose into
feasibility for the averaged forest, and the aggregate is the quantity that is
evaluated.

| `LAMBDA_SELECT` | How lambda is chosen |
| --- | --- |
| `oob_slope` | **the selection reported.** Pick lambda so that the out-of-bag calibration slope equals 1. The corrected predictions are affine in lambda, so this has a closed form from two evaluations, `lam* = (1 - eta_0) / (eta_1 - eta_0)`. Training data only |
| `fixed` | use `LAM` as given; `LAM = 1.0` is exact out-of-bag satisfaction with no tuning |
| `oob_cv` | the same slope-matching target, cross-fitted over K folds |

The out-of-bag slope is optimistic relative to the test slope at the same
lambda, so `lam*` chosen this way lands the test slope below 1. Choosing a
larger lambda by looking at the test slope would not be a legitimate procedure,
and the driver does not do it; `lambda_sweep()` reports both curves so that the
gap is visible.

Two backends are available. `sklearn` (the default) bags
`DecisionTreeRegressor`, drawing the bootstrap indices here so that exact
out-of-bag sets are available; sklearn reads leaf predictions from `tree_.value`
at predict time, so in-place edits there *are* the leaf correction, and it is
about 7.6x faster at this problem size. `custom` is the pure-NumPy reference
implementation, and is required for the constrained-split variant (ALG 3,
`CONSTRAINED_SPLITS = True`).

There is no GPU backend, by construction: every correction here rewrites
individual leaf values, and neither cuML nor XGBoost exposes writable leaves
from Python. Parallelism over replications is the speedup that matters.

The forest is usable directly:

```python
from run_simulation import UnbiasedForest

f = UnbiasedForest(correction="forest_lambda",
                   lambda_select="oob_slope").fit(X, y)
y_hat = f.predict(X_new)

f.lam_used            # the selected lambda
f.oob_pred            # the aggregated out-of-bag prediction
f.set_lambda(1.5)     # re-correct at a new lambda without refitting
```

`set_lambda` re-applies the forest-level correction in time linear in the
number of leaves, which is what makes a lambda sweep cheap.

---

## 5. The padding comparator

`run_padding.py` is the post-hoc alternative the paper contrasts the
constrained forest with: a staged residual forest that appends the running
prediction to the design matrix as an extra ("padded") feature.

```
Stage 0 :  f0 = RF(X -> y)
Stage k :  X_k = [X, f_{k-1}] ;  f_k = f_{k-1} + RF(X_k -> y - f_{k-1})
```

Stages run until the correction stops moving rather than for a fixed count --
a negligible stage increment, or no improvement in the training residual, for
`STAGE_PATIENCE` consecutive stages -- with `MAX_STAGES` as a hard cap. Which
of these ended a fit is recorded per row as `stop_reason` (`converged`,
`stalled`, `diverged` or `max_stages`), so a run that hit the cap is never
mistaken for one that settled. Replications therefore differ in length, which
is what the stage-trajectory figure is for.

Two variants are produced, each its own column in the tables and its own panel
in the figures:

| Method | What it does |
| --- | --- |
| `Padding` | in-sample residual targets and in-sample padded feature, faithful to the original code |
| `Padding-OOB` | out-of-bag predictions supply both the padded feature and the residual target at every stage, so the correction is fit on honest residuals |

The DGP block, the metrics and the proxy in `run_padding.py` are transcribed
from `run_simulation.py` -- same functions, same fixed coefficient vector at
the same `BETA_SEED`, same noise scale, same `RandomState` stream at the same
replication seed, and the same `W` stream. Replication `s` of the padding study
is therefore fitted on exactly the sample replication `s` of the CRF study saw,
which is what makes the columns comparable cell by cell. **If the DGP block in
`run_simulation.py` changes, copy it across again.**

`make_figures.py` draws `Padding` alongside the CRF and the baselines by
default and keeps `Padding-OOB` out of the main grids to hold the column count
down; both appear in every CSV, and setting `INCLUDE_PADDING_OOB = True` in
that file carries the second variant into the figures as well. Both are always
drawn in the stage-trajectory figure, since the gap between them is what it is
for.

---

## 6. Metrics

| Key | Quantity | Reference value |
| --- | --- | --- |
| `bias` | eta-hat, the OLS slope of the predicted outcome on the observed outcome | 1 |
| `rmse`, `mae` | prediction error | -- |
| `r2_pearson`, `r2_ols` | squared Pearson correlation, and `1 - SSR/SST` | -- |
| `r_U`, `r_L` | the two group residuals above | 0 |
| `theta_delta` | `theta_hat - theta_oracle`, the distortion a downstream association inherits from the predictions | 0 |

The downstream association uses a noisy outcome proxy `W = y + U(-a, a)` with
`a = sqrt(3) sd(y)`, so that `Var(W - y) = Var(y)` and the population slope of
`y` on `W` is 0.5. `theta_hat` is the slope of the predicted outcome on `W`;
`theta_oracle` is the same regression run with the observed outcome, which is
the association a downstream analysis would recover if the outcome were
available. The training standard deviation sets the noise level on both splits,
so the proxy is not rescaled by each split's own dispersion.

Reporting both r-squared variants is deliberate: they diverge under a slope
correction. A correction that fixes eta while inflating RMSE leaves
`r2_pearson` essentially unchanged and drives `r2_ols` down, which is the
bias-variance trade-off made visible.

For the CRF the Train row is the **out-of-bag** prediction on the training
sample (`CRF_TRAIN_ROW = "oob"`), which is the sample the correction is
calibrated on; the baselines and both padding variants report their in-sample
training fit. The Test row is the held-out sample for every method.

---

## 7. Outputs

The CRF study writes to `OUT_DIR`, which defaults to `sim_out/` beside
`run_simulation.py`:

| File | Contents |
| --- | --- |
| `results_table.csv` | **the reported table.** One row per (DGP, split, metric), one column per method, cells `mean (SD)` |
| `results_table.tex` | the same table as LaTeX (booktabs) |
| `summary_mean_sd.csv` | the same aggregation unformatted: mean, SD and count for every metric, at full precision |
| `results_per_sim.csv` | one row per (DGP, replication, method, split): every metric, the two group residuals, and the selected lambda |
| `predictions_sim<k>.csv` | the observed / predicted / `W` vectors behind the figures |
| `all_predictions.csv` | every replication's vectors (`CRF_SAVE_ALL=1`) |
| `lambda_sweep.csv` | the out-of-bag and test curves against lambda |
| `run_config.txt` | the configuration and the library versions of this run |

Its figures go one directory per DGP, `sim_out/<dgp>/`:

| File | Contents |
| --- | --- |
| `obs_pred_<dgp>.png` | observed versus predicted, all methods, Train and Test |
| `obs_pred_<dgp>_<method>.png` | the same for one method on its own |
| `metrics_box_<dgp>_<split>.png` | the distribution of each metric over replications |
| `constraints_<dgp>_<split>.png` | the two group residuals `r_U`, `r_L` |
| `association_<dgp>.png` | the downstream association: predicted outcome against `W`, with the oracle slope |
| `lambda_curve_<dgp>.png` | the out-of-bag and test curves against lambda |
| `tree/tree_<b>.png` | one tree before and after correction on the same axes, `TREE_FIGS_N` of them |

The padding study writes four files to `sim_out/padding/`:
`results_per_sim.csv` and `predictions_sim0.csv` in the same schema as above,
`summary_mean_sd.csv`, and `stage_metrics.csv` -- the per-stage trajectory of
every metric, plus `rel_step`, the size of each stage's increment.

---

## 8. Stacked figures

`make_figures.py` reads finished result files and redraws the key figures with
the three DGPs stacked vertically and larger fonts, which is the form the paper
uses. It takes the run directory as its only argument and looks for each CSV in
`STUDY_DIRS` in turn -- `v3/`, `padding/`, then the run directory itself -- so
one call picks up both studies. Rows carrying the same DGP, method and split in
more than one of them are resolved in favour of the earlier entry.

```bash
python make_figures.py sim_out
```

Output goes to `<root>/restacked_figures/`: the observed-versus-predicted grids
(full, Train, Test), the metric boxplots, the group-residual panels, the
downstream association slopes, the lambda curves, and the padding stage
trajectories. Anything whose input file is missing is skipped with a message
rather than failing the run, so the script is safe to point at a partial
directory -- running it before `run_padding.py` simply omits the stage figure.

---

## 9. Randomness and determinism

Replication `s` uses `seed = SEED_BASE + s = 42 + s` for the data draw, the
bootstrap indices, the forest's `random_state` and the `random_state` of every
baseline, so any single replication can be rerun in isolation and reproduce
exactly. Three seed streams are kept separate on purpose:

* `BETA_SEED = 20260724` draws the linear DGP's coefficient vector once, so it
  is identical in every replication and in every spawned worker process.
* `TUNE_SEED = 20260815` draws the pilot sample used to select the base
  forest's depth and `mtry`. It sits outside the study seeds, so the tuning
  sample is never one of the replications, and the selection is deterministic
  -- `--figures` and `--tables` recover exactly the configuration the study
  used, without refitting.
* `W_SEED_BASE = 42` gives the downstream proxy its own stream, keyed so that
  methods sharing the same observed outcomes face the same realised `W` and
  their `theta_hat` values are comparable within a replication.

`run_padding.py` uses the same three streams at the same values, which is what
puts the two studies on identical samples (Section 5). Its own forests are
seeded at `SEED_BASE + s` as well, so it too reproduces replication by
replication.
