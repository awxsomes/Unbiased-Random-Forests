# Unbiased Predictions for Continuous Outcomes Using Constrained Random Forests

This package contains the complete source code for the experiments reported in
the paper: the constrained random forest (CRF), the leaf-correction it is
built from, the post-hoc padding comparator, the three comparison estimators,
the three data-generating processes, the evaluation metrics and the figure
code. Everything runs from scratch with no external inputs. One command reproduces the synthetic experiments, results table, per-replication file, and relevant figures.

---

## 1. Layout

```
run_simulation.py   the study: three DGPs x 100 replications, fitting CRF,
                    the post-hoc, RF, NN and XGBoost
make_figures.py     redraws the key figures with the three DGPs stacked
                    vertically at publication font sizes
supplementary_checks.py
                    the two supplementary analyses reported in the
                    paper: sensitivity to the number of trees B, and
                    the downstream slope with the prediction used as a
                    covariate rather than as the response
requirements.txt    dependencies
```

`run_simulation.py` is fully self-contained. The trees, corrections, forest, padding comparator, metrics, DGPs, baselines, study loop, result files, and figures are all defined in one file, organized in the order of a single replication with section banners for navigation. The configuration block appears at the top under `CONFIG`, and no setting that affects the reported results is supplied through the command line.

| Banner | Contents |
| --- | --- |
| `CONFIG` | All settings used in the simulation. |
| `CUSTOM TREE` | The tree implementations: `CustomTree`, a pure-NumPy reference implementation, and `SkTreeWrapper`, which uses bagged `DecisionTreeRegressor` trees in the default backend. |
| `FOREST` | `UnbiasedForest`, including bagging, exact out-of-bag sets, the forest-level correction, and lambda selection. |
| `METRICS` | The evaluation metrics: $\hat{\eta}$, RMSE, MAE, and the downstream association slope. |
| `THE PADDING COMPARATOR` | `PaddedForest`, a staged residual-forest comparator that applies a post-hoc adjustment rather than a constraint and does not modify leaf values. |
| `DATA-GENERATING PROCESSES` | The three simulation DGPs and the procedure for selecting the base forest. |
| `ONE REPLICATION` | `run_one`, which carries out a single simulation replication and is the unit distributed across the process pool. |
| `LAMBDA SWEEP` | The lambda-sensitivity analysis invoked with `--sweep`. |
| `FIGURES`, `TABLES`, `STUDY DRIVER`, `ENTRY POINT` | Code for producing output files and figures, running the full simulation study, and handling command-line execution. |

Every method is fitted inside the same replication, on the same sample, so
each column of the results table faces identical data.

---

## 2. Requirements

The versions below are the ones that produced the reported results.

```
numpy==2.5.2
pandas==3.0.5
scikit-learn==1.9.0
scipy==1.18.0
joblib==1.5.3
threadpoolctl==3.6.0
matplotlib==3.11.1
xgboost==3.4.1
```

```
pip install -r requirements.txt
```

<!-- `xgboost` is required. The gradient-boosting column of the results table is
produced by `xgboost.XGBRegressor` at the hyperparameters in `XGB_PARAMS`;
scikit-learn's `HistGradientBoostingRegressor` is a different model at
different settings, so there is no silent fallback -- a missing `xgboost` stops
the run with a message rather than filling the column with another estimator.
Dropping `"XGBoost"` from `BASELINES` is the explicit opt-out. -->

The driver is spawn-safe (Windows and macOS) and runs on Python 3 and no additional data files are needed: the three DGPs are generated at the seeds in
the config block.

---

## 3. Running

```bash
# the study: three DGPs x 100 replications        -> sim_out/
python run_simulation.py

# the stacked figures                             -> sim_out/restacked_figures/
python make_figures.py sim_out

# supplementary analyses                          -> supplementary_out/
python supplementary_checks.py trees 20          # B in {50,...,1000}
python supplementary_checks.py downstream 100    # covariate-role slope
```

`make_figures.py` reads only the run directory it is given. Result
folders left by earlier layouts (`v3/`, `padding/`) are ignored, so a
fresh study is never overridden by older files.

| Command | Effect |
| --- | --- |
| `python run_simulation.py` | the full study at the configured `N_SIMS` |
| `python run_simulation.py 10` | a short run, 10 replications |
| `python run_simulation.py --figures` | redraw the figures from saved results |
| `python run_simulation.py --tables` | rebuild `results_table.csv` / `.tex` from `results_per_sim.csv` |
| `python run_simulation.py --sweep` | the lambda sweep, all DGPs |
| `python run_simulation.py --dgps=friedman1,friedman2` | fit only these DGPs |
| `python run_simulation.py --methods=Padding` | fit only these methods |
| `python run_simulation.py --merge` | merge this run's rows into the existing result files |

`--dgps` and `--methods` restrict what is **fitted**. The rows they did not
touch are carried over from the existing files rather than refitted, so adding
one method to a finished study costs one method and not the whole study --
`--methods=Padding` refits the comparator alone and merges it in beside the CRF
rows already scored. Merging is valid only while the seeds and the DGP
definitions are unchanged; `merge_csv()` states exactly what must hold.

Two environment variables exist because worker processes are spawned and
re-import the file, so these must reach them without a mutated module global:

| Variable | Effect |
| --- | --- |
| `CRF_SMOKE=1` | 2 replications, 40 trees, 40 padding stages, short budgets -- a dry run that exercises every code path in about a minute. Not the reported configuration |
| `CRF_SAVE_ALL=1` | also write every replication's prediction vectors, not only the one behind the figures |

Replications are independent and run in a process pool of `N_WORKERS` workers.
Wall-clock time is printed at the end of the run, and the configuration and
library versions that produced it are written to `run_config.txt`.

The padding comparator is the most computationally expensive part of a run because it continues adding stages until the correction stabilizes rather than stopping after a fixed number of stages. In the reported study every fit converged, after about 50 stages (49 on average). Runtime can be reduced by lowering `PAD_MAX_STAGES` or `PAD_STAGE_PATIENCE`, or by excluding the padding comparator through `--methods`.
---

## 4. The method

**`UnbiasedForest`** grows a bagged forest, keeps the exact out-of-bag set of
every tree, and rewrites leaf values so that the aggregated out-of-bag
prediction satisfies the two group-mean constraints defined in the Methods
section of the paper such that mean prediction error is zero below and above the outcome
mean. A tree's prediction is linear in its leaf values, so the correction is a
single projection in leaf space.

| `CORRECTION` | What it does |
| --- | --- |
| `forest_lambda` | One lambda-scaled update over all leaves of all trees, driven by the residuals of the aggregated out-of-bag prediction. `lam = 0` is no correction, `lam = 1` is exact satisfaction on the out-of-bag sample, `lam > 1` is deliberate overcorrection |
| `none` | the plain forest |

| `LAMBDA_SELECT` | How lambda is chosen |
| --- | --- |
| `oob_slope` | **the selection reported.** Pick lambda so that the out-of-bag calibration slope equals 1; closed form from two evaluations, `lam* = (1 - eta_0) / (eta_1 - eta_0)`. Training data only |
| `fixed` | use `LAM` as given; `LAM = 1.0` is exact out-of-bag satisfaction with no tuning |
| `oob_cv` | the same target, cross-fitted over K folds |

Two backends are available: `sklearn` (the default, and faster) and `custom`,
the pure-NumPy reference implementation required for
`CONSTRAINED_SPLITS = True`, which imposes stratum balance during tree growth
rather than as a post-fit correction; it is implemented but not part of the
reported study.

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

`PaddedForest` is the post-hoc alternative the paper contrasts the constrained
forest with: a staged residual forest that appends the running prediction to
the design matrix as an extra ("padded") feature.

```
Stage 0 :  f0 = RF(X -> y)
Stage k :  X_k = [X, f_{k-1}] ;  f_k = f_{k-1} + RF(X_k -> y - f_{k-1})
```

Stages run until there is a 
a negligible increment between stages, or no improvement in the training residual, for
`PAD_STAGE_PATIENCE` consecutive stages, or until `PAD_MAX_STAGES` as a hard cap.
Which of these ended a fit is recorded as `stop_reason` (`converged`,
`stalled`, `diverged` or `max_stages`), so a fit that hit the cap is never
mistaken for one that settled. Replications therefore differ in length, which
is what the stage-trajectory figure is for.

**`Padding`** is the method reported according to the
original Zhang and Lu (2012) code by taking its residual targets in sample, so its training predictions
interpolate: a training eta-hat of 1 and a training RMSE of 0 in the table are
that, not a bug. The Test row is the informative one.

Two columns of `results_per_sim.csv` belong to the comparator alone,
`n_stages` and `stop_reason`; they are empty for every other method, and
nothing aggregates them, so the shared schema is unaffected. The test sample is
passed to `PaddedForest.fit()` only so that the stage trace can carry test
metrics alongside training ones -- no stage is selected on it.


---

## 6. Metrics

| Key | Quantity | Reference value |
| --- | --- | --- |
| `bias` | eta-hat, the OLS slope of the predicted outcome on the observed outcome | 1 |
| `rmse`, `mae` | prediction error | -- |
| `theta_delta` | `theta_hat - theta_oracle`, the distortion a downstream association inherits from the predictions | 0 |

The downstream association uses a noisy outcome proxy `W = y + U(-a, a)` with
`a = sqrt(3) sd(y)`, so that `Var(W - y) = Var(y)` and the population slope of
`y` on `W` is 0.5. `theta_hat` is the slope of the predicted outcome on `W`;
`theta_oracle` is the same regression run with the observed outcome, which is
the association a downstream analysis would recover if the outcome were
available. The training standard deviation sets the noise level on both splits,
so the proxy is not rescaled by each split's own dispersion.

Squared Pearson correlation, `1 - SSR/SST`, and the two group residuals
`r_U`/`r_L` were computed in earlier runs -- they're the natural diagnostics
for a method built around a group-mean constraint -- but none of them appears
in the paper, so they were dropped from the metrics, the tables and the
figures below rather than carried as dead columns.

---

## 7. Outputs

Everything is written to `OUT_DIR`, which defaults to `sim_out/` beside
`run_simulation.py`:

| File | Contents |
| --- | --- |
| `results_table.csv` | **the reported table.** One row per (DGP, split, metric), one column per method, cells `mean (SD)` |
| `results_table.tex` | the same table as LaTeX (booktabs) |
| `summary_mean_sd.csv` | the same aggregation unformatted: mean, SD, median and count for every metric, at full precision |
| `results_per_sim.csv` | one row per (DGP, replication, method, split): every metric, the selected lambda, and the padding stage count and stop reason |
| `predictions_sim<k>.csv` | the observed / predicted / `W` vectors behind the figures |
| `all_predictions.csv` | every replication's vectors (`CRF_SAVE_ALL=1`) |
| `lambda_sweep.csv` | the out-of-bag and test curves against lambda |
| `stage_metrics.csv` | the padding comparator's per-stage trajectory of every metric, plus `rel_step`, the size of each stage's increment |
| `run_config.txt` | the configuration and the library versions of this run |

Figures go one directory per DGP, `sim_out/<dgp>/`:

| File | Contents |
| --- | --- |
| `obs_pred_<dgp>.png` | observed versus predicted, all methods, Train and Test |
| `obs_pred_<dgp>_<method>.png` | the same for one method on its own |
| `metrics_box_<dgp>_<split>.png` | the distribution of each metric over replications |
| `association_<dgp>.png` | the downstream association: predicted outcome against `W`, with the oracle slope |
| `lambda_curve_<dgp>.png` | the out-of-bag and test curves against lambda |
| `tree/tree_<b>.png` | one tree before and after correction on the same axes, `TREE_FIGS_N` of them |

---

## 8. Figures

`make_figures.py` reads the finished result files and redraws the key figures
with the three DGPs stacked vertically and larger fonts, which is the form the
paper uses. It takes the run directory as its only argument.

```bash
python make_figures.py sim_out
```

Output goes to `<root>/restacked_figures/`: the observed-versus-predicted grids
(full, Train, Test), the metric boxplots, the downstream association slopes,
the lambda curves, and the padding stage trajectories. Anything whose input
file is missing is skipped with a message rather than failing the run, so the
script is safe to point at a partial directory.

It draws `Padding` alongside the CRF and the baselines, including in the
stage-trajectory figure.

---

## 9. Randomness and determinism

Replication `s` uses `seed = SEED_BASE + s = 42 + s` for the data draw, the
bootstrap indices, the forest's `random_state`, the padding forests' seeds and
the `random_state` of every baseline, so any single replication can be rerun in
isolation and reproduce exactly. Three seed streams are kept separate on
purpose:

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

Within a replication, stage `k` of a padding fit is seeded at
`SEED_BASE + s + k`, so a fit of a given length is reproducible stage by stage.