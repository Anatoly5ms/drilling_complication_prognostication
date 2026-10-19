# Drilling Complication Prognostication

Predicting the risk of drilling complications (lost circulation, kicks, stuck pipe, wellbore instability) for each depth interval of a well, using open daily drilling reports from the Equinor Volve field.

> **Status:** preliminary study complete (v0.1). Data pipeline, labeling, feature engineering, evaluation, five models, the ensemble and a one-time held-out test are done. Next steps are listed in the roadmap.

## Background

This project revisits my Master's thesis (Saint Petersburg Mining University, 2019), *"Development of an Algorithm for Calculating the Probability of Complications in the Drilling Process"*. The original method estimated complication risk at the well design stage from geological and drilling-fluid parameters, using products of per-parameter probabilities computed from 70+ well construction projects in Russian oil fields. It won the 3rd International Conference "Drilling Wells in Complicated Conditions" (2018), received a UMNIK federal innovation grant (2020) and was registered as software (RU 2020614466).

The original field data is no longer available, and the method had known limitations: it assumed independent parameters and estimated P(parameter | complication) rather than P(complication | parameters), because it was built on complication cases only. This project rebuilds the approach on open data with a proper ML formulation: negative examples, well-grouped validation and explicit leakage control. The 2019 heuristic is kept as the baseline.

## Data

**Source:** [Volve daily drilling reports](https://huggingface.co/datasets/bengsoon/volve_daily_drilling_report), a parsed version of the WITSML daily drilling reports from the Equinor Volve data set: 1,759 reports, 26 wellbores, 11 wells.

**Licence and attribution:** the data is distributed under the Equinor Open Data Licence. Data ownership: Equinor and the former Volve licence partners. The data is not included in this repository; `scripts/download_data.py` downloads it.

## Pipeline

```
DataBuilder -> Labeler -> FeatureBuilder -> models -> ModelSelector (rank-averaging ensemble) -> HoldoutTester
                                              ↑ Evaluator (shared evaluation tool)
```

### 1. DataBuilder
Flattens nested WITSML records into tables (`reports`, `status`, `activity`, `fluid`, `lithology`, `pore_pressure`, `survey`), removes duplicate report versions, casts types and converts WITSML null codes (`-999.99`, `-999.25`) to missing values.

### 2. Labeler
Labels complications from the operations log, for all four complication types at once (multi-label):
- **Matching:** operation codes (e.g. `interruption -- lost circulation`) plus regular-expression rules on free-text comments, with *strong* patterns (unambiguous wording) and *weak* patterns (candidates for planned LLM-based verification). Negations and irrelevant contexts are excluded at phrase level ("no losses", "losses over shakers", "kick drill", equipment stuck vs. drill string stuck).
- **Code verification:** codes are accepted only with text confirmation where they proved unreliable (e.g. well-control codes used for weather waiting).
- **Events:** consecutive operations of one type in one wellbore are merged into a single event; event depth is the hole depth at the event start, not the bit depth (losses are typically noticed while tripping or circulating, far above the loss zone).
- **Intervals:** each wellbore is split into 10 m depth intervals. An interval is labelled positive if it contains an event or lies directly above it. Intervals below the event depth are not labelled: they did not exist when the event started.

| Complication | Events in drilled range | Positive intervals | Prevalence |
|---|---|---|---|
| Lost circulation | 47 | 69 | 1.0% |
| Well control (kicks) | below training threshold | — | — |
| Stuck pipe | 44 | 95* | 1.3%* |
| Wellbore instability | 110 | 218* | 3.0%* |

\* before the asymmetric depth tolerance was introduced; will be updated. Well-control events in Volve are too few to train a model, and the pipeline refuses to train below a configurable threshold.

### 3. FeatureBuilder
Builds features for each interval:

| Feature | Source |
|---|---|
| TVD, inclination, inclination change | survey (interpolated) |
| hole diameter, open-hole length, FIT/LOT | well status |
| mud density, plastic viscosity, yield point | mud checks |
| pore pressure (EMW) | pore pressure estimates |
| lithology group, shale share | lithology descriptions (normalised to 10 rock groups) |
| fracture margin = FIT − mud density | derived |
| overbalance = mud density − pore pressure | derived |

Physically impossible values are set to missing by configurable valid ranges; open-hole length for sidetracks is measured from the kick-off point.

### 4. Evaluation
- **Well-grouped validation:** sidetracks of one well share the upper hole section, so all splits are by well, not by wellbore.
- **Held-out test:** 3 of 11 wells, chosen so that both the interval share and the positive share are close to 20%.
- **Cross-validation:** `StratifiedGroupKFold` by well inside the training set; folds are constructed so that every validation fold contains positive examples.
- **Metrics:** PR-AUC on pooled out-of-fold predictions is the primary metric (robust to the ~1% positive rate); ROC-AUC, Brier score (vs. a constant-prevalence predictor) and F1 at a threshold tuned on out-of-fold predictions are reported as well. Test-set metrics will include bootstrap confidence intervals.

## Leakage control

The first version of the pipeline took interval features from the last measurement before the bit *left* the interval. A dedicated diagnostic (`scripts/diagnose_labels.py`) showed that this let the drillers' response to the complication leak into the features:

| | Before fix | After fix |
|---|---|---|
| Positive intervals with mud features measured after the loss started | **53%** | **4%** |

Two changes fixed it: features are now taken as of the moment the bit **enters** the interval (the prediction must be available before the interval is drilled), and intervals below the event depth are no longer labelled.

The leak had a visible effect: before the fix, logistic regression assigned a **positive** coefficient to fracture margin (more margin → more risk), which contradicts the physics of lost circulation. After the fix the sign became negative.

## Cross-validation results (lost circulation, 8 training wells)

All models are evaluated on the same 5 folds.

| Model | OOF PR-AUC | Median fold PR-AUC | OOF ROC-AUC | OOF F1 |
|---|---|---|---|---|
| Random | 0.010 | 0.010 | 0.50 | — |
| Baseline: 2019 thesis heuristic | 0.034 | 0.044 | 0.713 | 0.115 |
| Logistic regression | 0.018 | 0.035 | 0.731 | 0.043 |
| Random forest | 0.047 | 0.115 | 0.735 | 0.155 |
| LightGBM | **0.056** | 0.090 | 0.720 | **0.175** |
| CatBoost | 0.055 | 0.070 | 0.735 | 0.152 |
| **Ensemble (random forest + LightGBM + CatBoost)** | 0.049 | **0.119** | **0.745** | 0.163 |

**Final model: rank-averaging ensemble** of the models that beat the baseline. Each model's score is converted to a percentile of its own out-of-fold scores, and the percentiles are averaged, so models with different probability scales (class-weighted random forest vs. boosting) contribute equally. Nothing is fitted on the data at this step, and the ensemble is evaluated on the same out-of-fold predictions without retraining.

**Observations so far:**
- All three tree models beat the 2019 heuristic, while logistic regression does not. Trees split features by thresholds and learn feature interactions (e.g. high overbalance *and* a long open hole), which neither the heuristic's independent value groups nor logistic regression's straight lines can capture.
- No single tree model is the best: CatBoost wins two folds, random forest two others; random forest ranks intervals best *within* a well, while LightGBM produces risk levels that are more comparable *across* wells. The ensemble combines these strengths: it has the best median fold PR-AUC and ROC-AUC and the second-best F1. Its pooled PR-AUC is slightly lower than LightGBM's and CatBoost's — rank averaging compresses the top of the scale, so an interval reaches the very top only if all models agree — but the difference is smaller than the fold-to-fold variance.
- Class weights matter: without them random forest's ROC-AUC rises slightly, but PR-AUC drops by a third. For a driller, a model that raises many false alarms is useless, so precision at the top of the list (PR-AUC) is preferred over overall ordering (ROC-AUC).
- Three different tree algorithms agree on the most important features: open-hole length, TVD, inclination, pore pressure and fracture margin. TVD likely acts partly as a proxy for stratigraphy, since formations in one field lie at similar depths.
- Logistic regression without missing-value indicators gives coefficients with the signs expected from drilling engineering (longer open hole, higher mud density, yield point and overbalance increase risk; higher pore pressure, FIT and fracture margin decrease it), but little predictive power; missing-value indicators carry real information (e.g. no FIT yet at the start of a section).
- One validation well is consistently hard for every model (ROC-AUC 0.41–0.55): losses there occurred under conditions unlike those in the other wells.

## Held-out test (lost circulation, 3 wells never used in training or model selection)

Evaluated once, after all modelling decisions were frozen. Threshold-based metrics use each model's threshold from cross-validation; 95% confidence intervals by bootstrap over intervals (somewhat optimistic, since neighbouring intervals of one well are dependent).

| Model | PR-AUC [95% CI] | ROC-AUC [95% CI] | Precision | Recall |
|---|---|---|---|---|
| Random | 0.009 | 0.50 | — | — |
| Baseline: 2019 thesis heuristic | 0.022 [0.010; 0.056] | 0.671 [0.551; 0.791] | 0.021 | 0.071 |
| **Ensemble (final model)** | **0.048** [0.023; 0.111] | **0.879** [0.836; 0.922] | 0.047 | 0.357 |
| LightGBM | 0.097 [0.029; 0.288] | 0.889 [0.837; 0.936] | 0.045 | 0.429 |
| Logistic regression | 0.071 [0.021; 0.232] | 0.833 [0.732; 0.921] | 0.022 | 0.857 |
| Random forest | 0.065 [0.030; 0.147] | 0.895 [0.839; 0.941] | 0.070 | 0.214 |
| CatBoost | 0.065 [0.017; 0.219] | 0.819 [0.748; 0.890] | 0.036 | 0.286 |

Test set: 1,503 intervals, 14 positive. Ensemble by well:

| Well | Intervals | Positive | PR-AUC | ROC-AUC |
|---|---|---|---|---|
| NO 15/9-F-11 | 773 | 8 | 0.058 | 0.879 |
| NO 15/9-F-14 | 363 | 4 | 0.038 | 0.831 |
| NO 15/9-F-5 | 367 | 2 | 0.072 | 0.948 |

**Interpretation:**
- On wells it has never seen, the ensemble ranks depth intervals by lost-circulation risk with ROC-AUC 0.88 versus 0.67 for the 2019 heuristic; the confidence intervals do not overlap. Its PR-AUC is about five times the random level and twice the heuristic's, but with 14 positive intervals the PR-AUC confidence intervals overlap.
- The ranking is consistent across all three test wells (ROC-AUC 0.83–0.95), so the result does not rest on one favourable well.
- The single models other than the baseline are reported for context only. LightGBM's higher test PR-AUC is within the noise (its confidence interval fully covers the ensemble's), and the final model was not changed after seeing the test results.
- Test ROC-AUC is higher than in cross-validation (0.88 vs. 0.75), most likely because all three test wells are development (F) wells similar to most training wells, while cross-validation included a consistently hard well.
- **This is a proof of concept, not an operational tool:** at the cross-validation threshold, about 1 in 20 alarms is a real loss interval, and about a third of loss intervals are caught. The signal in the data is real and the model finds it, but precision at an operational threshold is still far from what a driller would trust, which motivates stratigraphy and well-log features as the next step.

## Roadmap

1. **Stratigraphy features:** daily reports explicitly name loss-prone formations (Ty, Grid, Balder), so the formation of each interval is expected to be the strongest missing feature, and should take over part of TVD's importance.
2. Hyperparameter tuning of the boosting models (nested, without touching the validation folds).
3. Well-log features (gamma ray, density, neutron porosity) from the full Volve data set, taken from offset wells to avoid leakage.
4. LLM verification of weak-pattern events, validated against a manually labelled sample.
5. Models for stuck pipe and wellbore instability.
6. Applicability to modern data: Volve reports use WITSML 1.4.1, while current systems increasingly use WITSML 2.x / ETP and each operator fills reports differently, so production use will require an input adapter and a check on current-format data.
7. Streamlit demo application.

## How to run

Requires Python 3.12+ (developed on 3.13).

```bash
python -m venv .venv                 # create a virtual environment
source .venv/bin/activate            # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m scripts.download_data      # downloads the Volve daily drilling reports
python main.py                       # full training pipeline
python -m scripts.diagnose_labels    # leakage diagnostic
```

The complication type and all thresholds, rules and model parameters are set in `config/config.yaml`.

## Project structure

```
config/config.yaml           # all parameters: data, labeling rules, features, evaluation, models
scripts/                     # data download, exploration and diagnostics
src/components/              # DataBuilder, Labeler, FeatureBuilder, models
src/configs/                 # config dataclasses
src/pipeline/                # training pipeline
src/utils/                   # data I/O, evaluator, train/test splitting
main.py
```

## Author

Anatolii Grigorovich — drilling engineer (MSc, Saint Petersburg Mining University) and technical project manager in AI/ML.