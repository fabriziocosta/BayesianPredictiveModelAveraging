# Scientific Questions and Experimental Plan

This document turns the claims behind Bayesian Predictive Model Averaging (BPMA) into testable questions. The goal is not to demonstrate that BPMA can fit a model, but to determine when its predictive weighting, prior structure, subset sampling, and adaptive allocation are scientifically useful.

The hypotheses below are proposed in advance. They should be treated as hypotheses to falsify rather than as expected conclusions.

## 1. Scientific questions

### Q1. Does predictive model averaging improve generalization?

**Question.** When several estimator families are plausible, does BPMA achieve better out-of-sample prediction than selecting one model or one family?

**Hypothesis.** On problems with genuine model or parameter uncertainty, BPMA should improve held-out log loss for classification and held-out predictive log score for regression. The gain should be largest when competing families make complementary errors. On simple problems with one clearly superior family, BPMA should not incur a large penalty.

**Key ablations.** Compare the full method with:

- the best single model selected by nested cross-validation;
- the best single family selected by nested cross-validation;
- equal-weight averaging of the same sampled models;
- score-weighted averaging without the declared prior;
- a non-adaptive BPMA run using the same total draw budget.

The comparison must keep the candidate family registry, parameter support, fitting budget, and outer test splits fixed.

### Q2. Does the declared prior have a useful and interpretable effect?

**Question.** Do prior choices influence predictions in the intended way, and are those effects helpful rather than accidental consequences of the implementation?

**Hypothesis.** A well-calibrated prior should improve performance or calibration when data are scarce, while its influence should diminish as the amount of data and predictive evidence increase. A deliberately misspecified prior should be detectable through sensitivity analysis and should not be hidden by the reporting of predictive weights.

**Key ablations.** Use the same data and random seeds with:

1. the intended family and parameter priors;
2. uniform family and parameter priors;
3. a prior concentrated on the wrong family;
4. a prior concentrated on the family that generated synthetic data;
5. prior draws with the prior term removed from the final weight.

Measure the change in predictive performance, calibration, family mass, parameter mass, and effective sample size. Report prior-predictive summaries before fitting so that a prior that is already implausible is not judged only by its posterior outcome.

### Q3. Is cross-validated predictive evidence a useful surrogate for generalization?

**Question.** Does the cross-validated score $L(\theta)$ rank candidate models in a way that predicts performance on fresh data?

**Hypothesis.** Cross-validated predictive scores should correlate more strongly with fresh-test performance than training scores or a single non-nested validation split. The relationship may weaken when the validation design is poorly matched to the data-generating process.

**Key ablations.** For every sampled draw, record several scores without changing the fitted model:

- the implemented cross-validated score;
- training-set score;
- a single holdout score;
- repeated cross-validation score;
- an oracle score on a large independently generated test set for synthetic data.

Compare rank correlation, top-k overlap, weight concentration, and the predictive performance of the resulting ensembles. For low-dimensional synthetic models with a tractable likelihood, also compare BPMA’s weights with a reference Bayesian posterior or a high-accuracy numerical integration.

### Q4. Does sampling training subsets improve robustness?

**Question.** Does averaging over CV-admissible training subsets capture useful data uncertainty, or does it mainly add variance and discard information?

**Hypothesis.** Subset averaging should help when samples are limited, noisy, outlier-prone, or locally heterogeneous. It may hurt when the full training set is already large and the best estimator benefits strongly from every observation.

**Key ablations.** Hold the family and parameter draws fixed while comparing:

- full-data fitting for every draw;
- a fixed subset size;
- the default subset-size prior;
- a narrow subset-size prior;
- a broad subset-size prior;
- bootstrap or subsampling baselines with equal fitting budgets.

Evaluate both average prediction quality and the variability of predictions across data resamples. On synthetic data, vary sample size, label noise, outlier rate, and covariate shift independently.

### Q5. How should the target temperature be chosen?

**Question.** Does the target temperature provide a useful bias-variance and concentration control, and is its best value stable across datasets?

**Hypothesis.** Very small temperatures will over-concentrate on a few high-scoring draws and may overfit the CV score. Very large temperatures will underuse predictive evidence and approach prior or equal averaging. Intermediate temperatures should often improve log score and calibration.

**Key ablations.** Sweep `temperature` over a pre-registered grid, for example `0.1`, `0.25`, `0.5`, `1`, `2`, and `5`, with all draws reused where possible. Select a temperature only inside the training portion of each outer split. Report:

- test log loss or predictive log score;
- accuracy, RMSE, and MAE as secondary task metrics;
- calibration and predictive entropy;
- effective sample size and maximum normalized weight;
- sensitivity of family and parameter shares.

The primary result should be a temperature-performance curve, not only the best temperature.

### Q6. Does adaptive sampling improve allocation without changing the target?

**Question.** When the total number of model fits is fixed, does adaptive importance sampling spend more computation on useful regions while preserving the declared predictive target?

**Hypothesis.** Adaptive sampling should reach a given predictive quality or effective sample size with fewer draws when family performance is uneven. With correct deterministic-mixture correction, it should not systematically inflate the final mass of families that were oversampled.

**Key ablations.** Compare, at matched total draw counts and matched wall-clock budgets:

- non-adaptive prior sampling;
- adaptive sampling with deterministic-mixture correction;
- adaptive sampling with correction intentionally disabled;
- adaptive sampling with different `defensive_prior_weight` values;
- adaptive sampling with different `adaptation_temperature` values;
- an oracle allocation based on known synthetic test performance.

The uncorrected adaptive run is a negative control: it should expose allocation bias, not serve as a proposed method. Track performance after every completed round, family proposal probabilities, family posterior shares, ESS, proposal distance, prediction change, and the number of fits required to reach a fixed quality threshold.

### Q7. Is deterministic-mixture correction numerically and statistically correct?

**Question.** Does the importance-weight calculation recover the same target when the proposal changes across rounds?

**Hypothesis.** Under a known finite model space, corrected adaptive estimates should converge to the target distribution induced by the declared prior and predictive score. The error should decrease with the number of draws and remain bounded when the defensive component keeps every family proposal positive.

**Experimental test.** Construct a small synthetic model space for which every model’s prior probability and predictive score can be computed exactly. Run adaptive sampling many times, then compare estimated family masses with the exact target using total variation distance, KL divergence, bias, and Monte Carlo variance. Repeat with increasingly aggressive adaptation and with the defensive component removed in a diagnostic failure condition.

This test should be separated from end-to-end prediction so that a good or bad predictive result cannot conceal a weighting error.

### Q8. Does BPMA improve uncertainty calibration, not only point prediction?

**Question.** Are BPMA probabilities and across-draw predictive distributions better calibrated than predictions from a selected model?

**Hypothesis.** Averaging should reduce overconfident predictions when several plausible models disagree. Improvements should be visible in proper scoring rules and calibration curves, even when accuracy changes little.

**Experimental test.** For classification, measure log loss, Brier score, expected calibration error, adaptive calibration error, reliability diagrams, and classwise calibration. For regression, retain predictions from each fitted draw and form weighted predictive quantiles; measure interval coverage, interval width, and proper interval scores. Compare BPMA with the selected model, equal averaging, and post-hoc calibrated single models.

Calibration must be evaluated on data not used to choose the family registry, temperature, or calibration map.

### Q9. Do the convergence diagnostics identify stable ensembles?

**Question.** Do prediction change, proposal distance, ESS, and stopping patience provide reliable signals that more draws will not materially change the result?

**Hypothesis.** Prediction stability should be the strongest direct indicator of ensemble stability. ESS and proposal stability should be useful complementary diagnostics, but neither should be treated as proof that predictions have converged.

**Key ablations.** Run long non-stopping trajectories and replay the stopping rules offline with different:

- `prediction_tolerance` values;
- `proposal_tolerance` values;
- `ess_target_fraction` values;
- `stopping_patience` values;
- convergence-subset sizes.

Define the reference prediction as the final prediction from the longest run. Measure stopping-time error, false-stopping rate, excess computation, and the probability that the stopped prediction lies within a pre-specified tolerance of the reference.

### Q10. How sensitive is BPMA to the cross-validation design?

**Question.** Does the method remain valid when the data are grouped, temporal, imbalanced, or otherwise non-exchangeable?

**Hypothesis.** BPMA should inherit the strengths and failures of its validation design. Random folds should be competitive for exchangeable data but can produce misleading weights under group or temporal leakage. Matching the splitter to the deployment process should improve external performance and calibration.

**Experimental test.** Use synthetic data with known groups, time order, and distribution shift, then compare random, grouped, blocked, and rolling validation. Include an intentional leakage condition as a negative control. Evaluate performance on a future or held-out group rather than only on randomly sampled test rows.

### Q11. Does the method remain useful under family misspecification?

**Question.** What happens when the true predictive mechanism is absent from the family registry, or when the registry contains many irrelevant families?

**Hypothesis.** BPMA should still average the best available approximations, but posterior shares should not be interpreted as evidence that a family is scientifically true. Adding irrelevant families should reduce efficiency more than it changes the target when their prior mass is small.

**Key ablations.** For each synthetic data-generating process, compare:

- a registry containing the generating family;
- a registry excluding the generating family;
- a registry augmented with weak and strong distractor families;
- a registry with duplicated or near-duplicate families;
- a deliberately over-flexible family with a controlled prior.

Report predictive performance, calibration, family mass, ESS, and computational cost. This separates predictive usefulness from claims of model identification.

### Q12. Are improvements worth the computational cost?

**Question.** Does BPMA deliver a meaningful quality gain per fit or per unit wall time compared with simpler baselines?

**Hypothesis.** Adaptive sampling should be most attractive when model families have highly unequal predictive value and each fit is expensive. On homogeneous or very easy tasks, simpler selection or equal averaging may dominate on cost-effectiveness.

**Experimental test.** Produce quality-versus-budget curves by varying the number of draws and record CPU time, peak memory, number of CV fits, number of final fits, ESS, and convergence status. Compare serial and parallel execution under fixed seeds and verify that parallelism changes wall time but not predictions beyond numerical tolerance.

## 2. Proposed experimental setup

### 2.1 Benchmark matrix

Use a two-stage benchmark so that mechanism-level tests are not confused with application-level performance.

**Synthetic mechanism benchmarks** should include:

- linear and smoothly nonlinear regression;
- heteroscedastic regression;
- linearly separable and overlapping classification;
- curved decision boundaries such as moons and circles;
- mixtures with changing numbers of components;
- small-sample, noisy, outlier, grouped, and temporally ordered variants.

Synthetic generators should expose the data-generating mechanism, Bayes-optimal predictions where available, and a large independent test set. Repeat each condition over at least 20 data seeds so that conclusions do not depend on one sampled dataset.

**Real-data benchmarks** should contain a balanced set of classification and regression tasks with different sample sizes, feature counts, class imbalance, and likely nonlinear structure. Pin dataset versions and preprocessing. Do not use the final test split to select the family registry, priors, temperature, or stopping criteria.

### 2.2 Candidate methods

Use one fixed default family registry for the primary comparison and separate controlled registries for ablations. The primary baselines should be:

1. BPMA with prior sampling and the default target temperature;
2. BPMA with adaptive importance sampling and correction;
3. a single model selected by nested cross-validation;
4. a single family selected by nested cross-validation;
5. equal-weight averaging over the same sampled models;
6. score-weighted averaging with the prior contribution removed;
7. bagging or random-subset averaging with an equivalent fitting budget.

Whenever possible, use the same fitted draws for methods that only differ in weighting. This makes the ablation isolate weighting rather than changing the random model population.

### 2.3 Data splitting and tuning

Use an outer evaluation split for every reported result. All choices of prior, temperature, family registry, convergence thresholds, and calibration procedure must be made inside the corresponding training portion. Use a fixed list of outer splits shared by all methods and repeat the complete experiment over multiple random seeds.

For each outer split:

1. fit every method using only the outer training data;
2. tune allowed hyperparameters with an inner split or pre-registered grid;
3. refit the selected configuration on the complete outer training data;
4. evaluate once on the untouched outer test data;
5. save predictions, draw metadata, weights, diagnostics, and runtime.

The test set must never determine which scientific question appears to have the most favorable answer.

### 2.4 Fixed budgets and randomization

Pre-register a primary budget, such as a fixed number of total model draws and a fixed maximum number of CV fits. Use the same budget for non-adaptive and adaptive methods. For cost comparisons, add a wall-clock budget but report both fit count and time because parallel hardware can change the latter.

Use common random numbers where the comparison permits it: share data splits, base seeds, and sampled draws across weighting ablations. For adaptive versus non-adaptive sampling, use independent proposal draws but repeat enough seeds to estimate Monte Carlo variability.

### 2.5 Primary metrics

Use proper predictive scoring rules as the primary outcomes:

- classification: log loss and Brier score;
- regression: predictive log score when a predictive distribution is available, otherwise RMSE as the primary point-prediction metric.

Use accuracy, AUROC, MAE, and median absolute error only as secondary metrics. Always report calibration and uncertainty metrics alongside point metrics:

- classification: reliability, expected calibration error, Brier score, and predictive entropy;
- regression: interval coverage, interval width, and a proper interval score when weighted predictive quantiles are available;
- sampling diagnostics: ESS fraction, maximum weight, family-mass error, proposal distance, and prediction change;
- efficiency: CV fits, final fits, wall time, peak memory, and quality per unit budget.

Report paired differences against each baseline on the same outer split. Include confidence intervals across data seeds and outer splits, not only the mean rank across datasets.

### 2.6 Required ablation reports

Every ablation should include four views:

1. **Predictive outcome:** the primary test metric and its uncertainty.
2. **Distributional outcome:** calibration, uncertainty, and prediction disagreement.
3. **Sampling outcome:** ESS, weight concentration, family shares, and convergence history.
4. **Cost outcome:** number of fits, time, memory, and stopping point.

For adaptive experiments, plot each quantity by completed round. For temperature and prior experiments, plot the full sensitivity curve rather than reporting only the chosen setting. For correction experiments, compare estimated family masses with the exact target on finite synthetic model spaces.

### 2.7 Decision criteria

The approach should be considered supported only if the results satisfy pre-registered criteria such as:

- BPMA improves or matches the primary proper score across the majority of relevant conditions, with no severe degradation on simple conditions;
- calibration improves or remains competitive when point accuracy is unchanged;
- corrected adaptive sampling improves quality per budget without a systematic shift in target family mass;
- convergence diagnostics reduce computation while keeping stopping error below a pre-specified tolerance;
- prior sensitivity is visible, explainable, and reduced as data become more informative.

Failure is informative. In particular, evidence that adaptive allocation changes the target, that CV scores poorly predict fresh performance, or that prior sensitivity dominates data evidence would identify limitations of the method rather than merely indicate an inconvenient hyperparameter setting.

## 3. Reproducibility checklist

Each experiment should save:

- package version, Python version, dependency versions, and hardware details;
- dataset version, preprocessing, and exact train/test indices;
- family registry, prior definitions, temperature, CV splitter, and all stopping settings;
- base random seed and per-draw child seeds where available;
- every draw’s family, parameters, subset, score, prior, proposal, importance weight, and posterior weight;
- proposal and convergence histories for adaptive runs;
- predictions on every outer test set;
- fit counts, elapsed time, peak memory, exceptions, and stopping reason.

The complete analysis should be rerunnable from these records without relying on the current default registry or undocumented random-state behavior.
