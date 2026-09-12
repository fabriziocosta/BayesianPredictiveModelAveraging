# Bayesian Predictive Model Averaging (BPMA)

## Abstract

Bayesian Predictive Model Averaging (BPMA) is a Bayesian-inspired ensemble framework in which models are sampled from prior distributions over model families and hyperparameters, and weighted according to estimates of their predictive evidence obtained through cross-validation, rather than exact marginal likelihoods. BPMA is designed for heterogeneous machine-learning estimators whose parameter spaces may be structured, conditional, or computationally inaccessible to a traditional marginal-likelihood calculation.

The framework samples complete fitted models, including an estimator family, an admissible training subset, and family-specific parameters. Each draw is scored by held-out predictive performance and contributes to an ensemble in proportion to its normalized predictive weight. An optional adaptive importance-sampling mode allocates later rounds toward promising estimator families while retaining the declared prior as the target distribution and correcting allocation bias with deterministic-mixture importance weights.

## 1. Motivation

Machine-learning model selection usually commits to one algorithm and one hyperparameter configuration. That commitment can hide substantial uncertainty: different families may fit different regions of the data, and several configurations may have similar predictive performance. Averaging predictions over a prior-supported collection of models provides a direct way to represent this uncertainty while retaining the practical interfaces of scikit-learn.

BPMA treats the estimator family itself as uncertain. A family can be a k-nearest-neighbour model, a linear or gated linear mixture, a Gaussian model, a mixture model, a neural network, a random forest, or a user-provided adapter. The same engine can therefore average models with very different parameter spaces without putting family-specific logic into the ensemble core.

## 2. Relationship to classical Bayesian Model Averaging

Classical Bayesian Model Averaging computes posterior model probabilities using marginal likelihoods obtained by integrating over model parameters. BPMA instead targets predictive performance by replacing exact marginal likelihoods with cross-validated estimates of predictive evidence. This substitution enables averaging over arbitrary machine-learning algorithms whose parameter spaces are inaccessible or computationally intractable.

BPMA should therefore be described as a predictive variant of Bayesian model averaging or as a Bayesian-inspired predictive model averaging framework. It is not presented as an approximation to classical Bayesian Model Averaging: the quantity being estimated is different. The word “Bayesian” refers to explicit priors over families, subsets, and hyperparameters; “predictive” identifies the cross-validated target; and “model averaging” describes the weighted prediction ensemble.

## 3. Definition and notation

### A complete model draw

A complete model draw is a full specification of one candidate model before it is fitted. It includes the estimator family, the observations used for training, and the family-specific parameters. We write this specification as

$\theta = (f, s, \phi)$

The symbol $\theta$ therefore identifies a particular point in the model space, not just a fitted estimator. Two draws can use the same family and parameter values but still represent different models if they use different training subsets. After the draw is sampled, BPMA fits the corresponding estimator on $s$ and evaluates its held-out predictive performance.

The three components have distinct roles. The family $f$ selects an estimator adapter, such as k-nearest neighbours, a mixture model, a neural network, or a random forest. The subset $s$ contains the training observations for the draw, including its sampled size and indices. It must be CV-admissible: every validation design used for that draw must have enough training observations to support the selected family and task. The parameter collection $\phi$ contains every family-specific choice sampled by the adapter, including hyperparameters and any other values needed to build the estimator. Because different families expose different parameters, $\phi$ is conditional on both $f$ and $s$.

### Prior factorization

The declared prior factorizes as

$p(\theta) = p(f)\,p(s)\,p(\phi\mid f,s)$

The family prior $p(f)$ expresses how much prior mass is assigned to each estimator family. Conditional on the family, the subset prior $p(s)$ controls the training size and the choice of observations; the implementation restricts this draw to subsets that support the requested cross-validation scheme. Finally, $p(\phi\mid f,s)$ is owned by the selected adapter, so it can enforce family-specific constraints and parameter dependencies. For example, a parameter choice can depend on the sampled subset size, and an architecture choice can determine which regularization values are valid.

This factorization is a modelling declaration rather than an assumption that all parameters share one common representation. It lets the core combine heterogeneous estimator families while leaving the details of each family’s parameter space with its adapter. The log prior contribution recorded for a draw is the sum of the log probabilities from these factors.

### Predictive score and temperature

For a fitted draw, cross-validation produces a predictive log-evidence score $L(\theta)$. In classification, this is based on the log probability assigned to each held-out target after class probabilities are aligned and smoothed. In regression, it is a Gaussian residual contribution using a variance floor. The score is aggregated across validation observations, so it measures expected held-out predictive performance under the selected CV design.

$L(\theta)$ is intentionally a pseudo-likelihood-like quantity. It summarizes out-of-sample prediction for a complete sampled model, but it is not an exact likelihood and does not result from integrating a parameter likelihood to obtain a marginal likelihood. Its value depends on the data, the cross-validation splitter, the scoring rule, and the numerical safeguards used by the task.

With positive target temperature $T$, BPMA combines the declared prior and predictive score through the unnormalized target score

$\log \pi_T(\theta) = \log p(\theta) + L(\theta) / T$

where an additive normalizing constant is omitted. A smaller $T$ makes differences in predictive performance more influential and concentrates weight on higher-scoring draws. A larger $T$ flattens those differences and keeps the resulting ensemble closer to the prior-supported collection of models. The target temperature controls final model weighting; it does not control the concentration of adaptive proposals, whose separate control is `adaptation_temperature`.

## 4. Prior-supported model draws

The family registry declares positive prior weights. The registry is normalized after filtering to families compatible with the requested task. Each adapter owns its conditional parameter prior and returns both sampled parameters and their log prior probability. The core does not interpret individual parameter names.

The subset prior samples a training size and an ordered subset while respecting the minimum data requirements imposed by cross-validation. Adapter priors can be conditional: for example, a tree depth can determine which leaf-size choices are valid, and a neural-network architecture can determine which regularization parameters are meaningful. This design preserves a clean separation between generic sampling and family-specific prior logic.

Every draw is fitted independently. Its record contains the family, sampled parameters, subset information, predictive log-evidence, prior contribution, proposal contribution, and final normalized weight. The resulting collection is the material from which BPMA predictions and diagnostics are computed.

## 5. Predictive evidence

For classification, each CV validation fold is scored from predicted class probabilities after aligning class columns to the global class order. Probability smoothing controlled by `alpha` prevents invalid logarithms. For regression, the scorer uses a Gaussian predictive residual contribution with a variance floor controlled by `epsilon`; the training-fold residual scale supplies the predictive concentration when needed.

The aggregate score is intentionally cross-validated. It measures how well a complete sampled model predicts held-out observations under the selected validation design. Consequently, the score depends on the CV splitter, the available data, and the scoring assumptions. It should be interpreted as predictive evidence, not as a claim that the model has a particular exact posterior probability.

## 6. BPMA weighting and prediction

When the proposal equals the declared prior, the prior terms cancel in the importance ratio. The score-only form is therefore

$\log w_i = L(\theta_i) / T; \bar{w}_i = \exp(\log w_i) / \sum_j \exp(\log w_j)$

This is the default behavior and preserves the existing non-adaptive numerical path. Classification predictions are the weighted average of globally aligned probability vectors. Regression predictions are the weighted average of fitted regression outputs. Family and parameter summaries aggregate these normalized predictive weights and are useful for describing which prior-supported regions of model space contributed to the ensemble.

## 7. Adaptive importance sampling

Adaptive sampling is opt-in through `adaptive_importance_sampling=True`. The first round samples from the normalized declared family prior. Later rounds adapt only the family probabilities; conditional subset and adapter-parameter draws remain under their original priors in this implementation.

Let $p(f)$ be the declared family prior and let $\hat q_t(f)$ be the normalized weighted family mass estimated from all draws before round $t$. The next family proposal is

$q_t(f) = \varepsilon p(f) + (1-\varepsilon)\hat q_t(f)$

where $\varepsilon$ is `defensive_prior_weight`. Before mixing, the estimated mass can be transformed by `adaptation_temperature`; smaller values concentrate more strongly on families with larger estimated predictive mass. Every prior-supported family retains positive proposal probability because of the defensive component.

All completed rounds are retained. If $\alpha_t$ is the fraction of retained draws generated in round $t$, the deterministic-mixture proposal for a draw with family $f$ is

$q_{\mathrm{mix}}(\theta) = p(s) p(\phi \mid f,s) \sum_t \alpha_t q_t(f)$

The centralized importance calculation is

$\log w(\theta) = \log p(\theta) + L(\theta) / T - \log q_{\mathrm{mix}}(\theta)$

Because only family probabilities adapt, the mixture denominator is evaluated exactly from the draw’s conditional prior contribution and the recorded family proposal probabilities. Oversampling a high-scoring family therefore changes computational allocation without artificially increasing that family’s final predictive mass.

Round construction is deterministic at the boundary: a complete batch is generated from child seeds derived from global draw indices, scored and fitted in parallel, and incorporated only after the batch finishes. This makes serial and parallel execution equivalent for a fixed seed and keeps proposal updates independent of worker completion order.

## 8. Automatic stopping and diagnostics

Adaptive fitting can stop after `min_rounds` when enabled convergence conditions hold for `stopping_patience` consecutive rounds. Proposal stability is always available through `proposal_tolerance`, measured by total variation distance between consecutive family proposals. Prediction stability is enabled by `prediction_tolerance` and compares predictions on the existing fixed convergence subset. ESS quality is enabled by `ess_target_fraction`.

The hard limits `max_estimators` and `max_rounds` always terminate the process when reached. In adaptive mode, `max_estimators` is cumulative across rounds; `round_size` controls the nominal size of each batch. ESS remains a diagnostic unless an explicit ESS target is configured.

Fitted adaptive estimators expose:

- `proposal_history_` and `round_history_`;
- `n_rounds_`;
- `effective_sample_size_` and `effective_sample_size_fraction_`;
- `adaptive_converged_`;
- `stopping_reason_`.

Each round record includes new and cumulative draw counts, proposal probabilities, predictive family mass, proposal distance, optional prediction change, ESS, ESS fraction, maximum normalized weight, convergence status, and the stopping reason. Histories use ordinary JSON-compatible values so they can be persisted with experiment results.

## 9. Software interface

The public estimators are:

```python
from bayesian_predictive_model_averaging import (
    BayesianPredictiveModelAveragingClassifier,
    BayesianPredictiveModelAveragingRegressor,
)

classifier = BayesianPredictiveModelAveragingClassifier(
    adaptive_importance_sampling=True,
    round_size=50,
    max_estimators=500,
    random_state=7,
)
classifier.fit(X_train, y_train)
probabilities = classifier.predict_proba(X_test)

regressor = BayesianPredictiveModelAveragingRegressor(
    n_estimators=100,
    random_state=7,
)
regressor.fit(X_train, y_train)
predictions = regressor.predict(X_test)
```

Both estimators follow scikit-learn conventions, including cloning, pipelines, and `GridSearchCV`. Family registrations are explicit and can be passed as runtime parameters, allowing experiments to compare different prior-supported family sets without modifying the core engine.

## 10. Reproducibility and computation

Each model draw receives a deterministic child seed based on the base seed and its global draw index. Adaptive rounds therefore preserve reproducibility when `n_jobs=1` is changed to a parallel setting. The round boundary also provides a natural synchronization point for updating proposals and diagnostics.

The method remains computationally demanding: every draw can require multiple CV fits, followed by a final fit on its sampled subset. Parallelism reduces wall time but does not change the target weighting calculation.

## 11. Limitations and future work

BPMA does not compute exact marginal likelihoods and its normalized predictive weights should not be read as classical posterior model probabilities. Results depend on the cross-validation design, predictive scoring rule, prior choices, temperature, and numerical safeguards. A predictive evidence score can favor a model that generalizes well under the selected validation scheme without being a good description of a data-generating posterior.

Adaptive proposals currently change only estimator-family probabilities. Subset sizes and adapter-owned parameters continue to be sampled from their original priors, which keeps proposal evaluation exact and preserves the adapter protocol. Future work can add adapter-level proposals, richer diagnostics for continuous parameter adaptation, and calibrated predictive uncertainty benchmarks.

## Conclusion

Bayesian Predictive Model Averaging combines explicit prior structure with cross-validated predictive evidence and weighted ensemble prediction. It acknowledges the practical value of Bayesian model uncertainty while making a clear methodological distinction from classical marginal-likelihood-based Bayesian Model Averaging. Optional adaptive importance sampling improves the allocation of model draws while deterministic-mixture correction preserves the declared prior target.
