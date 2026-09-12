# Bayesian Predictive Model Averaging (BPMA)

Bayesian Predictive Model Averaging (BPMA) is a scikit-learn-compatible
ensemble for classification and regression. Instead of selecting one
algorithm and one hyperparameter configuration, BPMA samples complete models
from an explicit prior and averages their predictions according to
cross-validated predictive performance.

The model space can include several estimator families, training subsets, and
family-specific hyperparameters. This makes model uncertainty part of the
prediction rather than an implicit consequence of choosing a single winning
model.

## What the method does

For each draw, BPMA samples a complete candidate model

~~~
estimator family + training subset + family-specific parameters
~~~

and then:

1. Fits the candidate on each training fold of the selected cross-validation
   design.
2. Scores its held-out predictions.
3. Fits the candidate once more on its sampled training subset.
4. Gives the fitted candidate a normalized predictive weight.
5. Averages the predictions of all fitted candidates using those weights.

The declared prior factorizes as

~~~
p(model) = p(family) * p(training subset) * p(parameters | family, subset)
~~~

The family registry supplies `p(family)`. BPMA samples CV-admissible subset
sizes and observations, and each estimator-family adapter owns its conditional
parameter prior. Priors can therefore be structured: for example, the valid
number of neighbours can depend on the sampled subset and the valid number of
Gaussian-mixture components can depend on the available observations.

### Predictive evidence and weights

BPMA uses cross-validated predictive evidence rather than an exact marginal
likelihood. For a model draw `theta`, let `L(theta)` be its mean held-out
predictive log score. With target temperature `T`, the target weighting rule
is

~~~
log target(theta) = log p(theta) + L(theta) / T
~~~

When models are sampled directly from the declared prior, prior sampling is
already represented by the draw population and the normalized weights reduce
to a stable softmax of `L(theta) / T`. A lower `temperature` concentrates the
ensemble on better-scoring draws; a higher value keeps the ensemble closer to
the prior-supported model collection.

For classification, the score is the mean held-out log probability after class
alignment and probability smoothing. Predictions are weighted averages of
class-probability vectors. For regression, the score is a Gaussian residual
log score with a variance floor, and predictions are weighted averages of the
fitted regressors.

BPMA is therefore best described as Bayesian-inspired predictive model
averaging. It is not classical Bayesian Model Averaging: its normalized
weights are predictive weights based on cross-validation, not exact posterior
model probabilities from integrated likelihoods.

### Optional adaptive importance sampling

Set `adaptive_importance_sampling=True` to fit in rounds. The first round uses
the declared family prior. Later rounds can allocate more draws to families
with higher estimated predictive mass while retaining a defensive amount of
the original prior for every family. Deterministic-mixture importance weights
correct the changed allocation, so oversampling a family does not by itself
inflate its final predictive share.

Adaptive sampling currently changes family proposals only; subset sizes and
adapter parameters remain under their declared conditional priors. It is
useful when some families are expensive and substantially more promising than
others.

## Installation

BPMA requires Python 3.10 or newer.

### From PyPI

~~~bash
python -m pip install bayesian-predictive-model-averaging
~~~

### From a local checkout

~~~bash
git clone <repository-url>
cd BayesianPredictiveModelAveraging
python -m pip install -e .
~~~

Install optional notebook and plotting dependencies with:

~~~bash
python -m pip install -e ".[notebook]"
~~~

The optional `recursive-partition` extra installs the dependency needed by the
recursive-partition adapters:

~~~bash
python -m pip install bayesian-predictive-model-averaging[recursive-partition]
~~~

For development, including tests and linting:

~~~bash
python -m pip install -e ".[test,dev]"
~~~

The core dependencies are NumPy, SciPy, scikit-learn, and joblib.

## Quick start: classification

The following example uses the built-in family registry. The registry is a
uniform prior over k-nearest neighbours, gated linear mixtures,
class-conditional Gaussian mixtures, multilayer perceptrons, and random
forests.

~~~python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from bayesian_predictive_model_averaging import (
    BayesianPredictiveModelAveragingClassifier,
)

data = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.25,
    random_state=7,
    stratify=data.target,
)

model = BayesianPredictiveModelAveragingClassifier(
    n_estimators=100,
    cv=5,
    temperature=1.0,
    random_state=7,
    n_jobs=-1,
)
model.fit(X_train, y_train)

labels = model.predict(X_test)
probabilities = model.predict_proba(X_test)

print(f"accuracy: {model.score(X_test, y_test):.3f}")
print(probabilities.shape)  # (n_test_samples, n_classes)
~~~

Use `n_estimators="auto"` when you want the ensemble to grow in batches of
`20, 40, 80, ...` until its predictions stabilize or `max_estimators` is
reached:

~~~python
model = BayesianPredictiveModelAveragingClassifier(
    n_estimators="auto",
    max_estimators=640,
    tolerance=1e-3,
    convergence_metric="max",
    random_state=7,
)
model.fit(X_train, y_train)

print(model.n_estimators_)  # number of draws actually retained
print(model.converged_)
print(model.convergence_history_)
~~~

## Quick start: regression

The regressor follows the usual scikit-learn interface and returns the
weighted point prediction from all sampled models.

~~~python
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

from bayesian_predictive_model_averaging import (
    BayesianPredictiveModelAveragingRegressor,
)

data = load_diabetes()
X_train, X_test, y_train, y_test = train_test_split(
    data.data,
    data.target,
    test_size=0.25,
    random_state=7,
)

model = BayesianPredictiveModelAveragingRegressor(
    n_estimators=100,
    cv=5,
    random_state=7,
    n_jobs=-1,
)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
print(f"R^2: {model.score(X_test, y_test):.3f}")
~~~

The default regression registry uses the families that support regression:
k-nearest neighbours, gated linear mixtures, multilayer perceptrons, and
random forests. Classification-only families are filtered out automatically.

## Choosing estimator families and prior weights

Pass `family_registry` to replace the default registry. Entries can be
adapters or `FamilyRegistration` objects. Relative positive weights are
normalized into the family prior.

~~~python
from bayesian_predictive_model_averaging import (
    BayesianPredictiveModelAveragingClassifier,
    DecisionTreeAdapter,
    FamilyRegistration,
    LinearAdapter,
    RandomForestAdapter,
)

model = BayesianPredictiveModelAveragingClassifier(
    family_registry=[
        FamilyRegistration(LinearAdapter(), prior_weight=2.0),
        FamilyRegistration(DecisionTreeAdapter(), prior_weight=1.0),
        FamilyRegistration(RandomForestAdapter(), prior_weight=1.0),
    ],
    n_estimators=100,
    random_state=7,
)
model.fit(X_train, y_train)
~~~

Available built-in adapters include:

| Adapter | Classification | Regression |
| --- | :---: | :---: |
| `KNNAdapter` | yes | yes |
| `LinearAdapter` | yes | yes |
| `LinearMixtureAdapter` | yes | yes |
| `GaussianAdapter` | yes | yes |
| `GaussianMixtureAdapter` | yes | no |
| `MLPAdapter` | yes | yes |
| `DecisionTreeAdapter` | yes | yes |
| `RandomForestAdapter` | yes | yes |
| `RecursivePartitionLinearAdapter` | optional | no |
| `RecursivePartitionQuadraticAdapter` | optional | no |
| `RecursivePartitionQDAAdapter` | optional | no |
| `RecursivePartitionRBFAdapter` | optional | no |

The default registry contains `KNNAdapter`, `LinearMixtureAdapter`,
`GaussianMixtureAdapter`, `MLPAdapter`, and `RandomForestAdapter`. Use an
explicit registry when you want a fixed family, a controlled comparison, or a
different prior. Supplying `family_registry` replaces the default; it does not
append to it.

## Controlling the prior

The shared `scale_prior` controls ordered discrete choices such as sampled
subset size and, where applicable, neighbourhood size. The default is a
logistic scale prior. It can be configured with a mapping or a prior object:

~~~python
from bayesian_predictive_model_averaging import (
    BayesianPredictiveModelAveragingClassifier,
    LogisticScalePrior,
)

model = BayesianPredictiveModelAveragingClassifier(
    scale_prior=LogisticScalePrior(beta_shape=2.0, beta_scale=1.0),
    min_subset_size=100,
    max_subset_size=500,
    n_estimators=100,
    random_state=7,
)
~~~

The same prior can be supplied as:

~~~python
model = BayesianPredictiveModelAveragingClassifier(
    scale_prior={
        "family": "logistic",
        "beta_shape": 2.0,
        "beta_scale": 1.0,
    },
    random_state=7,
)
~~~

Other public prior helpers include `CategoricalPrior`,
`IntegerChoicePrior`, `LogUniformPrior`, `LogisticLogScalePrior`,
`SimplicityCategoricalPrior`, and `GaussianCovariancePrior`. Family adapters
use these to define valid conditional hyperparameter draws and record their
log prior probabilities.

`min_subset_size` and `max_subset_size` bound the sampled training subset. If
they are omitted, BPMA uses CV-admissible defaults. Classification subsets are
sampled with enough observations per class to support the selected CV design.

## Adaptive importance sampling

Adaptive mode runs in complete batches and updates the family proposal between
rounds. `max_estimators` is the total draw budget in this mode, while
`round_size` controls each batch.

~~~python
from bayesian_predictive_model_averaging import (
    BayesianPredictiveModelAveragingClassifier,
)

model = BayesianPredictiveModelAveragingClassifier(
    adaptive_importance_sampling=True,
    round_size=50,
    max_estimators=300,
    min_rounds=3,
    max_rounds=6,
    defensive_prior_weight=0.2,
    proposal_tolerance=1e-3,
    prediction_tolerance=1e-3,
    ess_target_fraction=0.5,
    stopping_patience=2,
    random_state=7,
    n_jobs=-1,
)
model.fit(X_train, y_train)

print(model.n_rounds_)
print(model.stopping_reason_)
print(model.effective_sample_size_)
print(model.effective_sample_size_fraction_)
~~~

The final predictions still use corrected predictive weights. The adaptive
diagnostics are available in:

~~~python
model.proposal_history_  # proposal probabilities by round
model.round_history_     # proposal distance, ESS, predictive mass, and stopping state
model.adaptive_converged_
~~~

Use `adaptation_temperature` to control how sharply later proposals follow
the estimated family mass. This is separate from `temperature`, which
controls final model weighting.

## Inspecting sampled models and predictive shares

Every fitted draw is retained as a serializable dictionary. It includes the
family, sampled parameters, subset, prior terms, CV score, importance terms,
round information, and final normalized weight.

~~~python
draws = model.get_model_draws()

best_draw = max(draws, key=lambda draw: draw["posterior_weight"])
print(best_draw["family_name"])
print(best_draw["parameters"])
print(best_draw["cv_log_pseudo_likelihood"])
print(best_draw["posterior_weight"])
~~~

Aggregate predictive shares by family and by parameter value with:

~~~python
masses = model.get_model_masses()
print(masses["family"])
print(masses["parameter"])
~~~

Family shares sum to one. Parameter shares are conditional within each family,
which makes them comparable even when families expose different parameters.

## scikit-learn compatibility

The estimators implement the standard `fit`, `predict`, `score`,
`get_params`, and `set_params` conventions. The classifier also implements
`predict_proba`. They can be cloned and used in pipelines and
`GridSearchCV`:

~~~python
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

pipeline = make_pipeline(
    StandardScaler(),
    BayesianPredictiveModelAveragingClassifier(
        n_estimators=40,
        random_state=7,
        n_jobs=-1,
    ),
)

search = GridSearchCV(
    pipeline,
    {
        "bayesianpredictivemodelaveragingclassifier__temperature": [0.5, 1.0, 2.0],
    },
    cv=3,
)
search.fit(X_train, y_train)
~~~

As with any estimator containing many candidate fits, nested cross-validation
can be computationally expensive. Keep `n_estimators` and the search grid
small while developing, then increase the draw budget for final experiments.

## Reproducibility and computational cost

Each draw receives a deterministic child seed derived from `random_state` and
its global draw index. Consequently, changing `n_jobs` does not change the
sampled models or their predictions for a fixed configuration. Adaptive rounds
update only after a complete batch finishes, so proposal updates are also
independent of worker completion order.

Each draw can require one fit per CV fold plus a final fit on its sampled
subset. Parallelism reduces wall-clock time but not the number of fits. For a
quick smoke test, use a small explicit `n_estimators` value and `n_jobs=1`.

## Development

From a checkout, run the tests and lint checks with:

~~~bash
python -m pip install -e ".[test,dev]"
python -m pytest
ruff check bayesian_predictive_model_averaging tests
~~~

The repository also includes:

- [WHITE_PAPER.md](WHITE_PAPER.md), the full methodological description;
- [ARCHITECTURE.md](ARCHITECTURE.md), implementation and extension details;
- [notebooks/simple_library_usage.ipynb](notebooks/simple_library_usage.ipynb),
  a complete notebook example;
- [notebooks/](notebooks/), reproducible experiment scaffolds;
- [EXPERIMENTS/README.md](EXPERIMENTS/README.md), experiment-specific
  instructions.

## License

This project is licensed under the MIT License.
