# BPMA question notebooks

Each notebook corresponds to one question in [`../QUESTIONS.md`](../QUESTIONS.md). They are intentionally designed as reproducible experiment scaffolds: the hypothesis and ablation are pre-registered in Markdown cells, the first run is small enough to execute locally, and the final cells are where longer grids and repeated seeds should be enabled.

Run notebooks from the repository root or from this directory. The notebooks locate the project package automatically and share small helpers from [`common.py`](common.py).

| Notebook | Question |
| --- | --- |
| `01_generalization.ipynb` | Predictive averaging versus selection |
| `02_prior_sensitivity.ipynb` | Effect of the declared prior |
| `03_cv_score_surrogate.ipynb` | CV score versus fresh generalization |
| `04_subset_sampling.ipynb` | Training-subset averaging |
| `05_temperature.ipynb` | Target-temperature sensitivity |
| `06_adaptive_allocation.ipynb` | Adaptive allocation and target preservation |
| `07_mixture_correction.ipynb` | Deterministic-mixture correction |
| `08_uncertainty_calibration.ipynb` | Calibration and uncertainty |
| `09_convergence_diagnostics.ipynb` | Stability and stopping rules |
| `10_cv_design.ipynb` | Validation-design sensitivity |
| `11_family_misspecification.ipynb` | Missing and distractor families |
| `12_compute_efficiency.ipynb` | Quality per computational budget |

The notebooks do not contain claims or committed results. Save executed copies or exported result tables separately so the source notebooks remain clean and reproducible.
