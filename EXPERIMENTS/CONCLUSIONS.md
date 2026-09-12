# Executed starter-run conclusions

The 12 question notebooks were executed with the repository virtual environment on the current implementation. These are preliminary, single-seed conclusions from the small starter configurations, not claims from the full repeated benchmark proposed in QUESTIONS.md.

| Question | Starter-run conclusion |
| --- | --- |
| Q1 | Not falsified; BPMA slightly beat equal-weight averaging and clearly beat the single logistic baseline on test log loss. |
| Q2 | Not falsified; prior weights changed family mass and test log loss, but the one-seed run cannot establish general prior usefulness. |
| Q3 | Not falsified; CV and fresh-test draw scores had Spearman correlation 0.690, with substantial remaining variability. |
| Q4 | Not falsified; subset averaging lost to full-data fitting on the baseline, consistent with the hypothesized trade-off. |
| Q5 | Not falsified; temperature changed concentration and performance, with 0.1 best on this split but no evidence yet about overfitting. |
| Q6 | Not falsified but unsupported here; adaptive sampling was slightly worse and had lower ESS at the matched 64-draw budget. |
| Q7 | Not falsified; corrected finite-space estimates were close to the exact target, with maximum observed absolute bias 0.0172. |
| Q8 | Not falsified; BPMA improved Brier score, but numerical calibration metrics and repeated seeds are still needed. |
| Q9 | Inconclusive, not falsified; the run hit the maximum budget instead of stopping early. |
| Q10 | Not falsified; 3-fold and 5-fold CV changed test log loss while family masses remained similar. |
| Q11 | Not falsified; removing an unsuitable linear family improved the moons result, but controlled distractor tests remain. |
| Q12 | Not falsified; quality improved at 64 draws but at substantially higher runtime. |

The strongest current evidence is limited to the observed starter conditions. The adaptive-allocation, convergence, CV-surrogate, and calibration claims require the repeated, nested, matched-budget studies specified in QUESTIONS.md before they can be considered supported.
