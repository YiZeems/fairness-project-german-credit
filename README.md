# Responsible AI, German Credit Dataset

IA708 project, Télécom Paris, 2026.

> French version: [README.fr.md](README.fr.md)

## Team

- Julien GIMENEZ <julien.gimenez@telecom-paris.fr>
- Hugo FANCHINI <hugo.fanchini@telecom-paris.fr>
- Paul CINTRA <paul.cintra@telecom-paris.fr>
- Yimou ZHANG <yimou.zhang@telecom-paris.fr>
- Zaher HAMADEH <zaher.hamadeh@telecom-paris.fr>

## Contents of this folder

| File | Role |
|------|------|
| `responsiveAI-german-credit_rendu.ipynb` | Main notebook (12 sections, 92 cells) |
| `responsiveAI-german-credit_rapport.pdf` | Long pedagogical report (~30 pages) |
| `responsiveAI-german-credit_rapport.tex` | Report source code |
| `responsiveAI-german-credit_rapport_compile.sh` | Report compile script (LaTeX, 2 passes) |
| `responsibleAI-4.pdf` | Project assignment |
| `requirements.txt` | Python dependencies |
| `data/` | UCI German Credit raw data |

The PDF report is the primary written deliverable. The notebook contains the full computational pipeline and reproduces every number cited in the report.

## Subject

Audit of a credit scoring model (UCI German Credit, 1000 rows) along the three axes of `responsibleAI-4.pdf`:

- **Fairness**: audit (DP, EO, Disparate Impact), intersectional audit (gender × age), pre-processing reweighing (Kamiran & Calders 2012), per-group post-processing thresholds (Hardt et al. 2016), `fairlearn ThresholdOptimizer`.
- **Interpretability**: SHAP global and local (`LinearExplainer`, exact for logistic regression), permutation importance, proxy detection (univariate correlation and multivariate logistic regression), SHAP comparison with vs without sensitive attributes.
- **Uncertainty quantification**: bootstrap confidence intervals (B = 500 on baseline, B = 100 on three model variants).

Per the instructor's recommendation, the adversarial robustness axis from the assignment is replaced by uncertainty quantification via bootstrap.

## Three model variants

The whole study compares three versions of the model side by side:

- **V1** — with sensitive attributes (`gender`, `age_in_years`)
- **V2** — without sensitive attributes (baseline)
- **V3** — without sensitive attributes AND without univariate proxies (`|r| > 0.15`)

V3 is the main result: it improves both performance (AUC 0.785 → 0.790) and equity per attribute (DI age 0.69 → 0.87, above the EEOC threshold) without any in/post-model intervention. The intersectional bias (gender × age, DI = 0.48) remains unresolved.

## Stack

- [`scikit-learn`](https://scikit-learn.org/) — models, metrics, pipelines, permutation importance
- [`fairlearn`](https://fairlearn.org/) — `MetricFrame`, `demographic_parity_difference`, `equalized_odds_difference`, `ThresholdOptimizer`
- [`shap`](https://shap.readthedocs.io/) — `LinearExplainer` (exact form for linear models)
- `pandas`, `numpy`, `matplotlib`
- `papermill` for reproducible execution

## Installation

```bash
pip install -r requirements.txt
```

The `german.data` file must be present in `data/raw/` (downloadable from [UCI](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data)).

## Running the notebook

### Interactive

Open `responsiveAI-german-credit_rendu.ipynb` in Jupyter and run cells in order.

### Automated (papermill)

```bash
papermill responsiveAI-german-credit_rendu.ipynb responsiveAI-german-credit_rendu.ipynb
```

In-place execution: outputs are saved back into the source notebook. `random_state = 42` is fixed everywhere, so the run is deterministic.

Notebook execution generates an `outputs/` folder containing all figures (`00_convergence.png` through `19_bootstrap_triple.png`).

## Recompiling the report

```bash
bash responsiveAI-german-credit_rapport_compile.sh
```

This calls `pdflatex` twice (for table of contents and references). The script auto-detects `pdflatex` or `xelatex`. The `.tex` source references figures from `outputs/`, so the notebook should be executed first if the figures need to be regenerated.

## Notebook structure

1. Loading and label translation: UCI `Axx` codes mapped to French labels
2. Exploration and sensitive attributes: distributions, default rates per group (age threshold = 25 years)
   - 2.1 Proxy detection (univariate correlation `|r| > 0.15`, multivariate AUC features → sensitive attribute)
3. Data preparation: `ColumnTransformer`, 60 / 20 / 20 stratified split
4. Baseline model: L2 logistic regression, convergence check, comparison to trivial baselines
   - 4.1 Performance visualizations
   - 4.2 Business cost (UCI matrix)
5. Performance and fairness metrics: AUC, BalAcc, |DP|, |EO|, Disparate Impact (via fairlearn)
   - 5.1 Intersectional audit
6. Pre-processing mitigation: reweighing $w_i = P(S) \cdot P(Y) / P(S, Y)$
7. Post-processing mitigation: per-group thresholds (custom + `fairlearn ThresholdOptimizer`)
8. Comparison: 4 configurations × 3 variants, impossibility theorem
9. Interpretability: SHAP global, SHAP local, comparison with/without sensitive attributes, proxy detection
10. Uncertainty quantification: bootstrap B = 500 + B = 100 on three variants, individual prediction intervals
11. Synthesis and conclusion: methodological reproducibility, decision pathway, recommendations
12. References

## Methodological reproducibility

Beyond the specific numbers, the project provides a transferable canvas applicable to any binary classification with sensitive attributes (insurance scoring, hiring, social aid allocation, predictive justice, health scoring): identify sensitive attributes, audit the baseline, detect proxies, compare V1/V2/V3, evaluate classical mitigations, interpret globally and locally, quantify uncertainty, summarize and choose a motivated decision pathway.

## Main result

Removing both sensitive attributes and the 17 univariate proxies (`|r| > 0.15`) produces a model that is simultaneously more performant (AUC +0.005), more equitable per attribute (DP age divided by 3, DI age above the EEOC 80 % threshold), and more statistically stable (tightest bootstrap CI). On this dataset, upstream proxy removal is more effective than in/post-model interventions; on other datasets, V3 might cost performance or fail to improve fairness — the transferable lesson is to always compare V1/V2/V3 explicitly rather than assume the effect of a removal a priori.
