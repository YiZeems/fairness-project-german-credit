# German Credit — Responsible AI Pipeline

> **Course:** IADATA708 — Machine Learning Equitable et Interpretable  
> **Institution:** Télécom Paris — Mastère Spécialisé IA Multimodale  
> **Dataset:** UCI Statlog German Credit (1 000 observations, 20 attributes)

---

## Table of contents

1. [Project overview](#1-project-overview)
2. [Scientific context and motivations](#2-scientific-context-and-motivations)
3. [Dataset](#3-dataset)
4. [ML architecture — baseline model](#4-ml-architecture--baseline-model)
5. [Fairness theory and methods](#5-fairness-theory-and-methods)
6. [Interpretability theory and methods](#6-interpretability-theory-and-methods)
7. [Robustness evaluation](#7-robustness-evaluation)
8. [Results and analysis](#8-results-and-analysis)
9. [Design decisions](#9-design-decisions)
10. [Project structure](#10-project-structure)
11. [Installation and usage](#11-installation-and-usage)
12. [Outputs reference](#12-outputs-reference)
13. [Dependencies](#13-dependencies)
14. [Known limitations](#14-known-limitations)

---

## 1. Project overview

This project implements a complete **Responsible AI (RAI) pipeline** for credit risk classification. Starting from a raw dataset, it covers every stage of a fairness-aware ML workflow:

- **Baseline** — train a logistic regression classifier and evaluate performance
- **Fairness** — detect and mitigate bias with two complementary interventions (pre-processing and post-processing)
- **Interpretability** — explain predictions globally with exact linear SHAP and permutation importance
- **Robustness** — stress-test all models under controlled feature perturbations

All components are implemented from scratch (no scikit-learn) to make every algorithmic choice explicit and auditable. The pipeline is reproducible end-to-end via a single CLI command.

---

## 2. Scientific context and motivations

### Why fairness in credit scoring?

Automated credit decisions directly affect individuals' financial lives. Historically, credit scoring systems have been shown to discriminate along gender, age, ethnicity, and other sensitive attributes — sometimes as a direct consequence of historical patterns in training data, sometimes through proxy variables that correlate with protected characteristics.

Two legal/ethical frameworks motivate the fairness criteria used in this project:

| Framework | Fairness criterion | Intuition |
|---|---|---|
| Anti-classification | Do not use the protected attribute as input | Minimal but insufficient alone |
| Demographic parity | Equal selection rates across groups | "Equal outcomes" |
| Equal opportunity | Equal true positive rates across groups | "Equal access to credit for creditworthy applicants" |
| Equalized odds | Equal TPR **and** FPR across groups | Strictest, often incompatible with accuracy |

### Why interpretability?

Beyond regulatory requirements (GDPR right to explanation), interpretability serves three purposes in this project:

1. **Debugging** — verify that the model relies on legitimate signals, not protected proxies
2. **Trust** — allow domain experts to validate predictions
3. **Fairness auditing** — identify which features drive disparate predictions between groups

### Why robustness?

A model that is fair and interpretable on the test set but sensitive to small input perturbations provides weak real-world guarantees. Robustness testing checks whether fairness properties are stable when data quality degrades (noise, measurement error, category misclassification).

---

## 3. Dataset

- **Source:** UCI Machine Learning Repository — Statlog (German Credit Data)
- **Auto-download:** the pipeline fetches the raw file if not present (`--download-if-missing`)
- **Size:** 1 000 observations × 20 original attributes
- **Target:** `default` (binary)
  - `1` = bad credit risk (positive class, ~30 % of data)
  - `0` = good credit risk (negative class, ~70 % of data)

### Sensitive attributes

| Attribute | Groups | Decision rationale |
|---|---|---|
| **Gender** | Female / Male (inferred from `personal_status`) | Gender is a paradigmatic protected attribute in credit law |
| **Age** | Young (< threshold) / Older (≥ threshold, default 25) | Age discrimination is common in automated scoring |

Gender was chosen as the primary sensitive attribute because the dataset documentation explicitly flags `personal_status` as a potential source of gender bias and it produces clearer group imbalances than age at the default threshold.

### Data split

| Split | Proportion | Stratified |
|---|---|---|
| Train | 60 % | Yes (on target) |
| Validation | 20 % | Yes |
| Test | 20 % | Yes |

Stratification is critical here because the positive class (default = 1) represents only ~30 % of examples. Without it, small splits could have substantially different class ratios, which would corrupt fairness metric estimates.

### Preprocessing

- **Numeric features** — z-score normalization using train-set statistics only (no leakage)
- **Categorical features** — one-hot encoding; unknown test-set categories are silently dropped
- **Missing values** — none in this dataset

---

## 4. ML architecture — baseline model

### Choice of model: Logistic Regression

Logistic regression was selected deliberately over more powerful models (trees, neural networks) for three reasons:

1. **Inherent interpretability** — coefficients map directly to log-odds contributions; SHAP values are exact and closed-form
2. **Transparency for fairness analysis** — it is easy to trace which features drive predictions for each demographic group
3. **Sufficient capacity** — the German Credit dataset is small (1 000 samples) and linearly separable enough for logistic regression to be competitive

### Custom Adam implementation

Rather than using a library optimizer, the project implements Adam from scratch to keep the full training loop auditable:

```
m_t = β₁ · m_{t-1} + (1 − β₁) · g_t          (first moment)
v_t = β₂ · v_{t-1} + (1 − β₂) · g_t²         (second moment)
m̂_t = m_t / (1 − β₁ᵗ)                         (bias correction)
v̂_t = v_t / (1 − β₂ᵗ)                         (bias correction)
θ_t = θ_{t-1} − α · m̂_t / (√v̂_t + ε)         (parameter update)
```

**Hyperparameters:**

| Parameter | Value | Rationale |
|---|---|---|
| Learning rate α | 0.03 | Tuned on validation log-loss; higher values caused instability |
| L2 regularization | 0.01 | Prevents overfitting on the small training set |
| Max epochs | 3 500 | Sufficient for convergence with early stopping |
| Early stopping patience | 300 epochs | Avoids premature stopping while preventing overfitting |
| β₁ | 0.9 | Adam default |
| β₂ | 0.999 | Adam default |

### Threshold selection

The default classification threshold is chosen to maximize **Youden's index** (balanced accuracy) on the validation set, rather than using the fixed 0.5. This is important because the class imbalance (~30/70) means 0.5 systematically under-predicts the positive class.

---

## 5. Fairness theory and methods

### Definitions

Let S be the sensitive attribute (e.g., gender), Ŷ the model's prediction, and Y the true label.

**Demographic Parity Difference (DP):**
```
ΔDP = |P(Ŷ=1 | S=comparison) − P(Ŷ=1 | S=privileged)|
```
Measures whether both groups are assigned positive outcomes at equal rates, regardless of true labels.

**Equal Opportunity Difference (EO):**
```
ΔEO = |TPR_comparison − TPR_privileged|
     = |P(Ŷ=1 | Y=1, S=comparison) − P(Ŷ=1 | Y=1, S=privileged)|
```
Measures whether truly creditworthy individuals from both groups are equally likely to receive credit.

**Average Odds Difference (AOD):**
```
ΔAOD = 0.5 × (|FPR_comparison − FPR_privileged| + |TPR_comparison − TPR_privileged|)
```
Combines both false positive and true positive rate differences — the strictest of the three metrics.

**Why Equal Opportunity over Demographic Parity?**  
In a credit context, EO is often the more ethically relevant criterion: we want individuals who are genuinely creditworthy to have equal chances of approval regardless of their group. DP requires equal approval rates even if group base rates differ, which can conflict with accuracy.

### Method 1 — Reweighing (pre-processing)

**Type:** Data-level intervention applied during training.

**Mechanism:** Each training sample (xᵢ, yᵢ, sᵢ) receives a weight:

```
w(s, y) = P(S=s) · P(Y=y) / P(S=s, Y=y)
```

This reweights the joint distribution P(S, Y) toward independence, without modifying the data or the model architecture.

**Why reweighing?**
- It acts before the model sees the data, which means no changes to model training code
- It is computationally cheap (one pass over training labels)
- It is interpretable: the weights tell us exactly which (group, label) combinations are under- or over-represented
- It was introduced by Kamiran & Calders (2012) and has strong theoretical grounding

**Expected effect:** Reduce ΔDP and ΔEO by correcting the imbalanced representation of (group, label) combinations in training data.

### Method 2 — Per-group threshold calibration (post-processing)

**Type:** Decision-level intervention applied at inference time.

**Mechanism:** Instead of using a single global classification threshold, the model applies a distinct threshold per sensitive group:

```
Ŷ(x) = 1  if  f(x) ≥ θ(S(x))
```

Thresholds are calibrated on the **validation set** by grid search (181 candidates from 0.05 to 0.95) to minimize either ΔDP or ΔEO.

**Why post-processing?**
- It does not require retraining — useful when the model is a black box
- It directly targets a specific fairness criterion
- It is transparent: the per-group thresholds are explicit and auditable

**Limitation:** Thresholds are calibrated on validation data. If the validation set is small or unrepresentative, the calibrated thresholds may not generalize well to the test set — which is precisely what we observe in the results.

---

## 6. Interpretability theory and methods

### Method 1 — Exact linear SHAP

**Theory:** SHAP (SHapley Additive exPlanations) is a game-theoretic framework that assigns each feature a contribution φᵢ(x) to the model output, satisfying three axioms: efficiency, symmetry, and dummy player. For linear models, the SHAP values are exact and do not require sampling or approximation.

**Formula:**
```
φᵢ(x) = wᵢ · (xᵢ − E_train[xᵢ])
```

where wᵢ is the model coefficient and E_train[xᵢ] is the mean of feature i in the training set (the reference baseline).

**Aggregation for categorical features:**  
One-hot encoding creates one dummy column per category. The SHAP values of all dummy columns belonging to the same original feature are **summed** to produce a single contribution per original feature. This makes the output interpretable in terms of the original feature space rather than the encoded one.

**Why SHAP over plain coefficients?**  
Raw coefficients depend on feature scale and are not comparable across features. SHAP values are in the output space (log-odds units) and directly comparable, even for one-hot encoded categoricals.

### Method 2 — Permutation importance

**Theory:** A model-agnostic method that measures the importance of feature i as the drop in ROC-AUC when feature i is randomly shuffled on the test set (breaking its relationship with the target).

**Algorithm:**
```
For each feature i:
    baseline_auc = ROC_AUC(model, X_test)
    For r = 1 to R (R=10 repeats):
        X_shuffled = X_test with column i permuted
        importance_r = baseline_auc − ROC_AUC(model, X_shuffled)
    importance[i] = mean(importance_r over R repeats)
```

**Why two interpretability methods?**  
SHAP and permutation importance capture complementary signals:

| Aspect | SHAP | Permutation Importance |
|---|---|---|
| Scope | Local (per sample) and global | Global only |
| Basis | Model coefficients | Prediction change under shuffling |
| Sensitivity | To feature magnitude × coefficient | To feature–target correlation |
| Computational cost | O(n × d) | O(n × d × R) |

If both methods agree on the top features, that convergence is strong evidence for their true importance. If they disagree, it may indicate non-linearity, feature correlations, or scale effects.

---

## 7. Robustness evaluation

**Motivation:** A model that passes fairness checks on a clean test set but is fragile under small perturbations provides weak guarantees in production, where data quality is imperfect.

**Perturbation strategy:**

| Feature type | Perturbation | Parameters |
|---|---|---|
| Numeric | Add Gaussian noise: x̃ = x + ε, ε ~ N(0, σ²·std(x)²) | σ = 0.2 |
| Categorical | Replace with random training-set value at probability p | p = 0.1 |

**Design choices:**
- σ = 0.2 represents ~20 % of each feature's standard deviation — realistic measurement noise
- p = 0.1 models ~10 % category misclassification — plausible data entry error rate
- The same perturbation is applied identically to all three models so results are directly comparable

**Evaluation:** All three models (baseline, reweighing, post-processing) are evaluated on both clean and perturbed test sets. Performance and fairness metrics are reported for both, allowing direct comparison of robustness across interventions.

**Key question:** Do fairness interventions make the model more or less robust? (Hypothesis: no systematic effect expected.)

---

## 8. Results and analysis

### Gender-based fairness analysis (primary run)

#### Clean test set

| Model | ROC-AUC | Balanced Acc. | F1 | \|ΔDP\| | \|ΔEO\| |
|---|---|---|---|---|---|
| Baseline | 0.7900 | 0.7071 | 0.5915 | 0.1071 | 0.0741 |
| Reweighing | **0.7917** | **0.7155** | **0.6014** | **0.0998** | **0.0438** |
| Post-processing | 0.7900 | 0.7333 | 0.6216 | 0.2038 | 0.2222 |

#### Analysis

**Reweighing** achieves the best balance:
- Marginal AUC improvement (+0.17 %) — likely statistical noise on 200 test samples
- Slight reduction in ΔDP (−0.73 %) and meaningful reduction in ΔEO (−0.30 pp)
- Demonstrates that correcting training distribution imbalances can improve fairness without sacrificing performance

**Post-processing** produces unexpected results:
- Perfect AUC preservation (as expected — the decision boundary is unchanged)
- Significant worsening of ΔDP (+9.7 pp) and ΔEO (+14.8 pp)
- **Explanation:** The group sizes in the validation set are small (~40 females in 200 validation samples). The threshold calibration overfits to the validation set and does not generalize to the test set. This is a well-known failure mode of post-processing methods on small, imbalanced groups.

#### Robustness under perturbation

| Model | AUC (clean) | AUC (perturbed) | AUC drop |
|---|---|---|---|
| Baseline | 0.7900 | 0.7651 | −2.49 % |
| Reweighing | 0.7917 | 0.7665 | −2.51 % |
| Post-processing | 0.7900 | 0.7651 | −2.49 % |

All three models degrade similarly under perturbation. **Fairness interventions do not increase fragility.** This is an important finding: there is no robustness–fairness trade-off in this setting.

### Feature importance (top 5)

Both interpretability methods agree on the most important features:

| Rank | SHAP (mean \|φ\|) | Permutation Importance (ΔAUC) |
|---|---|---|
| 1 | checking_status (0.576) | checking_status (0.088) |
| 2 | savings_account_bonds (0.346) | purpose (0.028) |
| 3 | credit_history (0.290) | savings_account_bonds (0.025) |
| 4 | installment_rate (0.251) | duration_in_month (0.023) |
| 5 | purpose (0.251) | credit_history (0.021) |

**Interpretation:**
- `checking_status` (the balance in the applicant's checking account) is by far the most predictive feature in both methods — a clear and financially legitimate signal
- `credit_history` and `savings_account_bonds` also rank highly in both — coherent with credit risk theory
- The gender variable does not appear in the top features, suggesting the bias does not come from a direct gender effect but from correlations between gender and other features (e.g., loan purpose, amount)

---

## 9. Design decisions

| Decision | Choice | Rationale |
|---|---|---|
| Model family | Logistic regression (custom) | Exact SHAP, transparent coefficients, sufficient for 1 000 samples |
| No scikit-learn | Pure NumPy/Pandas | Auditable, portable, no implicit behavior |
| Optimizer | Adam | Faster convergence than SGD; handles sparse one-hot features well |
| Threshold selection | Youden index on validation | Accounts for class imbalance without arbitrary cutoff |
| SHAP | Exact linear (not approximate) | Linear model allows closed-form; no Monte Carlo sampling error |
| Categorical SHAP aggregation | Sum dummy SHAP values | Required for interpretability in original feature space |
| Robustness noise | 20 % of std (numeric), 10 % swap (categorical) | Realistic noise levels; parameterizable via CLI |
| Sensitive attribute | Gender (primary), Age (secondary) | Gender has clearest legal relevance; age allows comparison |
| Fairness criterion for calibration | Equal Opportunity (default) | More ethically grounded for credit access than demographic parity |
| Stratified split | Yes | Required to preserve label distribution in small splits |
| Random seed | 42 (default, configurable) | Full reproducibility |

---

## 10. Project structure

```
projet/
├── README.md                          # This file
├── pyproject.toml                     # Package metadata and dependencies
├── uv.lock                            # Locked dependency versions
│
├── german_credit_rai/                 # Main Python package
│   ├── __init__.py                    # Package version (0.2.0)
│   ├── __main__.py                    # python -m entry point
│   ├── cli.py                         # Argument parser and command dispatch
│   └── pipeline.py                    # Core implementation (~1 350 lines)
│       ├── TabularPreprocessor        # Normalization + one-hot encoding
│       ├── LogisticRegressionGD       # Custom logistic regression with Adam
│       ├── compute_linear_shap()      # Exact SHAP for linear models
│       ├── aggregate_shap_by_raw_feature()  # Aggregate dummy SHAP to features
│       ├── compute_group_thresholds() # Per-group threshold calibration
│       ├── compute_fairness_metrics() # DP, EO, AOD, per-group breakdown
│       └── run_pipeline()             # Full pipeline orchestration
│
├── notebook.ipynb                     # Interactive Jupyter analysis
│
├── data/
│   └── raw/
│       └── german.data                # UCI raw dataset (auto-downloaded)
│
└── outputs/                           # Results (git-ignored in large runs)
    ├── final_run/                     # Primary gender-based analysis
    │   ├── metrics.json               # Full results + config (JSON)
    │   ├── metrics_overview.csv       # Performance & fairness summary
    │   ├── shap_summary_*.csv         # Per-feature SHAP importance
    │   ├── shap_values_*.csv          # Per-sample SHAP values
    │   ├── permutation_importance_*.csv
    │   ├── predictions_clean.csv      # Predictions on clean test set
    │   ├── predictions_perturbed.csv  # Predictions on perturbed test set
    │   ├── *.png                      # Feature importance plots
    │   ├── summary.md                 # Markdown report
    │   └── run.log                    # Execution log
    └── final_run_age/                 # Secondary age-based analysis
```

---

## 11. Installation and usage

> **Python version required: 3.11.x** (exact — enforced by `uv.lock` and `.python-version`)

This section covers how to get an identical working environment on **macOS**, **Linux**, and **Windows**.  
Two paths are offered: **uv** (fast, handles Python version automatically) and **pip** (standard, requires Python 3.11 already installed).

---

### Option A — uv (recommended, all platforms)

`uv` is a fast Python package manager that installs the correct Python version automatically.

#### macOS / Linux

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Restart your shell or run:
source $HOME/.cargo/env

# 2. Clone the repo
git clone https://github.com/YiZeems/fairness-project-german-credit.git
cd fairness-project-german-credit

# 3. Create the environment and install all dependencies
#    uv reads .python-version (3.11) and uv.lock automatically
uv sync

# 4. (Optional) add Jupyter support
uv sync --extra notebook
```

#### Windows (PowerShell)

```powershell
# 1. Install uv
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Restart PowerShell, then:

# 2. Clone the repo
git clone https://github.com/YiZeems/fairness-project-german-credit.git
cd fairness-project-german-credit

# 3. Create the environment
uv sync

# 4. (Optional) add Jupyter support
uv sync --extra notebook
```

---

### Option B — pip with Python 3.11

Use this if you already have Python 3.11 installed and prefer the standard toolchain.

#### macOS / Linux — pip

```bash
# Verify you have Python 3.11
python3.11 --version   # must print 3.11.x

# Clone the repo
git clone https://github.com/YiZeems/fairness-project-german-credit.git
cd fairness-project-german-credit

# Create and activate a virtual environment
python3.11 -m venv .venv
source .venv/bin/activate

# Install the package in editable mode
pip install -e .

# (Optional) add Jupyter support
pip install -e ".[notebook]"
```

#### Windows — pip (PowerShell or CMD)

```powershell
# Verify you have Python 3.11
python --version   # must print 3.11.x

# Clone the repo
git clone https://github.com/YiZeems/fairness-project-german-credit.git
cd fairness-project-german-credit

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate

# Install the package in editable mode
pip install -e .

# (Optional) add Jupyter support
pip install -e ".[notebook]"
```

> **Where to get Python 3.11 on Windows:**  
> Download the installer from [python.org/downloads](https://www.python.org/downloads/release/python-3119/) and tick *"Add python.exe to PATH"* during installation.

---

### Verify the installation

After either option, confirm everything works (same command on all platforms):

```bash
# With uv
uv run python -m german_credit_rai --help

# With pip (venv activated)
python -m german_credit_rai --help
```

Expected output starts with `usage: german_credit_rai [-h] {run,show} ...`

---

### Run the full pipeline

The commands below use `python -m german_credit_rai`. If you used `uv`, prefix them with `uv run`:

```bash
# macOS / Linux — gender-based fairness analysis
python -m german_credit_rai run \
  --download-if-missing \
  --sensitive-attribute gender \
  --output-dir outputs/run_gender

# Windows (PowerShell) — same command, no backslash continuation needed
python -m german_credit_rai run --download-if-missing --sensitive-attribute gender --output-dir outputs/run_gender

# Age-based fairness analysis (all platforms)
python -m german_credit_rai run \
  --download-if-missing \
  --sensitive-attribute age \
  --age-threshold 25 \
  --output-dir outputs/run_age

# Display results from a previous run
python -m german_credit_rai show outputs/run_gender/metrics.json
```

---

### All CLI options

| Option | Default | Description |
|---|---|---|
| `--download-if-missing` | off | Fetch german.data from UCI if not present |
| `--sensitive-attribute` | `gender` | `gender` or `age` |
| `--age-threshold` | `25` | Age cutoff for young/old split |
| `--postprocessing-criterion` | `equal_opportunity` | `demographic_parity` or `equal_opportunity` |
| `--noise-scale` | `0.2` | Gaussian noise σ as fraction of feature std |
| `--category-swap-prob` | `0.1` | Probability of replacing a categorical value |
| `--learning-rate` | `0.03` | Adam learning rate |
| `--epochs` | `3500` | Maximum training epochs |
| `--l2` | `0.01` | L2 regularization coefficient |
| `--random-seed` | `42` | Global random seed |
| `--output-dir` | `outputs/run` | Directory for all output files |
| `--log-level` | `INFO` | Logging verbosity |

---

### Interactive notebook

```bash
# With uv
uv run jupyter notebook notebook.ipynb

# With pip (venv activated)
jupyter notebook notebook.ipynb
```

> **Windows note:** if the browser does not open automatically, copy the URL printed in the terminal (e.g. `http://localhost:8888/?token=...`) and paste it into your browser.

The notebook follows the same structure as the CLI pipeline and adds visual exploration of the data distribution, training curves, and per-group fairness metrics.

---

## 12. Outputs reference

Every run writes the following files to `--output-dir`:

| File | Description |
|---|---|
| `run.log` | Full execution log with timestamps |
| `metrics.json` | Complete results: config, per-model performance and fairness metrics |
| `metrics_overview.csv` | Flat CSV summary of all models × scenarios |
| `shap_summary_baseline.csv` | Mean \|SHAP\| per feature for the baseline model |
| `shap_summary_fair.csv` | Mean \|SHAP\| per feature for the reweighing model |
| `shap_values_baseline.csv` | Per-sample SHAP values (baseline) |
| `shap_values_fair.csv` | Per-sample SHAP values (reweighing) |
| `permutation_importance_baseline.csv` | AUC drop per feature (baseline) |
| `permutation_importance_fair.csv` | AUC drop per feature (reweighing) |
| `predictions_clean.csv` | Predictions (all models) on the clean test set |
| `predictions_perturbed.csv` | Predictions (all models) on the perturbed test set |
| `shap_baseline.png` | SHAP bar chart — baseline model |
| `shap_fair.png` | SHAP bar chart — reweighing model |
| `permutation_importance_baseline.png` | Permutation importance bar chart — baseline |
| `permutation_importance_fair.png` | Permutation importance bar chart — reweighing |
| `summary.md` | Human-readable markdown report |

---

## 13. Dependencies

| Package | Version | Role |
|---|---|---|
| `numpy` | ≥ 1.26 | All numerical computation (model, SHAP, metrics) |
| `pandas` | ≥ 2.2 | Data loading, preprocessing, CSV output |
| `requests` | ≥ 2.32 | Dataset download from UCI |
| `matplotlib` | ≥ 3.8 | Optional — feature importance plots |
| `jupyter` | ≥ 1.0 | Optional — interactive notebook |

Python ≥ 3.11 required. **No scikit-learn dependency** by design.

---

## 14. Known limitations

1. **Linear model only** — logistic regression cannot capture feature interactions; a gradient boosted tree might improve AUC by ~5–10 pp on this dataset
2. **Single sensitive attribute at a time** — no intersectional analysis (e.g., young women vs. older men); intersectional fairness requires much larger datasets to estimate reliably
3. **No confidence intervals** — with only 200 test samples, metric differences of < 2 pp should be interpreted with caution; bootstrap CIs would be needed for rigorous comparison
4. **Post-processing overfitting on small groups** — the validation set contains ~40 female examples; calibrated thresholds are noisy and do not generalize (observed in results)
5. **Fairness–accuracy impossibility** — it is mathematically impossible to satisfy both demographic parity and equal opportunity simultaneously unless base rates are equal; the project does not attempt joint optimization
6. **Synthetic perturbations** — the robustness evaluation uses simple i.i.d. noise; real distribution shifts (covariate shift, label shift, adversarial attacks) may produce qualitatively different effects

---

## References

- Kamiran, F. & Calders, T. (2012). *Data preprocessing techniques for classification without discrimination.* Knowledge and Information Systems, 33(1), 1–33.
- Hardt, M., Price, E., & Srebro, N. (2016). *Equality of opportunity in supervised learning.* NeurIPS 2016.
- Lundberg, S. M. & Lee, S.-I. (2017). *A unified approach to interpreting model predictions.* NeurIPS 2017.
- Dua, D. & Graff, C. (2019). UCI Machine Learning Repository. German Credit dataset.
- Kingma, D. P. & Ba, J. (2015). *Adam: A method for stochastic optimization.* ICLR 2015.
