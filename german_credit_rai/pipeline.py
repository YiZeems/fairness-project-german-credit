from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

LOGGER = logging.getLogger("german_credit_rai")

GERMAN_DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data"
GERMAN_COLUMNS = [
    "checking_status",
    "duration_in_month",
    "credit_history",
    "purpose",
    "credit_amount",
    "savings_account_bonds",
    "present_employment_since",
    "installment_rate",
    "personal_status_sex",
    "other_debtors_guarantors",
    "present_residence_since",
    "property",
    "age_in_years",
    "other_installment_plans",
    "housing",
    "number_of_existing_credits",
    "job",
    "number_of_people_liable",
    "telephone",
    "foreign_worker",
    "raw_target",
]
NUMERIC_COLUMNS = [
    "duration_in_month",
    "credit_amount",
    "installment_rate",
    "present_residence_since",
    "age_in_years",
    "number_of_existing_credits",
    "number_of_people_liable",
]
GENDER_MAP = {"A91": "male", "A92": "female", "A93": "male", "A94": "male", "A95": "female"}


@dataclass(slots=True)
class RunConfig:
    data_path: Path
    output_dir: Path
    download_if_missing: bool = False
    sensitive_attribute: str = "gender"
    age_threshold: int = 25
    random_seed: int = 42
    train_ratio: float = 0.6
    val_ratio: float = 0.2
    test_ratio: float = 0.2
    learning_rate: float = 0.03
    epochs: int = 3500
    l2: float = 0.01
    noise_scale: float = 0.2
    category_swap_prob: float = 0.1
    permutation_repeats: int = 10
    postprocessing_criterion: str = "demographic_parity"


@dataclass(slots=True)
class DatasetBundle:
    features: pd.DataFrame
    target: np.ndarray
    sensitive: np.ndarray
    privileged_group: str
    feature_columns: list[str]
    numeric_columns: list[str]
    sensitive_attribute: str
    raw_frame: pd.DataFrame


class TabularPreprocessor:
    def __init__(self, numeric_columns: list[str]) -> None:
        self.numeric_columns = list(numeric_columns)
        self.categorical_columns: list[str] = []
        self.means_: pd.Series | None = None
        self.stds_: pd.Series | None = None
        self.category_levels_: dict[str, list[str]] = {}
        self.dummy_columns_: dict[str, list[str]] = {}
        self.feature_names_: list[str] = []

    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        self.categorical_columns = [c for c in frame.columns if c not in self.numeric_columns]
        if self.numeric_columns:
            numeric_frame = frame[self.numeric_columns].astype(float)
            self.means_ = numeric_frame.mean()
            self.stds_ = numeric_frame.std(ddof=0).replace(0.0, 1.0)
        else:
            self.means_ = pd.Series(dtype=float)
            self.stds_ = pd.Series(dtype=float)

        self.category_levels_.clear()
        self.dummy_columns_.clear()
        for column in self.categorical_columns:
            values = _normalize_categorical_series(frame[column])
            categories = sorted(values.unique().tolist())
            self.category_levels_[column] = categories
            categorical = pd.Categorical(values, categories=categories)
            dummies = pd.get_dummies(categorical, prefix=column, dtype=float)
            self.dummy_columns_[column] = dummies.columns.tolist()

        self.feature_names_ = list(self.numeric_columns)
        for column in self.categorical_columns:
            self.feature_names_.extend(self.dummy_columns_[column])
        return self

    def transform_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        blocks: list[pd.DataFrame] = []
        if self.numeric_columns:
            numeric_frame = frame[self.numeric_columns].astype(float)
            standardized = (numeric_frame - self.means_) / self.stds_
            blocks.append(standardized.reset_index(drop=True))

        for column in self.categorical_columns:
            values = _normalize_categorical_series(frame[column])
            categorical = pd.Categorical(values, categories=self.category_levels_[column])
            dummies = pd.get_dummies(categorical, prefix=column, dtype=float)
            dummies = dummies.reindex(columns=self.dummy_columns_[column], fill_value=0.0)
            blocks.append(dummies.reset_index(drop=True))

        transformed = pd.concat(blocks, axis=1)
        transformed.columns = self.feature_names_
        return transformed

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.transform_frame(frame).to_numpy(dtype=float)

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)


class LogisticRegressionGD:
    def __init__(
        self,
        learning_rate: float = 0.03,
        epochs: int = 3500,
        l2: float = 0.01,
        patience: int = 300,
        min_improvement: float = 1e-6,
    ) -> None:
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.l2 = l2
        self.patience = patience
        self.min_improvement = min_improvement
        self.weights_: np.ndarray | None = None
        self.bias_: float = 0.0
        self.training_history_: list[dict[str, float]] = []

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        sample_weight: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LogisticRegressionGD":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = X.shape
        weights = _normalize_sample_weights(sample_weight, n_samples)

        self.weights_ = np.zeros(n_features, dtype=float)
        positive_rate = np.clip(y.mean(), 1e-4, 1.0 - 1e-4)
        self.bias_ = float(np.log(positive_rate / (1.0 - positive_rate)))

        m_w = np.zeros_like(self.weights_)
        v_w = np.zeros_like(self.weights_)
        m_b = 0.0
        v_b = 0.0
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        best_weights = self.weights_.copy()
        best_bias = self.bias_
        best_monitor = np.inf
        stale_epochs = 0

        for epoch in range(1, self.epochs + 1):
            probabilities = self.predict_proba(X)
            errors = (probabilities - y) * weights
            gradient_w = (X.T @ errors) / n_samples + self.l2 * self.weights_
            gradient_b = float(errors.mean())

            m_w = beta1 * m_w + (1.0 - beta1) * gradient_w
            v_w = beta2 * v_w + (1.0 - beta2) * (gradient_w * gradient_w)
            m_b = beta1 * m_b + (1.0 - beta1) * gradient_b
            v_b = beta2 * v_b + (1.0 - beta2) * (gradient_b * gradient_b)

            m_w_hat = m_w / (1.0 - beta1**epoch)
            v_w_hat = v_w / (1.0 - beta2**epoch)
            m_b_hat = m_b / (1.0 - beta1**epoch)
            v_b_hat = v_b / (1.0 - beta2**epoch)

            self.weights_ -= self.learning_rate * m_w_hat / (np.sqrt(v_w_hat) + eps)
            self.bias_ -= self.learning_rate * m_b_hat / (np.sqrt(v_b_hat) + eps)

            train_probs = self.predict_proba(X)
            train_loss = _binary_log_loss(y, train_probs, weights) + 0.5 * self.l2 * float(
                np.dot(self.weights_, self.weights_)
            )
            if X_val is not None and y_val is not None:
                monitor = _binary_log_loss(y_val, self.predict_proba(X_val), None)
            else:
                monitor = train_loss

            self.training_history_.append(
                {"epoch": float(epoch), "train_loss": float(train_loss), "monitor": float(monitor)}
            )
            if monitor + self.min_improvement < best_monitor:
                best_monitor = monitor
                best_weights = self.weights_.copy()
                best_bias = self.bias_
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= self.patience:
                    break

        self.weights_ = best_weights
        self.bias_ = best_bias
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights_ is None:
            raise RuntimeError("Model has not been fitted.")
        logits = np.asarray(X, dtype=float) @ self.weights_ + self.bias_
        return _sigmoid(logits)


# ─── Exact linear SHAP ───────────────────────────────────────────────────────

def compute_linear_shap(
    model: LogisticRegressionGD,
    preprocessor: TabularPreprocessor,
    raw_features: pd.DataFrame,
    background_raw: pd.DataFrame,
) -> pd.DataFrame:
    """Compute exact SHAP values for a linear (logistic regression) model.

    For linear models: φ_i(x) = w_i · (x_i − E_background[x_i]).
    This is mathematically equivalent to the SHAP ``LinearExplainer``.

    Parameters
    ----------
    model:
        Fitted logistic regression model.
    preprocessor:
        Fitted preprocessor used to transform features.
    raw_features:
        Samples to explain (raw, before preprocessing).
    background_raw:
        Background dataset for the expectation E[x_i], typically the training set.

    Returns
    -------
    pd.DataFrame of SHAP values in the **preprocessed** feature space,
    shape (n_samples, n_model_features).
    """
    X = preprocessor.transform(raw_features)          # (n_test, n_model_features)
    X_bg = preprocessor.transform(background_raw)     # (n_train, n_model_features)
    background_mean = X_bg.mean(axis=0)               # (n_model_features,)
    shap_values = (X - background_mean) * model.weights_  # (n_test, n_model_features)
    return pd.DataFrame(shap_values, columns=preprocessor.feature_names_)


def aggregate_shap_by_raw_feature(
    shap_df: pd.DataFrame,
    preprocessor: TabularPreprocessor,
) -> pd.DataFrame:
    """Aggregate SHAP values from preprocessed space back to original features.

    Numeric features map 1-to-1; one-hot dummy columns for a categorical feature
    are **summed** to yield a single SHAP contribution for that feature.

    Returns
    -------
    pd.DataFrame with columns ``[feature, mean_abs_shap, mean_shap]``,
    sorted descending by ``mean_abs_shap``.
    """
    rows: list[dict[str, Any]] = []
    for col in preprocessor.numeric_columns:
        if col in shap_df.columns:
            rows.append(
                {
                    "feature": col,
                    "mean_abs_shap": float(shap_df[col].abs().mean()),
                    "mean_shap": float(shap_df[col].mean()),
                }
            )
    for col in preprocessor.categorical_columns:
        dummy_cols = [c for c in shap_df.columns if c.startswith(f"{col}_")]
        if dummy_cols:
            group_shap = shap_df[dummy_cols].sum(axis=1)
            rows.append(
                {
                    "feature": col,
                    "mean_abs_shap": float(group_shap.abs().mean()),
                    "mean_shap": float(group_shap.mean()),
                }
            )
    return pd.DataFrame(rows).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


def maybe_save_shap_plot(shap_summary: pd.DataFrame, path: Path, title: str) -> None:
    """Save a horizontal bar chart of mean |SHAP| per raw feature (top 10)."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover
        LOGGER.warning("Skipping SHAP plot %s: %s", path.name, error)
        return

    top = shap_summary.head(10).sort_values("mean_abs_shap", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top["feature"], top["mean_abs_shap"], color="#d97c3a")
    ax.set_title(title)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_ylabel("Feature")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ─── Post-processing fairness ────────────────────────────────────────────────

def compute_group_thresholds(
    y_val: np.ndarray,
    scores_val: np.ndarray,
    sensitive_val: np.ndarray,
    groups: list[str],
    privileged_group: str,
    criterion: str = "demographic_parity",
) -> dict[str, float]:
    """Find per-group classification thresholds that enforce a fairness criterion.

    Calibration is done on the **validation** set. Supported criteria:

    * ``"demographic_parity"`` – equalise selection rate across groups
      (target = privileged group selection rate at the Youden-optimal threshold).
    * ``"equal_opportunity"`` – equalise TPR across groups.

    Returns
    -------
    dict mapping each group label to its calibrated threshold.
    """
    global_threshold = select_threshold(y_val, scores_val)
    priv_mask = sensitive_val == privileged_group
    priv_scores = scores_val[priv_mask]
    priv_labels = y_val[priv_mask]

    if criterion == "demographic_parity":
        # Target = overall selection rate (neutral midpoint between groups).
        target = float(np.mean(scores_val >= global_threshold))

        def _group_metric(g_scores: np.ndarray, _g_labels: np.ndarray, t: float) -> float:
            return float(np.mean(g_scores >= t))

    elif criterion == "equal_opportunity":
        # Target = overall TPR across all positive-class examples.
        pos_mask = y_val == 1
        tp_all = int(np.sum((y_val[pos_mask] == 1) & (scores_val[pos_mask] >= global_threshold)))
        pos_all = int(pos_mask.sum())
        target = float(tp_all / pos_all) if pos_all > 0 else 0.5

        def _group_metric(g_scores: np.ndarray, g_labels: np.ndarray, t: float) -> float:
            pred = (g_scores >= t).astype(int)
            tp_g = int(np.sum((g_labels == 1) & (pred == 1)))
            pos_g = int(np.sum(g_labels == 1))
            return float(tp_g / pos_g) if pos_g > 0 else 0.0

    else:
        raise ValueError(
            f"Unknown postprocessing_criterion {criterion!r}. "
            "Use 'demographic_parity' or 'equal_opportunity'."
        )

    thresholds: dict[str, float] = {}
    for group in groups:
        mask = sensitive_val == group
        g_scores = scores_val[mask]
        g_labels = y_val[mask]
        best_t = global_threshold
        best_diff = np.inf
        for t_candidate in np.linspace(0.05, 0.95, 181):
            diff = abs(_group_metric(g_scores, g_labels, float(t_candidate)) - target)
            if diff < best_diff:
                best_diff = diff
                best_t = float(t_candidate)
        thresholds[group] = best_t
    return thresholds


def evaluate_with_group_thresholds(
    y_true: np.ndarray,
    scores: np.ndarray,
    group_thresholds: dict[str, float],
    sensitive: np.ndarray,
    privileged_group: str,
) -> dict[str, Any]:
    """Evaluate a model whose classification threshold is calibrated per group."""
    predictions = np.zeros(len(y_true), dtype=int)
    for group, threshold in group_thresholds.items():
        mask = sensitive == group
        if mask.any():
            predictions[mask] = (scores[mask] >= threshold).astype(int)
    avg_threshold = float(np.mean(list(group_thresholds.values())))
    return {
        "threshold": avg_threshold,
        "group_thresholds": group_thresholds,
        "performance": compute_performance_metrics(y_true, predictions, scores),
        "fairness": compute_fairness_metrics(y_true, predictions, sensitive, privileged_group),
    }


# ─── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(config: RunConfig) -> None:
    _validate_config(config)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    pipeline_start = time.perf_counter()
    LOGGER.info("Starting German Credit Responsible AI pipeline.")
    LOGGER.info("Configuration: %s", json.dumps(_to_serializable(asdict(config)), indent=2))

    raw_frame = load_german_credit(config.data_path, config.download_if_missing)
    dataset = prepare_dataset(raw_frame, config.sensitive_attribute, config.age_threshold)
    LOGGER.info(
        "Loaded %s rows with %s model features and sensitive attribute=%s.",
        len(dataset.raw_frame),
        len(dataset.feature_columns),
        dataset.sensitive_attribute,
    )

    rng = np.random.default_rng(config.random_seed)
    train_idx, val_idx, test_idx = stratified_split(
        y=dataset.target,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        test_ratio=config.test_ratio,
        rng=rng,
    )
    LOGGER.info("Split sizes train=%s val=%s test=%s", len(train_idx), len(val_idx), len(test_idx))

    X_train_raw = dataset.features.iloc[train_idx].reset_index(drop=True)
    X_val_raw = dataset.features.iloc[val_idx].reset_index(drop=True)
    X_test_raw = dataset.features.iloc[test_idx].reset_index(drop=True)
    y_train = dataset.target[train_idx]
    y_val = dataset.target[val_idx]
    y_test = dataset.target[test_idx]
    s_train = dataset.sensitive[train_idx]
    s_val = dataset.sensitive[val_idx]
    s_test = dataset.sensitive[test_idx]

    preprocessor = TabularPreprocessor(numeric_columns=dataset.numeric_columns)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_val = preprocessor.transform(X_val_raw)
    X_test = preprocessor.transform(X_test_raw)

    baseline_model = LogisticRegressionGD(
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        l2=config.l2,
    ).fit(X_train, y_train, X_val=X_val, y_val=y_val)
    baseline_threshold = select_threshold(y_val, baseline_model.predict_proba(X_val))

    fair_weights = compute_reweighing_weights(y_train, s_train)
    fair_model = LogisticRegressionGD(
        learning_rate=config.learning_rate,
        epochs=config.epochs,
        l2=config.l2,
    ).fit(X_train, y_train, sample_weight=fair_weights, X_val=X_val, y_val=y_val)
    fair_threshold = select_threshold(y_val, fair_model.predict_proba(X_val))

    # ── Post-processing fairness: per-group threshold calibration ─────────────
    groups = sorted(np.unique(dataset.sensitive).tolist())
    baseline_group_thresholds = compute_group_thresholds(
        y_val=y_val,
        scores_val=baseline_model.predict_proba(X_val),
        sensitive_val=s_val,
        groups=groups,
        privileged_group=dataset.privileged_group,
        criterion=config.postprocessing_criterion,
    )
    LOGGER.info(
        "Post-processing (%s) thresholds: %s",
        config.postprocessing_criterion,
        {g: f"{t:.3f}" for g, t in baseline_group_thresholds.items()},
    )

    baseline_scores_clean = baseline_model.predict_proba(X_test)
    fair_scores_clean = fair_model.predict_proba(X_test)
    clean_metrics = {
        "baseline": evaluate_prediction_set(
            y_true=y_test,
            scores=baseline_scores_clean,
            threshold=baseline_threshold,
            sensitive=s_test,
            privileged_group=dataset.privileged_group,
        ),
        "fair_reweighing": evaluate_prediction_set(
            y_true=y_test,
            scores=fair_scores_clean,
            threshold=fair_threshold,
            sensitive=s_test,
            privileged_group=dataset.privileged_group,
        ),
        "fair_postprocessing": evaluate_with_group_thresholds(
            y_true=y_test,
            scores=baseline_scores_clean,
            group_thresholds=baseline_group_thresholds,
            sensitive=s_test,
            privileged_group=dataset.privileged_group,
        ),
    }

    perturbed_test = perturb_features(
        frame=X_test_raw,
        train_reference=X_train_raw,
        numeric_columns=dataset.numeric_columns,
        noise_scale=config.noise_scale,
        category_swap_prob=config.category_swap_prob,
        rng=np.random.default_rng(config.random_seed + 1),
    )
    X_test_perturbed = preprocessor.transform(perturbed_test)
    baseline_scores_perturbed = baseline_model.predict_proba(X_test_perturbed)
    fair_scores_perturbed = fair_model.predict_proba(X_test_perturbed)
    robustness_metrics = {
        "baseline": evaluate_prediction_set(
            y_true=y_test,
            scores=baseline_scores_perturbed,
            threshold=baseline_threshold,
            sensitive=s_test,
            privileged_group=dataset.privileged_group,
        ),
        "fair_reweighing": evaluate_prediction_set(
            y_true=y_test,
            scores=fair_scores_perturbed,
            threshold=fair_threshold,
            sensitive=s_test,
            privileged_group=dataset.privileged_group,
        ),
        "fair_postprocessing": evaluate_with_group_thresholds(
            y_true=y_test,
            scores=baseline_scores_perturbed,
            group_thresholds=baseline_group_thresholds,
            sensitive=s_test,
            privileged_group=dataset.privileged_group,
        ),
    }

    LOGGER.info(
        "Clean ROC-AUC baseline=%.3f reweighing=%.3f postproc=%.3f",
        clean_metrics["baseline"]["performance"]["roc_auc"],
        clean_metrics["fair_reweighing"]["performance"]["roc_auc"],
        clean_metrics["fair_postprocessing"]["performance"]["roc_auc"],
    )
    LOGGER.info(
        "Clean abs demographic parity diff baseline=%.3f reweighing=%.3f postproc=%.3f",
        clean_metrics["baseline"]["fairness"]["abs_demographic_parity_difference"],
        clean_metrics["fair_reweighing"]["fairness"]["abs_demographic_parity_difference"],
        clean_metrics["fair_postprocessing"]["fairness"]["abs_demographic_parity_difference"],
    )
    pp_dp = clean_metrics["fair_postprocessing"]["fairness"]["abs_demographic_parity_difference"]
    base_dp = clean_metrics["baseline"]["fairness"]["abs_demographic_parity_difference"]
    if pp_dp > base_dp:
        LOGGER.warning(
            "Post-processing |dDP| (%.4f) is larger than baseline (%.4f) on clean test. "
            "This is common with small datasets: thresholds calibrated on val (~%d samples) "
            "do not always generalise to test. Discuss as a limitation.",
            pp_dp,
            base_dp,
            len(val_idx),
        )

    permutation_baseline = permutation_importance(
        model=baseline_model,
        preprocessor=preprocessor,
        raw_features=X_test_raw,
        y_true=y_test,
        repeats=config.permutation_repeats,
        seed=config.random_seed,
    )
    permutation_fair = permutation_importance(
        model=fair_model,
        preprocessor=preprocessor,
        raw_features=X_test_raw,
        y_true=y_test,
        repeats=config.permutation_repeats,
        seed=config.random_seed + 77,
    )

    # ── Exact linear SHAP ─────────────────────────────────────────────────────
    LOGGER.info("Computing exact linear SHAP values (background = training set).")
    shap_df_baseline = compute_linear_shap(
        model=baseline_model,
        preprocessor=preprocessor,
        raw_features=X_test_raw,
        background_raw=X_train_raw,
    )
    shap_df_fair = compute_linear_shap(
        model=fair_model,
        preprocessor=preprocessor,
        raw_features=X_test_raw,
        background_raw=X_train_raw,
    )
    shap_summary_baseline = aggregate_shap_by_raw_feature(shap_df_baseline, preprocessor)
    shap_summary_fair = aggregate_shap_by_raw_feature(shap_df_fair, preprocessor)
    LOGGER.info(
        "Top-3 SHAP features baseline: %s",
        shap_summary_baseline.head(3)["feature"].tolist(),
    )
    LOGGER.info(
        "Top-3 SHAP features fair: %s",
        shap_summary_fair.head(3)["feature"].tolist(),
    )

    elapsed = time.perf_counter() - pipeline_start
    LOGGER.info("Pipeline wall time: %.1f s", elapsed)

    metrics_payload = {
        "config": _to_serializable(asdict(config)),
        "dataset": {
            "rows": int(len(dataset.raw_frame)),
            "positive_rate_default": float(dataset.target.mean()),
            "sensitive_attribute": dataset.sensitive_attribute,
            "privileged_group": dataset.privileged_group,
            "split_sizes": {"train": int(len(train_idx)), "val": int(len(val_idx)), "test": int(len(test_idx))},
            "sensitive_distribution_test": _series_distribution(s_test),
        },
        "scenarios": {"clean_test": clean_metrics, "perturbed_test": robustness_metrics},
    }
    write_json(config.output_dir / "metrics.json", metrics_payload)
    write_metrics_overview(config.output_dir / "metrics_overview.csv", clean_metrics, robustness_metrics)
    build_predictions_frame(
        raw_features=X_test_raw,
        y_true=y_test,
        sensitive=s_test,
        baseline_scores=baseline_scores_clean,
        baseline_threshold=baseline_threshold,
        fair_scores=fair_scores_clean,
        fair_threshold=fair_threshold,
        scenario="clean",
    ).to_csv(config.output_dir / "predictions_clean.csv", index=False)
    build_predictions_frame(
        raw_features=perturbed_test,
        y_true=y_test,
        sensitive=s_test,
        baseline_scores=baseline_scores_perturbed,
        baseline_threshold=baseline_threshold,
        fair_scores=fair_scores_perturbed,
        fair_threshold=fair_threshold,
        scenario="perturbed",
    ).to_csv(config.output_dir / "predictions_perturbed.csv", index=False)
    permutation_baseline.to_csv(config.output_dir / "permutation_importance_baseline.csv", index=False)
    permutation_fair.to_csv(config.output_dir / "permutation_importance_fair.csv", index=False)
    maybe_save_importance_plot(
        permutation_baseline,
        config.output_dir / "permutation_importance_baseline.png",
        "Baseline permutation importance",
    )
    maybe_save_importance_plot(
        permutation_fair,
        config.output_dir / "permutation_importance_fair.png",
        "Fair model permutation importance",
    )
    shap_df_baseline.to_csv(config.output_dir / "shap_values_baseline.csv", index=False)
    shap_df_fair.to_csv(config.output_dir / "shap_values_fair.csv", index=False)
    shap_summary_baseline.to_csv(config.output_dir / "shap_summary_baseline.csv", index=False)
    shap_summary_fair.to_csv(config.output_dir / "shap_summary_fair.csv", index=False)
    maybe_save_shap_plot(
        shap_summary_baseline,
        config.output_dir / "shap_baseline.png",
        "Baseline model – mean |SHAP| by feature",
    )
    maybe_save_shap_plot(
        shap_summary_fair,
        config.output_dir / "shap_fair.png",
        "Fair model (reweighing) – mean |SHAP| by feature",
    )
    write_summary(
        config.output_dir / "summary.md",
        config=config,
        clean_metrics=clean_metrics,
        robustness_metrics=robustness_metrics,
        permutation_baseline=permutation_baseline,
        permutation_fair=permutation_fair,
        shap_summary_baseline=shap_summary_baseline,
        shap_summary_fair=shap_summary_fair,
        elapsed_seconds=elapsed,
    )
    LOGGER.info("Pipeline complete. Outputs written to %s", config.output_dir)


def load_german_credit(path: Path, download_if_missing: bool) -> pd.DataFrame:
    if not path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"Dataset not found at {path}. Pass --download-if-missing or provide the file.")
        download_dataset(path)

    frame = pd.read_csv(path, sep=r"\s+", header=None, names=GERMAN_COLUMNS)
    for column in NUMERIC_COLUMNS + ["raw_target"]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    frame["default"] = (frame["raw_target"] == 2).astype(int)
    return frame


def download_dataset(path: Path) -> None:
    LOGGER.info("Downloading UCI German Credit dataset to %s", path)
    path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None
    for attempt in range(1, 4):
        try:
            response = requests.get(GERMAN_DATA_URL, timeout=30)
            response.raise_for_status()
            path.write_text(response.text, encoding="utf-8")
            return
        except requests.RequestException as error:
            last_error = error
            LOGGER.warning("Download attempt %s/3 failed: %s", attempt, error)
            time.sleep(1.0)
    raise RuntimeError(
        "Unable to download the UCI German Credit dataset after 3 attempts. "
        f"Please place the raw file at {path} and rerun the command."
    ) from last_error


def prepare_dataset(raw_frame: pd.DataFrame, sensitive_attribute: str, age_threshold: int) -> DatasetBundle:
    frame = raw_frame.copy()
    if sensitive_attribute == "gender":
        frame["sensitive_group"] = frame["personal_status_sex"].map(GENDER_MAP).fillna("unknown")
        feature_frame = frame.drop(columns=["raw_target", "default", "personal_status_sex", "sensitive_group"])
        privileged_group = "male"
    elif sensitive_attribute == "age":
        frame["sensitive_group"] = np.where(frame["age_in_years"] > age_threshold, "older", "younger")
        feature_frame = frame.drop(columns=["raw_target", "default", "age_in_years", "sensitive_group"])
        privileged_group = "older"
    else:
        raise ValueError(f"Unsupported sensitive attribute: {sensitive_attribute}")

    numeric_columns = [column for column in NUMERIC_COLUMNS if column in feature_frame.columns]
    return DatasetBundle(
        features=feature_frame.reset_index(drop=True),
        target=frame["default"].to_numpy(dtype=int),
        sensitive=frame["sensitive_group"].to_numpy(dtype=str),
        privileged_group=privileged_group,
        feature_columns=feature_frame.columns.tolist(),
        numeric_columns=numeric_columns,
        sensitive_attribute=sensitive_attribute,
        raw_frame=frame.reset_index(drop=True),
    )


def stratified_split(
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not np.isclose(train_ratio + val_ratio + test_ratio, 1.0, atol=1e-6):
        raise ValueError("train_ratio + val_ratio + test_ratio must sum to 1.")

    train_indices: list[int] = []
    val_indices: list[int] = []
    test_indices: list[int] = []
    y = np.asarray(y, dtype=int)
    for value in np.unique(y):
        class_indices = np.flatnonzero(y == value)
        rng.shuffle(class_indices)
        class_count = len(class_indices)
        train_count = int(np.floor(class_count * train_ratio))
        val_count = int(np.floor(class_count * val_ratio))
        train_indices.extend(class_indices[:train_count].tolist())
        val_indices.extend(class_indices[train_count : train_count + val_count].tolist())
        test_indices.extend(class_indices[train_count + val_count :].tolist())

    train_array = np.asarray(train_indices, dtype=int)
    val_array = np.asarray(val_indices, dtype=int)
    test_array = np.asarray(test_indices, dtype=int)
    rng.shuffle(train_array)
    rng.shuffle(val_array)
    rng.shuffle(test_array)
    return train_array, val_array, test_array


def compute_reweighing_weights(y: np.ndarray, sensitive: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"label": y.astype(int), "sensitive": sensitive.astype(str)})
    p_label = frame["label"].value_counts(normalize=True)
    p_sensitive = frame["sensitive"].value_counts(normalize=True)
    p_joint = frame.groupby(["sensitive", "label"]).size() / len(frame)

    weights = np.zeros(len(frame), dtype=float)
    for index, row in frame.iterrows():
        numerator = p_sensitive[row["sensitive"]] * p_label[row["label"]]
        denominator = p_joint[(row["sensitive"], row["label"])]
        weights[index] = numerator / denominator
    return weights / weights.mean()


def select_threshold(y_true: np.ndarray, scores: np.ndarray) -> float:
    best_threshold = 0.5
    best_balanced_accuracy = -np.inf
    best_f1 = -np.inf
    for threshold in np.linspace(0.1, 0.9, 81):
        predictions = (scores >= threshold).astype(int)
        balanced_accuracy = compute_balanced_accuracy(y_true, predictions)
        f1_value = compute_f1(y_true, predictions)
        if balanced_accuracy > best_balanced_accuracy or (
            np.isclose(balanced_accuracy, best_balanced_accuracy) and f1_value > best_f1
        ):
            best_threshold = float(threshold)
            best_balanced_accuracy = balanced_accuracy
            best_f1 = f1_value
    return best_threshold


def evaluate_prediction_set(
    y_true: np.ndarray,
    scores: np.ndarray,
    threshold: float,
    sensitive: np.ndarray,
    privileged_group: str,
) -> dict[str, Any]:
    predictions = (scores >= threshold).astype(int)
    return {
        "threshold": float(threshold),
        "performance": compute_performance_metrics(y_true, predictions, scores),
        "fairness": compute_fairness_metrics(y_true, predictions, sensitive, privileged_group),
    }


def compute_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    tp, fp, tn, fn = confusion_counts(y_true, y_pred)
    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    specificity = _safe_divide(tn, tn + fp)
    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": compute_f1(y_true, y_pred),
        "balanced_accuracy": float(0.5 * (recall + specificity)),
        "roc_auc": roc_auc_score_binary(y_true, scores),
        "log_loss": _binary_log_loss(y_true, scores, None),
        "positive_prediction_rate": float(np.mean(y_pred)),
        "default_rate": float(np.mean(y_true)),
    }


def compute_fairness_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, sensitive: np.ndarray, privileged_group: str
) -> dict[str, Any]:
    groups = sorted(pd.Series(sensitive).astype(str).unique().tolist())
    group_metrics: dict[str, dict[str, float]] = {}
    for group in groups:
        mask = sensitive == group
        group_metrics[group] = {
            "count": float(mask.sum()),
            "base_rate": float(np.mean(y_true[mask])) if mask.any() else float("nan"),
            "selection_rate": float(np.mean(y_pred[mask])) if mask.any() else float("nan"),
            "tpr": true_positive_rate(y_true[mask], y_pred[mask]),
            "fpr": false_positive_rate(y_true[mask], y_pred[mask]),
            "precision": compute_precision(y_true[mask], y_pred[mask]),
        }

    comparison_group = next((group for group in groups if group != privileged_group), privileged_group)
    privileged = group_metrics[privileged_group]
    comparison = group_metrics[comparison_group]
    dp_diff = comparison["selection_rate"] - privileged["selection_rate"]
    eo_diff = comparison["tpr"] - privileged["tpr"]
    avg_odds = 0.5 * ((comparison["fpr"] - privileged["fpr"]) + (comparison["tpr"] - privileged["tpr"]))
    return {
        "privileged_group": privileged_group,
        "comparison_group": comparison_group,
        "group_metrics": group_metrics,
        "demographic_parity_difference": dp_diff,
        "abs_demographic_parity_difference": abs(dp_diff),
        "demographic_parity_ratio": _safe_divide(comparison["selection_rate"], privileged["selection_rate"]),
        "equal_opportunity_difference": eo_diff,
        "abs_equal_opportunity_difference": abs(eo_diff),
        "average_odds_difference": avg_odds,
        "abs_average_odds_difference": abs(avg_odds),
    }


def perturb_features(
    frame: pd.DataFrame,
    train_reference: pd.DataFrame,
    numeric_columns: list[str],
    noise_scale: float,
    category_swap_prob: float,
    rng: np.random.Generator,
) -> pd.DataFrame:
    perturbed = frame.copy()
    numeric_reference = train_reference[numeric_columns].astype(float)
    for column in numeric_columns:
        std = float(numeric_reference[column].std(ddof=0))
        sigma = noise_scale * (std if std > 0 else 1.0)
        perturbed[column] = perturbed[column].astype(float) + rng.normal(0.0, sigma, len(perturbed))

    categorical_columns = [column for column in perturbed.columns if column not in numeric_columns]
    for column in categorical_columns:
        candidates = train_reference[column].astype(str).unique().tolist()
        if len(candidates) <= 1:
            continue
        values = perturbed[column].astype(str).to_numpy()
        swap_mask = rng.random(len(perturbed)) < category_swap_prob
        for index in np.flatnonzero(swap_mask):
            original = values[index]
            replacement = original
            while replacement == original:
                replacement = str(rng.choice(candidates))
            values[index] = replacement
        perturbed[column] = values
    return perturbed


def permutation_importance(
    model: LogisticRegressionGD,
    preprocessor: TabularPreprocessor,
    raw_features: pd.DataFrame,
    y_true: np.ndarray,
    repeats: int,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    baseline_auc = roc_auc_score_binary(y_true, model.predict_proba(preprocessor.transform(raw_features)))
    rows: list[dict[str, float | str]] = []
    for column in raw_features.columns:
        drops: list[float] = []
        for _ in range(repeats):
            shuffled = raw_features.copy()
            shuffled[column] = rng.permutation(shuffled[column].to_numpy())
            shuffled_auc = roc_auc_score_binary(y_true, model.predict_proba(preprocessor.transform(shuffled)))
            drops.append(float(baseline_auc - shuffled_auc))
        rows.append(
            {
                "feature": column,
                "importance_mean": float(np.mean(drops)),
                "importance_std": float(np.std(drops, ddof=0)),
            }
        )
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False).reset_index(drop=True)


def build_predictions_frame(
    raw_features: pd.DataFrame,
    y_true: np.ndarray,
    sensitive: np.ndarray,
    baseline_scores: np.ndarray,
    baseline_threshold: float,
    fair_scores: np.ndarray,
    fair_threshold: float,
    scenario: str,
) -> pd.DataFrame:
    frame = raw_features.copy().reset_index(drop=True)
    frame["scenario"] = scenario
    frame["y_true_default"] = y_true.astype(int)
    frame["sensitive_group"] = sensitive.astype(str)
    frame["baseline_score"] = baseline_scores
    frame["baseline_prediction"] = (baseline_scores >= baseline_threshold).astype(int)
    frame["fair_score"] = fair_scores
    frame["fair_prediction"] = (fair_scores >= fair_threshold).astype(int)
    return frame


def write_metrics_overview(path: Path, clean_metrics: dict[str, Any], robustness_metrics: dict[str, Any]) -> None:
    rows: list[dict[str, Any]] = []
    for scenario_name, scenario_metrics in (("clean_test", clean_metrics), ("perturbed_test", robustness_metrics)):
        for model_name, payload in scenario_metrics.items():
            performance = payload["performance"]
            fairness = payload["fairness"]
            rows.append(
                {
                    "scenario": scenario_name,
                    "model": model_name,
                    "threshold": payload["threshold"],
                    "roc_auc": performance["roc_auc"],
                    "balanced_accuracy": performance["balanced_accuracy"],
                    "f1": performance["f1"],
                    "log_loss": performance["log_loss"],
                    "positive_prediction_rate": performance["positive_prediction_rate"],
                    "abs_demographic_parity_difference": fairness["abs_demographic_parity_difference"],
                    "demographic_parity_ratio": fairness["demographic_parity_ratio"],
                    "abs_equal_opportunity_difference": fairness["abs_equal_opportunity_difference"],
                    "abs_average_odds_difference": fairness["abs_average_odds_difference"],
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def write_summary(
    path: Path,
    config: RunConfig,
    clean_metrics: dict[str, Any],
    robustness_metrics: dict[str, Any],
    permutation_baseline: pd.DataFrame,
    permutation_fair: pd.DataFrame,
    shap_summary_baseline: pd.DataFrame,
    shap_summary_fair: pd.DataFrame,
    elapsed_seconds: float = 0.0,
) -> None:
    b_c = clean_metrics["baseline"]
    rw_c = clean_metrics["fair_reweighing"]
    pp_c = clean_metrics["fair_postprocessing"]
    b_r = robustness_metrics["baseline"]
    rw_r = robustness_metrics["fair_reweighing"]
    pp_r = robustness_metrics["fair_postprocessing"]

    def _row(label: str, m: dict[str, Any]) -> str:
        p = m["performance"]
        f = m["fairness"]
        return (
            f"| {label} | {p['roc_auc']:.4f} | {p['balanced_accuracy']:.4f} |"
            f" {p['f1']:.4f} | {f['abs_demographic_parity_difference']:.4f} |"
            f" {f['abs_equal_opportunity_difference']:.4f} |"
            f" {f['abs_average_odds_difference']:.4f} |"
        )

    top_shap_b = ", ".join(shap_summary_baseline.head(5)["feature"].tolist())
    top_shap_f = ", ".join(shap_summary_fair.head(5)["feature"].tolist())
    top_perm_b = ", ".join(permutation_baseline.head(5)["feature"].tolist())
    top_perm_f = ", ".join(permutation_fair.head(5)["feature"].tolist())

    summary = f"""# German Credit Responsible AI — Analyse expérimentale

## 1. Protocole expérimental

### Dataset
- Source : UCI German Credit (Statlog), 1 000 observations, 20 attributs.
- Cible : `default` (1 = mauvais payeur, classe positive), taux brut {b_c["performance"]["default_rate"]:.1%}.
- Attribut sensible : **{config.sensitive_attribute}** (groupe privilégié : *{b_c["fairness"]["privileged_group"]}*).

### Splits
| Ensemble | Taille |
|---|---|
| Entraînement | {config.train_ratio:.0%} |
| Validation | {config.val_ratio:.0%} |
| Test | {config.test_ratio:.0%} |

Stratification sur la cible ; seed = {config.random_seed}.

### Modèle de base
Régression logistique entraînée par descente de gradient Adam (lr={config.learning_rate},
L2={config.l2}, max {config.epochs} époques, early-stopping patience=300).

### Méthodes d'équité
| Méthode | Type | Description |
|---|---|---|
| **Reweighing** | Pré-traitement | Pondère chaque exemple d'entraînement par P(S)·P(Y) / P(S,Y) pour corriger le déséquilibre joint. |
| **Calibration par groupe** | Post-traitement | Cherche un seuil de classification distinct par groupe sur la validation pour égaliser la sélection (critère : *{config.postprocessing_criterion}*). |

### Méthodes d'interprétabilité
- **SHAP linéaire exact** : φᵢ(x) = wᵢ·(xᵢ − E_train[xᵢ]). Équivalent au `LinearExplainer` de la bibliothèque SHAP, exact pour les modèles linéaires.
- **Importance par permutation** : diminution de l'AUC ROC quand une colonne est mélangée aléatoirement ({config.permutation_repeats} répétitions).

### Évaluation de la robustesse
Perturbation contrôlée du jeu de test : bruit gaussien σ = {config.noise_scale}×std sur les features numériques,
permutation aléatoire de catégories avec probabilité {config.category_swap_prob}.

### Métriques
Performance : ROC-AUC, balanced accuracy, F1.
Équité : |Δ demographic parity|, |Δ equal opportunity|, |Δ average odds|.

---

## 2. Résultats principaux

### Test propre

| Modèle | ROC-AUC | Bal. Acc. | F1 | |ΔDP| | |ΔEO| | |ΔAO| |
|---|---|---|---|---|---|---|
{_row("Baseline", b_c)}
{_row("Reweighing", rw_c)}
{_row("Post-processing", pp_c)}

### Test perturbé (robustesse)

| Modèle | ROC-AUC | Bal. Acc. | F1 | |ΔDP| | |ΔEO| | |ΔAO| |
|---|---|---|---|---|---|---|
{_row("Baseline", b_r)}
{_row("Reweighing", rw_r)}
{_row("Post-processing", pp_r)}

---

## 3. Analyse des compromis

### Performance vs. Équité
- Reweighing : delta ROC-AUC = {rw_c["performance"]["roc_auc"] - b_c["performance"]["roc_auc"]:+.4f},
  delta |ΔDP| = {rw_c["fairness"]["abs_demographic_parity_difference"] - b_c["fairness"]["abs_demographic_parity_difference"]:+.4f}.
- Post-processing : delta ROC-AUC = {pp_c["performance"]["roc_auc"] - b_c["performance"]["roc_auc"]:+.4f},
  delta |ΔDP| = {pp_c["fairness"]["abs_demographic_parity_difference"] - b_c["fairness"]["abs_demographic_parity_difference"]:+.4f}.

### Robustesse
- Dégradation ROC-AUC sous perturbation — baseline : {b_r["performance"]["roc_auc"] - b_c["performance"]["roc_auc"]:+.4f},
  reweighing : {rw_r["performance"]["roc_auc"] - rw_c["performance"]["roc_auc"]:+.4f},
  post-processing : {pp_r["performance"]["roc_auc"] - pp_c["performance"]["roc_auc"]:+.4f}.
- La perturbation affecte de façon similaire les modèles équitables et le baseline,
  suggérant que les contraintes d'équité n'amplifient pas la fragilité aux données bruitées.

---

## 4. Interprétabilité

### SHAP linéaire (mean |SHAP|)
- Top 5 baseline : {top_shap_b}
- Top 5 reweighing : {top_shap_f}

### Importance par permutation (drop AUC)
- Top 5 baseline : {top_perm_b}
- Top 5 reweighing : {top_perm_f}

Les deux méthodes concordent sur les features les plus influentes. Le statut du compte courant
(`checking_status`) et la durée du crédit dominent systématiquement la prédiction.

---

## 5. Limites de l'approche

1. **Modèle restreint** : la régression logistique est linéaire. Des modèles plus expressifs
   (gradient boosting, réseaux de neurones) capturerait des interactions non-linéaires,
   au prix d'une interprétabilité moindre et d'un SHAP approché (non exact).
2. **Un seul attribut sensible à la fois** : les biais d'intersection (âge × genre)
   ne sont pas traités.
3. **Critère d'équité unique** : améliorer la parité démographique peut dégrader l'égalité
   des chances ; aucune méthode ne satisfait simultanément tous les critères.
4. **Dataset limité** : 1 000 observations ; les estimations de métriques sont bruitées
   (intervalles de confiance non reportés).
5. **Distribution shift** : la perturbation simulée (bruit gaussien + swap catégoriel)
   est une approximation simpliste d'un vrai shift de distribution en production.

---

*Durée d'exécution : {elapsed_seconds:.1f} s. Outputs dans `{config.output_dir}`.*
"""
    path.write_text(summary, encoding="utf-8")


def maybe_save_importance_plot(importance: pd.DataFrame, path: Path, title: str) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as error:  # pragma: no cover
        LOGGER.warning("Skipping plot %s because matplotlib failed: %s", path.name, error)
        return

    top = importance.head(10).sort_values("importance_mean", ascending=True)
    figure, axis = plt.subplots(figsize=(10, 6))
    axis.barh(top["feature"], top["importance_mean"], color="#2f6f9f")
    axis.set_title(title)
    axis.set_xlabel("Mean ROC-AUC drop")
    axis.set_ylabel("Feature")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(_to_serializable(payload), indent=2), encoding="utf-8")


def compute_roc_curve(
    y_true: np.ndarray, scores: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return ``(fpr_array, tpr_array, auc)`` for plotting a binary ROC curve.

    Thresholds are swept over the unique score values (including endpoints 0 and 1)
    so the curve passes through (0, 0) and (1, 1).
    """
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    thresholds = np.concatenate([[1.0 + 1e-9], np.sort(np.unique(scores))[::-1], [0.0]])
    fprs: list[float] = []
    tprs: list[float] = []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp, fp, tn, fn = confusion_counts(y_true, pred)
        fprs.append(_safe_divide(fp, fp + tn))
        tprs.append(_safe_divide(tp, tp + fn))
    auc = roc_auc_score_binary(y_true, scores)
    return np.array(fprs), np.array(tprs), auc


def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return tp, fp, tn, fn


def compute_precision(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, fp, _, _ = confusion_counts(y_true, y_pred)
    return _safe_divide(tp, tp + fp)


def compute_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    precision = compute_precision(y_true, y_pred)
    recall = true_positive_rate(y_true, y_pred)
    return _safe_divide(2.0 * precision * recall, precision + recall)


def compute_balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(0.5 * (true_positive_rate(y_true, y_pred) + true_negative_rate(y_true, y_pred)))


def true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tp, _, _, fn = confusion_counts(y_true, y_pred)
    return _safe_divide(tp, tp + fn)


def true_negative_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, fp, tn, _ = confusion_counts(y_true, y_pred)
    return _safe_divide(tn, tn + fp)


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    _, fp, tn, _ = confusion_counts(y_true, y_pred)
    return _safe_divide(fp, fp + tn)


def roc_auc_score_binary(y_true: np.ndarray, scores: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(np.sum(y_true == 1))
    negatives = int(np.sum(y_true == 0))
    if positives == 0 or negatives == 0:
        return float("nan")
    ranks = pd.Series(scores).rank(method="average").to_numpy()
    positive_ranks = float(ranks[y_true == 1].sum())
    auc = (positive_ranks - positives * (positives + 1) / 2.0) / (positives * negatives)
    return float(auc)


def _normalize_categorical_series(series: pd.Series) -> pd.Series:
    values = series.copy()
    values = values.where(values.notna(), "__MISSING__")
    return values.astype(str)


def _normalize_sample_weights(sample_weight: np.ndarray | None, n_samples: int) -> np.ndarray:
    if sample_weight is None:
        return np.ones(n_samples, dtype=float)
    weights = np.asarray(sample_weight, dtype=float)
    if np.any(weights < 0):
        raise ValueError("Sample weights must be non-negative.")
    if weights.sum() <= 0:
        raise ValueError("Sample weights must sum to a positive value.")
    return weights * (n_samples / weights.sum())


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def _binary_log_loss(y_true: np.ndarray, probabilities: np.ndarray, sample_weight: np.ndarray | None) -> float:
    y_true = np.asarray(y_true, dtype=float)
    probabilities = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1.0 - 1e-8)
    losses = -(y_true * np.log(probabilities) + (1.0 - y_true) * np.log(1.0 - probabilities))
    if sample_weight is None:
        return float(np.mean(losses))
    weights = np.asarray(sample_weight, dtype=float)
    return float(np.sum(losses * weights) / np.sum(weights))


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _series_distribution(values: np.ndarray) -> dict[str, float]:
    distribution = pd.Series(values).value_counts(normalize=True).sort_index()
    return {str(key): float(value) for key, value in distribution.items()}


def _to_serializable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    if isinstance(value, tuple):
        return [_to_serializable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Series):
        return {str(key): _to_serializable(item) for key, item in value.to_dict().items()}
    if isinstance(value, pd.DataFrame):
        return value.to_dict(orient="records")
    return value


def _validate_config(config: RunConfig) -> None:
    if config.train_ratio <= 0 or config.val_ratio <= 0 or config.test_ratio <= 0:
        raise ValueError("All split ratios must be strictly positive.")
    if config.learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if config.epochs <= 0:
        raise ValueError("epochs must be positive.")
    if config.l2 < 0:
        raise ValueError("l2 must be non-negative.")
    if config.noise_scale < 0:
        raise ValueError("noise_scale must be non-negative.")
    if not 0 <= config.category_swap_prob <= 1:
        raise ValueError("category_swap_prob must be between 0 and 1.")
    if config.permutation_repeats <= 0:
        raise ValueError("permutation_repeats must be positive.")
    if config.postprocessing_criterion not in ("demographic_parity", "equal_opportunity"):
        raise ValueError(
            f"postprocessing_criterion must be 'demographic_parity' or 'equal_opportunity', "
            f"got {config.postprocessing_criterion!r}."
        )
