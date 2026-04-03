from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from german_credit_rai.pipeline import RunConfig, run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="german_credit_rai",
        description=(
            "Responsible AI pipeline for the UCI German Credit dataset.\n\n"
            "Implements a logistic regression baseline, two fairness methods "
            "(reweighing pre-processing and per-group threshold post-processing), "
            "exact linear SHAP interpretability, permutation importance, and "
            "controlled robustness evaluation."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ── run ──────────────────────────────────────────────────────────────────
    run_parser = subparsers.add_parser(
        "run",
        help="Run the full experiment pipeline and write all outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    run_parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("data/raw/german.data"),
        help="Path to the UCI German Credit raw data file.",
    )
    run_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/german_credit_run"),
        help="Directory where metrics, logs, plots, and reports are written.",
    )
    run_parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download the UCI dataset when --data-path does not exist.",
    )
    run_parser.add_argument(
        "--sensitive-attribute",
        choices=("gender", "age"),
        default="gender",
        help="Sensitive attribute used for fairness evaluation.",
    )
    run_parser.add_argument(
        "--age-threshold",
        type=int,
        default=25,
        help="Age split threshold when --sensitive-attribute=age (younger < threshold ≤ older).",
    )
    run_parser.add_argument("--random-seed", type=int, default=42, help="Global random seed.")
    run_parser.add_argument("--train-ratio", type=float, default=0.6, help="Training split ratio.")
    run_parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    run_parser.add_argument("--test-ratio", type=float, default=0.2, help="Test split ratio.")
    run_parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.03,
        help="Adam learning rate for the logistic regression.",
    )
    run_parser.add_argument(
        "--epochs",
        type=int,
        default=3500,
        help="Maximum optimisation epochs (early stopping with patience=300).",
    )
    run_parser.add_argument(
        "--l2",
        type=float,
        default=0.01,
        help="L2 regularisation strength.",
    )
    run_parser.add_argument(
        "--noise-scale",
        type=float,
        default=0.2,
        help="Gaussian noise scale (×feature std) applied to numeric features during robustness evaluation.",
    )
    run_parser.add_argument(
        "--category-swap-prob",
        type=float,
        default=0.1,
        help="Probability of randomly replacing a categorical value during robustness evaluation.",
    )
    run_parser.add_argument(
        "--permutation-repeats",
        type=int,
        default=10,
        help="Number of shuffles per feature for permutation importance.",
    )
    run_parser.add_argument(
        "--postprocessing-criterion",
        choices=("demographic_parity", "equal_opportunity"),
        default="demographic_parity",
        help=(
            "Fairness criterion used for per-group threshold calibration (post-processing). "
            "'demographic_parity' equalises selection rates; 'equal_opportunity' equalises TPR."
        ),
    )
    run_parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
        help="Logging verbosity.",
    )

    # ── show ─────────────────────────────────────────────────────────────────
    show_parser = subparsers.add_parser(
        "show",
        help="Display a formatted summary from a previous run's metrics.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    show_parser.add_argument(
        "metrics_path",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Path to metrics.json produced by a previous `run`. "
            "Defaults to the latest timestamped output directory if omitted."
        ),
    )
    show_parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/german_credit_run"),
        help="Output directory to look for metrics.json when metrics_path is not given.",
    )

    return parser


def configure_logging(level: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(getattr(logging, level))
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    file_handler = logging.FileHandler(output_dir / "run.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)


def _show_metrics(metrics_path: Path) -> None:
    if not metrics_path.exists():
        print(f"[ERROR] metrics.json not found at {metrics_path}", file=sys.stderr)
        sys.exit(1)

    with metrics_path.open(encoding="utf-8") as fh:
        data = json.load(fh)

    cfg = data.get("config", {})
    ds = data.get("dataset", {})
    scenarios = data.get("scenarios", {})
    clean = scenarios.get("clean_test", {})
    perturbed = scenarios.get("perturbed_test", {})

    def _fmt_model(name: str, payload: dict) -> str:
        p = payload.get("performance", {})
        f = payload.get("fairness", {})
        return (
            f"  {name:<20s}  AUC={p.get('roc_auc', float('nan')):.4f}  "
            f"BalAcc={p.get('balanced_accuracy', float('nan')):.4f}  "
            f"|dDP|={f.get('abs_demographic_parity_difference', float('nan')):.4f}  "
            f"|dEO|={f.get('abs_equal_opportunity_difference', float('nan')):.4f}"
        )

    lines = [
        "=" * 70,
        "German Credit RAI — Run summary",
        "=" * 70,
        f"  Sensitive attribute : {cfg.get('sensitive_attribute', '?')}",
        f"  Dataset rows        : {ds.get('rows', '?')}",
        f"  Privileged group    : {ds.get('privileged_group', '?')}",
        f"  Split               : train={cfg.get('train_ratio', '?')} "
        f"val={cfg.get('val_ratio', '?')} test={cfg.get('test_ratio', '?')}",
        "",
        "Clean test results:",
    ]
    for model_name, payload in clean.items():
        lines.append(_fmt_model(model_name, payload))
    lines += ["", "Perturbed test results:"]
    for model_name, payload in perturbed.items():
        lines.append(_fmt_model(model_name, payload))
    lines.append("=" * 70)
    print("\n".join(lines))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "run":
        configure_logging(args.log_level, args.output_dir)
        config = RunConfig(
            data_path=args.data_path,
            output_dir=args.output_dir,
            download_if_missing=args.download_if_missing,
            sensitive_attribute=args.sensitive_attribute,
            age_threshold=args.age_threshold,
            random_seed=args.random_seed,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            learning_rate=args.learning_rate,
            epochs=args.epochs,
            l2=args.l2,
            noise_scale=args.noise_scale,
            category_swap_prob=args.category_swap_prob,
            permutation_repeats=args.permutation_repeats,
            postprocessing_criterion=args.postprocessing_criterion,
        )
        run_pipeline(config)

    elif args.command == "show":
        metrics_path = args.metrics_path or (args.output_dir / "metrics.json")
        _show_metrics(metrics_path)
