"""Command-line tool with three commands: train, evaluate and predict."""

import argparse
import sys
from pathlib import Path

from student_risk.config import DEFAULT_CONFIG_PATH, Config, load_config
from student_risk.data import DatasetLoader
from student_risk.evaluate import final_scores
from student_risk.models import Trainer, baseline_scores
from student_risk.predict import predict_new_case


def run_train(config: Config) -> None:
    """Fit on the training split, save the reports, the model and its threshold.

    Args:
        config: Project settings.
    """
    X_train, _X_test, y_train, _y_test = DatasetLoader(config).load_and_split()
    config.reports_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training set: {len(y_train)} students, fail rate = {y_train.mean():.3f}")

    baseline = baseline_scores(X_train, y_train, config)
    baseline.to_csv(config.reports_dir / "baseline.csv")
    print(f"\nBaselines ({config.cv_splits}-fold stratified CV):\n{baseline.to_string()}")

    trainer = Trainer(config).fit(X_train, y_train)
    trainer.results.to_csv(config.reports_dir / "model_comparison.csv")
    scores = trainer.results.drop(columns="best_params").astype(float).round(3)
    print(f"\nModel comparison (GridSearchCV, same folds):\n{scores.to_string()}")
    print(f"\nBest parameters:\n{trainer.results['best_params'].to_string()}")
    print(f"\nChosen model: {trainer.best_name}. {trainer.selection_reason}")

    comparison = trainer.threshold_comparison(y_train)
    comparison.to_csv(config.reports_dir / "threshold_report.csv")
    print(f"\nOut-of-fold results of {trainer.best_name}:\n{comparison.round(3).to_string()}")

    tradeoff = trainer.threshold_tradeoff(y_train)
    tradeoff.to_csv(config.reports_dir / "threshold_tradeoff.csv", index=False)
    print(f"\nRecall / precision trade-off (out-of-fold):\n{tradeoff.round(3).to_string()}")

    trainer.save(config.model_path)
    print(f"\nSaved the model and its threshold to {config.model_path}")


def run_evaluate(config: Config) -> None:
    """Score the saved model once on the test set and print the metrics.

    Args:
        config: Project settings.
    """
    trainer = Trainer.load(config.model_path, config)
    _X_train, X_test, _y_train, y_test = DatasetLoader(config).load_and_split()
    proba = trainer.best_model.predict_proba(X_test)[:, 1]
    scores = final_scores(y_test, proba, trainer.threshold)

    print(f"Test set: {len(y_test)} students, model {trainer.best_name}")
    print(f"ROC-AUC            {scores['roc_auc']:.3f}")
    print(f"Average precision  {scores['average_precision']:.3f}")
    print(f"At threshold {scores['threshold']:.3f}:")
    print(f"  Precision        {scores['precision']:.3f}")
    print(f"  Recall           {scores['recall']:.3f}")
    print(f"  F1               {scores['f1']:.3f}")
    print(f"  Students flagged {scores['flagged_share']:.1%}")
    print("Confusion matrix (rows = truth, columns = decision):")
    print("                  pred pass  pred fail")
    print(f"  true pass       {scores['tn']:9d}  {scores['fp']:9d}")
    print(f"  true fail       {scores['fn']:9d}  {scores['tp']:9d}")


def run_predict(config: Config, input_path: Path) -> None:
    """Print the probability of failing and the decision for one JSON case.

    Args:
        config: Project settings.
        input_path: JSON file describing one student.
    """
    result = predict_new_case(config, input_path)
    decision = "AT RISK - flag for tutoring" if result["at_risk"] else "not at risk"
    print(f"Probability of failing: {result['probability_fail']:.3f}")
    print(f"Decision (threshold {result['threshold']:.3f}): {decision}")


def build_parser() -> argparse.ArgumentParser:
    """Describe the command-line arguments.

    Returns:
        The parser for: [--config FILE] {train, evaluate, predict --input FILE}.
    """
    parser = argparse.ArgumentParser(
        prog="student-risk",
        description="Early-warning tool: flag students at risk of failing (final grade < 10).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="path to the YAML settings file (default: config.yaml of the project)",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("train", help="compare models on the training set and save the best one")
    commands.add_parser("evaluate", help="score the saved model once on the test set")
    predict = commands.add_parser("predict", help="predict one new student from a JSON file")
    predict.add_argument("--input", type=Path, required=True, help="JSON file with one student")
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run one command and return an exit code (0 = success, 1 = error).

    Args:
        argv: Command-line arguments without the program name. None means sys.argv.

    Returns:
        0 on success, 1 if a file is missing or an input is invalid.
    """
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    try:
        if args.command == "train":
            run_train(config)
        elif args.command == "evaluate":
            run_evaluate(config)
        else:
            run_predict(config, args.input)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
