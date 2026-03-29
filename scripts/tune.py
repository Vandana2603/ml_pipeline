"""
scripts/tune.py
Hyperparameter tuning — grid or random search.
All trials logged to LocalTracker (or MLflow if available).

Usage:
    python scripts/tune.py
    python scripts/tune.py --strategy random --n_trials 4
"""

import sys, copy, random, argparse, logging, itertools, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.pipeline import load_config, MLFLOW_AVAILABLE, TORCH_AVAILABLE
from data_pipeline.loader import DataLoader
from data_pipeline.preprocessor import Preprocessor

logging.basicConfig(
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S", level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_args():
    p = argparse.ArgumentParser(description="Hyperparameter Tuning")
    p.add_argument("--config", default="configs/config.yaml")
    p.add_argument("--strategy", choices=["grid", "random"], default=None)
    p.add_argument("--n_trials", type=int, default=None)
    return p.parse_args()


def build_grid(param_grid):
    keys = list(param_grid.keys())
    return [dict(zip(keys, combo)) for combo in itertools.product(*[param_grid[k] for k in keys])]


def apply_overrides(base, overrides):
    cfg = copy.deepcopy(base)
    if "learning_rate" in overrides:
        cfg["training"]["learning_rate"] = overrides["learning_rate"]
    if "batch_size" in overrides:
        cfg["training"]["batch_size"] = overrides["batch_size"]
    if "hidden_dims" in overrides:
        cfg["model"]["hidden_dims"] = overrides["hidden_dims"]
    cfg["training"]["epochs"] = min(cfg["training"]["epochs"], 10)
    cfg["training"]["distributed"] = False
    return cfg


def main():
    args = parse_args()
    base_config = load_config(args.config)

    tune_cfg = base_config.get("tuning", {})
    strategy = args.strategy or tune_cfg.get("strategy", "grid")
    n_trials  = args.n_trials or tune_cfg.get("n_trials", 6)
    param_grid = tune_cfg.get("param_grid", {
        "learning_rate": [0.05, 0.1, 0.2],
        "hidden_dims":   [[256, 128], [512, 256, 128]],
    })

    all_trials = build_grid(param_grid)
    if strategy == "random":
        random.shuffle(all_trials)
        all_trials = all_trials[:n_trials]

    logger.info(f"Tuning | strategy={strategy} | trials={len(all_trials)}")

    # Load data once
    loader = DataLoader(base_config)
    X_raw, y_raw, metadata = loader.load()
    preprocessor = Preprocessor(base_config)
    X_train, X_val, X_test, y_train, y_val, y_test, split_info = preprocessor.fit_transform(X_raw, y_raw)
    n_features = split_info["n_features"]
    n_classes  = split_info["n_classes"]
    logger.info(f"Data ready | features={n_features} classes={n_classes}")

    all_results = []
    best = {"acc": 0.0, "trial": None, "overrides": None}

    for idx, overrides in enumerate(all_trials):
        logger.info(f"\n{'='*55}\nTRIAL {idx+1}/{len(all_trials)}: {overrides}\n{'='*55}")
        trial_cfg = apply_overrides(base_config, overrides)

        if MLFLOW_AVAILABLE:
            from monitoring.tracker import ExperimentTracker
            tracker = ExperimentTracker(trial_cfg)
        else:
            from monitoring.local_tracker import LocalTracker
            tracker = LocalTracker(trial_cfg)

        run_id = tracker.start_run(
            run_name=f"tune-{idx+1}",
            tags={"tuning": "true", "trial": str(idx+1)},
        )
        tracker.log_config_as_params()
        tracker.log_params(overrides)

        try:
            t0 = time.time()

            if TORCH_AVAILABLE:
                from training.trainer import Trainer
                trainer = Trainer(trial_cfg)
            else:
                from training.sklearn_trainer import SklearnTrainer
                trainer = SklearnTrainer(trial_cfg)

            history = trainer.train(
                X_train, y_train, X_val, y_val,
                n_features=n_features, n_classes=n_classes,
                on_epoch_end=tracker.epoch_callback(),
            )

            if TORCH_AVAILABLE:
                from evaluation.evaluator import Evaluator
                ev = Evaluator(trial_cfg)
            else:
                from evaluation.sklearn_evaluator import SklearnEvaluator
                ev = SklearnEvaluator(trial_cfg)

            result = ev.evaluate(X_test, y_test, n_features, n_classes, run_id=run_id)
            metrics = result["metrics"]
            elapsed = time.time() - t0

            tracker.log_metrics({f"test/{k}": v for k, v in metrics.items() if v is not None})
            tracker.end_run("FINISHED")

            summary = {
                "trial": idx+1, "overrides": overrides,
                "test_accuracy": metrics["accuracy"],
                "test_f1": metrics["f1_weighted"],
                "roc_auc": metrics.get("roc_auc"),
                "duration_s": round(elapsed, 1), "run_id": run_id,
            }
            all_results.append(summary)
            logger.info(f"Trial {idx+1} | acc={metrics['accuracy']:.4f} | "
                        f"f1={metrics['f1_weighted']:.4f} | {elapsed:.1f}s")

            if metrics["accuracy"] > best["acc"]:
                best = {"acc": metrics["accuracy"], "trial": idx+1,
                        "overrides": overrides, "run_id": run_id}

        except Exception as e:
            logger.error(f"Trial {idx+1} failed: {e}")
            tracker.end_run("FAILED")

    print("\n" + "=" * 55)
    print("HYPERPARAMETER TUNING COMPLETE")
    print("=" * 55)
    print(f"\nAll {len(all_results)} trials (sorted by accuracy):\n")
    for r in sorted(all_results, key=lambda x: x["test_accuracy"], reverse=True):
        roc = f"{r['roc_auc']:.4f}" if r.get("roc_auc") else "N/A"
        print(f"  Trial {r['trial']:2d} | acc={r['test_accuracy']:.4f} | "
              f"f1={r['test_f1']:.4f} | roc={roc} | {r['overrides']}")
    print(f"\n🏆  Best: Trial {best['trial']}  acc={best['acc']:.4f}")
    print(f"    Config : {best['overrides']}")
    print(f"    Run ID : {best['run_id']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
