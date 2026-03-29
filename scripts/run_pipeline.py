"""
scripts/run_pipeline.py
Entry point for running the full pipeline or individual stages.

Usage:
    python scripts/run_pipeline.py                    # full pipeline
    python scripts/run_pipeline.py --stage data       # data only
    python scripts/run_pipeline.py --stage train      # train only
    python scripts/run_pipeline.py --stage evaluate   # evaluate only
    python scripts/run_pipeline.py --config custom.yaml
"""

import sys
import argparse
import logging
from pathlib import Path

# Make sure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from orchestration.pipeline import Pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="ML Training Pipeline")
    parser.add_argument(
        "--stage",
        choices=["data", "train", "evaluate", "all"],
        default="all",
        help="Which stage to run (default: all)",
    )
    parser.add_argument(
        "--config",
        default="configs/config.yaml",
        help="Path to config YAML (default: configs/config.yaml)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    pipeline = Pipeline(config_path=args.config)

    if args.stage == "all":
        result = pipeline.run_all()
        metrics = result["metrics"]
        print("\n" + "=" * 60)
        print("✅  PIPELINE COMPLETE")
        print("=" * 60)
        print(f"  Accuracy  : {metrics['accuracy']:.4f}")
        print(f"  F1 Score  : {metrics['f1_weighted']:.4f}")
        print(f"  Precision : {metrics['precision_weighted']:.4f}")
        print(f"  Recall    : {metrics['recall_weighted']:.4f}")
        if metrics.get("roc_auc"):
            print(f"  ROC-AUC   : {metrics['roc_auc']:.4f}")
        print("=" * 60)

    elif args.stage == "data":
        version_id = pipeline.run_data_stage()
        print(f"\n✅  Data stage complete. Version ID: {version_id}")

    elif args.stage == "train":
        pipeline.run_data_stage()
        run_id = pipeline.run_train_stage()
        print(f"\n✅  Training complete. MLflow Run ID: {run_id}")

    elif args.stage == "evaluate":
        pipeline.run_data_stage()
        pipeline.run_train_stage()
        result = pipeline.run_evaluate_stage()
        metrics = result["metrics"]
        print(f"\n✅  Evaluation complete.")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  F1       : {metrics['f1_weighted']:.4f}")


if __name__ == "__main__":
    main()
