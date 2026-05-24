import argparse
import os

from dcse.utils.dataset_config import load_dataset_config

from dcse.dataset.build_dataset import run_dataset_generation
from dcse.labeling.dataset_labeler import label_datasets
from dcse.postprocessing.dataset_finalizer import run_postprocessing

from dcse.training.trainer import train_model

from dcse.evaluation.evaluate import evaluate_model


def run_sequences(cfg):

    print("\n=== SEQUENCES MODE ===")

    run_dataset_generation(cfg)

    label_datasets(cfg.root_dir)

    run_postprocessing(cfg.root_dir)


def run_training(dataset_name):

    print("\n=== TRAIN MODE ===")

    dataset_dir = os.path.join(
        "outputs",
        "dataset",
        dataset_name
    )

    if not os.path.exists(dataset_dir):

        raise FileNotFoundError(
            f"Dataset not found: {dataset_dir}"
        )

    train_model(dataset_dir)


def run_evaluation(dataset_name):

    print("\n=== EVALUATION MODE ===")

    dataset_dir = os.path.join(
        "outputs",
        "dataset",
        dataset_name
    )

    if not os.path.exists(dataset_dir):

        raise FileNotFoundError(
            f"Dataset not found: {dataset_dir}"
        )

    evaluate_model(dataset_dir)


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mode",
        choices=[
            "sequences",
            "train",
            "evaluate"
        ],
        required=True
    )

    parser.add_argument(
        "--config",
        default="configs/dataset.yaml"
    )

    parser.add_argument(
        "--dataset",
        help="Dataset name for training/evaluation"
    )

    args = parser.parse_args()


    if args.mode == "sequences":

        cfg = load_dataset_config(
            args.config
        )

        run_sequences(cfg)

        return

    if args.mode == "train":

        if not args.dataset:

            raise ValueError(
                "--dataset is required for train mode"
            )

        run_training(args.dataset)

        return


    if args.mode == "evaluate":

        if not args.dataset:

            raise ValueError(
                "--dataset is required for evaluation mode"
            )

        run_evaluation(args.dataset)

        return


if __name__ == "__main__":

    main()