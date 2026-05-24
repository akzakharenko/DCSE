import argparse

from dcse.labeling.dataset_labeler import label_datasets


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-root",
        default="outputs/datasets",
        help="Root folder containing dataset_XX folders",
    )

    args = parser.parse_args()

    print("\n=== LABELING MODE ===")
    print(f"Dataset root: {args.dataset_root}")

    label_datasets(args.dataset_root)


if __name__ == "__main__":
    main()