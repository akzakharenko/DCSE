from dcse.utils.dataset_config import (
    load_dataset_config,
)

from dcse.dataset.build_dataset import (
    run_dataset_generation,
)


def main():
    cfg = load_dataset_config(
        "configs/dataset.yaml"
    )

    run_dataset_generation(cfg)


if __name__ == "__main__":
    main()