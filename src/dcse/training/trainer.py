import gc
import json
import os
import numpy as np
import tensorflow as tf
import yaml

from sklearn.model_selection import train_test_split

from dcse.model.model import build_model

from dcse.training.data import (
    build_dataset,
    compute_y_min_max,
    get_max_sequence_length,
    load_data,
)

from dcse.training.losses import hsp_mask_loss
from dcse.training.plotting import save_history_plots
from dcse.training.utils import set_seed


def require(config, key):
    if key not in config:
        raise KeyError(f"Missing required config key in training.yaml: '{key}'")
    return config[key]


def train_model(dataset_dir):

    global_config_path = os.path.abspath("configs/training.yaml")

    print(f"Loading config: {global_config_path}")

    if not os.path.exists(global_config_path):
        raise FileNotFoundError(f"Missing config file: {global_config_path}")

    with open(global_config_path) as f:
        config = yaml.safe_load(f)

    if not config:
        raise ValueError("training.yaml is empty or invalid")

    print("Loaded config:")
    print(config)

    dataset_config_path = os.path.join(dataset_dir, "training.yaml")

    if os.path.exists(dataset_config_path):

        with open(dataset_config_path) as f:
            dataset_config = yaml.safe_load(f)

        if dataset_config:
            config.update(dataset_config)

    required_keys = [
        "learning_rate",
        "batch_size",
        "epochs",
        "patience",
        "seed",
        "mask_loss_pos_weight",
        "mask_loss_tv_weight",
        "mask_loss_sparsity",
        "dense_units_hsp",
    ]

    for k in required_keys:
        if k not in config:
            raise ValueError(f"Missing required config key: {k}")

    fasta_path = os.path.join(dataset_dir, "train_sequences.fasta")
    tsv_path = os.path.join(dataset_dir, "train_labels.tsv")

    out_root = os.path.join(dataset_dir, "training_run")
    os.makedirs(out_root, exist_ok=True)

    max_seq_len = get_max_sequence_length(fasta_path)
    y_min_max = compute_y_min_max(tsv_path)

    metadata = {
        "max_sequence_length": int(max_seq_len),
        "hsp_identity": {
            "min": float(y_min_max["hsp_identity"][0]),
            "max": float(y_min_max["hsp_identity"][1]),
        },
        "global_identity": {
            "min": float(y_min_max["global_identity"][0]),
            "max": float(y_min_max["global_identity"][1]),
        }
    }

    with open(os.path.join(dataset_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    pairs = load_data(
        fasta_path,
        tsv_path,
        max_seq_len,
        y_min_max
    )

    idx = np.arange(len(pairs))

    train_idx, val_idx = train_test_split(
        idx,
        test_size=0.3,
        random_state=config["seed"]
    )

    train_pairs = [pairs[i] for i in train_idx]
    val_pairs = [pairs[i] for i in val_idx]

    tf.keras.backend.clear_session()
    gc.collect()
    set_seed(config["seed"])

    train_ds = build_dataset(
        train_pairs,
        batch_size=config["batch_size"],
        shuffle=True
    )

    val_ds = build_dataset(
        val_pairs,
        batch_size=config["batch_size"],
        shuffle=False
    )

    loss = {
        "hsp_identity": tf.keras.losses.Huber(delta=0.1),
        "global_identity": tf.keras.losses.Huber(delta=0.1),
        "qmask": hsp_mask_loss(
            config["mask_loss_pos_weight"],
            config["mask_loss_tv_weight"],
            config["mask_loss_sparsity"]
        ),
        "smask": hsp_mask_loss(
            config["mask_loss_pos_weight"],
            config["mask_loss_tv_weight"],
            config["mask_loss_sparsity"]
        ),
    }

    metrics = {
        "hsp_identity": tf.keras.metrics.MeanAbsoluteError(),
        "global_identity": tf.keras.metrics.MeanAbsoluteError(),
    }

    model = build_model(
        max_seq_len,
        config
    )

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=config["learning_rate"]
        ),
        loss=loss,
        metrics=metrics
    )

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_global_identity_mean_absolute_error",
            mode="min",
            patience=config["patience"],
            restore_best_weights=True
        )
    ]

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=config["epochs"],
        callbacks=callbacks,
        verbose=1
    )

    model.save(os.path.join(out_root, "model.keras"))

    save_history_plots(
        history,
        os.path.join(out_root, "history")
    )

    print("Training complete.")