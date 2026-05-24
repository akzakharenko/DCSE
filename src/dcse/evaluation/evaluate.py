import json
import os

import numpy as np
import tensorflow as tf

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

from dcse.model.model import (
    AbsDiff,
    CrossAttention,
    MaskCoverage,
    MaskedFeatures,
    MaskWeightedPooling,
    OutputLayer,
    ResidualScaling,
    StructuralModelF
)

from dcse.training.data import load_test_data
from dcse.evaluation.plotting import scatter_plot

def mask_iou(true_mask, pred_mask, threshold=0.5):

    pred_mask_bin = (
        pred_mask > threshold
    ).astype(np.float32)

    intersection = np.sum(
        true_mask * pred_mask_bin,
        axis=1
    )

    union = np.sum(
        np.maximum(
            true_mask,
            pred_mask_bin
        ),
        axis=1
    ) + 1e-7

    return np.mean(
        intersection / union
    )


def evaluate_model(dataset_dir):

    print("\n=== EVALUATION ===")

    metadata_path = os.path.join(
        dataset_dir,
        "metadata.json"
    )

    model_path = os.path.join(
        dataset_dir,
        "training_run",
        "model.keras"
    )

    test_fasta = os.path.join(
        dataset_dir,
        "test_sequences.fasta"
    )

    test_tsv = os.path.join(
        dataset_dir,
        "test_labels.tsv"
    )

    output_dir = os.path.join(
        dataset_dir,
        "evaluation"
    )

    os.makedirs(output_dir, exist_ok=True)

    with open(metadata_path) as f:

        metadata = json.load(f)

    max_len = metadata[
        "max_sequence_length"
    ]

    hmin = metadata[
        "hsp_identity"
    ]["min"]

    hmax = metadata[
        "hsp_identity"
    ]["max"]

    gmin = metadata[
        "global_identity"
    ]["min"]

    gmax = metadata[
        "global_identity"
    ]["max"]


    (
        q_test,
        s_test,
        qmask_true,
        smask_true,
        hsp_true,
        glob_true,
    ) = load_test_data(
        test_fasta,
        test_tsv,
        max_len,
        hmin,
        hmax,
        gmin,
        gmax,
    )

    tf.keras.backend.clear_session()

    model = tf.keras.models.load_model(
        model_path,
        custom_objects={
            "AbsDiff": AbsDiff,
            "MaskedFeatures": MaskedFeatures,
            "OutputLayer": OutputLayer,
            "CrossAttention": CrossAttention,
            "MaskWeightedPooling": MaskWeightedPooling,
            "ResidualScaling": ResidualScaling,
            "MaskCoverage": MaskCoverage,
            "StructuralModelF": StructuralModelF,
        },
        compile=False
    )


    preds = model.predict(
        {
            "q_seq": q_test,
            "s_seq": s_test
        },
        batch_size=32,
        verbose=1
    )

    hsp_pred = preds[
        "hsp_identity"
    ].squeeze()

    glob_pred = preds[
        "global_identity"
    ].squeeze()

    qmask_pred = preds[
        "qmask"
    ].squeeze()

    smask_pred = preds[
        "smask"
    ].squeeze()


    hsp_true_orig = (
        hsp_true * (hmax - hmin)
    ) + hmin

    hsp_pred_orig = (
        hsp_pred * (hmax - hmin)
    ) + hmin

    glob_true_orig = (
        glob_true * (gmax - gmin)
    ) + gmin

    glob_pred_orig = (
        glob_pred * (gmax - gmin)
    ) + gmin

    metrics = {

        "iou_q": float(
            mask_iou(
                qmask_true,
                qmask_pred
            )
        ),

        "iou_s": float(
            mask_iou(
                smask_true,
                smask_pred
            )
        ),

        "mae_hsp": float(
            mean_absolute_error(
                hsp_true_orig,
                hsp_pred_orig
            )
        ),

        "rmse_hsp": float(
            np.sqrt(
                mean_squared_error(
                    hsp_true_orig,
                    hsp_pred_orig
                )
            )
        ),

        "mae_global": float(
            mean_absolute_error(
                glob_true_orig,
                glob_pred_orig
            )
        ),

        "rmse_global": float(
            np.sqrt(
                mean_squared_error(
                    glob_true_orig,
                    glob_pred_orig
                )
            )
        ),

        "pearson_r": float(
            np.corrcoef(
                glob_true_orig,
                glob_pred_orig
            )[0, 1]
        ),

        "r2": float(
            r2_score(
                glob_true_orig,
                glob_pred_orig
            )
        ),
    }

    metrics_path = os.path.join(
        output_dir,
        "metrics.json"
    )

    with open(metrics_path, "w") as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    scatter_plot(
        hsp_true_orig,
        hsp_pred_orig,
        "HSP Identity",
        os.path.join(
            output_dir,
            "scatter_hsp_identity.png"
        )
    )

    scatter_plot(
        glob_true_orig,
        glob_pred_orig,
        "Global Identity",
        os.path.join(
            output_dir,
            "scatter_global_identity.png"
        )
    )

    print("\n=== RESULTS ===")

    for k, v in metrics.items():

        print(f"{k}: {v:.4f}")

    print("\nSaved evaluation to:")
    print(output_dir)