import tensorflow as tf


def hsp_mask_loss(
    pos_weight=10.0,
    tv_weight=0.05,
    sparsity_weight=0.002
):

    def loss(y_true, y_pred):

        y_true = tf.cast(y_true, tf.float32)

        bce = tf.keras.backend.binary_crossentropy(
            y_true,
            y_pred
        )

        weights = 1.0 + y_true * (
            pos_weight - 1.0
        )

        bce = tf.reduce_mean(bce * weights)

        tv = tf.reduce_mean(
            tf.square(
                y_pred[:, 1:] - y_pred[:, :-1]
            )
        )

        sparsity = tf.reduce_mean(y_pred)

        return (
            bce
            + tv_weight * tv
            + sparsity_weight * sparsity
        )

    return loss