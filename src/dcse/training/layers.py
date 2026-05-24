import tensorflow as tf


class AbsDiff(tf.keras.layers.Layer):

    def call(self, inputs):

        a, b = inputs
        return tf.abs(a - b)


class OutputLayer(tf.keras.layers.Layer):

    def call(self, inputs):

        return inputs


class MaskCoverage(tf.keras.layers.Layer):

    def call(self, mask):

        return tf.reduce_mean(mask, axis=1)


class MaskedFeatures(tf.keras.layers.Layer):

    def call(self, inputs):

        features, mask = inputs

        if len(mask.shape) == 2:
            mask = tf.expand_dims(mask, -1)

        return features * mask


class CrossAttention(tf.keras.layers.Layer):

    def call(self, inputs):

        q_enc, s_enc = inputs

        scores = tf.matmul(
            q_enc,
            s_enc,
            transpose_b=True
        )

        probs = tf.nn.softmax(scores, axis=-1)

        return tf.matmul(probs, s_enc)


class MaskWeightedPooling(tf.keras.layers.Layer):

    def call(self, inputs):

        features, mask = inputs

        mask = tf.cast(mask, tf.float32)

        if len(mask.shape) == 2:
            mask = tf.expand_dims(mask, -1)

        numerator = tf.reduce_sum(
            features * mask,
            axis=1
        )

        denominator = tf.reduce_sum(
            mask,
            axis=1
        ) + 1e-6

        return numerator / denominator


class ResidualScaling(tf.keras.layers.Layer):

    def __init__(
        self,
        scale=0.1,
        **kwargs
    ):

        super().__init__(**kwargs)

        self.scale = scale

    def call(self, x):

        return self.scale * x


class StructuralModelF(tf.keras.layers.Layer):

    def build(self, input_shape):

        self.b1 = self.add_weight(
            shape=(1,),
            initializer="ones"
        )

        self.b2 = self.add_weight(
            shape=(1,),
            initializer="zeros"
        )

        self.b3 = self.add_weight(
            shape=(1,),
            initializer="zeros"
        )

        self.b4 = self.add_weight(
            shape=(1,),
            initializer="zeros"
        )

        self.b5 = self.add_weight(
            shape=(1,),
            initializer="zeros"
        )

        self.bias = self.add_weight(
            shape=(1,),
            initializer="zeros"
        )

    def call(self, inputs):

        hsp_identity, coverage = inputs

        return (
            self.b1 * hsp_identity
            + self.b2 * coverage
            + self.b3 * (hsp_identity * coverage)
            + self.b4 * tf.square(hsp_identity)
            + self.b5 * tf.square(coverage)
            + self.bias
        )