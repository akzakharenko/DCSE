import tensorflow as tf

from dcse.training.layers import (
    AbsDiff,
    CrossAttention,
    MaskCoverage,
    MaskedFeatures,
    MaskWeightedPooling,
    OutputLayer,
    ResidualScaling,
    StructuralModelF,
)


def build_positional_encoder(seq_len, vocab_size=4, filters=32, depth=5, use_residuals=True, use_dilation=True):
    inp = tf.keras.Input(shape=(seq_len, vocab_size))
    x = tf.keras.layers.Conv1D(filters, 7, padding="same", activation="relu")(inp)

    for i in range(depth):
        res = x
        dilation = 2 ** i if use_dilation else 1
        x = tf.keras.layers.SeparableConv1D(filters, 3, padding="same", dilation_rate=dilation, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)

        if use_residuals:
            x = tf.keras.layers.Add()([x, res])

    pooled = tf.keras.layers.GlobalMaxPooling1D()(x)
    return tf.keras.Model(inp, [x, pooled], name="positional_encoder")


def build_identity_encoder(seq_len, vocab_size=4, filters=64):
    inp = tf.keras.Input(shape=(seq_len, vocab_size))
    x = tf.keras.layers.Conv1D(filters, 7, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv1D(filters, 5, padding="same", activation="relu")(x)
    return tf.keras.Model(inp, x, name="identity_encoder")


def build_positional_decoder(seq_len, channels, hidden_1=64, hidden_2=32):
    inp = tf.keras.Input((seq_len, channels))
    x = tf.keras.layers.Conv1D(hidden_1, 3, padding="same", activation="relu")(inp)
    x = tf.keras.layers.Conv1D(hidden_2, 3, padding="same", activation="relu")(x)
    mask = tf.keras.layers.Conv1D(1, 1, activation="sigmoid", dtype="float32", name="mask")(x)
    return tf.keras.Model(inp, mask, name="mask_decoder")



def build_model(seq_len, config, vocab_size=4):
    pos_filters = config.get("pos_filters", 32)
    pos_depth = config.get("pos_depth", 5)
    identity_filters = config.get("filters_identity", 64)

    dense_units = config.get("dense_units_hsp", (128, 64))
    residual_scale = config.get("residual_scale", 0.05)

    decoder_hidden = config.get("decoder_hidden", (64, 32))

    q_in = tf.keras.Input((seq_len, vocab_size), name="q_seq")
    s_in = tf.keras.Input((seq_len, vocab_size), name="s_seq")

    positional_encoder = build_positional_encoder(
        seq_len,
        vocab_size=vocab_size,
        filters=pos_filters,
        depth=pos_depth,
        use_residuals=True,
        use_dilation=True
    )

    q_seq, _ = positional_encoder(q_in)
    s_seq, _ = positional_encoder(s_in)

    attn_qs = CrossAttention()([q_seq, s_seq])
    attn_sq = CrossAttention()([s_seq, q_seq])

    decoder = build_positional_decoder(seq_len, q_seq.shape[-1] * 4, hidden_1=decoder_hidden[0], hidden_2=decoder_hidden[1])

    def decode(seq, attn):
        x = tf.keras.layers.Concatenate()([
            seq,
            attn,
            AbsDiff()([seq, attn]),
            tf.keras.layers.Multiply()([seq, attn])
        ])
        return decoder(x)

    qmask = decode(q_seq, attn_qs)
    smask = decode(s_seq, attn_sq)


    qmask = OutputLayer(name="qmask")(qmask)
    smask = OutputLayer(name="smask")(smask)

    identity_encoder_model = build_identity_encoder(seq_len, vocab_size, filters=identity_filters)

    q_features = identity_encoder_model(q_in)
    s_features = identity_encoder_model(s_in)

    q_features = MaskedFeatures()([q_features, qmask])
    s_features = MaskedFeatures()([s_features, smask])

    q_hsp = MaskWeightedPooling()([q_features, qmask])
    s_hsp = MaskWeightedPooling()([s_features, smask])

    x_hsp = AbsDiff()([q_hsp, s_hsp])
    x_hsp = tf.keras.layers.Dense(dense_units[0], activation="relu")(x_hsp)
    x_hsp = tf.keras.layers.Dense(dense_units[1], activation="relu")(x_hsp)

    hsp_identity = tf.keras.layers.Dense(1, name="hsp_identity")(x_hsp)

    q_cov = MaskCoverage()(qmask)
    s_cov = MaskCoverage()(smask)
    mean_cov = (q_cov + s_cov) / 2.0

    base = StructuralModelF()([hsp_identity, mean_cov])

    residual = tf.keras.layers.Dense(32, activation="relu")(x_hsp)
    residual = tf.keras.layers.Dense(1)(residual)
    residual = ResidualScaling(scale=residual_scale)(residual)

    global_identity = tf.keras.layers.Add(name="global_identity")([base, residual])

    return tf.keras.Model(
        inputs=[q_in, s_in],
        outputs={
            "qmask": qmask,
            "smask": smask,
            "hsp_identity": hsp_identity,
            "global_identity": global_identity
        }
    )

