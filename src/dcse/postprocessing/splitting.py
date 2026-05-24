import pandas as pd


def split_train_test(df, seq_lengths, test_percent=0.5, n_bins=20):

    df = df.copy()

    df["pair"] = df.apply(
        lambda x: tuple(sorted([x["seq_id_1"], x["seq_id_2"]])),
        axis=1
    )

    df["len_1"] = df["seq_id_1"].map(seq_lengths)
    df["len_2"] = df["seq_id_2"].map(seq_lengths)
    df["pair_max_len"] = df[["len_1", "len_2"]].max(axis=1)
    df.drop(columns=["len_1", "len_2"], inplace=True)

    df["hsp_bin"] = pd.qcut(df["hsp_identity"], n_bins, labels=False, duplicates="drop")
    df["global_bin"] = pd.qcut(df["global_identity"], n_bins, labels=False, duplicates="drop")
    df["qstart_bin"] = pd.qcut(df["qstart"], n_bins, labels=False, duplicates="drop")
    df["sstart_bin"] = pd.qcut(df["sstart"], n_bins, labels=False, duplicates="drop")

    pairs = df[[
        "pair",
        "pair_max_len",
        "hsp_bin",
        "global_bin",
        "qstart_bin",
        "sstart_bin"
    ]].drop_duplicates()

    pairs = pairs.sort_values("pair_max_len", ascending=False)

    num_test_target = int(len(pairs) * test_percent)

    train_pairs = set()
    test_pairs = set()

    train_max_len = pairs.iloc[0]["pair_max_len"]

    first = pairs.iloc[0]
    train_pairs.add(first["pair"])

    remaining = pairs.iloc[1:].sample(frac=1)

    for _, row in remaining.iterrows():
        p = row["pair"]
        plen = row["pair_max_len"]

        if plen > train_max_len:
            train_pairs.add(p)
            train_max_len = plen
            continue

        if len(test_pairs) < num_test_target:
            test_pairs.add(p)
        else:
            train_pairs.add(p)
            train_max_len = max(train_max_len, plen)

    df_train = df[df["pair"].isin(train_pairs)].copy()
    df_test = df[df["pair"].isin(test_pairs)].copy()

    df_train.drop(columns=["hsp_bin", "global_bin", "qstart_bin", "sstart_bin"], inplace=True)
    df_test.drop(columns=["hsp_bin", "global_bin", "qstart_bin", "sstart_bin"], inplace=True)

    return df_train, df_test