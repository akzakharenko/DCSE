import pandas as pd

def clean_hsp_table(df: pd.DataFrame):
    df = df.copy()

    df["pair"] = df.apply(
        lambda x: tuple(sorted([x["seq_id_1"], x["seq_id_2"]])),
        axis=1
    )

    df = df.drop_duplicates(
        subset=[
            "pair",
            "qstart", "qend",
            "sstart", "send",
            "hsp_identity",
            "global_identity"
        ]
    )

    hsp_counts = df.groupby("pair").size()
    total_pairs = len(hsp_counts)
    multi_hsp_pairs = (hsp_counts > 1).sum()

    stats = {
        "total_pairs": total_pairs,
        "multi_hsp_pairs": int(multi_hsp_pairs),
        "multi_hsp_percent": float(100 * multi_hsp_pairs / total_pairs) if total_pairs else 0.0,
        "hsp_count_distribution": hsp_counts.value_counts().sort_index().to_dict()
    }

    valid_pairs = set(hsp_counts[hsp_counts == 1].index)
    df = df[df["pair"].isin(valid_pairs)].copy()

    df = df.drop(columns=["pair"])

    return df, stats