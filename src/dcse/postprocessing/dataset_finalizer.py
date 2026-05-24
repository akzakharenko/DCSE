import os
import pandas as pd

from dcse.postprocessing.hsp_cleaning import clean_hsp_table
from dcse.postprocessing.splitting import split_train_test
from dcse.postprocessing.io import (
    load_metadata,
    save_metadata,
    load_sequences,
    write_fasta
)


def process_dataset(dataset_dir: str):
    print(f"\nProcessing {dataset_dir}...")

    labels_path = os.path.join(dataset_dir, "all_pairs_labels.tsv")
    fasta_path = os.path.join(dataset_dir, "sequences.fasta")
    meta_path = os.path.join(dataset_dir, "metadata.json")

    if not os.path.exists(labels_path):
        print("  Skipping (no labels.tsv)")
        return

    df = pd.read_csv(labels_path, sep="\t")

    df, stats = clean_hsp_table(df)

    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"  Multi-HSP pairs: {stats['multi_hsp_pairs']} ({stats['multi_hsp_percent']:.3f}%)")

    clean_path = os.path.join(dataset_dir, "clean_labels.tsv")
    df.to_csv(clean_path, sep="\t", index=False)

    records, seq_dict, seq_lengths = load_sequences(fasta_path)

    seq_to_species = {}
    for r in records:
        tokens = r.description.split()
        if len(tokens) >= 3:
            seq_to_species[r.id] = f"{tokens[1]} {tokens[2]}"
        else:
            seq_to_species[r.id] = "unknown"

    df_train, df_test = split_train_test(df, seq_lengths)

    print(f"  Train pairs: {len(df_train)}")
    print(f"  Test pairs: {len(df_test)}")

    train_path = os.path.join(dataset_dir, "train_labels.tsv")
    test_path = os.path.join(dataset_dir, "test_labels.tsv")

    df_train.to_csv(train_path, sep="\t", index=False)
    df_test.to_csv(test_path, sep="\t", index=False)

    train_ids = set(df_train["seq_id_1"]) | set(df_train["seq_id_2"])
    test_ids = set(df_test["seq_id_1"]) | set(df_test["seq_id_2"])

    train_records = [seq_dict[i] for i in train_ids if i in seq_dict]
    test_records = [seq_dict[i] for i in test_ids if i in seq_dict]

    train_fasta = os.path.join(dataset_dir, "train_sequences.fasta")
    test_fasta = os.path.join(dataset_dir, "test_sequences.fasta")

    write_fasta(train_records, train_fasta)
    write_fasta(test_records, test_fasta)

    print(f"  Train FASTA: {len(train_records)} sequences")
    print(f"  Test FASTA: {len(test_records)} sequences")

    meta = load_metadata(meta_path)

    meta.update({
        "clean_pairs": len(df),
        "train_pairs": len(df_train),
        "test_pairs": len(df_test),
        "num_train_sequences": len(train_records),
        "num_test_sequences": len(test_records),
    })

    save_metadata(meta_path, meta)

    print(f"  Updated metadata saved")
    print(f"Finished {dataset_dir}")


def run_postprocessing(dataset_root: str):
    dataset_dirs = sorted(
        d for d in os.listdir(dataset_root)
        if d.startswith("dataset_")
    )

    for d in dataset_dirs:
        process_dataset(os.path.join(dataset_root, d))

    print("\nAll datasets processed successfully.")