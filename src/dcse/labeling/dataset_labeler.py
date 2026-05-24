import json
import csv
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm
from Bio import SeqIO

from dcse.labeling.global_identity import global_identity
from dcse.labeling.blast import run_blast
from dcse.pairing.graph_pairs import generate_pairs


def _init_worker(rec_dict, seq_dict):
    global RECS, SEQS
    RECS = rec_dict
    SEQS = seq_dict


def process_pair(pair):
    id1, id2, g_id = pair

    hsps = run_blast(RECS[id1], RECS[id2])

    rows = []
    for h in hsps:
        rows.append([
            id1,
            id2,
            g_id,
            h["qlen"],
            h["slen"],
            h["qstart"],
            h["qend"],
            h["sstart"],
            h["send"],
            h["hsp_identity"],
        ])

    return rows


def label_datasets(dataset_root: str):
    dataset_root = Path(dataset_root)

    dataset_dirs = sorted([
        d for d in dataset_root.iterdir()
        if d.is_dir() and d.name.startswith("dataset_")
    ])

    for dset in dataset_dirs:
        print(f"\nProcessing {dset}...")

        dataset_dir = dset

        fasta_path = dataset_dir / "sequences.fasta"
        metadata_path = dataset_dir / "metadata.json"

        if not metadata_path.exists():
            print(f"Skipping {dataset_dir} (no metadata.json)")
            continue

        if not fasta_path.exists():
            print(f"Skipping {dataset_dir} (no sequences.fasta)")
            continue

        with open(metadata_path) as f:
            metadata = json.load(f)

        params = metadata["sampling_parameters"]
        K = params["K"]
        seed = params["random_seed"]

        print(f"K from metadata: {K}")
        print(f"Seed from metadata: {seed}")

        records = list(SeqIO.parse(str(fasta_path), "fasta"))

        seq_dict = {r.id: str(r.seq) for r in records}
        id_to_rec = {r.id: r for r in records}

        raw_pairs = generate_pairs(records, K, seed)

        pairs = []
        for id1, id2 in tqdm(raw_pairs, desc="Computing global identity"):
            gid = global_identity(seq_dict[id1], seq_dict[id2])
            pairs.append((id1, id2, gid))

        out_path = dataset_dir / "all_pairs_labels.tsv"

        with Pool(
            cpu_count(),
            initializer=_init_worker,
            initargs=(id_to_rec, seq_dict),
        ) as pool, open(out_path, "w", newline="") as f:

            writer = csv.writer(f, delimiter="\t")

            writer.writerow([
                "seq_id_1",
                "seq_id_2",
                "global_identity",
                "qlen",
                "slen",
                "qstart",
                "qend",
                "sstart",
                "send",
                "hsp_identity",
            ])

            for rows in tqdm(
                pool.imap(process_pair, pairs, chunksize=8),
                total=len(pairs),
                desc="Computing HSPs",
            ):
                for row in rows:
                    writer.writerow(row)

    print("\nAll datasets processed successfully.")