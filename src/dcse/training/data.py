import csv
from collections import defaultdict

import numpy as np
import tensorflow as tf

from Bio import SeqIO
from tqdm import tqdm


def one_hot_encode(seq, max_len):

    mapping = {
        "A": 0,
        "C": 1,
        "G": 2,
        "T": 3
    }

    arr = np.zeros((max_len, 4), dtype=np.float32)

    for i, base in enumerate(seq[:max_len]):

        if base in mapping:
            arr[i, mapping[base]] = 1.0

    return arr


def get_max_sequence_length(fasta_file):

    return max(
        len(str(rec.seq))
        for rec in SeqIO.parse(fasta_file, "fasta")
    )


def compute_y_min_max(pairs_file):

    mins = defaultdict(lambda: float("inf"))
    maxs = defaultdict(lambda: float("-inf"))

    with open(pairs_file) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in reader:

            for key in ["hsp_identity", "global_identity"]:

                val_str = row[key].strip()

                if not val_str:
                    continue

                val = float(val_str)

                mins[key] = min(mins[key], val)
                maxs[key] = max(maxs[key], val)

    return {
        "hsp_identity": (
            mins["hsp_identity"],
            maxs["hsp_identity"]
        ),
        "global_identity": (
            mins["global_identity"],
            maxs["global_identity"]
        ),
    }


def load_data(
    fasta_file,
    pairs_file,
    max_len,
    y_min_max
):

    sequences = {
        r.id: str(r.seq)
        for r in SeqIO.parse(fasta_file, "fasta")
    }

    hmin, hmax = y_min_max["hsp_identity"]
    gmin, gmax = y_min_max["global_identity"]

    data = []

    with open(pairs_file) as f:

        reader = csv.DictReader(f, delimiter="\t")

        for row in tqdm(reader):

            qid = row["seq_id_1"]
            sid = row["seq_id_2"]

            if qid not in sequences:
                continue

            if sid not in sequences:
                continue

            if not row["hsp_identity"].strip():
                continue

            hsp_val = float(row["hsp_identity"])
            glob_val = float(row["global_identity"])

            hsp_identity = np.float32(
                (hsp_val - hmin) / (hmax - hmin)
            )

            global_identity = np.float32(
                (glob_val - gmin) / (gmax - gmin)
            )

            q = one_hot_encode(
                sequences[qid],
                max_len
            )

            s = one_hot_encode(
                sequences[sid],
                max_len
            )

            qmask = np.zeros(max_len, dtype=np.float32)
            smask = np.zeros(max_len, dtype=np.float32)

            qstart = int(row["qstart"])
            qend = int(row["qend"])

            sstart = int(row["sstart"])
            send = int(row["send"])

            qmask[qstart:qend + 1] = 1.0
            smask[sstart:send + 1] = 1.0

            data.append(
                (
                    q,
                    s,
                    qmask,
                    smask,
                    hsp_identity,
                    global_identity
                )
            )

    return data


def build_dataset(
    pairs,
    batch_size=32,
    shuffle=True
):

    q, s, qm, sm, hsp_id, glob_id = zip(*pairs)

    ds = tf.data.Dataset.from_tensor_slices(
        (
            {
                "q_seq": np.array(q),
                "s_seq": np.array(s)
            },
            {
                "qmask": np.array(qm),
                "smask": np.array(sm),
                "hsp_identity": np.array(hsp_id),
                "global_identity": np.array(glob_id),
            }
        )
    )

    if shuffle:

        ds = ds.shuffle(
            len(pairs),
            reshuffle_each_iteration=True
        )

    return ds.batch(batch_size).prefetch(
        tf.data.AUTOTUNE
    )




def load_test_data(
    fasta_file,
    pairs_file,
    max_len,
    hmin,
    hmax,
    gmin,
    gmax
):

    sequences = {
        r.id: str(r.seq)
        for r in SeqIO.parse(fasta_file, "fasta")
    }

    q_seqs = []
    s_seqs = []

    qmask_true = []
    smask_true = []

    hsp_true = []
    glob_true = []

    with open(pairs_file) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t"
        )

        for row in tqdm(reader):

            qid = row["seq_id_1"]
            sid = row["seq_id_2"]

            if qid not in sequences:
                continue

            if sid not in sequences:
                continue

            if not row["hsp_identity"].strip():
                continue

            q = one_hot_encode(
                sequences[qid],
                max_len
            )

            s = one_hot_encode(
                sequences[sid],
                max_len
            )

            qmask = np.zeros(
                max_len,
                dtype=np.float32
            )

            smask = np.zeros(
                max_len,
                dtype=np.float32
            )

            qstart = int(row["qstart"])
            qend = int(row["qend"])

            sstart = int(row["sstart"])
            send = int(row["send"])

            qmask[qstart:qend + 1] = 1.0
            smask[sstart:send + 1] = 1.0

            hsp_val = float(
                row["hsp_identity"]
            )

            glob_val = float(
                row["global_identity"]
            )

            hsp_norm = (
                (hsp_val - hmin)
                / (hmax - hmin)
            )

            glob_norm = (
                (glob_val - gmin)
                / (gmax - gmin)
            )

            q_seqs.append(q)
            s_seqs.append(s)

            qmask_true.append(qmask)
            smask_true.append(smask)

            hsp_true.append(hsp_norm)
            glob_true.append(glob_norm)

    return (
        np.array(q_seqs, dtype=np.float32),
        np.array(s_seqs, dtype=np.float32),

        np.array(qmask_true, dtype=np.float32),
        np.array(smask_true, dtype=np.float32),

        np.array(hsp_true, dtype=np.float32),
        np.array(glob_true, dtype=np.float32),
    )