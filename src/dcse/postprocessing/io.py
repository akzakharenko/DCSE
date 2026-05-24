import json
from Bio import SeqIO


def load_metadata(path):
    with open(path) as f:
        return json.load(f)


def save_metadata(path, metadata):
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)


def load_sequences(fasta_path):
    records = list(SeqIO.parse(fasta_path, "fasta"))
    seq_dict = {r.id: r for r in records}
    seq_lengths = {r.id: len(r.seq) for r in records}
    return records, seq_dict, seq_lengths


def write_fasta(records, path):
    with open(path, "w") as f:
        for r in records:
            f.write(f">{r.id}\n{str(r.seq)}\n")