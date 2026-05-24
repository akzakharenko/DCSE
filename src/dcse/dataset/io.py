import json

def write_fasta(records, path):
    with open(path, "w") as f:
        for sp, rec_id, seq in records:
            f.write(f">{rec_id} {sp}\n{seq}\n")


def write_metadata(path, dataset_id, params, selected, species_counts):
    max_seq_len = max(len(seq) for _, _, seq in selected)

    metadata = {
        "dataset_id": dataset_id,
        "sampling_parameters": params,
        "actual_sequence_count": len(selected),
        "max_sequence_length": max_seq_len,
        "species_counts_preliminary": dict(sorted(species_counts.items())),
    }

    with open(path, "w") as f:
        json.dump(metadata, f, indent=2)