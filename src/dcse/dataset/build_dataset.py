import os
import random
import numpy as np
from collections import defaultdict
from Bio import SeqIO
import pandas as pd

from dcse.dataset.sampling import sample_parameters
from dcse.dataset.fasta_utils import is_valid_dna, extract_species
from dcse.dataset.io import write_fasta, write_metadata


def build_dataset(species_to_records, params):
    random.seed(params["random_seed"])
    np.random.seed(params["random_seed"])

    target_count = params["target_count"]
    distribution_type = params["distribution_type"]

    eligible_species = [s for s in species_to_records if species_to_records[s]]
    if not eligible_species:
        raise ValueError("No species with sequences available")

    num_species = min(params["num_species"], len(eligible_species))
    chosen_species = random.sample(eligible_species, num_species)

    if distribution_type == "uniform":
        base = target_count // num_species
        remainder = target_count % num_species

        species_counts = {sp: base for sp in chosen_species}
        for sp in chosen_species[:remainder]:
            species_counts[sp] += 1

    elif distribution_type == "random":
        proportions = np.random.dirichlet([1.0] * num_species)
        raw_counts = np.floor(proportions * target_count).astype(int)

        species_counts = dict(zip(chosen_species, raw_counts))
        diff = target_count - sum(species_counts.values())

        for sp in chosen_species[:diff]:
            species_counts[sp] += 1
    else:
        raise ValueError("distribution_type must be 'uniform' or 'random'")

    selected = []

    for sp in chosen_species:
        count = species_counts[sp]
        records = species_to_records[sp]

        if len(records) < count:
            sampled = random.choices(records, k=count)
        else:
            sampled = random.sample(records, count)

        for rec_id, seq in sampled:
            selected.append((sp, rec_id, seq))

    random.shuffle(selected)

    if len(selected) != target_count:
        raise RuntimeError("Dataset size mismatch")

    return selected


def run_dataset_generation(cfg):
    os.makedirs(cfg.root_dir, exist_ok=True)

    species_to_records = defaultdict(list)
    removed_invalid = 0

    print("Pass 1: Cleaning sequences...")

    for record in SeqIO.parse(cfg.input_fasta, "fasta"):
        species = extract_species(record.description)

        if species not in cfg.target_species:
            continue

        seq = str(record.seq).upper()

        if not is_valid_dna(seq):
            removed_invalid += 1
            continue

        species_to_records[species].append((record.id, seq))

    total_available = sum(len(v) for v in species_to_records.values())
    print(f"Valid sequences: {total_available}")
    print(f"Removed invalid: {removed_invalid}")

    num_uniform = int(cfg.num_datasets * cfg.uniform_percent)
    distribution_plan = (
        ["uniform"] * num_uniform +
        ["random"] * (cfg.num_datasets - num_uniform)
    )
    random.shuffle(distribution_plan)

    summary = []

    for i in range(cfg.num_datasets):
        dataset_id = f"dataset_{i+1:02d}"
        dataset_dir = os.path.join(cfg.root_dir, dataset_id)
        os.makedirs(dataset_dir, exist_ok=True)

        params = sample_parameters(
            cfg.param_space,
            cfg.min_pairs,
            cfg.max_pairs,
            len(species_to_records),
        )
        params["distribution_type"] = distribution_plan[i]

        if total_available < params["target_count"]:
            raise ValueError("Not enough sequences")

        selected = build_dataset(species_to_records, params)

        fasta_path = os.path.join(dataset_dir, "sequences.fasta")
        meta_path = os.path.join(dataset_dir, "metadata.json")

        species_counts = defaultdict(int)

        for sp, rec_id, seq in selected:
            species_counts[sp] += 1

        write_fasta(selected, fasta_path)
        write_metadata(meta_path, dataset_id, params, selected, species_counts)

        summary.append({
            "dataset": dataset_id,
            "sequences": len(selected),
            "species": len(species_counts),
            "distribution": params["distribution_type"],
            "K": params["K"],
            "pairs": len(selected) * params["K"] // 2,
        })

    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(os.path.join(cfg.root_dir, "dataset_summary.csv"), index=False)

    print("\nDataset generation complete.")