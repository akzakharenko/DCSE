

## Overview

The example input file ncbi_16s_rRNA_46b.fasta contains 16S rRNA nucleotide sequences spanning multiple bacterial species and collected from the NCBI Nucleotide database.

The preprocessing pipeline provides functionality for generating a configurable number of datasets from the input FASTA collection.

The dataset generator creates synthetic sequence collections by sampling the following configurable parameters defined in dataset.yaml:

| Parameter         | Description                                             |
| ----------------- | ------------------------------------------------------- |
| `target_count`    | Number of sequences sampled into a dataset              |
| `K`               | Degree of the random regular pairing graph              |
| `min_pairs`       | Minimum allowed pair count                              |
| `max_pairs`       | Maximum allowed pair count                              |
| `uniform_percent` | Fraction of datasets using uniform species distribution |

Species distributions can be either uniform or randomly sampled using a Dirichlet distribution.

For every generated sequence pair, four labels are computed:

| Label             | Description                      |
| ----------------- | -------------------------------- |
| `global_identity` | Full-sequence alignment identity |
| `hsp_identity`    | Local BLAST HSP identity         |
| `qmask`           | Query homologous region          |
| `smask`           | Subject homologous region        |

Global identity is computed using Needleman–Wunsch alignment implemented with parasail. Local homologous regions are extracted using BLASTN in megablast mode.

The model is trained using the default parameters defined in training.yaml, which can be modified depending on the experiment configuration.

Evaluation is automatically performed on the generated test split.

The evaluation stage computes:
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Pearson Correlation
- R² Score
- Intersection over Union (IoU) for homologous region prediction

Additionally, scatter plots are generated for both HSP and global identity.

Each generated dataset is stored under outputs/dataset/dataset_XX/

The directory structure is organized as follows:

```txt
dataset_XX/
├── sequences.fasta
├── metadata.json
├── all_pairs_labels.tsv
├── clean_labels.tsv
├── train_labels.tsv
├── test_labels.tsv
├── train_sequences.fasta
├── test_sequences.fasta
│
├── training_run/
│   ├── model.keras
│   └── history/
│
└── evaluation/
    ├── metrics.json
    ├── scatter_hsp_identity.png
    └── scatter_global_identity.png
```


| File                    | Description                               |
| ----------------------- | ----------------------------------------- |
| `sequences.fasta`       | Sampled sequences for the dataset         |
| `metadata.json`         | Dataset configuration and statistics      |
| `all_pairs_labels.tsv`  | Raw pairwise BLAST labeling output        |
| `clean_labels.tsv`      | Filtered pair labels after postprocessing |
| `train_labels.tsv`      | Training pair labels                      |
| `test_labels.tsv`       | Evaluation pair labels                    |
| `train_sequences.fasta` | Training subset sequences                 |
| `test_sequences.fasta`  | Test subset sequences                     |
| `model.keras`           | Trained TensorFlow model                  |
| `metrics.json`          | Evaluation metrics                        |
| `scatter_*.png`         | Prediction scatter plots                  |



## Installation

1. Install the package in editable mode:
```bash
pip install -e .
```

3. Generate datasets:
```bash
dcse --mode sequences
```

3. Train a model on specified dataset:
```bash
dcse --mode train --dataset dataset_XX
```

4. Evaluate a trained model:
```bash   
dcse --mode evaluate --dataset dataset_XX
```
