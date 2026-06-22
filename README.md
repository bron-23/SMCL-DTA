# SMCL-DTA

**Surface-aware multi-modal contrastive learning for drug-target affinity prediction**

SMCL-DTA is a reproducibility package for drug-target affinity (DTA) prediction experiments. The repository contains model code, benchmark scripts, documented data splits, example usage, pretrained-checkpoint references, and external-evaluation workflows used to reproduce the results reported for SMCL-DTA.

## Highlights

- Reproducible KIBA benchmark workflow with validation scripts and expected metric ranges.
- Surface-aware molecular and protein representations for DTA modeling.
- Independent MMAtt-DTA kinase evaluation with imputation, new-compound, and new-compound-plus-new-target scenarios.
- PDBbind/GIGN structure-centric benchmark scripts for the 2013 and 2016 core sets.
- Minimal model-loading example and detailed documentation for splits, features, and external evaluations.

## Repository Layout

```text
SMCL-DTA/
|-- README.md
|-- examples/
|   |-- minimal_example.py
|   `-- expected_output.txt
|-- docs/
|   |-- SURFACE_FEATURES.md
|   |-- SPLITS.md
|   |-- MMATT_INDEPENDENT_EVALUATION.md
|   `-- PDBBIND_GIGN_BENCHMARK.md
|-- scripts/
|   |-- step1_prepare_mmatt_s1.py
|   |-- step2_add_uniprot_sequences.py
|   |-- step3_prepare_smcl_external_raw.py
|   |-- step4_build_basic_pyg_external.py
|   |-- step6b_external_validation_inference.py
|   |-- step7f_finetune_mmatt_from_kiba.py
|   `-- ...
|-- scripts_pdbbind/
|   |-- step8b_make_gign_exact_manifest.py
|   |-- step8c_make_gign_test2013_manifest.py
|   |-- run_gign_exact_5seeds.sh
|   |-- evaluate_gign_test2013_5seeds.py
|   `-- ...
|-- splits/
|   |-- README.md
|   |-- kiba/
|   |   |-- standard/
|   |   `-- cold_start/
|   |-- davis/
|   |   |-- standard/
|   |   `-- cold_start/
|   `-- pdbbind_gign/
|-- src/
|   |-- model_0428_16_dual.py
|   |-- dataset.py
|   |-- metrics.py
|   `-- ...
|-- checkpoints/
|-- paper_standard_reproduction.py
|-- paper_reproducibility_validation.py
|-- reproduce_paper_results.py
|-- preprocessing.py
`-- utils.py
```

## Documentation Index

| File | Description |
| --- | --- |
| [`docs/SURFACE_FEATURES.md`](docs/SURFACE_FEATURES.md) | Surface-feature extraction for proteins and ligands, including MSMS/PyMesh processing, electrostatic features, hydrogen-bond features, and hydrophobicity. |
| [`docs/SPLITS.md`](docs/SPLITS.md) | Released KIBA, Davis, and PDBbind/GIGN split organization. |
| [`splits/README.md`](splits/README.md) | Dataset split file index. |
| [`splits/kiba/README.md`](splits/kiba/README.md) | KIBA standard and cold-start split files. |
| [`splits/davis/README.md`](splits/davis/README.md) | Davis standard and cold-start split files. |
| [`docs/MMATT_INDEPENDENT_EVALUATION.md`](docs/MMATT_INDEPENDENT_EVALUATION.md) | Independent MMAtt-DTA kinase evaluation protocol. |
| [`docs/PDBBIND_GIGN_BENCHMARK.md`](docs/PDBBIND_GIGN_BENCHMARK.md) | PDBbind/GIGN benchmark protocol and reported results. |
| [`examples/minimal_example.py`](examples/minimal_example.py) | Minimal model-loading example with expected output. |
| [`environment_versions.txt`](environment_versions.txt) | Reference software and hardware environment used for SMCL-DTA experiments. |

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/bron-23/SMCL-DTA.git
cd SMCL-DTA
```

### 2. Create an Environment

The exact CUDA, PyTorch, and PyTorch Geometric versions should be selected for your hardware. A typical setup is:

```bash
conda create -n smcl-dta python=3.10
conda activate smcl-dta

pip install torch torchvision torchaudio
pip install torch-geometric
pip install scikit-learn numpy pandas rdkit-pypi
```

### 3. Run the Minimal Example

```bash
python examples/minimal_example.py
```

Compare the output with:

```bash
cat examples/expected_output.txt
```

## Core Reproduction Files

### Main Scripts

| File | Purpose |
| --- | --- |
| `paper_standard_reproduction.py` | Recommended entry point for reproducing the main KIBA results. |
| `paper_reproducibility_validation.py` | Runs reproducibility checks and writes a validation report. |
| `reproduce_paper_results.py` | Convenience script for reproducing paper-level outputs. |
| `train_kiba_optimized.py` | Optimized KIBA training pipeline. |
| `advanced_breakthrough_final.py` | Advanced ensemble and multi-stage calibration pipeline. |

### Core Source Files

| File | Purpose |
| --- | --- |
| `src/model_0428_16_dual.py` | Main MGraphDTA/SMCL-DTA model architecture. |
| `src/dataset.py` | Dataset loading and preprocessing utilities. |
| `src/metrics.py` | Evaluation metrics. |
| `preprocessing.py` | Data preprocessing helpers. |
| `utils.py` | Shared utility functions. |

### Pretrained Checkpoints

The `checkpoints/` directory is expected to contain the pretrained KIBA checkpoints used for ensemble prediction and calibration. Large model files may need to be downloaded or restored separately depending on the distribution channel.

## KIBA Reproduction Workflow

### Step 1: Prepare the Dataset

The repository includes KIBA split CSV files under:

```text
splits/kiba/
|-- standard/
|   |-- all.csv
|   |-- train.csv
|   `-- test.csv
`-- cold_start/
    |-- train.csv
    |-- test1.csv
    |-- test2.csv
    `-- test3.csv
```

Use `splits/kiba/standard/train.csv` and `splits/kiba/standard/test.csv` for the standard KIBA reproduction workflow. Use the `cold_start/` files for cold-start evaluation.

If a script expects processed data under `data/kiba/`, copy or link the required split files from `splits/kiba/` into the expected runtime data directory.

### Step 2: Run Standard Reproduction

```bash
python paper_standard_reproduction.py
```

Expected results should be close to:

| Metric | Expected value | Tolerance |
| --- | ---: | ---: |
| MSE | 0.1310 | +/- 0.002 |
| CI | 0.8886 | +/- 0.005 |
| R2 | 0.8035 | +/- 0.010 |

### Step 3: Validate Reproducibility

```bash
python paper_reproducibility_validation.py
cat reproducibility_report.json
```

A run is considered successfully reproduced when the metrics fall within the tolerance ranges above.

## Reported KIBA Results

| Method | MSE | CI | R2 | Description |
| --- | ---: | ---: | ---: | --- |
| Base training | 0.1330 | 0.8886 | 0.7746 | Optimized base training. |
| Model ensemble | 0.1321 | 0.8891 | 0.7805 | Ensemble of four high-performing checkpoints. |
| Prediction calibration | **0.1310** | **0.8886** | **0.8035** | Isotonic calibration. |
| Advanced ensemble | 0.1303 | 0.8883 | 0.8053 | Stacking and multi-stage calibration. |


## Dataset Splits

The exact standard and cold-start data splits used in this study are provided under:

```text
splits/
├── davis/
│   ├── standard/
│   │   ├── all.csv
│   │   ├── train.csv
│   │   └── test.csv
│   └── cold_start/
│       ├── train.csv
│       ├── test1.csv
│       ├── test2.csv
│       └── test3.csv
└── kiba/
    ├── standard/
    │   ├── all.csv
    │   ├── train.csv
    │   └── test.csv
    └── cold_start/
        ├── train.csv
        ├── test1.csv
        ├── test2.csv
        └── test3.csv
````

The numbers of drug--target pairs in each released split are summarized below.

| Dataset | Standard all | Standard train | Standard test | Cold train | Target-cluster test (`test1`) | Drug-scaffold test (`test2`) | Pair-level test (`test3`) |
| ------- | -----------: | -------------: | ------------: | ---------: | ---------------------------: | ----------------------------: | ------------------------: |
| Davis   |       30,056 |         25,046 |         5,010 |     23,258 |                        2,378 |                         4,010 |                       410 |
| KIBA    |      118,254 |         98,545 |        19,709 |     94,157 |                        7,431 |                        15,451 |                     1,215 |


For both datasets, the standard training and test sets cover the complete dataset. The cold-start training set and the three cold-start test sets also cover the complete dataset without overlap in the released partition files.

The three cold-start scenarios are defined as follows:

* `test1.csv`: target protein cluster-based cold-start evaluation;
* `test2.csv`: drug scaffold-based cold-start evaluation;
* `test3.csv`: pair-level cold-start evaluation involving unseen ligand and target groups.


## Reproducibility Controls

The reported SMCL-DTA results are averaged across multiple independent runs to reduce the effect of random initialization, data shuffling, and stochastic optimization.

```python
SEEDS = [1, 2, 3, 4, 5]
```

Each run initializes the main random sources:

```python
import random
import numpy as np
import torch


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
```

Summary statistics are computed as:

```python
mean_result = np.mean(results)
std_result = np.std(results, ddof=1)
```

Minor numerical differences can still occur across GPU models, CUDA versions, PyTorch versions, and PyTorch Geometric implementations.

## Validation Criteria

| Metric | Reported value | Tolerance | Expected range |
| --- | ---: | ---: | ---: |
| MSE | 0.1310 | +/- 0.002 | 0.1290-0.1330 |
| CI | 0.8886 | +/- 0.005 | 0.8836-0.8936 |
| R2 | 0.8035 | +/- 0.010 | 0.7935-0.8135 |

If reproduced results are outside these ranges, check the environment, dataset path, checkpoint path, and random-seed settings.

## MMAtt-DTA Independent External Evaluation

SMCL-DTA also includes an independent external evaluation on the kinase subset of the supplementary testing data released with MMAtt-DTA. The evaluation contains three scenarios:

| Scenario | Description | Original sample count |
| --- | --- | ---: |
| A) Imputation | Seen compounds and seen targets with unseen compound-target pairings. | 215 |
| B) New compound | Unseen compounds paired with seen targets. | 41,378 |
| C) New compound + new target | Both compounds and targets are unseen. | 607 |

The original MMAtt-DTA supplementary testing data are not redistributed in this repository. Download the supplementary file from the MMAtt-DTA publication and place it under:

```text
data/mmatt_dta/Supplementary_File_1.csv
```

### MMAtt Workflow

```bash
python scripts/step1_prepare_mmatt_s1.py
python scripts/step2_add_uniprot_sequences.py
python scripts/step3_prepare_smcl_external_raw.py
python scripts/step4_build_basic_pyg_external.py
python scripts/step5a_check_surface_reuse_feasibility.py
python scripts/step5c_build_external_surface_masif_overlap.py
python scripts/step6b_external_validation_inference.py
```

Additional `scripts/step7*`, `scripts/step8*`, and `scripts/step9*` files provide seeded fine-tuning, scenario alignment, sensitivity analyses, and missing-target surface-processing utilities.

Expected MMAtt output files are written under:

```text
results/mmatt_dta/
|-- mmatt_kinase_imputation_summary.csv
|-- mmatt_kinase_new_compound_summary.csv
|-- mmatt_kinase_new_compound_new_target_summary.csv
`-- mmatt_kinase_5seeds_summary.txt
```

After feature construction and validity checking, all 215 imputation samples and 40,649 new-compound samples were retained. For the new-compound-plus-new-target scenario, protein surface representations were generated for uncovered targets, enabling evaluation on all 607 samples.

## PDBbind/GIGN Benchmark

The repository also provides scripts for a structure-centric PDBbind benchmark based on the official split released with GIGN. The benchmark evaluates SMCL-DTA on the PDBbind 2013 core set and the PDBbind 2016 core set after training on the official GIGN training/validation split.

Raw PDBbind structural data are not redistributed because of licensing restrictions. Download PDBbind v2016 from the official PDBbind website and organize it as:

```text
data/pdbbind/raw/v2016/
|-- general-set-except-refined/
|-- refined-set/
|-- index/
`-- ...
```

The official GIGN split files are provided under:

```text
splits/pdbbind_gign/
|-- train.csv
|-- valid.csv
|-- test2013.csv
`-- test2016.csv
```

### PDBbind Workflow

```bash
# Build the PDBbind/GIGN manifest and prepare metadata.
python scripts_pdbbind/step1_make_pdbbind2016_manifest.py
python scripts_pdbbind/step2_add_smiles_to_manifest.py
python scripts_pdbbind/step3_resplit_valid_manifest.py
python scripts_pdbbind/step4_add_protein_sequence.py

# Create and inspect the official GIGN-aligned manifests.
python scripts_pdbbind/step8a_inspect_gign_split_overlap.py
python scripts_pdbbind/step8b_make_gign_exact_manifest.py
python scripts_pdbbind/step8c_make_gign_test2013_manifest.py

# Build, sanitize, standardize, and filter surface-feature datasets.
python scripts_pdbbind/step7_build_pdbbind_pilot_surface_pt.py
python scripts_pdbbind/step7b_sanitize_pdbbind_pt.py
python scripts_pdbbind/step7c_standardize_pdbbind_surface_pt.py
python scripts_pdbbind/step7d_filter_bad_surface_samples.py

# Train/evaluate over five random seeds.
bash scripts_pdbbind/run_gign_exact_5seeds.sh
python scripts_pdbbind/evaluate_gign_test2013_5seeds.py
```

### PDBbind Results

Using the official GIGN split and five random seeds, SMCL-DTA obtained:

| Test set | RMSE | MAE | Pearson Rp | Spearman Rs |
| --- | ---: | ---: | ---: | ---: |
| PDBbind 2013 core | 1.4441 +/- 0.0345 | 1.1575 +/- 0.0317 | 0.7930 +/- 0.0116 | 0.7882 +/- 0.0145 |
| PDBbind 2016 core | 1.2785 +/- 0.0358 | 1.0092 +/- 0.0270 | 0.8141 +/- 0.0117 | 0.8050 +/- 0.0095 |

After SMCL-DTA feature construction, the final evaluation used 107 processable complexes from the 2013 core set and 283 processable complexes from the 2016 core set.

The five-seed summaries are saved to:

```text
results/pdbbind_gign/
|-- gign_exact_smcl_5seeds_e200_pat60_summary.csv
|-- gign_exact_smcl_5seeds_e200_pat60_summary.txt
|-- gign_exact_smcl_test2013_5seeds_summary.csv
`-- gign_exact_smcl_test2013_5seeds_summary.txt
```

## Data and Licensing Notes

This repository includes the KIBA and Davis split CSV files under `splits/kiba/` and `splits/davis/`. Third-party raw datasets that require separate access or licensing, including raw PDBbind structural data and MMAtt-DTA supplementary files, are not redistributed here. Please obtain those resources from their official sources and place them in the paths described above.

Large generated files, processed datasets, cached surface features, and checkpoints may be excluded from Git tracking by `.gitignore`. Restore or regenerate them before running the full benchmark workflows.

## Citation

If you use this repository, please cite the associated SMCL-DTA manuscript and the original datasets or benchmark resources used in your experiments.
