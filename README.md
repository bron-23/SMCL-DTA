# SMCL-DTA Reproducibility Package

## Repository Structure

```text
SMCL-DTA/
├── README.md
├── environment.yml
├── examples/
│   ├── minimal_example.py
│   └── expected_output.txt
├── docs/
│   ├── SURFACE_FEATURES.md
│   ├── SPLITS.md
|   ├── PDBBIND_GIGN_BENCHMARK.md 
│   ├── MMATT_INDEPENDENT_EVALUATION.md
│   └── VIRTUAL_SCREENING.md
├── splits/
│   ├── classification/
│   ├── regression/
│   ├── mmatt_independent/
|   └── pdbbind_gign/
├── src/
│   ├── model_0428_16_dual.py
│   ├── dataset.py
│   ├── metrics.py
│   └── ...
├── checkpoints/



This repository provides the file organization, core scripts, pretrained checkpoints, expected outputs, and reproducibility instructions for the benchmark experiments reported in our manuscript. It is intended to help researchers reproduce the main SMCL-DTA results in a transparent and standardized manner.

```markdown 
## Documentation Index 
The following documentation files provide additional details for reproducing the complete SMCL-DTA pipeline: 
| File | Description |
|---|---|
| `docs/SURFACE_FEATURES.md` | Describes molecular and protein surface-feature extraction, including MSMS-based protein surface generation, PyMesh processing, electrostatic properties, hydrogen-bonding features, and hydrophobicity. | 
| `docs/SPLITS.md` | Describes scaffold-based ligand splits and protein-cluster-based cold-start splits. | 
| `docs/MMATT_INDEPENDENT_EVALUATION.md` | Provides instructions for reproducing the independent MMAtt-DTA kinase evaluation. | 
| `docs/PDBBIND_GIGN_BENCHMARK.md` | Provides instructions for reproducing the PDBbind datasets. | 
| `docs/VIRTUAL_SCREENING.md` | Describes the virtual screening workflow, including compound preprocessing, model inference, ranking, and candidate selection. |
| `examples/minimal_example.py` | Provides a minimal model-loading example with expected output. |
| `environment.yml` | Specifies the conda environment and main package dependencies. |

---

## 1. Core Reproducibility Files

### 1.1 Main Reproduction Scripts

```bash
paper_standard_reproduction.py      # Standard reproduction script for the reported paper results
paper_reproducibility_validation.py # Reproducibility validation script
```

`paper_standard_reproduction.py` is the recommended entry point for reproducing the main KIBA results reported in the manuscript.
`paper_reproducibility_validation.py` provides additional checks for reproducibility and generates a validation report.

---

### 1.2 Training and Optimization Scripts

```bash
train_kiba_optimized.py             # Base KIBA training script for optimized training
final_breakthrough_simple.py        # Ensemble and calibration script for the best reported result
advanced_breakthrough_final.py      # Advanced ensemble and multi-stage calibration strategy
```

These scripts correspond to different stages of the KIBA optimization pipeline, including base model training, model ensemble, prediction calibration, and advanced ensemble strategies.

---

### 1.3 Core Model and Utility Files

```bash
src/
├── model_0428_16_dual.py  # MGraphDTA/SMCL-DTA model architecture
├── dataset.py             # Dataset loader and preprocessing utilities
├── metrics.py             # Evaluation metrics
```

These files define the model architecture, data loading pipeline, and evaluation metrics required for reproducing the reported experiments.

---

### 1.4 Pretrained Checkpoints

```bash
checkpoints/
├── epoch-1344, LR-0.000009, MSEloss-0.1232, cindex-0.8529, r2-0.7146, test1: [MSEloss-0.1328, cindex:0.8885, r2:0.7805].pt
├── epoch-1323, LR-0.000011, MSEloss-0.1231, cindex-0.8546, r2-0.7115, test1: [MSEloss-0.1329, cindex:0.8884, r2:0.7780].pt
├── epoch-1400, LR-0.000004, MSEloss-0.1223, cindex-0.8561, r2-0.7152, test1: [MSEloss-0.1327, cindex:0.8888, r2:0.7760].pt
└── epoch-1317, LR-0.000012, MSEloss-0.1233, cindex-0.8547, r2-0.7152, test1: [MSEloss-0.1331, cindex:0.8886, r2:0.7775].pt
```

These checkpoints correspond to the best-performing KIBA models used for ensemble prediction and calibration.

---

## 2. Reported KIBA Results

### 2.1 Main Performance Metrics

| Method                 |        MSE |         CI |         R2 | Description                          |
| ---------------------- | ---------: | ---------: | ---------: | ------------------------------------ |
| Base training          |     0.1330 |     0.8886 |     0.7746 | Optimized base training              |
| Model ensemble         |     0.1321 |     0.8891 |     0.7805 | Ensemble of four best checkpoints    |
| Prediction calibration | **0.1310** | **0.8886** | **0.8035** | Isotonic calibration                 |
| Advanced ensemble      |     0.1303 |     0.8883 |     0.8053 | Stacking and multi-stage calibration |


---

## 3. Reproduction Workflow

### Step 1: Environment Setup

```bash
# Create a conda environment
conda create -n kiba_reproduction python=3.10
conda activate kiba_reproduction

# Install core dependencies
pip install torch torchvision torchaudio
pip install torch-geometric
pip install scikit-learn numpy pandas
pip install rdkit-pypi
```

Please make sure that the installed PyTorch and PyTorch Geometric versions are compatible with your CUDA version.

---

### Step 2: Data Preparation

Please place the KIBA dataset under the following directory:

```bash
/path/to/SMCL-DTA/data/kiba/
```

The expected data directory should contain the processed KIBA files required by the dataset loader.

---

### Step 3: Run the Standard Reproduction Script

```bash
python paper_standard_reproduction.py
```

Expected outputs should be close to the reported values:

```bash
MSE: approximately 0.1310, tolerance ±0.002
CI:  approximately 0.8886, tolerance ±0.005
R2:  approximately 0.8035, tolerance ±0.01
```

---

### Step 4: Validate Reproducibility

```bash
python paper_reproducibility_validation.py
```

After execution, check the generated report:

```bash
cat reproducibility_report.json
```

The report summarizes whether the reproduced results fall within the predefined tolerance range.

---

## 4. Reproducibility Controls

### 4.1 Multiple Random Seeds and Averaged Results To make the reported performance more robust and reduce the influence of random initialization, data shuffling, and stochastic optimization, the main SMCL-DTA results are obtained by averaging independent runs with multiple random seeds. In our experiments, the model was independently trained and evaluated using the following random seeds: ```python SEEDS = [1, 2, 3, 4, 5]
````

For each run, all major random sources were initialized using the corresponding seed:

import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Improve reproducibility where possible.
    torch.backends.cudnn.benchmark = False

The final reported performance is computed as the average over all repeated runs:

mean_result = np.mean(results)
std_result = np.std(results, ddof=1)

where results denotes the metric values obtained from independent runs with different random seeds.

This repeated-run protocol is consistent with the manuscript, where the reported SMCL-DTA results are averaged across multiple independent runs to reduce stochastic variation. Please note that minor numerical differences may still occur across different GPU models, CUDA versions, PyTorch versions, and PyTorch Geometric implementations.

```

### 4.2 Tolerance Range

The following tolerance ranges are used for reproducibility validation:

| Metric | Tolerance |
| ------ | --------: |
| MSE    |    ±0.002 |
| CI     |    ±0.005 |
| R2     |     ±0.01 |

---

### 4.3 Expected Reproduction Stability

| Setting                                | Expected outcome                    |
| -------------------------------------- | ----------------------------------- |
| Same hardware and software environment | Highly consistent reproduction      |
| Different GPU or CUDA versions         | Minor numerical variation may occur |
| Results within tolerance range         | Considered successfully reproduced  |

---

## 5. Recommended Manuscript Description

### 5.1 Experimental Setting

SMCL-DTA was evaluated on the KIBA benchmark using the MGraphDTA-based architecture with surface-aware representations and contrastive learning. The base model was optimized through long-term training, followed by ensemble prediction using four high-performing checkpoints. Isotonic regression was further applied for prediction calibration.

### 5.2 Result Reporting

The reproduced KIBA performance is reported as follows:

```text
SMCL-DTA achieved MSE = 0.1310, CI = 0.8886, and R2 = 0.8035 on the KIBA test set after ensemble prediction and calibration.
```

### 5.3 Reproducibility Statement

To facilitate reproducibility, we provide the source code, model architecture, pretrained checkpoints, and standardized reproduction scripts. The reproduced results are considered valid if they fall within the predefined tolerance ranges.

---

## 6. Required Files for Reproduction

The following files are required for reproducing the reported KIBA results:

1. `paper_standard_reproduction.py`
   Standard script for reproducing the main reported results.

2. `paper_reproducibility_validation.py`
   Script for checking whether reproduced results fall within the tolerance range.

3. `checkpoints/`
   Directory containing the four pretrained model checkpoints.

4. `src/model_0428_16_dual.py`
   Model architecture file.

5. `src/dataset.py` and `src/metrics.py`
   Dataset loading and metric computation files.

6. `data/kiba/`
   KIBA dataset directory.

---

## 7. Environment Requirements

The recommended environment is:

```bash
Python >= 3.10
PyTorch >= 1.12
torch-geometric
scikit-learn
numpy
pandas
RDKit
```
Users are encouraged to use the same CUDA, PyTorch, and PyTorch Geometric versions as those used in the original experiments whenever possible.

---

## 8. Validation Criteria

A reproduction run is considered successful if the results fall within the following ranges:

| Metric | Reported value | Tolerance | Expected range |
|---|---:|---:|---:| 
| MSE | 0.1310 | ±0.002 | 0.1290–0.1330 | 
| CI | 0.8886 | ±0.005 | 0.8836–0.8936 | 
| R2 | 0.8035 | ±0.010 | 0.7935–0.8135 |

If reproduced results are outside these ranges, please check the environment configuration, dataset path, checkpoint path, and random seed settings.

---

````markdown
## PDBbind/GIGN Benchmark Reproduction

We additionally provide scripts for reproducing the structure-centric PDBbind benchmark based on the official split released with GIGN. This benchmark evaluates SMCL-DTA on the PDBbind 2013 core set and the PDBbind 2016 core set after training on the official GIGN training/validation split.

### Required external data

Due to licensing restrictions, the raw PDBbind structural data are not redistributed with this repository. Users should download PDBbind v2016 from the official PDBbind website and organize the raw files as follows:

```text
/data/pdbbind/raw/v2016/
├── general-set-except-refined/
├── refined-set/
├── index/
└── ...
````

The official GIGN split files are provided under:

```text
splits/pdbbind_gign/
├── train.csv
├── valid.csv
├── test2013.csv
└── test2016.csv
```

### Reproduction workflow

The PDBbind/GIGN benchmark can be reproduced using the following steps:

```bash
# 1. Generate the SMCL-DTA manifest aligned with the official GIGN split
python scripts_pdbbind/01_make_gign_exact_manifest.py

# 2. Build SMCL-DTA-compatible graph and surface-feature datasets
python scripts_pdbbind/02_build_pdbbind_surface_dataset.py \
  --manifest manifests/pdbbind2016_gign_exact_manifest.csv \
  --out_dir processed/pdbbind_gign_raw \
  --protein_source pocket

# 3. Sanitize numerical features
python scripts_pdbbind/03_sanitize_pdbbind_dataset.py \
  --in_dir processed/pdbbind_gign_raw/processed \
  --out_dir processed/pdbbind_gign_sanitized/processed \
  --clamp_abs 5000

# 4. Filter failed feature-construction cases
python scripts_pdbbind/04_filter_bad_surface_samples.py \
  --in_dir processed/pdbbind_gign_sanitized/processed \
  --out_dir processed/pdbbind_gign_filtered/processed

# 5. Standardize surface-related features using training-set statistics
python scripts_pdbbind/05_standardize_surface_features.py \
  --in_dir processed/pdbbind_gign_filtered/processed \
  --out_dir processed/pdbbind_gign_filtered_std/processed \
  --clamp_abs 10

# 6. Train SMCL-DTA over five random seeds and summarize 2016 core-set results
bash scripts_pdbbind/06_train_gign_exact_5seeds.sh
```

### Test2013 evaluation

The PDBbind 2013 core set is evaluated using the same trained checkpoints. The surface-feature standardization statistics are computed only from the GIGN training split and then applied to the 2013 core set to avoid test-set leakage.

```bash
python scripts_pdbbind/07_make_gign_test2013_manifest.py

python scripts_pdbbind/02_build_pdbbind_surface_dataset.py \
  --manifest manifests/pdbbind2016_gign_test2013_manifest.csv \
  --out_dir processed/pdbbind_gign_test2013_raw \
  --protein_source pocket

python scripts_pdbbind/03_sanitize_pdbbind_dataset.py \
  --in_dir processed/pdbbind_gign_test2013_raw/processed \
  --out_dir processed/pdbbind_gign_test2013_sanitized/processed \
  --clamp_abs 5000

python scripts_pdbbind/04_filter_bad_surface_samples.py \
  --in_dir processed/pdbbind_gign_test2013_sanitized/processed \
  --out_dir processed/pdbbind_gign_test2013_filtered/processed

python scripts_pdbbind/08_apply_train_surface_standardization.py \
  --in_dir processed/pdbbind_gign_test2013_filtered/processed \
  --out_dir processed/pdbbind_gign_test2013_filtered_std/processed \
  --stats_json processed/pdbbind_gign_filtered_std/surface_standardization_stats.json \
  --clamp_abs 10

python scripts_pdbbind/09_evaluate_test2013_5seeds.py
```

### Expected results

Using the official GIGN split and five random seeds, SMCL-DTA obtained:

| Test set          |            RMSE |             MAE |      Pearson Rp |     Spearman Rs |
| ----------------- | --------------: | --------------: | --------------: | --------------: |
| PDBbind 2013 core | 1.4441 ± 0.0345 | 1.1575 ± 0.0317 | 0.7930 ± 0.0116 | 0.7882 ± 0.0145 |
| PDBbind 2016 core | 1.2785 ± 0.0358 | 1.0092 ± 0.0270 | 0.8141 ± 0.0117 | 0.8050 ± 0.0095 |

After SMCL-DTA feature construction, the final evaluation used 107 processable complexes from the 2013 core set and 283 processable complexes from the 2016 core set.

### Output files

The five-seed summaries are saved to:

```text
results/pdbbind_gign/
├── gign_exact_smcl_5seeds_e200_pat60_summary.csv
├── gign_exact_smcl_5seeds_e200_pat60_summary.txt
├── gign_exact_smcl_test2013_5seeds_summary.csv
└── gign_exact_smcl_test2013_5seeds_summary.txt
```

```
```
