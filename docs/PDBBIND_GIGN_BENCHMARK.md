````markdown
# PDBbind/GIGN Benchmark

This document describes how to reproduce the additional PDBbind benchmark experiments reported for SMCL-DTA.

## Benchmark protocol

We follow the official PDBbind split released with GIGN. The benchmark contains:

| Split | Source file | Number of complexes in GIGN split |
|---|---:|---:|
| Training | `train.csv` | 11,904 |
| Validation | `valid.csv` | 1,000 |
| 2013 core test | `test2013.csv` | 107 |
| 2016 core test | `test2016.csv` | 285 |

SMCL-DTA uses the GIGN training and validation splits for model training and checkpoint selection. The best checkpoint is selected according to validation RMSE. The selected checkpoints from five random seeds are then evaluated on the 2013 and 2016 core sets.

## Feature construction

For each PDBbind complex, SMCL-DTA constructs the following features:

1. ligand molecular graph;
2. protein sequence representation;
3. protein pocket surface features;
4. ligand surface features;
5. ligand global surface descriptors.

Protein and ligand surface features are extracted offline and cached before model training. During inference, the model only loads precomputed features and performs neural-network forward propagation.

## Sanitization and filtering

After feature construction, numerical features are sanitized by replacing NaN and infinite values with finite values. Samples with failed surface construction or all-zero ligand/protein surface descriptors are filtered before training or evaluation.

The final SMCL-DTA evaluation used:

| Test set | Processable complexes |
|---|---:|
| PDBbind 2013 core | 107 |
| PDBbind 2016 core | 283 |

## Feature standardization

Surface-related features are standardized using statistics computed from the training split only. The same training-set statistics are applied to the validation, 2013 core, and 2016 core sets to avoid test-set leakage.

## Repeated-run evaluation

SMCL-DTA is trained and evaluated over five random seeds:

```python
SEEDS = [42, 43, 44, 45, 46]
````

The final metrics are reported as mean ± standard deviation.

## Reported results

| Test set          |            RMSE |             MAE |      Pearson Rp |     Spearman Rs |
| ----------------- | --------------: | --------------: | --------------: | --------------: |
| PDBbind 2013 core | 1.4441 ± 0.0345 | 1.1575 ± 0.0317 | 0.7930 ± 0.0116 | 0.7882 ± 0.0145 |
| PDBbind 2016 core | 1.2785 ± 0.0358 | 1.0092 ± 0.0270 | 0.8141 ± 0.0117 | 0.8050 ± 0.0095 |

These results were used for the additional structure-centric benchmark comparison in the revised manuscript.

```
```
