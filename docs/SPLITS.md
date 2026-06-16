# Dataset Split Protocols

This document describes the split files released with SMCL-DTA and how they are organized in the repository.

## Released Split Directories

| Directory | Dataset / benchmark | Description |
| --- | --- | --- |
| `splits/kiba/` | KIBA | Standard train/test split and cold-start split files. |
| `splits/davis/` | Davis | Standard train/test split and cold-start split files. |
| `splits/pdbbind_gign/` | PDBbind/GIGN | Official GIGN-aligned PDBbind train/validation/test splits. |

## KIBA and Davis Splits

KIBA and Davis use the same directory layout:

```text
splits/<dataset>/
|-- standard/
|   |-- all.csv
|   |-- train.csv
|   `-- test.csv
|-- cold_start/
|   |-- train.csv
|   |-- test1.csv
|   |-- test2.csv
|   |-- test3.csv
|   |-- <dataset>_proteins.csv
|   `-- legacy/
|       |-- train.csv
|       |-- test1.csv
|       `-- test2.csv
`-- README.md
```

`standard/` contains the regular train/test split. The main schema is:

```text
compound_iso_smiles,target_sequence,affinity
```

`cold_start/` contains the cold-start evaluation split files copied from the local dataset split folder. These files preserve the original exported columns and may contain leading index columns.

`cold_start/legacy/` keeps the older cold-start split files to make previous experiments reproducible.

## Split Sizes

| Dataset | Standard all | Standard train | Standard test | Cold train | Cold test1 | Cold test2 | Cold test3 | Protein rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| KIBA | 118,254 | 98,545 | 19,709 | 94,157 | 7,431 | 15,451 | 1,215 | 229 |
| Davis | 118,254 | 98,545 | 19,709 | 94,157 | 7,431 | 15,451 | 1,215 | 442 |

Legacy cold-start files are also retained:

| Dataset | Legacy train | Legacy test1 | Legacy test2 |
| --- | ---: | ---: | ---: |
| KIBA | 97,956 | 19,621 | 677 |
| Davis | 97,956 | 19,621 | 677 |

## Cold-Start Evaluation Notes

The cold-start files support evaluation settings where part of the compound/target space is held out from training. Use the corresponding `cold_start/train.csv` and `cold_start/test*.csv` files when reproducing cold-start experiments.

For PDBbind/GIGN split details, see `docs/PDBBIND_GIGN_BENCHMARK.md`.
