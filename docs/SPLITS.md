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


