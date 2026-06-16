# Davis Split Files

This directory contains the Davis split CSV files copied from the local source folder `C:\Users\26331\OneDrive\Desktop\data\davis`.

## Layout

```text
splits/davis/
|-- standard/
|   |-- all.csv
|   |-- train.csv
|   `-- test.csv
|-- cold_start/
|   |-- train.csv
|   |-- test1.csv
|   |-- test2.csv
|   |-- test3.csv
|   |-- davis_proteins.csv
|   `-- legacy/
|       |-- train.csv
|       |-- test1.csv
|       `-- test2.csv
`-- README.md
```

## File Summary

| File | Rows | Description |
| --- | ---: | --- |
| `standard/all.csv` | 118,254 | Full Davis interaction table used to derive the standard split. |
| `standard/train.csv` | 98,545 | Standard training split. |
| `standard/test.csv` | 19,709 | Standard test split. |
| `cold_start/train.csv` | 94,157 | Cold-start training split. |
| `cold_start/test1.csv` | 7,431 | Cold-start test split 1. |
| `cold_start/test2.csv` | 15,451 | Cold-start test split 2. |
| `cold_start/test3.csv` | 1,215 | Cold-start test split 3. |
| `cold_start/davis_proteins.csv` | 442 | Protein list/metadata used by the Davis cold-start split. |
| `cold_start/legacy/train.csv` | 97,956 | Legacy cold-start training split retained from the local source folder. |
| `cold_start/legacy/test1.csv` | 19,621 | Legacy cold-start test split 1. |
| `cold_start/legacy/test2.csv` | 677 | Legacy cold-start test split 2. |

## Schema

The standard split files use:

```text
compound_iso_smiles,target_sequence,affinity
```

Cold-start CSV files preserve the original exported columns and may include `Unnamed: 0` and `index` before the SMILES, sequence, and affinity fields.
