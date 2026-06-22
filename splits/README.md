# Dataset Split Files

This directory contains benchmark split files used by SMCL-DTA.

## Available Splits

| Directory | Dataset / benchmark | Contents |
| --- | --- | --- |
| `kiba/` | KIBA | Standard train/test split and cold-start split files. |
| `davis/` | Davis | Standard train/test split and cold-start split files. |
| `pdbbind_gign/` | PDBbind/GIGN | Official GIGN-aligned PDBbind split files. |

## KIBA and Davis Organization

The KIBA and Davis split files are organized consistently:

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
`-- README.md
```

`standard/` contains the regular random train/test split. `cold_start/` contains held-out evaluation splits for cold-start style evaluation. `legacy/` preserves the older cold-start split files from the local source folder for reproducibility.

The main CSV schema is:

```text
compound_iso_smiles,target_sequence,affinity
```

Cold-start files may also contain index columns such as `Unnamed: 0` and `index`, inherited from their original CSV export.
