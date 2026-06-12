#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Add canonical SMILES to PDBbind manifest using RDKit.

Priority:
1. ligand_sdf
2. ligand_mol2

Output:
  pdbbind2016_final_manifest_with_smiles.csv
  pdbbind2016_final_manifest_with_smiles_valid.csv
  pdbbind2016_ligand_parse_failures.csv
"""

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem


def mol_from_sdf(path):
    try:
        suppl = Chem.SDMolSupplier(str(path), sanitize=True, removeHs=False)
        if suppl is None or len(suppl) == 0:
            return None
        mol = suppl[0]
        return mol
    except Exception:
        return None


def mol_from_mol2(path):
    try:
        return Chem.MolFromMol2File(str(path), sanitize=True, removeHs=False)
    except Exception:
        return None


def canonical_smiles_from_paths(sdf_path, mol2_path):
    mol = None
    source = None

    sdf_path = Path(str(sdf_path))
    mol2_path = Path(str(mol2_path))

    if sdf_path.exists():
        mol = mol_from_sdf(sdf_path)
        if mol is not None:
            source = "sdf"

    if mol is None and mol2_path.exists():
        mol = mol_from_mol2(mol2_path)
        if mol is not None:
            source = "mol2"

    if mol is None:
        return None, None, None

    try:
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        num_atoms = mol.GetNumAtoms()
        if not smiles or num_atoms <= 0:
            return None, None, None
        return smiles, source, num_atoms
    except Exception:
        return None, None, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="/data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_final_manifest.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="/data_C/sdb1/lww/pdbbind/manifests",
    )
    args = parser.parse_args()

    manifest = Path(args.manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(manifest)

    smiles_list = []
    source_list = []
    atom_count_list = []

    for i, row in df.iterrows():
        smiles, source, n_atoms = canonical_smiles_from_paths(
            row["ligand_sdf"],
            row["ligand_mol2"],
        )
        smiles_list.append(smiles)
        source_list.append(source)
        atom_count_list.append(n_atoms)

        if (i + 1) % 1000 == 0:
            print(f"[INFO] processed {i + 1}/{len(df)}")

    df["canonical_smiles"] = smiles_list
    df["ligand_parse_source"] = source_list
    df["ligand_num_atoms"] = atom_count_list
    df["ligand_rdkit_ok"] = df["canonical_smiles"].notna()

    out_all = out_dir / "pdbbind2016_final_manifest_with_smiles.csv"
    out_valid = out_dir / "pdbbind2016_final_manifest_with_smiles_valid.csv"
    out_fail = out_dir / "pdbbind2016_ligand_parse_failures.csv"

    df.to_csv(out_all, index=False)
    df[df["ligand_rdkit_ok"]].to_csv(out_valid, index=False)
    df[~df["ligand_rdkit_ok"]].to_csv(out_fail, index=False)

    print("=" * 100)
    print("[SUMMARY]")
    print("total:", len(df))
    print("rdkit valid:", int(df["ligand_rdkit_ok"].sum()))
    print("rdkit failed:", int((~df["ligand_rdkit_ok"]).sum()))
    print()
    print("[VALID BY SPLIT]")
    print(df[df["ligand_rdkit_ok"]]["split"].value_counts())
    print()
    print("[FAILED BY SPLIT]")
    print(df[~df["ligand_rdkit_ok"]]["split"].value_counts())
    print()
    print("[SOURCE COUNTS]")
    print(df["ligand_parse_source"].value_counts(dropna=False))
    print()
    print("[ATOM COUNT STATS]")
    print(df[df["ligand_rdkit_ok"]].groupby("split")["ligand_num_atoms"].describe())
    print()
    print("[OUT]", out_all)
    print("[OUT]", out_valid)
    print("[OUT]", out_fail)


if __name__ == "__main__":
    main()
