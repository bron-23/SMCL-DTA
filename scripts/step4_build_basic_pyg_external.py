#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 4: Build basic PyTorch Geometric processed .pt file for MMAtt-DTA S1 kinase external validation.

This script converts:
    compound_iso_smiles
    target_sequence
    affinity

into PyG Data objects containing:
    x
    edge_index
    edge_attr
    y
    target
    protein_id

This is a basic graph + sequence processed file.
It does NOT include ligand_surface, ligand_global, or protein_surface.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig

from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset

import os.path as osp
from tqdm import tqdm


VOCAB_PROTEIN = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23,
    "X": 24, "Z": 25
}


fdef_name = osp.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


def seqs2int(sequence: str):
    sequence = str(sequence).strip()
    return [VOCAB_PROTEIN.get(s, VOCAB_PROTEIN["X"]) for s in sequence]


def encode_target_sequence(sequence: str, target_len: int = 1200):
    target = seqs2int(sequence)

    if len(target) < target_len:
        target = np.pad(target, (0, target_len - len(target)))
    else:
        target = target[:target_len]

    return torch.LongTensor([target])


def get_nodes(g):
    feat = []

    for n, d in g.nodes(data=True):
        h_t = []

        h_t += [int(d["a_type"] == x) for x in ["H", "C", "N", "O", "F", "Cl", "S", "Br", "I"]]
        h_t.append(d["a_num"])
        h_t.append(d["acceptor"])
        h_t.append(d["donor"])
        h_t.append(int(d["aromatic"]))

        h_t += [
            int(d["hybridization"] == x)
            for x in (
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
            )
        ]

        h_t.append(d["num_h"])
        h_t.append(d["ExplicitValence"])
        h_t.append(d["FormalCharge"])
        h_t.append(d["ImplicitValence"])
        h_t.append(d["NumExplicitHs"])
        h_t.append(d["NumRadicalElectrons"])

        feat.append((n, h_t))

    feat.sort(key=lambda item: item[0])
    node_attr = torch.FloatTensor([item[1] for item in feat])

    return node_attr


def get_edges(g):
    e = {}

    for n1, n2, d in g.edges(data=True):
        e_t = [
            int(d["b_type"] == x)
            for x in (
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
            )
        ]

        e_t.append(int(d["IsConjugated"] == False))
        e_t.append(int(d["IsConjugated"] == True))

        e[(n1, n2)] = e_t

    if len(e) == 0:
        return torch.LongTensor([[0], [0]]), torch.FloatTensor([[0, 0, 0, 0, 0, 0]])

    edge_index = torch.LongTensor(list(e.keys())).transpose(0, 1)
    edge_attr = torch.FloatTensor(list(e.values()))

    return edge_index, edge_attr


def mol2graph(mol):
    if mol is None:
        return None

    feats = chem_feature_factory.GetFeaturesForMol(mol)
    g = nx.DiGraph()

    for i in range(mol.GetNumAtoms()):
        atom_i = mol.GetAtomWithIdx(i)

        g.add_node(
            i,
            a_type=atom_i.GetSymbol(),
            a_num=atom_i.GetAtomicNum(),
            acceptor=0,
            donor=0,
            aromatic=atom_i.GetIsAromatic(),
            hybridization=atom_i.GetHybridization(),
            num_h=atom_i.GetTotalNumHs(),
            ExplicitValence=atom_i.GetExplicitValence(),
            FormalCharge=atom_i.GetFormalCharge(),
            ImplicitValence=atom_i.GetImplicitValence(),
            NumExplicitHs=atom_i.GetNumExplicitHs(),
            NumRadicalElectrons=atom_i.GetNumRadicalElectrons(),
        )

    for feat in feats:
        if feat.GetFamily() == "Donor":
            for n in feat.GetAtomIds():
                g.nodes[n]["donor"] = 1
        elif feat.GetFamily() == "Acceptor":
            for n in feat.GetAtomIds():
                g.nodes[n]["acceptor"] = 1

    for i in range(mol.GetNumAtoms()):
        for j in range(mol.GetNumAtoms()):
            e_ij = mol.GetBondBetweenAtoms(i, j)
            if e_ij is not None:
                g.add_edge(
                    i,
                    j,
                    b_type=e_ij.GetBondType(),
                    IsConjugated=int(e_ij.GetIsConjugated()),
                )

    node_attr = get_nodes(g)
    edge_index, edge_attr = get_edges(g)

    return node_attr, edge_index, edge_attr


def normalize_node_features(x: torch.Tensor):
    min_val = x.min()
    max_val = x.max()
    denom = max_val - min_val

    if denom.abs() < 1e-8:
        return x

    return (x - min_val) / denom


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default="external_validation/mmatt_s1/mmatt_s1_kinase_smcl_ready.csv",
        help="Input SMCL-ready CSV from Step 3"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="external_validation/mmatt_s1/smcl_processed_basic",
        help="Output directory for processed .pt file"
    )
    parser.add_argument(
        "--target_len",
        type=int,
        default=1200,
        help="Maximum protein sequence length"
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Loading: {input_path}")
    df = pd.read_csv(input_path)

    required_cols = ["compound_iso_smiles", "target_sequence", "affinity"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"[INFO] Total rows: {len(df)}")

    smiles_list = df["compound_iso_smiles"].astype(str).str.strip().unique()
    print(f"[INFO] Unique SMILES: {len(smiles_list)}")

    graph_dict = {}
    failed_smiles = []

    print("[INFO] Building molecular graphs...")
    for smi in tqdm(smiles_list, total=len(smiles_list)):
        mol = Chem.MolFromSmiles(smi)
        graph = mol2graph(mol)

        if graph is None:
            failed_smiles.append(smi)
            continue

        graph_dict[smi] = graph

    print(f"[INFO] Graphs built: {len(graph_dict)}")
    print(f"[INFO] Failed SMILES: {len(failed_smiles)}")

    data_list = []
    failed_rows = []

    print("[INFO] Building PyG Data objects...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        smi = str(row["compound_iso_smiles"]).strip()

        if smi not in graph_dict:
            failed_rows.append(idx)
            continue

        x, edge_index, edge_attr = graph_dict[smi]
        x = normalize_node_features(x)

        target = encode_target_sequence(row["target_sequence"], target_len=args.target_len)
        label = float(row["affinity"])

        protein_id = str(row["target_id"]) if "target_id" in row else ""

        try:
            data = DATA.Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.FloatTensor([label]),
                target=target,
                protein_id=protein_id,
            )
            data_list.append(data)
        except Exception as e:
            print(f"[WARN] Failed row {idx}: {e}")
            failed_rows.append(idx)

    print(f"[INFO] Valid PyG samples: {len(data_list)}")
    print(f"[INFO] Failed rows: {len(failed_rows)}")

    if len(data_list) == 0:
        raise RuntimeError("No valid PyG samples were generated.")

    data, slices = InMemoryDataset.collate(data_list)

    pt_path = out_dir / "processed_data_mmatt_s1_kinase_basic.pt"
    torch.save((data, slices), pt_path)

    failed_smiles_path = out_dir / "failed_smiles.txt"
    failed_smiles_path.write_text("\n".join(failed_smiles), encoding="utf-8")

    failed_rows_path = out_dir / "failed_rows.txt"
    failed_rows_path.write_text("\n".join(map(str, failed_rows)), encoding="utf-8")

    summary_lines = []
    summary_lines.append("Step 4 basic PyG external validation processing summary")
    summary_lines.append("=" * 70)
    summary_lines.append(f"Input file: {input_path}")
    summary_lines.append(f"Original rows: {len(df)}")
    summary_lines.append(f"Unique SMILES: {len(smiles_list)}")
    summary_lines.append(f"Graphs built: {len(graph_dict)}")
    summary_lines.append(f"Failed SMILES: {len(failed_smiles)}")
    summary_lines.append(f"Valid PyG samples: {len(data_list)}")
    summary_lines.append(f"Failed rows: {len(failed_rows)}")
    summary_lines.append("")
    summary_lines.append("Output")
    summary_lines.append("-" * 70)
    summary_lines.append(str(pt_path))
    summary_lines.append(str(failed_smiles_path))
    summary_lines.append(str(failed_rows_path))

    summary_path = out_dir / "step4_basic_pyg_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("[DONE] Step 4 finished.")
    print(f"[OUT] Processed basic .pt: {pt_path}")
    print(f"[OUT] Summary: {summary_path}")


if __name__ == "__main__":
    main()