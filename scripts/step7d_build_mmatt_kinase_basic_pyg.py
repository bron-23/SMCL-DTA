#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 7D: Build basic PyG .pt files for MMAtt-DTA kinase train/validation.

Inputs:
    mmatt_kinase_train_with_sequence.csv
    mmatt_kinase_val_with_sequence.csv

Outputs:
    processed_data_mmatt_kinase_train_basic.pt
    processed_data_mmatt_kinase_val_basic.pt
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import networkx as nx
from tqdm import tqdm

from rdkit import Chem
from rdkit.Chem import ChemicalFeatures
from rdkit import RDConfig
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset

import os.path as osp


VOCAB_PROTEIN = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,
    "Z": 25
}


def seqs2int(sequence):
    return [VOCAB_PROTEIN.get(s, 24) for s in str(sequence)]


fdef_name = osp.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


class BasicGraphBuilder:
    def get_nodes(self, g):
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
        return torch.FloatTensor([item[1] for item in feat])

    def get_edges(self, g):
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

    def mol2graph(self, mol):
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

        node_attr = self.get_nodes(g)

        # Match previous preprocessing normalization while avoiding zero division.
        denom = node_attr.max() - node_attr.min()
        if denom > 0:
            node_attr = (node_attr - node_attr.min()) / denom

        edge_index, edge_attr = self.get_edges(g)
        return node_attr, edge_index, edge_attr


def build_graph_cache(smiles_list):
    builder = BasicGraphBuilder()
    graph_dict = {}
    failed = []

    for smi in tqdm(smiles_list, desc="Building molecular graphs"):
        mol = Chem.MolFromSmiles(smi)
        graph = builder.mol2graph(mol)
        if graph is None:
            failed.append(smi)
        else:
            graph_dict[smi] = graph

    return graph_dict, failed


def dataframe_to_data_list(df, graph_dict, target_len=1200):
    data_list = []
    failed_rows = []

    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Building PyG Data"):
        smi = str(row["compound_iso_smiles"])
        sequence = str(row["target_sequence"])
        label = float(row["affinity"])
        target_id = str(row["target_id"])

        if smi not in graph_dict:
            failed_rows.append((idx, smi, "missing_graph"))
            continue

        x, edge_index, edge_attr = graph_dict[smi]

        target = seqs2int(sequence)
        if len(target) < target_len:
            target = np.pad(target, (0, target_len - len(target)))
        else:
            target = target[:target_len]

        data = DATA.Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.FloatTensor([label]),
            target=torch.LongTensor([target]),
            protein_id=target_id,
        )

        data_list.append(data)

    return data_list, failed_rows


def save_data_list(data_list, out_path):
    dataset = InMemoryDataset(".")
    data, slices = dataset.collate(data_list)
    torch.save((data, slices), out_path, _use_new_zipfile_serialization=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train_csv",
        type=str,
        default="/data_C/sdb1/lww/mmatt_training/smcl_sequence/mmatt_kinase_train_with_sequence.csv",
    )
    parser.add_argument(
        "--val_csv",
        type=str,
        default="/data_C/sdb1/lww/mmatt_training/smcl_sequence/mmatt_kinase_val_with_sequence.csv",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/data_C/sdb1/lww/mmatt_training/smcl_processed_basic",
    )
    parser.add_argument("--limit_train", type=int, default=0)
    parser.add_argument("--limit_val", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(args.train_csv)
    val_df = pd.read_csv(args.val_csv)

    if args.limit_train > 0:
        train_df = train_df.iloc[:args.limit_train].copy()
    if args.limit_val > 0:
        val_df = val_df.iloc[:args.limit_val].copy()

    required = ["compound_iso_smiles", "target_id", "target_sequence", "affinity"]
    for name, df in [("train", train_df), ("val", val_df)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")

    all_smiles = sorted(
        set(train_df["compound_iso_smiles"].astype(str)) |
        set(val_df["compound_iso_smiles"].astype(str))
    )

    print("=" * 80)
    print("Step 7D basic PyG construction")
    print("=" * 80)
    print(f"Train rows: {len(train_df)}")
    print(f"Validation rows: {len(val_df)}")
    print(f"Unique SMILES: {len(all_smiles)}")

    graph_dict, failed_smiles = build_graph_cache(all_smiles)

    print(f"Graphs built: {len(graph_dict)}")
    print(f"Failed SMILES: {len(failed_smiles)}")

    train_list, train_failed_rows = dataframe_to_data_list(train_df, graph_dict)
    val_list, val_failed_rows = dataframe_to_data_list(val_df, graph_dict)

    train_pt = out_dir / "processed_data_mmatt_kinase_train_basic.pt"
    val_pt = out_dir / "processed_data_mmatt_kinase_val_basic.pt"

    print("Saving train .pt...")
    save_data_list(train_list, train_pt)

    print("Saving validation .pt...")
    save_data_list(val_list, val_pt)

    failed_smiles_path = out_dir / "failed_smiles.txt"
    failed_rows_path = out_dir / "failed_rows.txt"
    summary_path = out_dir / "step7d_basic_pyg_summary.txt"

    failed_smiles_path.write_text("\n".join(failed_smiles), encoding="utf-8")

    failed_lines = []
    for item in train_failed_rows:
        failed_lines.append(f"train\t{item[0]}\t{item[1]}\t{item[2]}")
    for item in val_failed_rows:
        failed_lines.append(f"val\t{item[0]}\t{item[1]}\t{item[2]}")
    failed_rows_path.write_text("\n".join(failed_lines), encoding="utf-8")

    summary = []
    summary.append("Step 7D MMAtt-DTA kinase basic PyG summary")
    summary.append("=" * 80)
    summary.append(f"Train input: {args.train_csv}")
    summary.append(f"Validation input: {args.val_csv}")
    summary.append("")
    summary.append(f"Train rows input: {len(train_df)}")
    summary.append(f"Validation rows input: {len(val_df)}")
    summary.append(f"Unique SMILES: {len(all_smiles)}")
    summary.append(f"Graphs built: {len(graph_dict)}")
    summary.append(f"Failed SMILES: {len(failed_smiles)}")
    summary.append("")
    summary.append(f"Train PyG samples: {len(train_list)}")
    summary.append(f"Train failed rows: {len(train_failed_rows)}")
    summary.append(f"Validation PyG samples: {len(val_list)}")
    summary.append(f"Validation failed rows: {len(val_failed_rows)}")
    summary.append("")
    summary.append("Output files")
    summary.append("-" * 80)
    summary.append(str(train_pt))
    summary.append(str(val_pt))
    summary.append(str(failed_smiles_path))
    summary.append(str(failed_rows_path))

    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print("[DONE] Step 7D finished.")


if __name__ == "__main__":
    main()