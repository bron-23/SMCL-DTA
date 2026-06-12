#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDConfig
from rdkit.Chem import ChemicalFeatures
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
from tqdm import tqdm


VOCAB_PROTEIN = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24,
    "Z": 25,
}

fdef_name = os.path.join(RDConfig.RDDataDir, "BaseFeatures.fdef")
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(fdef_name)


class _Collater(InMemoryDataset):
    def __init__(self):
        pass


def load_module(path):
    path = str(path)
    module_dir = os.path.dirname(path)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("preprocessing_suf_pdbbind", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seqs2int(sequence, target_len=1200):
    sequence = str(sequence).strip()
    arr = [VOCAB_PROTEIN.get(ch, VOCAB_PROTEIN["X"]) for ch in sequence]
    if len(arr) < target_len:
        arr = np.pad(arr, (0, target_len - len(arr)))
    else:
        arr = arr[:target_len]
    return torch.LongTensor([arr])


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
    return torch.FloatTensor([item[1] for item in feat])


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


def mol2graph(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
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
            bond = mol.GetBondBetweenAtoms(i, j)
            if bond is not None:
                g.add_edge(
                    i,
                    j,
                    b_type=bond.GetBondType(),
                    IsConjugated=int(bond.GetIsConjugated()),
                )

    x = get_nodes(g)
    edge_index, edge_attr = get_edges(g)

    if torch.isfinite(x).all() and float(x.max() - x.min()) > 0:
        x = (x - x.min()) / (x.max() - x.min())

    return x, edge_index, edge_attr


def make_ligand_surface(smiles, extractor, rng, num_points=80):
    try:
        surface_features = extractor.get_surface_features(str(smiles))
        if surface_features is None:
            return torch.zeros((num_points, 6), dtype=torch.float), torch.zeros(264, dtype=torch.float), False

        atom_features = torch.tensor(surface_features["atom_features"], dtype=torch.float)
        global_features = torch.tensor(surface_features["global_features"], dtype=torch.float)

        if atom_features.dim() != 2 or atom_features.shape[1] != 6 or atom_features.shape[0] == 0:
            return torch.zeros((num_points, 6), dtype=torch.float), torch.zeros(264, dtype=torch.float), False

        cur_len = atom_features.size(0)
        if cur_len >= num_points:
            indices = rng.choice(cur_len, size=num_points, replace=False)
            ligand_surface = atom_features[indices]
        else:
            repeat_times = (num_points + cur_len - 1) // cur_len
            ligand_surface = atom_features.repeat((repeat_times, 1))[:num_points]

        if global_features.numel() != 264:
            global_features = torch.zeros(264, dtype=torch.float)

        return ligand_surface, global_features.float(), True

    except Exception as e:
        print(f"[WARN] ligand surface failed: {smiles} | {type(e).__name__}: {e}")
        return torch.zeros((num_points, 6), dtype=torch.float), torch.zeros(264, dtype=torch.float), False


def make_protein_surface(row, module, protein_source="pocket"):
    pdb_id = str(row["pdb_id"]).lower()
    seq = str(row["protein_sequence"])

    pocket_pdb = str(row.get("pocket_pdb", ""))
    protein_pdb = str(row.get("protein_pdb", ""))

    if protein_source == "pocket":
        pdb_file = pocket_pdb if os.path.exists(pocket_pdb) else protein_pdb
        cache_id = f"pdbbind_{pdb_id}_pocket"
    else:
        pdb_file = protein_pdb if os.path.exists(protein_pdb) else pocket_pdb
        cache_id = f"pdbbind_{pdb_id}_protein"

    if not os.path.exists(pdb_file):
        return torch.zeros((512, 9), dtype=torch.float), False, "missing_pdb"

    try:
        feat = module.extract_protein_features(
            protein_id=cache_id,
            protein_sequence=seq,
            pdb_file=pdb_file,
            include_masif=True,
            chain_id=None,
        )

        masif = feat.get("masif", None)
        if masif is None:
            return torch.zeros((512, 9), dtype=torch.float), False, "no_masif"

        ps = torch.tensor(np.asarray(masif), dtype=torch.float)

        if ps.shape != (512, 9):
            return torch.zeros((512, 9), dtype=torch.float), False, f"bad_shape_{tuple(ps.shape)}"

        if not torch.isfinite(ps).all():
            return torch.zeros((512, 9), dtype=torch.float), False, "nonfinite"

        if float(torch.sum(torch.abs(ps))) <= 1e-8:
            return ps, False, "zero_surface"

        return ps, True, "ok"

    except Exception as e:
        return torch.zeros((512, 9), dtype=torch.float), False, f"{type(e).__name__}:{e}"


def build_split(df, split_name, module, ligand_extractor, rng, protein_source):
    sub = df[df["split"] == split_name].copy()
    data_list = []
    records = []

    graph_cache = {}

    for i, row in tqdm(sub.iterrows(), total=len(sub), desc=f"build {split_name}"):
        pdb_id = str(row["pdb_id"]).lower()
        smiles = str(row["canonical_smiles"])
        label = float(row["label"])

        rec = {
            "pdb_id": pdb_id,
            "split": split_name,
            "status": "ok",
            "ligand_surface_ok": False,
            "protein_surface_ok": False,
            "protein_surface_status": "",
        }

        try:
            if smiles not in graph_cache:
                graph_cache[smiles] = mol2graph(smiles)

            graph = graph_cache[smiles]
            if graph is None:
                rec["status"] = "bad_smiles_graph"
                records.append(rec)
                continue

            x, edge_index, edge_attr = graph

            target = seqs2int(row["protein_sequence"], target_len=1200)

            ligand_surface, ligand_global, ligand_ok = make_ligand_surface(smiles, ligand_extractor, rng)
            protein_surface, protein_ok, protein_status = make_protein_surface(
                row=row,
                module=module,
                protein_source=protein_source,
            )

            data = DATA.Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=torch.FloatTensor([label]),
                target=target,
                protein_id=pdb_id,
                ligand_surface=ligand_surface,
                ligand_global=ligand_global,
                protein_surface=protein_surface,
            )

            data_list.append(data)

            rec["ligand_surface_ok"] = bool(ligand_ok)
            rec["protein_surface_ok"] = bool(protein_ok)
            rec["protein_surface_status"] = protein_status
            records.append(rec)

        except Exception as e:
            rec["status"] = f"{type(e).__name__}:{e}"
            records.append(rec)
            continue

    return data_list, pd.DataFrame(records)


def save_pt(data_list, out_pt):
    out_pt = Path(out_pt)
    out_pt.parent.mkdir(parents=True, exist_ok=True)

    collater = _Collater()
    data, slices = collater.collate(data_list)
    torch.save((data, slices), out_pt)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        default="/data_C/sdb1/lww/pdbbind/manifests/pdbbind2016_pilot_manifest.csv",
    )
    parser.add_argument(
        "--out_dir",
        default="/data_C/sdb1/lww/pdbbind/processed/pilot",
    )
    parser.add_argument(
        "--preprocessing_suf",
        default="/home/lww/learn_project/MGraphDTA-dev/regression/preprocessing_suf.py",
    )
    parser.add_argument(
        "--protein_source",
        choices=["pocket", "protein"],
        default="pocket",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit_per_split", type=int, default=0)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    module = load_module(args.preprocessing_suf)

    # Redirect caches and temp paths to PDBbind-specific directories.
    cache_root = out_dir / f"cache_{args.protein_source}"
    cache_root.mkdir(parents=True, exist_ok=True)

    module.PROTEIN_SURFACE_CACHE_DIR = str(cache_root / "protein_surface_cache")
    os.makedirs(module.PROTEIN_SURFACE_CACHE_DIR, exist_ok=True)

    module.MASIF_OPTS["tmp_dir"] = str(cache_root / "tmp")
    module.MASIF_OPTS["raw_pdb_dir"] = "/data_C/sdb1/lww/pdbbind/raw/v2016"
    module.MASIF_OPTS["ply_chain_dir"] = str(cache_root / "ply_chain")
    module.MASIF_OPTS["pdb_chain_dir"] = str(cache_root / "pdb_chain")

    for k in ["tmp_dir", "ply_chain_dir", "pdb_chain_dir"]:
        os.makedirs(module.MASIF_OPTS[k], exist_ok=True)

    ligand_extractor = module.MolecularSurfaceExtractor(cache_dir=str(cache_root / "ligand_surface_cache"))

    df = pd.read_csv(args.manifest)

    if args.limit_per_split and args.limit_per_split > 0:
        parts = []
        for split in ["train", "val", "core2016"]:
            ss = df[df["split"] == split].copy()
            parts.append(ss.head(args.limit_per_split))
        df = pd.concat(parts, ignore_index=True)

    print("=" * 100)
    print("[INFO] Building PDBbind pilot surface .pt")
    print("[INFO] manifest:", args.manifest)
    print("[INFO] out_dir:", out_dir)
    print("[INFO] protein_source:", args.protein_source)
    print("[INFO] split counts:")
    print(df["split"].value_counts())
    print("=" * 100)

    all_summaries = []

    for split in ["train", "val", "core2016"]:
        data_list, summary = build_split(
            df=df,
            split_name=split,
            module=module,
            ligand_extractor=ligand_extractor,
            rng=rng,
            protein_source=args.protein_source,
        )

        out_pt = out_dir / f"processed_data_pdbbind2016_pilot_{split}_surface_masif.pt"
        out_csv = out_dir / f"processed_data_pdbbind2016_pilot_{split}_surface_masif_summary.csv"

        if len(data_list) == 0:
            print(f"[WARN] No data generated for split={split}; skip saving.")
            summary.to_csv(out_csv, index=False)
            all_summaries.append(summary)
            continue

        save_pt(data_list, out_pt)
        summary.to_csv(out_csv, index=False)
        all_summaries.append(summary)

        print()
        print(f"[DONE] {split}")
        print("samples saved:", len(data_list))
        print("out_pt:", out_pt)
        print("protein_surface_ok:", int(summary["protein_surface_ok"].sum()), "/", len(summary))
        print("ligand_surface_ok:", int(summary["ligand_surface_ok"].sum()), "/", len(summary))
        print(summary["protein_surface_status"].value_counts().head(10))

    if hasattr(ligand_extractor, "_save_cache"):
        ligand_extractor._save_cache()

    all_summary = pd.concat(all_summaries, ignore_index=True)
    all_summary.to_csv(out_dir / "processed_data_pdbbind2016_pilot_surface_masif_all_summary.csv", index=False)

    print("=" * 100)
    print("[ALL SUMMARY]")
    print(all_summary.groupby("split")[["protein_surface_ok", "ligand_surface_ok"]].sum())
    print()
    print(all_summary.groupby("split")["status"].value_counts())
    print("=" * 100)


if __name__ == "__main__":
    main()
