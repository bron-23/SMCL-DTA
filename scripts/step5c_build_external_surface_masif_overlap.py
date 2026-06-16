#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 5C: Build surface/MaSIF external validation .pt for MMAtt-DTA S1 kinase subset.

Strategy:
1. Load the basic external PyG .pt from Step 4.
2. Reuse existing protein_surface tensors from Davis/KIBA surface_masif processed files.
3. Generate ligand_surface and ligand_global from SMILES using the original MolecularSurfaceExtractor.
4. Keep only rows whose target_id has a real reusable protein_surface.
5. Save a surface_masif-compatible external validation .pt file.

Output:
    processed_data_mmatt_s1_kinase_surface_masif_overlap.pt
"""
import warnings
warnings.filterwarnings("ignore")

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass
import argparse
import importlib.util
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset


class LoadedDataset(InMemoryDataset):
    def __init__(self, pt_path):
        super().__init__(".")
        self.data, self.slices = torch.load(pt_path, weights_only=False)

    def len(self):
        return int(self.slices["y"].numel() - 1)


def load_molecular_surface_extractor(preprocessing_suf_path):
    preprocessing_suf_path = str(preprocessing_suf_path)
    module_dir = os.path.dirname(preprocessing_suf_path)

    import sys
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)

    spec = importlib.util.spec_from_file_location("preprocessing_suf_external", preprocessing_suf_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return module.MolecularSurfaceExtractor


def load_existing_protein_surface_lookup(pt_files):
    """
    Build target_id/protein_id -> protein_surface lookup from existing surface_masif .pt files.
    Prefer non-zero 512 x 9 tensors.
    """
    lookup = {}
    source_lookup = {}

    for pt_path in pt_files:
        pt_path = str(pt_path)
        if not os.path.exists(pt_path):
            print(f"[WARN] Missing existing pt: {pt_path}")
            continue

        print(f"[INFO] Loading existing surface file: {pt_path}")
        ds = LoadedDataset(pt_path)

        for i in tqdm(range(ds.len()), desc=f"Reading {Path(pt_path).name}"):
            try:
                d = ds.get(i)

                if not hasattr(d, "protein_id"):
                    continue
                if not hasattr(d, "protein_surface"):
                    continue

                pid = str(d.protein_id).strip()
                if not pid or pid.lower() == "nan":
                    continue

                ps = d.protein_surface
                if ps is None:
                    continue

                ps = ps.view(-1, 9)

                if ps.shape[0] != 512:
                    continue

                # Prefer non-zero real surface tensors
                is_nonzero = bool(torch.sum(torch.abs(ps)).item() > 1e-8)

                if pid not in lookup:
                    lookup[pid] = ps.detach().cpu()
                    source_lookup[pid] = pt_path
                else:
                    old_nonzero = bool(torch.sum(torch.abs(lookup[pid])).item() > 1e-8)
                    if is_nonzero and not old_nonzero:
                        lookup[pid] = ps.detach().cpu()
                        source_lookup[pid] = pt_path

            except Exception as e:
                print(f"[WARN] Failed reading item {i} from {pt_path}: {e}")

    return lookup, source_lookup


def make_ligand_surface(smiles, extractor, rng, num_points=80):
    """
    Generate ligand_surface [80, 6] and ligand_global [264].
    """
    try:
        surface_features = extractor.get_surface_features(smiles)

        if surface_features is None:
            return torch.zeros((num_points, 6), dtype=torch.float), torch.zeros(264, dtype=torch.float), False

        atom_features = torch.tensor(surface_features["atom_features"], dtype=torch.float)
        global_features = torch.tensor(surface_features["global_features"], dtype=torch.float)

        if atom_features.dim() != 2 or atom_features.shape[1] != 6:
            return torch.zeros((num_points, 6), dtype=torch.float), torch.zeros(264, dtype=torch.float), False

        cur_len = atom_features.size(0)

        if cur_len <= 0:
            return torch.zeros((num_points, 6), dtype=torch.float), torch.zeros(264, dtype=torch.float), False

        if cur_len >= num_points:
            indices = rng.choice(cur_len, size=num_points, replace=False)
            ligand_surface = atom_features[indices]
        else:
            repeat_times = (num_points + cur_len - 1) // cur_len
            ligand_surface = atom_features.repeat((repeat_times, 1))[:num_points]

        if global_features.numel() != 264:
            global_features = torch.zeros(264, dtype=torch.float)

        return ligand_surface, global_features, True

    except Exception as e:
        print(f"[WARN] Ligand surface failed for {smiles}: {e}")
        return torch.zeros((num_points, 6), dtype=torch.float), torch.zeros(264, dtype=torch.float), False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--external_csv",
        type=str,
        default="external_validation/mmatt_s1/mmatt_s1_kinase_smcl_ready.csv",
        help="External CSV from Step 3"
    )
    parser.add_argument(
        "--basic_pt",
        type=str,
        default="external_validation/mmatt_s1/smcl_processed_basic/processed_data_mmatt_s1_kinase_basic.pt",
        help="Basic PyG pt from Step 4"
    )
    parser.add_argument(
        "--preprocessing_suf",
        type=str,
        default="/home/lww/learn_project/MGraphDTA-dev/regression/preprocessing_suf.py",
        help="Original preprocessing_suf.py path"
    )
    parser.add_argument(
        "--existing_surface_pts",
        type=str,
        nargs="+",
        required=True,
        help="Existing Davis/KIBA surface_masif .pt files used to reuse protein_surface"
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="external_validation/mmatt_s1/smcl_processed_surface_masif",
        help="Output directory"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="For debugging only. If >0, process only the first N rows."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for ligand surface point sampling"
    )

    args = parser.parse_args()

    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading external CSV...")
    df = pd.read_csv(args.external_csv)

    if args.limit and args.limit > 0:
        df = df.iloc[:args.limit].copy()

    required_cols = ["compound_iso_smiles", "target_id"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"External CSV missing columns: {missing}")

    print("[INFO] Loading basic external PyG dataset...")
    basic_ds = LoadedDataset(args.basic_pt)

    if args.limit and args.limit > 0:
        n_total = min(args.limit, len(df), basic_ds.len())
    else:
        n_total = min(len(df), basic_ds.len())

    print(f"[INFO] External rows to consider: {n_total}")

    print("[INFO] Loading existing protein_surface lookup...")
    protein_surface_lookup, protein_source_lookup = load_existing_protein_surface_lookup(args.existing_surface_pts)
    print(f"[INFO] Protein surface lookup size: {len(protein_surface_lookup)}")

    print("[INFO] Loading MolecularSurfaceExtractor from original preprocessing_suf.py...")
    MolecularSurfaceExtractor = load_molecular_surface_extractor(args.preprocessing_suf)
    extractor = MolecularSurfaceExtractor()

    data_list = []
    kept_indices = []
    missing_protein_surface = []
    ligand_success = 0
    ligand_failed = 0

    print("[INFO] Building external surface_masif Data objects...")
    for i in tqdm(range(n_total)):
        row = df.iloc[i]
        target_id = str(row["target_id"]).strip()
        smiles = str(row["compound_iso_smiles"]).strip()

        if target_id not in protein_surface_lookup:
            missing_protein_surface.append(i)
            continue

        d = basic_ds.get(i).clone()

        ligand_surface, ligand_global, ok = make_ligand_surface(smiles, extractor, rng, num_points=80)
        if ok:
            ligand_success += 1
        else:
            ligand_failed += 1

        d.ligand_surface = ligand_surface
        d.ligand_global = ligand_global
        d.protein_surface = protein_surface_lookup[target_id].clone()
        d.protein_id = target_id

        data_list.append(d)
        kept_indices.append(i)

    if len(data_list) == 0:
        raise RuntimeError("No valid external samples with reusable protein_surface were generated.")

    data, slices = InMemoryDataset.collate(data_list)

    pt_out = out_dir / "processed_data_mmatt_s1_kinase_surface_masif_overlap.pt"
    torch.save((data, slices), pt_out)

    kept_df = df.iloc[kept_indices].copy()
    kept_csv = out_dir / "mmatt_s1_kinase_surface_masif_overlap_rows.csv"
    kept_df.to_csv(kept_csv, index=False)

    missing_df = df.iloc[missing_protein_surface].copy()
    missing_csv = out_dir / "mmatt_s1_kinase_missing_protein_surface_rows.csv"
    missing_df.to_csv(missing_csv, index=False)

    summary_lines = []
    summary_lines.append("Step 5C external surface_masif construction summary")
    summary_lines.append("=" * 80)
    summary_lines.append(f"External CSV: {args.external_csv}")
    summary_lines.append(f"Basic pt: {args.basic_pt}")
    summary_lines.append(f"Rows considered: {n_total}")
    summary_lines.append(f"Rows kept with reusable protein_surface: {len(data_list)}")
    summary_lines.append(f"Rows missing protein_surface: {len(missing_protein_surface)}")
    summary_lines.append("")
    summary_lines.append(f"Ligand surface success: {ligand_success}")
    summary_lines.append(f"Ligand surface failed/fallback zero: {ligand_failed}")
    summary_lines.append("")
    summary_lines.append(f"Unique targets kept: {kept_df['target_id'].nunique()}")
    summary_lines.append(f"Unique compounds kept: {kept_df['compound_iso_smiles'].nunique()}")
    summary_lines.append("")
    summary_lines.append("Output files")
    summary_lines.append("-" * 80)
    summary_lines.append(str(pt_out))
    summary_lines.append(str(kept_csv))
    summary_lines.append(str(missing_csv))

    summary_path = out_dir / "step5c_surface_masif_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")

    print("[DONE] Step 5C finished.")
    print(f"[OUT] Surface/MaSIF pt: {pt_out}")
    print(f"[OUT] Kept rows CSV: {kept_csv}")
    print(f"[OUT] Summary: {summary_path}")


if __name__ == "__main__":
    main()