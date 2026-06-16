#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merge Step 7E surface/MaSIF chunks into one final .pt file.
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm
from torch_geometric.data import InMemoryDataset


class LoadedPTDataset(InMemoryDataset):
    def __init__(self, pt_path):
        super().__init__(".")
        self.data, self.slices = torch.load(pt_path, weights_only=False)

    def len(self):
        return int(self.slices["y"].numel() - 1)


def save_data_list(data_list, out_path):
    dataset = InMemoryDataset(".")
    data, slices = dataset.collate(data_list)
    torch.save((data, slices), out_path, _use_new_zipfile_serialization=False)


def find_surface_pt(chunk_dir):
    pts = sorted(Path(chunk_dir).glob("processed_data_*surface_masif*.pt"))
    return pts[0] if pts else None


def find_rows_csv(chunk_dir):
    csvs = sorted(Path(chunk_dir).glob("*surface_masif*_rows.csv"))
    return csvs[0] if csvs else None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk_root", type=str, required=True)
    parser.add_argument("--out_pt", type=str, required=True)
    parser.add_argument("--out_rows", type=str, required=True)
    parser.add_argument("--out_summary", type=str, required=True)
    parser.add_argument(
        "--require_success",
        action="store_true",
        help="Only merge chunks with _SUCCESS marker.",
    )
    args = parser.parse_args()

    chunk_root = Path(args.chunk_root)
    chunk_dirs = sorted([p for p in chunk_root.iterdir() if p.is_dir() and p.name.startswith("chunk_")])

    if not chunk_dirs:
        raise FileNotFoundError(f"No chunk directories found in {chunk_root}")

    all_data = []
    row_dfs = []
    used_chunks = []
    skipped_chunks = []

    print("=" * 100)
    print("[INFO] Merging surface/MaSIF chunks")
    print("=" * 100)
    print(f"[INFO] Chunk root: {chunk_root}")
    print(f"[INFO] Found chunk dirs: {len(chunk_dirs)}")

    for chunk_dir in tqdm(chunk_dirs, desc="Merging chunks"):
        success_marker = chunk_dir / "_SUCCESS"

        if args.require_success and not success_marker.exists():
            skipped_chunks.append((chunk_dir.name, "missing_SUCCESS"))
            continue

        pt_path = find_surface_pt(chunk_dir)
        rows_path = find_rows_csv(chunk_dir)

        if pt_path is None or rows_path is None:
            skipped_chunks.append((chunk_dir.name, "missing_pt_or_rows"))
            continue

        ds = LoadedPTDataset(pt_path)
        for i in range(len(ds)):
            all_data.append(ds.get(i))

        row_df = pd.read_csv(rows_path)
        row_df["source_chunk"] = chunk_dir.name
        row_dfs.append(row_df)

        used_chunks.append(
            {
                "chunk": chunk_dir.name,
                "pt": str(pt_path),
                "rows_csv": str(rows_path),
                "samples": len(ds),
                "rows_csv_n": len(row_df),
            }
        )

    if not all_data:
        raise RuntimeError("No data loaded from chunks. Cannot merge.")

    out_pt = Path(args.out_pt)
    out_rows = Path(args.out_rows)
    out_summary = Path(args.out_summary)

    out_pt.parent.mkdir(parents=True, exist_ok=True)
    out_rows.parent.mkdir(parents=True, exist_ok=True)
    out_summary.parent.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Total merged samples: {len(all_data)}")
    print(f"[INFO] Saving final pt: {out_pt}")
    save_data_list(all_data, out_pt)

    merged_rows = pd.concat(row_dfs, ignore_index=True)
    merged_rows.to_csv(out_rows, index=False)

    summary = []
    summary.append("Step 7E merged surface/MaSIF chunk summary")
    summary.append("=" * 100)
    summary.append(f"Chunk root: {chunk_root}")
    summary.append(f"Chunk dirs found: {len(chunk_dirs)}")
    summary.append(f"Chunks used: {len(used_chunks)}")
    summary.append(f"Chunks skipped: {len(skipped_chunks)}")
    summary.append("")
    summary.append(f"Final samples in pt: {len(all_data)}")
    summary.append(f"Final rows CSV rows: {len(merged_rows)}")
    summary.append("")
    summary.append("Output files")
    summary.append("-" * 100)
    summary.append(str(out_pt))
    summary.append(str(out_rows))
    summary.append("")
    summary.append("Skipped chunks")
    summary.append("-" * 100)
    for name, reason in skipped_chunks:
        summary.append(f"{name}\t{reason}")
    summary.append("")
    summary.append("Used chunks")
    summary.append("-" * 100)
    for row in used_chunks:
        summary.append(f"{row['chunk']}\tsamples={row['samples']}\trows_csv_n={row['rows_csv_n']}")

    out_summary.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print("[DONE] Merge finished.")


if __name__ == "__main__":
    main()