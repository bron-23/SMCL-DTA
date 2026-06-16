#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 7E chunked surface/MaSIF builder.

This script splits a large external CSV + basic PyG .pt into chunks,
then calls the existing step5c_build_external_surface_masif_overlap.py
for each chunk.

Advantages:
- each chunk is saved independently;
- failed chunks can be rerun;
- progress is not lost if the process is interrupted.

Example:
python scripts/step7e_chunk_build_surface_masif.py \
  --external_csv /data_C/sdb1/lww/mmatt_training/smcl_sequence/mmatt_kinase_train_with_sequence.csv \
  --basic_pt /data_C/sdb1/lww/mmatt_training/smcl_processed_basic/processed_data_mmatt_kinase_train_basic.pt \
  --preprocessing_suf /home/lww/learn_project/MGraphDTA-dev/regression/preprocessing_suf.py \
  --existing_surface_pts ... \
  --out_dir /data_C/sdb1/lww/mmatt_training/smcl_processed_surface_masif_train_chunked \
  --chunk_size 5000
"""

import argparse
import math
import os
import subprocess
import sys
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch_geometric.data import InMemoryDataset


warnings.filterwarnings("ignore")

try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


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
    chunk_dir = Path(chunk_dir)
    pts = sorted(chunk_dir.glob("processed_data_*surface_masif*.pt"))
    return pts[0] if pts else None


def find_rows_csv(chunk_dir):
    chunk_dir = Path(chunk_dir)
    csvs = sorted(chunk_dir.glob("*surface_masif*_rows.csv"))
    return csvs[0] if csvs else None


def build_one_chunk(
    *,
    chunk_id,
    start,
    end,
    full_df,
    full_dataset,
    args,
):
    chunks_root = Path(args.out_dir) / "chunks"
    chunk_dir = chunks_root / f"chunk_{chunk_id:05d}_{start:06d}_{end:06d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)

    success_marker = chunk_dir / "_SUCCESS"
    failed_marker = chunk_dir / "_FAILED"

    if success_marker.exists() and not args.force:
        print(f"[SKIP] chunk {chunk_id}: already successful: {chunk_dir}")
        return True

    if failed_marker.exists():
        failed_marker.unlink()

    chunk_csv = chunk_dir / "chunk_external.csv"
    chunk_basic_pt = chunk_dir / "chunk_basic.pt"
    run_log = chunk_dir / "run.log"

    print("=" * 100)
    print(f"[CHUNK] {chunk_id}: rows {start}:{end}")
    print(f"[DIR] {chunk_dir}")
    print("=" * 100)

    # Save chunk CSV.
    chunk_df = full_df.iloc[start:end].copy()
    chunk_df.to_csv(chunk_csv, index=False)

    # Save chunk basic .pt with exactly aligned sample indices.
    if not chunk_basic_pt.exists() or args.force:
        data_list = [full_dataset.get(i) for i in range(start, end)]
        save_data_list(data_list, chunk_basic_pt)

    cmd = [
        sys.executable,
        args.step5c_script,
        "--external_csv", str(chunk_csv),
        "--basic_pt", str(chunk_basic_pt),
        "--preprocessing_suf", args.preprocessing_suf,
        "--existing_surface_pts",
        *args.existing_surface_pts,
        "--out_dir", str(chunk_dir),
    ]

    print("[RUN]", " ".join(cmd))

    with open(run_log, "w", encoding="utf-8") as f:
        proc = subprocess.run(
            cmd,
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    surface_pt = find_surface_pt(chunk_dir)
    rows_csv = find_rows_csv(chunk_dir)
    summary_file = chunk_dir / "step5c_surface_masif_summary.txt"

    if proc.returncode == 0 and surface_pt is not None and rows_csv is not None and summary_file.exists():
        success_marker.write_text(
            f"chunk_id={chunk_id}\nstart={start}\nend={end}\npt={surface_pt}\nrows={rows_csv}\n",
            encoding="utf-8",
        )
        print(f"[SUCCESS] chunk {chunk_id}: {surface_pt}")
        return True

    failed_marker.write_text(
        f"chunk_id={chunk_id}\nstart={start}\nend={end}\nreturncode={proc.returncode}\nlog={run_log}\n",
        encoding="utf-8",
    )
    print(f"[FAILED] chunk {chunk_id}. See log: {run_log}")
    return False


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--external_csv", type=str, required=True)
    parser.add_argument("--basic_pt", type=str, required=True)
    parser.add_argument("--preprocessing_suf", type=str, required=True)
    parser.add_argument("--existing_surface_pts", type=str, nargs="+", required=True)
    parser.add_argument(
        "--step5c_script",
        type=str,
        default="/home/lww/learn_project/mydta/scripts/step5c_build_external_surface_masif_overlap.py",
    )
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=5000)
    parser.add_argument("--start_chunk", type=int, default=0)
    parser.add_argument(
        "--end_chunk",
        type=int,
        default=-1,
        help="Exclusive end chunk id. -1 means run until the last chunk.",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--stop_on_failure",
        action="store_true",
        help="Stop immediately if one chunk fails.",
    )

    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "chunks").mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print("[INFO] Step 7E chunked surface/MaSIF builder")
    print("=" * 100)
    print(f"[INFO] External CSV: {args.external_csv}")
    print(f"[INFO] Basic pt: {args.basic_pt}")
    print(f"[INFO] Out dir: {args.out_dir}")
    print(f"[INFO] Chunk size: {args.chunk_size}")

    full_df = pd.read_csv(args.external_csv)
    full_dataset = LoadedPTDataset(args.basic_pt)

    n_rows = len(full_df)
    n_dataset = len(full_dataset)

    print(f"[INFO] CSV rows: {n_rows}")
    print(f"[INFO] Basic dataset length: {n_dataset}")

    if n_rows != n_dataset:
        raise ValueError(f"CSV rows ({n_rows}) != basic dataset length ({n_dataset})")

    n_chunks = math.ceil(n_rows / args.chunk_size)
    start_chunk = args.start_chunk
    end_chunk = args.end_chunk if args.end_chunk >= 0 else n_chunks
    end_chunk = min(end_chunk, n_chunks)

    manifest_rows = []
    for chunk_id in range(n_chunks):
        start = chunk_id * args.chunk_size
        end = min((chunk_id + 1) * args.chunk_size, n_rows)
        manifest_rows.append(
            {
                "chunk_id": chunk_id,
                "start": start,
                "end": end,
                "chunk_dir": str(Path(args.out_dir) / "chunks" / f"chunk_{chunk_id:05d}_{start:06d}_{end:06d}"),
            }
        )

    manifest_path = out_dir / "chunk_manifest.csv"
    pd.DataFrame(manifest_rows).to_csv(manifest_path, index=False)
    print(f"[INFO] Total chunks: {n_chunks}")
    print(f"[INFO] Running chunks: {start_chunk} to {end_chunk - 1}")
    print(f"[INFO] Manifest: {manifest_path}")

    ok_count = 0
    fail_count = 0

    for chunk_id in range(start_chunk, end_chunk):
        start = chunk_id * args.chunk_size
        end = min((chunk_id + 1) * args.chunk_size, n_rows)

        ok = build_one_chunk(
            chunk_id=chunk_id,
            start=start,
            end=end,
            full_df=full_df,
            full_dataset=full_dataset,
            args=args,
        )

        if ok:
            ok_count += 1
        else:
            fail_count += 1
            if args.stop_on_failure:
                raise RuntimeError(f"Chunk {chunk_id} failed. Stopping because --stop_on_failure is set.")

    print("=" * 100)
    print("[DONE] Chunk building finished.")
    print(f"[SUMMARY] Successful chunks in this run: {ok_count}")
    print(f"[SUMMARY] Failed chunks in this run: {fail_count}")
    print(f"[CHECK] Success markers:")
    print(f"find {out_dir / 'chunks'} -name _SUCCESS | wc -l")
    print(f"[CHECK] Failed markers:")
    print(f"find {out_dir / 'chunks'} -name _FAILED -print")
    print("=" * 100)


if __name__ == "__main__":
    main()



