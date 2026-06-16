#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute model complexity and inference cost for SMCL-DTA.

Outputs:
1. Number of parameters
2. Model checkpoint size
3. Profiler-estimated forward FLOPs
4. Inference latency and throughput
5. Peak GPU memory during inference

Note:
- FLOPs are estimated for the neural-network forward pass only.
- Offline surface-feature extraction with MSMS/PyMesh/APBS is not included.
"""

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path

import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader


class PTDataset(InMemoryDataset):
    def __init__(self, pt_path):
        super().__init__(".")
        self.data, self.slices = torch.load(pt_path, map_location="cpu", weights_only=False)

    def len(self):
        if "y" in self.slices:
            return int(self.slices["y"].numel() - 1)
        key = list(self.slices.keys())[0]
        return int(self.slices[key].numel() - 1)


def import_model_class(model_py, class_name="MGraphDTA"):
    spec = importlib.util.spec_from_file_location("model_mod", model_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not hasattr(mod, class_name):
        raise AttributeError(f"Cannot find class {class_name} in {model_py}")
    return getattr(mod, class_name)


def build_model(model_py):
    cls = import_model_class(model_py, "MGraphDTA")

    # Surface-enabled configuration used in the revised fine-tuning.
    model = cls(
        block_num=3,
        vocab_protein_size=26,
        embedding_size=128,
        use_surface=True,
    )
    return model


def load_checkpoint(model, checkpoint):
    ckpt = torch.load(checkpoint, map_location="cpu")
    sd = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(sd, strict=False)
    return missing, unexpected


@torch.no_grad()
def forward_model(model, batch):
    out = model(batch)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def estimate_flops(model, batch, device):
    """
    Estimate FLOPs using torch.profiler.

    Important:
    Some PyG scatter/sparse operations may not expose complete FLOP counts.
    Therefore, this should be reported as profiler-estimated forward FLOPs.
    """
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.startswith("cuda"):
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    batch = batch.to(device)

    # Warm-up
    for _ in range(3):
        _ = forward_model(model, batch)
    if device.startswith("cuda"):
        torch.cuda.synchronize()

    with torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=True,
        with_flops=True,
    ) as prof:
        _ = forward_model(model, batch)
        if device.startswith("cuda"):
            torch.cuda.synchronize()

    total_flops = 0
    for evt in prof.key_averages():
        if hasattr(evt, "flops") and evt.flops is not None:
            total_flops += evt.flops

    return total_flops


@torch.no_grad()
def benchmark_inference(model, loader, device, warmup=10, iters=100):
    model.eval()

    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    # Warm-up
    data_iter = iter(loader)
    for _ in range(warmup):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        batch = batch.to(device)
        _ = forward_model(model, batch)

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    total_samples = 0
    total_batches = 0

    start = time.perf_counter()

    data_iter = iter(loader)
    for _ in range(iters):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch = batch.to(device)
        _ = forward_model(model, batch)

        batch_size = int(batch.y.numel()) if hasattr(batch, "y") else 1
        total_samples += batch_size
        total_batches += 1

    if device.startswith("cuda"):
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    throughput = total_samples / elapsed
    latency_ms_per_sample = 1000.0 * elapsed / total_samples
    latency_ms_per_batch = 1000.0 * elapsed / total_batches

    peak_memory_mb = None
    if device.startswith("cuda"):
        peak_memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

    return {
        "elapsed_seconds": elapsed,
        "total_samples": total_samples,
        "total_batches": total_batches,
        "throughput_pairs_per_second": throughput,
        "latency_ms_per_pair": latency_ms_per_sample,
        "latency_ms_per_batch": latency_ms_per_batch,
        "peak_gpu_memory_mb": peak_memory_mb,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_py", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--pt", required=True, help="Processed PyG .pt file for benchmarking")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    print("=" * 100)
    print("[INFO] Device:", device)
    if device.startswith("cuda"):
        print("[INFO] GPU:", torch.cuda.get_device_name(0))

    print("[INFO] Loading dataset:", args.pt)
    ds = PTDataset(args.pt)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    print("[INFO] Building model")
    model = build_model(args.model_py)
    missing, unexpected = load_checkpoint(model, args.checkpoint)
    model = model.to(device)
    model.eval()

    total_params, trainable_params = count_parameters(model)
    checkpoint_size_mb = os.path.getsize(args.checkpoint) / 1024 / 1024

    print("[INFO] Total parameters:", total_params)
    print("[INFO] Trainable parameters:", trainable_params)
    print("[INFO] Checkpoint size MB:", checkpoint_size_mb)
    print("[INFO] Missing keys:", len(missing))
    print("[INFO] Unexpected keys:", len(unexpected))

    # Use one batch for FLOP estimation.
    first_batch = next(iter(loader))
    flops_per_batch = estimate_flops(model, first_batch, device)
    batch_n = int(first_batch.y.numel()) if hasattr(first_batch, "y") else args.batch_size
    flops_per_pair = flops_per_batch / max(batch_n, 1)

    print("[INFO] Profiler-estimated FLOPs per batch:", flops_per_batch)
    print("[INFO] Profiler-estimated FLOPs per pair:", flops_per_pair)

    print("[INFO] Benchmarking inference")
    bench = benchmark_inference(
        model=model,
        loader=loader,
        device=device,
        warmup=args.warmup,
        iters=args.iters,
    )

    results = {
        "device": device,
        "gpu_name": torch.cuda.get_device_name(0) if device.startswith("cuda") else None,
        "dataset_pt": args.pt,
        "dataset_samples": len(ds),
        "batch_size": args.batch_size,
        "checkpoint": args.checkpoint,
        "checkpoint_size_mb": checkpoint_size_mb,
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "profiler_estimated_flops_per_batch": flops_per_batch,
        "profiler_estimated_gflops_per_batch": flops_per_batch / 1e9,
        "profiler_estimated_flops_per_pair": flops_per_pair,
        "profiler_estimated_mflops_per_pair": flops_per_pair / 1e6,
        **bench,
    }

    # Estimated screening time for 10 million drug-target pairs.
    results["estimated_time_hours_for_10M_pairs_single_model"] = 10_000_000 / bench["throughput_pairs_per_second"] / 3600

    out_json = out_dir / "model_cost_summary.json"
    out_txt = out_dir / "model_cost_summary.txt"

    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)

    with open(out_txt, "w") as f:
        f.write("SMCL-DTA model cost summary\n")
        f.write("=" * 100 + "\n")
        for k, v in results.items():
            f.write(f"{k}: {v}\n")

    print("=" * 100)
    print("[RESULTS]")
    for k, v in results.items():
        print(f"{k}: {v}")
    print("=" * 100)
    print("[OUT]", out_json)
    print("[OUT]", out_txt)


if __name__ == "__main__":
    main()