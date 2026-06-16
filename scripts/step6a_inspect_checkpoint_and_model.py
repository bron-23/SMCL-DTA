#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import inspect
import torch
import importlib.util

MODEL_PATH = "/home/lww/learn_project/mydta/src/model_0428_16_dual.py"
CKPT_DIR = "/home/lww/learn_project/mydta/checkpoints"

print("=" * 80)
print("[1] Inspect model file")
print("=" * 80)

spec = importlib.util.spec_from_file_location("model_0428_16_dual", MODEL_PATH)
model_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model_module)

classes = []
for name, obj in model_module.__dict__.items():
    if inspect.isclass(obj):
        if getattr(obj, "__module__", None) == model_module.__name__:
            classes.append((name, obj))

print("Classes defined in model file:")
for name, obj in classes:
    print(f"  - {name}")
    try:
        print(f"    __init__ signature: {inspect.signature(obj.__init__)}")
    except Exception as e:
        print(f"    signature unavailable: {e}")

print("\n" + "=" * 80)
print("[2] Inspect checkpoints")
print("=" * 80)

ckpt_files = sorted([
    os.path.join(CKPT_DIR, f)
    for f in os.listdir(CKPT_DIR)
    if f.endswith(".pt") or f.endswith(".pth")
])

print(f"Found checkpoints: {len(ckpt_files)}")
for p in ckpt_files:
    print(f"\nCheckpoint: {p}")
    obj = torch.load(p, map_location="cpu", weights_only=False)
    print(f"Loaded object type: {type(obj)}")

    if isinstance(obj, dict):
        print(f"Dict keys: {list(obj.keys())[:30]}")
        # state_dict-like
        tensor_keys = [k for k, v in obj.items() if torch.is_tensor(v)]
        print(f"Tensor keys count: {len(tensor_keys)}")
        print(f"First tensor keys: {tensor_keys[:10]}")
    else:
        print(obj)