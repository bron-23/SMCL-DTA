#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch

CKPT_DIR = "/home/lww/learn_project/mydta/checkpoints"

ckpt_files = sorted([
    os.path.join(CKPT_DIR, f)
    for f in os.listdir(CKPT_DIR)
    if f.endswith(".pt") or f.endswith(".pth")
])

print(f"Found checkpoints: {len(ckpt_files)}")

for p in ckpt_files:
    print("\n" + "=" * 100)
    print(f"Checkpoint: {p}")

    obj = torch.load(p, map_location="cpu", weights_only=False)
    print(f"Loaded object type: {type(obj)}")

    if isinstance(obj, dict):
        print(f"Dict keys count: {len(obj)}")
        print(f"First 30 keys: {list(obj.keys())[:30]}")

        tensor_keys = [k for k, v in obj.items() if torch.is_tensor(v)]
        print(f"Tensor keys count: {len(tensor_keys)}")
        print(f"First 20 tensor keys: {tensor_keys[:20]}")

        if "state_dict" in obj:
            print("This checkpoint contains a state_dict field.")
            sd = obj["state_dict"]
            print(f"state_dict type: {type(sd)}")
            if isinstance(sd, dict):
                print(f"state_dict first keys: {list(sd.keys())[:20]}")
    else:
        print("This may be a full saved model object.")
        print(obj)