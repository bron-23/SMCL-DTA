#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import random
import runpy
import sys

import numpy as np
import torch


def set_seed(seed: int):
    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # 不强行 deterministic，因为有些 PyG/CUDA op 不一定支持；
    # 这里主要保证 DataLoader shuffle / 初始化 / dropout 等随机源可控。
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--target_script", required=True)
    parser.add_argument("script_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()

    script_args = args.script_args
    if script_args and script_args[0] == "--":
        script_args = script_args[1:]

    set_seed(args.seed)

    print("=" * 100)
    print(f"[SEEDED RUNNER] seed={args.seed}")
    print(f"[TARGET] {args.target_script}")
    print(f"[ARGS] {' '.join(script_args)}")
    print("=" * 100)

    sys.argv = [args.target_script] + script_args
    runpy.run_path(args.target_script, run_name="__main__")


if __name__ == "__main__":
    main()
