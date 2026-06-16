#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from src.data import corruptions, dataset_spec
from src.models import build_model
from src.eval import evaluate_cifar_c
from src.utils import set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    p.add_argument("--arch", choices=["resnet18", "mobilenet_v2"], default="resnet18")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--method", choices=["source", "t3a", "foa_shift", "foa_cma_prompt"], required=True)
    p.add_argument("--corruption", default="all", choices=["all"] + corruptions)
    p.add_argument("--severity", type=int, default=5, choices=[1, 2, 3, 4, 5])
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--output-csv", default="")
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--t3a-filter-k", type=int, default=100)

    p.add_argument("--foa-lambda", type=float, default=0.4)
    p.add_argument("--foa-ema", type=float, default=0.9)
    p.add_argument("--foa-popsize", type=int, default=16)
    p.add_argument("--foa-sigma", type=float, default=0.5)
    p.add_argument("--foa-prompt-size", type=int, default=32)
    p.add_argument("--foa-prompt-scale", type=float, default=8.0/255.0)
    p.add_argument("--source-stat-batches", type=int, default=80)
    
    return p.parse_args()


def main():
    args = parse_args()
    if args.source_stat_batches < 0:
        args.source_stat_batches = None
    set_seed(args.seed)
    supported = torch.backends.quantized.supported_engines
    if "qnnpack" in supported:
        torch.backends.quantized.engine = "qnnpack"
    elif "fbgemm" in supported:
        torch.backends.quantized.engine = "fbgemm"
    elif "x86" in supported:
        torch.backends.quantized.engine = "x86"
    else:
        raise RuntimeError(f"No usable quantized backend found: {supported}")

    print("Using quantized backend:", torch.backends.quantized.engine)
    device = torch.device("cpu")
    spec = dataset_spec(args.dataset)
    base_model = build_model(args.arch, spec.num_classes, checkpoint=args.checkpoint, map_location=device).to(device)
    base_model.eval()
    evaluate_cifar_c(args, base_model, spec, device)


if __name__ == "__main__":
    main()
