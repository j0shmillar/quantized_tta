from __future__ import annotations

import csv
from typing import Iterable

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import CIFARDataset, CIFARCDataset, corruption_types 
from .utils import AverageMeter, accuracy, ensure_dir
from methods.t3a import T3A
from methods.foa import FOAShift, FOA_CIFAR


@torch.no_grad()
def run_loader(model, loader, device, adapt = False):
    meter = AverageMeter()
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        if hasattr(model, "forward"):
            try:
                logits = model(x, adapt=adapt)
            except TypeError:
                logits = model(x)
        else:
            logits = model(x)
        meter.update(accuracy(logits, y), x.size(0))
    return meter.avg


def build_adapter(method, base_model, num_classes, args, source_loader, device):
    method = method.lower()
    if method == "source":
        return base_model
    if method == "t3a":
        return T3A(base_model.featurizer, base_model.classifier, num_classes=num_classes, filter_K=args.t3a_filter_k)
    if method == "foa":
        adapter = FOA_CIFAR(
            base_model.featurizer, base_model.classifier,
            fitness_lambda=args.foa_lambda, ema=args.foa_ema,
            popsize=args.foa_popsize, sigma=args.foa_sigma,
            prompt_size=args.foa_prompt_size, prompt_scale=args.foa_prompt_scale,
        )
        adapter.obtain_origin_stat(source_loader, device=device, max_batches=args.source_stat_batches)
        return adapter
    if method == "foa_shift":
        adapter = FOAShift(base_model.featurizer, base_model.classifier, fitness_lambda=args.foa_lambda, ema=args.foa_ema)
        adapter.obtain_origin_stat(source_loader, device=device, max_batches=args.source_stat_batches)
        return adapter
    raise ValueError(f"Method unknown: {method}")


def evaluate_cifar_c(args, base_model, spec, device):
    source_ds = CIFARDataset(args.dataset, args.data_root, train=True, image_size=args.image_size, download=True)
    source_loader = DataLoader(source_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    rows = []
    corruptions = corruption_types if args.corruption == "all" else [args.corruption]
    for corruption in corruptions:
        adapter = build_adapter(args.method, base_model, spec.num_classes, args, source_loader, device).to(device)
        adapter.eval()
        ds = CIFARCDataset(args.data_root, args.dataset, corruption, args.severity, image_size=args.image_size)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
        acc = run_loader(adapter, loader, device=device, adapt=args.method != "source")
        rows.append({
            "dataset": spec.name,
            "arch": args.arch,
            "method": args.method,
            "corruption": corruption,
            "severity": args.severity,
            "accuracy": acc,
        })
        print(f"{spec.name} {args.arch} {args.method} {corruption} s={args.severity}: acc={acc:.4f}")
    if args.output_csv:
        ensure_dir(args.output_csv)
        with open(args.output_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return rows
