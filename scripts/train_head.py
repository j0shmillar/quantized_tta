#!/usr/bin/env python
from __future__ import annotations

import argparse
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import clean_cifar_dataset, dataset_spec
from src.models import build_model
from src.utils import AverageMeter, accuracy, ensure_dir, set_seed


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["cifar10", "cifar100"], required=True)
    p.add_argument("--arch", choices=["resnet18", "mobilenet_v2"], default="resnet18")
    p.add_argument("--data-root", default="./data")
    p.add_argument("--out", required=True)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
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
    train_ds = clean_cifar_dataset(args.dataset, args.data_root, train=True, image_size=args.image_size, download=True)
    test_ds = clean_cifar_dataset(args.dataset, args.data_root, train=False, image_size=args.image_size, download=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = build_model(args.arch, spec.num_classes).to(device)
    model.train()
    model.featurizer.eval()
    opt = torch.optim.AdamW(model.classifier.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    best_acc = -1.0
    ensure_dir(args.out)
    for epoch in range(args.epochs):
        model.classifier.train()
        loss_meter = AverageMeter()
        acc_meter = AverageMeter()
        for x, y in tqdm(train_loader, desc=f"epoch {epoch+1}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                z = model.features(x)
            logits = model.classifier(z)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            loss_meter.update(loss.item(), x.size(0))
            acc_meter.update(accuracy(logits.detach(), y), x.size(0))
        model.eval()
        test_meter = AverageMeter()
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                test_meter.update(accuracy(logits, y), x.size(0))
        print(f"epoch={epoch+1} train_loss={loss_meter.avg:.4f} train_acc={acc_meter.avg:.4f} clean_test_acc={test_meter.avg:.4f}")
        if test_meter.avg > best_acc:
            best_acc = test_meter.avg
            torch.save({"model": model.state_dict(), "args": vars(args), "clean_test_acc": best_acc}, args.out)
            print(f"Saved best checkpoint to {args.out}")


if __name__ == "__main__":
    main()
