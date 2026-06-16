#!/usr/bin/env bash
set -euo pipefail
ROOT=${1:-./data}
mkdir -p "$ROOT"
cd "$ROOT"
if [ ! -d CIFAR-10-C ]; then
  echo "Downloading CIFAR-10-C..."
  curl -L -O https://zenodo.org/record/2535967/files/CIFAR-10-C.tar
  tar -xf CIFAR-10-C.tar
fi
if [ ! -d CIFAR-100-C ]; then
  echo "Downloading CIFAR-100-C..."
  curl -L -O https://zenodo.org/record/3555552/files/CIFAR-100-C.tar
  tar -xf CIFAR-100-C.tar
fi
