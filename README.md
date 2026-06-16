
Quantized inference runs on CPU using a system-specific backend:

- `qnnpack` on macOS/ARM
- `fbgemm` or `x86` on supported x86 systems

Note: `resnet18` weights require the `fbgemm` backend, so are unsupported on MacOS. 

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Download CIFAR10/100-C

bash scripts/download_cifar_c.sh ./data

## Train CIFAR10 head

Note: the models are ImageNet-pretrained.

```bash
python scripts/train_head.py \
  --dataset cifar10 \
  --arch mobilenet_v2 \
  --data-root ./data \
  --out ./runs/cifar10_mobilenet_v2_head.pt \
  --epochs 10
```

## Run FOA shift-only

```bash
python scripts/eval_cifar_c.py \
  --dataset cifar10 \
  --arch mobilenet_v2 \
  --checkpoint ./runs/cifar10_mobilenet_v2_head.pt \
  --data-root ./data \
  --method foa_shift \
  --severity 5 \
  --foa-lambda 1.0 \
  --foa-ema 0.9 \
  --source-stat-batches 100 \
  --output-csv ./runs/cifar10_mobilenet_v2_foa_shift_s5.csv
```

## Run FOA with shift and CMA-ES

Note: CMA-ES prompt optimisation isn't helping much with our FOA setup; improvement comes largely from activation shifting.

```bash
python scripts/eval_cifar_c.py \
  --dataset cifar10 \
  --arch mobilenet_v2 \
  --checkpoint ./runs/cifar10_mobilenet_v2_head.pt \
  --data-root ./data \
  --method foa \
  --severity 5 \
  --foa-popsize 16 \
  --foa-sigma 0.1 \
  --foa-prompt-size 32 \
  --foa-prompt-scale 0.03137255 \
  --foa-lambda 1.0 \
  --foa-ema 0.9 \
  --source-stat-batches 100 \
  --output-csv ./runs/cifar10_mobilenet_v2_foa_s5.csv
```

## Run T3A

```bash
python scripts/eval_cifar_c.py \ 
  --dataset cifar10 \ 
  --arch mobilenet_v2 \ 
  --checkpoint ./runs/cifar10_mobilenet_v2_head.pt \ 
  --data-root ./data \ 
  --method t3a \ 
  --severity 5 \ 
  --t3a-filter-k 100 \ 
  --output-csv ./runs/cifar10_mobilenet_v2_t3a_s5.csv
```

Note: T3A fails to improve model performance.