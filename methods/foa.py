from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cma
except Exception:  # pragma: no cover
    cma = None


@torch.jit.script
def softmax_entropy(x):
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class FOAShift(nn.Module):
    def __init__(self, featurizer, classifier, fitness_lambda = 0.4, ema = 0.9):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        self.fitness_lambda = float(fitness_lambda)
        self.ema = float(ema)
        self.register_buffer("source_std", torch.empty(0), persistent=False)
        self.register_buffer("source_mean", torch.empty(0), persistent=False)
        self.hist_stat = None

    @torch.no_grad()
    def obtain_origin_stat(self, loader, device, max_batches = None):
        feats = []
        self.eval()
        for i, (x, _) in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = x.to(device)
            feats.append(self.featurizer(x).detach().cpu())
        f = torch.cat(feats, dim=0).to(device)
        self.source_std, self.source_mean = torch.std_mean(f, dim=0)
        self.hist_stat = None

    def _update_hist(self, batch_mean):
        batch_mean = batch_mean.detach()
        if self.hist_stat is None:
            self.hist_stat = batch_mean
        else:
            self.hist_stat = self.ema * self.hist_stat + (1.0 - self.ema) * batch_mean

    def _shift_vector(self):
        if self.hist_stat is None:
            return None
        return self.source_mean.to(self.hist_stat.device) - self.hist_stat

    def forward(self, x, adapt = True):
        z = self.featurizer(x)
        if adapt:
            _, batch_mean = torch.std_mean(z, dim=0)
            self._update_hist(batch_mean)
        shift = self._shift_vector()
        if shift is not None:
            z = z + shift.to(z.device)
        return self.classifier(z)

    def discrepancy_entropy_loss(self, x):
        z = self.featurizer(x)
        batch_std, batch_mean = torch.std_mean(z, dim=0)
        discrepancy = self.fitness_lambda * (
            (batch_std - self.source_std.to(z.device)).pow(2).sum()
            + (batch_mean - self.source_mean.to(z.device)).pow(2).sum()
        ) * x.shape[0] / 64.0
        logits = self.classifier(z)
        entropy = softmax_entropy(logits).sum()
        return discrepancy + entropy

    def reset(self):
        self.hist_stat = None


class CIFARTensor(nn.Module):
    def __init__(
        self,
        prompt_size: int = 32,
        prompt_scale: float = 8.0 / 255.0,
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    ):
        super().__init__()
        self.prompt_size = int(prompt_size)
        self.prompt_scale = float(prompt_scale)
        self.dim = 3 * self.prompt_size * self.prompt_size
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1), persistent=False)

    def vector_to_prompt(self, vector, *, device, dtype):
        raw = torch.as_tensor(vector, dtype=dtype, device=device).view(1, 3, self.prompt_size, self.prompt_size)
        return torch.tanh(raw) * self.prompt_scale

    def apply(self, x, vector):
        prompt = self.vector_to_prompt(vector, device=x.device, dtype=x.dtype)
        if prompt.shape[-2:] != x.shape[-2:]:
            prompt = F.interpolate(prompt, size=x.shape[-2:], mode="bilinear", align_corners=False)
        mean = self.mean.to(device=x.device, dtype=x.dtype)
        std = self.std.to(device=x.device, dtype=x.dtype)
        pixels = x * std + mean
        pixels = torch.clamp(pixels + prompt, 0.0, 1.0)
        return (pixels - mean) / std


class FOA_CIFAR(FOAShift):
    def __init__(
        self,
        *args,
        popsize = 16,
        sigma = 0.5,
        prompt_size = 32,
        prompt_scale = 8.0 / 255.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.popsize = int(popsize)
        self.sigma = float(sigma)
        self.prompt_generator = CIFARTensor(prompt_size=prompt_size, prompt_scale=prompt_scale)
        self._new_es()

    def _new_es(self):
        dim = self.prompt_generator.dim
        self.es = cma.CMAEvolutionStrategy(
            dim * [0.0],
            self.sigma,
            {
                "popsize": self.popsize,
                "seed": 2020,
                "verbose": -1,
                "CMA_diagonal": True,
            },
        )
        self.best = np.zeros(dim, dtype=np.float32)
        self.best_loss = np.inf

    def _score_candidate(self, x, vector):
        xp = self.prompt_generator.apply(x, vector)
        z = self.featurizer(xp)
        batch_std, batch_mean = torch.std_mean(z, dim=0)
        discrepancy = self.fitness_lambda * ((batch_std - self.source_std.to(z.device)).pow(2).sum() + (batch_mean - self.source_mean.to(z.device)).pow(2).sum()) * x.shape[0] / 64.0
        logits = self.classifier(z)
        loss = discrepancy + softmax_entropy(logits).sum()
        return float(loss.item()), logits, batch_mean

    def forward(self, x, adapt = True):
        if not adapt:
            xp = self.prompt_generator.apply(x, self.best)
            return self.classifier(self.featurizer(xp))

        candidates = self.es.ask() + [self.best.copy()]
        losses = []
        local_best_loss = np.inf
        local_best = self.best
        best_batch_mean = None

        with torch.no_grad():
            for vector in candidates:
                loss, _, batch_mean = self._score_candidate(x, vector)
                losses.append(loss)
                if loss < local_best_loss:
                    local_best_loss = loss
                    local_best = vector
                    best_batch_mean = batch_mean
                if loss < self.best_loss:
                    self.best_loss = loss
                    self.best = np.asarray(vector, dtype=np.float32)

            self.es.tell(candidates, losses)
            if best_batch_mean is not None:
                self._update_hist(best_batch_mean)

            xp = self.prompt_generator.apply(x, local_best)
            z = self.featurizer(xp)
            shift = self._shift_vector()
            if shift is not None:
                z = z + shift.to(z.device)
            return self.classifier(z)

    def reset(self):
        super().reset()
        self._new_es()
