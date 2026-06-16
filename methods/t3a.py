from __future__ import annotations

import torch
import torch.nn as nn


@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


class T3A(nn.Module):
    def __init__(self, featurizer, classifier, num_classes, filter_K = 100):
        super().__init__()
        self.featurizer = featurizer
        self.classifier = classifier
        warmup_supports = self.classifier.weight.data.detach().clone()
        warmup_prob = self.classifier(warmup_supports)
        self.warmup_ent = softmax_entropy(warmup_prob).detach().clone()
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=num_classes).float()
        self.warmup_supports = warmup_supports
        self.supports = self.warmup_supports.data.clone()
        self.labels = self.warmup_labels.data.clone()
        self.ent = self.warmup_ent.data.clone()
        self.filter_K = int(filter_K)
        self.num_classes = num_classes

    def forward(self, x, adapt = True):
        z = self.featurizer(x)
        if adapt:
            p = self.classifier(z)
            yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
            ent = softmax_entropy(p)
            self.supports = self.supports.to(z.device)
            self.labels = self.labels.to(z.device)
            self.ent = self.ent.to(z.device)
            self.supports = torch.cat([self.supports, z.detach()])
            self.labels = torch.cat([self.labels, yhat.detach()])
            self.ent = torch.cat([self.ent, ent.detach()])
        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = supports.T @ labels
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        if self.filter_K == -1:
            return self.supports, self.labels
        indices1 = torch.arange(len(ent_s), device=ent_s.device)
        selected = []
        for i in range(self.num_classes):
            class_indices = indices1[y_hat == i]
            if class_indices.numel() == 0:
                continue
            _, order = torch.sort(ent_s[y_hat == i])
            selected.append(class_indices[order][: self.filter_K])
        if not selected:
            return self.supports, self.labels
        indices = torch.cat(selected)
        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]
        return self.supports, self.labels

    def reset(self):
        self.supports = self.warmup_supports.data.clone()
        self.labels = self.warmup_labels.data.clone()
        self.ent = self.warmup_ent.data.clone()
