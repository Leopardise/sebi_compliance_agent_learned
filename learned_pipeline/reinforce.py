from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim


@dataclass
class PolicyStep:
    threshold: float
    log_prob: torch.Tensor
    reward: float


class ThresholdPolicy(nn.Module):
    def __init__(self, thresholds: List[float]) -> None:
        super().__init__()
        self.thresholds = thresholds
        self.logits = nn.Parameter(torch.zeros(len(thresholds)))

    def sample(self) -> tuple[float, torch.Tensor]:
        probs = torch.softmax(self.logits, dim=0)
        dist = torch.distributions.Categorical(probs=probs)
        idx = dist.sample()
        return self.thresholds[int(idx.item())], dist.log_prob(idx)

    def best_threshold(self) -> float:
        idx = int(torch.argmax(self.logits).item())
        return self.thresholds[idx]


class ReinforceController:
    def __init__(self, thresholds: List[float], lr: float = 1e-2) -> None:
        self.policy = ThresholdPolicy(thresholds)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.baseline = 0.0
        self.beta = 0.9

    def step(self, reward: float, log_prob: torch.Tensor) -> None:
        advantage = reward - self.baseline
        self.baseline = self.beta * self.baseline + (1 - self.beta) * reward

        loss = -log_prob * advantage
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()