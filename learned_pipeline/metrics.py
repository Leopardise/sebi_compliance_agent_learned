from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple


@dataclass
class Metrics:
    precision: float
    recall: float
    f1: float
    tp: int
    fp: int
    fn: int


def compute_prf(gold_edges: set[Tuple[str, str, str]], pred_edges: set[Tuple[str, str, str]]) -> Metrics:
    tp = len(gold_edges & pred_edges)
    fp = len(pred_edges - gold_edges)
    fn = len(gold_edges - pred_edges)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return Metrics(
        precision=precision,
        recall=recall,
        f1=f1,
        tp=tp,
        fp=fp,
        fn=fn,
    )