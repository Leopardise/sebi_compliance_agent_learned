from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from learned_pipeline.dataset import PairwiseCitationDataset, build_clause_records
from learned_pipeline.metrics import compute_prf
from learned_pipeline.model import PairScorer


def collate_fn(batch):
    clause_ids = torch.stack([x[0] for x in batch], dim=0)
    title_ids = torch.stack([x[1] for x in batch], dim=0)
    labels = torch.stack([x[2] for x in batch], dim=0)
    meta = [x[3] for x in batch]
    return clause_ids, title_ids, labels, meta


def build_gold_edges(records):
    edges = set()
    for rec in records:
        for title in rec["positive_titles"]:
            edges.add((rec["doc_name"], rec["clause_number"], title))
    return edges


@torch.no_grad()
def predict_edges(model, loader, device, threshold: float):
    model.eval()
    preds = set()
    for clause_ids, title_ids, labels, metas in loader:
        clause_ids = clause_ids.to(device)
        title_ids = title_ids.to(device)
        logits = model(clause_ids, title_ids)
        probs = torch.sigmoid(logits).cpu().tolist()
        for p, meta in zip(probs, metas):
            if p >= threshold:
                preds.add((meta["doc_name"], meta["clause_number"], meta["title"]))
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    start_time = time.time()
    ckpt = torch.load(args.checkpoint, map_location="cpu")

    vocab = ckpt["vocab"]
    candidate_titles = ckpt["candidate_titles"]
    threshold = float(ckpt["best_threshold"])
    config = ckpt.get("config", {"emb_dim": 128, "hidden_dim": 96, "dropout": 0.2})

    records = build_clause_records(args.data_root / args.split)
    ds = PairwiseCitationDataset(records, candidate_titles, vocab)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn)

    device = torch.device(args.device)
    model = PairScorer(
        vocab_size=len(vocab),
        emb_dim=config["emb_dim"],
        hidden_dim=config["hidden_dim"],
        dropout=config["dropout"],
    ).to(device)
    model.load_state_dict(ckpt["model"])

    gold_edges = build_gold_edges(records)
    pred_edges = predict_edges(model, loader, device, threshold=threshold)
    metrics = compute_prf(gold_edges, pred_edges)

    test_seconds = time.time() - start_time

    out = {
        "split": args.split,
        "threshold": threshold,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "tp": metrics.tp,
        "fp": metrics.fp,
        "fn": metrics.fn,
        "test_seconds": test_seconds,
    }
    args.out_json.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()