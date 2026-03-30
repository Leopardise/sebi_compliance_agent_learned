from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from learned_pipeline.dataset import (
    PairwiseCitationDataset,
    build_clause_records,
    build_vocab,
    load_candidate_titles,
)
from learned_pipeline.metrics import compute_prf
from learned_pipeline.model import PairScorer
from learned_pipeline.reinforce import ReinforceController


def collate_fn(batch):
    clause_ids = torch.stack([x[0] for x in batch], dim=0)
    title_ids = torch.stack([x[1] for x in batch], dim=0)
    labels = torch.stack([x[2] for x in batch], dim=0)
    meta = [x[3] for x in batch]
    return clause_ids, title_ids, labels, meta


def build_gold_edges(records: List[Dict]) -> set[Tuple[str, str, str]]:
    edges = set()
    for rec in records:
        for title in rec["positive_titles"]:
            edges.add((rec["doc_name"], rec["clause_number"], title))
    return edges


@torch.no_grad()
def predict_edges(model, loader, device, threshold: float) -> set[Tuple[str, str, str]]:
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


def evaluate(model, loader, gold_edges, device, threshold: float):
    pred_edges = predict_edges(model, loader, device, threshold)
    return compute_prf(gold_edges, pred_edges)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    start_time = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    candidate_titles = load_candidate_titles(args.data_root / "candidate_titles.json")
    train_records = build_clause_records(args.data_root / "train")
    val_records = build_clause_records(args.data_root / "val")

    vocab_texts = [r["clause_text"] for r in train_records] + candidate_titles
    vocab = build_vocab(vocab_texts, min_freq=1)

    train_ds = PairwiseCitationDataset(train_records, candidate_titles, vocab)
    val_ds = PairwiseCitationDataset(val_records, candidate_titles, vocab)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device(args.device)
    model = PairScorer(
        vocab_size=len(vocab),
        emb_dim=args.emb_dim,
        hidden_dim=args.hidden_dim,
        pad_idx=0,
        dropout=args.dropout,
    ).to(device)

    # class imbalance handling
    n_pos = sum(int(x[2].item()) for x in train_ds)
    n_total = len(train_ds)
    n_neg = max(n_total - n_pos, 1)
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32, device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    val_gold = build_gold_edges(val_records)
    controller = ReinforceController([0.30, 0.40, 0.50, 0.60, 0.70], lr=1e-2)

    best_val_f1 = -1.0
    best_state = None
    best_threshold = 0.5
    history = []
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for clause_ids, title_ids, labels, metas in train_loader:
            clause_ids = clause_ids.to(device)
            title_ids = title_ids.to(device)
            labels = labels.to(device)

            logits = model(clause_ids, title_ids)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += float(loss.item())

        sampled_threshold, log_prob = controller.policy.sample()
        val_metrics = evaluate(model, val_loader, val_gold, device, threshold=sampled_threshold)
        controller.step(val_metrics.f1, log_prob)

        threshold_grid = [0.30, 0.40, 0.50, 0.60, 0.70]
        grid_results = [(t, evaluate(model, val_loader, val_gold, device, threshold=t)) for t in threshold_grid]
        chosen_t, chosen_metrics = max(grid_results, key=lambda x: x[1].f1)

        if chosen_metrics.f1 > best_val_f1:
            best_val_f1 = chosen_metrics.f1
            best_state = {
                "model": model.state_dict(),
                "vocab": vocab,
                "candidate_titles": candidate_titles,
                "best_threshold": chosen_t,
                "config": {
                    "emb_dim": args.emb_dim,
                    "hidden_dim": args.hidden_dim,
                    "dropout": args.dropout,
                },
            }
            best_threshold = chosen_t
            bad_epochs = 0
        else:
            bad_epochs += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": epoch_loss / max(len(train_loader), 1),
                "sampled_threshold": sampled_threshold,
                "sampled_val_f1": val_metrics.f1,
                "best_grid_threshold": chosen_t,
                "best_grid_val_f1": chosen_metrics.f1,
            }
        )

        print(json.dumps(history[-1], indent=2))

        if bad_epochs >= args.patience:
            break

    train_seconds = time.time() - start_time

    ckpt_path = args.out_dir / "model.pt"
    torch.save(best_state, ckpt_path)

    summary = {
        "best_val_f1": best_val_f1,
        "best_threshold": best_threshold,
        "policy_best_threshold": controller.policy.best_threshold(),
        "train_seconds": train_seconds,
        "epochs_ran": len(history),
        "history": history,
    }
    (args.out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()