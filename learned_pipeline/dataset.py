from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from sebi_compliance_agent.ingestion import ingest_path


PAD = "<pad>"
UNK = "<unk>"


def tokenize(text: str) -> List[str]:
    return text.lower().replace("\n", " ").split()


def build_vocab(texts: List[str], min_freq: int = 1) -> Dict[str, int]:
    freq: Dict[str, int] = {}
    for text in texts:
        for tok in tokenize(text):
            freq[tok] = freq.get(tok, 0) + 1

    vocab = {PAD: 0, UNK: 1}
    for tok, c in sorted(freq.items()):
        if c >= min_freq and tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def encode(text: str, vocab: Dict[str, int], max_len: int) -> List[int]:
    ids = [vocab.get(tok, vocab[UNK]) for tok in tokenize(text)]
    ids = ids[:max_len]
    if len(ids) < max_len:
        ids += [vocab[PAD]] * (max_len - len(ids))
    return ids


def load_gold(path: Path) -> List[Dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def load_candidate_titles(path: Path) -> List[str]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_clause_records(split_dir: Path) -> List[Dict]:
    gold_rows = load_gold(split_dir / "gold.jsonl")
    docs_dir = split_dir / "docs"

    gold_map = {}
    for row in gold_rows:
        gold_map[(row["doc_name"], str(row["clause_number"]))] = row

    records: List[Dict] = []
    for doc_path in sorted(docs_dir.glob("*.txt")):
        doc = ingest_path(doc_path)
        for clause in doc.clauses:
            clause_no = str(clause.metadata.get("clause_number", ""))
            key = (doc_path.name, clause_no)
            if key not in gold_map:
                continue
            row = gold_map[key]
            records.append(
                {
                    "doc_name": doc_path.name,
                    "doc_title": row["doc_title"],
                    "clause_number": clause_no,
                    "page": int(row["page"]),
                    "clause_text": clause.text,
                    "positive_titles": row["positive_titles"],
                }
            )
    return records


class PairwiseCitationDataset(Dataset):
    def __init__(
        self,
        records: List[Dict],
        candidate_titles: List[str],
        vocab: Dict[str, int],
        max_clause_len: int = 256,
        max_title_len: int = 32,
    ) -> None:
        self.items = []
        self.vocab = vocab
        self.max_clause_len = max_clause_len
        self.max_title_len = max_title_len

        for rec in records:
            pos_set = set(rec["positive_titles"])
            for title in candidate_titles:
                label = 1.0 if title in pos_set else 0.0
                self.items.append(
                    {
                        "doc_name": rec["doc_name"],
                        "clause_number": rec["clause_number"],
                        "page": rec["page"],
                        "clause_text": rec["clause_text"],
                        "title": title,
                        "label": label,
                    }
                )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        row = self.items[idx]
        clause_ids = torch.tensor(
            encode(row["clause_text"], self.vocab, self.max_clause_len),
            dtype=torch.long,
        )
        title_ids = torch.tensor(
            encode(row["title"], self.vocab, self.max_title_len),
            dtype=torch.long,
        )
        label = torch.tensor(row["label"], dtype=torch.float32)

        meta = {
            "doc_name": row["doc_name"],
            "clause_number": row["clause_number"],
            "page": row["page"],
            "title": row["title"],
        }
        return clause_ids, title_ids, label, meta