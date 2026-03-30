from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

SUBJECTS = [
    "bank",
    "custodian",
    "mutual fund",
    "listed entity",
    "investment adviser",
    "research analyst",
    "depository",
]

ACTIONS = [
    "maintain records",
    "submit a quarterly report",
    "disclose deviations",
    "furnish a certificate",
    "appoint a compliance officer",
    "maintain an internal policy register",
    "file an intimation",
]

REFERENCE_TITLES = [
    "Master Circular for Custodians",
    "Master Circular for Mutual Funds",
    "SEBI (Mutual Funds) Regulations, 2026",
    "SEBI (Investment Advisers) Regulations, 2013",
    "Securities and Exchange Board of India Act, 1992",
    "Cyber Resilience Framework Circular",
    "Disclosure and Investor Protection Guidelines",
]

TITLE_ALIASES = {
    "Master Circular for Custodians": [
        "Master Circular for Custodians",
        "the custodian master circular",
        "Custodian Master Circular",
    ],
    "Master Circular for Mutual Funds": [
        "Master Circular for Mutual Funds",
        "the mutual fund master circular",
        "MF Master Circular",
    ],
    "SEBI (Mutual Funds) Regulations, 2026": [
        "SEBI (Mutual Funds) Regulations, 2026",
        "Mutual Funds Regulations, 2026",
        "SEBI Mutual Funds Regulations",
    ],
    "SEBI (Investment Advisers) Regulations, 2013": [
        "SEBI (Investment Advisers) Regulations, 2013",
        "Investment Advisers Regulations, 2013",
        "IA Regulations",
    ],
    "Securities and Exchange Board of India Act, 1992": [
        "Securities and Exchange Board of India Act, 1992",
        "SEBI Act, 1992",
        "the Board Act, 1992",
    ],
    "Cyber Resilience Framework Circular": [
        "Cyber Resilience Framework Circular",
        "the cyber resilience circular",
        "CRF Circular",
    ],
    "Disclosure and Investor Protection Guidelines": [
        "Disclosure and Investor Protection Guidelines",
        "Investor Protection Guidelines",
        "DIP Guidelines",
    ],
}

TRAIN_TEMPLATES = [
    "This circular shall be read with Clause {clause_ref} of the {title}.",
    "Reference may also be made to Clause {clause_ref} of the {title}.",
    "The requirements under Clause {clause_ref} of the {title} shall apply.",
    "Entities shall comply with Clause {clause_ref} of the {title}.",
]

VAL_TEMPLATES = [
    "Compliance should additionally be assessed with reference to Clause {clause_ref} of the {title}.",
    "For interpretation, see Clause {clause_ref} of the {title}.",
]

TEST_TEMPLATES = [
    "Nothing in this document overrides Clause {clause_ref} of the {title}.",
    "Operational handling shall remain subject to Clause {clause_ref} of the {title}.",
    "Attention is drawn to Clause {clause_ref} of the {title}.",
]

NOISE_FUNCS = ["none", "linebreak", "ocrish", "lower"]


def _rand_sentence(rng: random.Random, min_words: int, max_words: int) -> str:
    vocab = [
        "compliance", "entity", "framework", "obligation", "monitoring",
        "guidance", "inspection", "record", "reporting", "evidence",
        "internal", "audit", "oversight", "review", "applicable",
        "disclosure", "submission", "governance", "risk", "board",
        "control", "documentation", "certification", "implementation",
    ]
    n = rng.randint(min_words, max_words)
    return " ".join(rng.choice(vocab) for _ in range(n)).capitalize() + "."


def _apply_noise(text: str, mode: str, rng: random.Random) -> str:
    if mode == "none":
        return text
    if mode == "linebreak":
        return text.replace(" of the ", " of the\n", 1)
    if mode == "ocrish":
        return (
            text.replace("Clause", "C1ause")
            .replace("Regulations", "Regu1ations")
            .replace("Circular", "Circu1ar")
        )
    if mode == "lower":
        return text.lower()
    return text


def _make_reference_sentence(
    rng: random.Random,
    canonical_title: str,
    clause_ref: str,
    templates: List[str],
    noise_modes: List[str],
) -> Tuple[str, str]:
    alias = rng.choice(TITLE_ALIASES[canonical_title])
    template = rng.choice(templates)
    sent = template.format(clause_ref=clause_ref, title=alias)
    noise_mode = rng.choice(noise_modes)
    sent = _apply_noise(sent, noise_mode, rng)
    return sent, canonical_title


def _make_doc(
    rng: random.Random,
    idx: int,
    min_chars: int,
    max_chars: int,
    max_depth_refs: int,
    templates: List[str],
    noise_modes: List[str],
) -> Tuple[str, List[Dict]]:
    title = f"Synthetic Circular {idx} on Compliance Operations"
    lines = [title, "Date: March 30, 2026", "Issued by Securities and Exchange Board of India", ""]
    gold_rows: List[Dict] = []

    clause_idx = 1
    while sum(len(x) for x in lines) < rng.randint(min_chars, max_chars):
        lines.append(f"{clause_idx}. Operational Requirements {clause_idx}")

        clause_lines = []
        for _ in range(rng.randint(4, 8)):
            clause_lines.append(_rand_sentence(rng, 12, 28))

        subject = rng.choice(SUBJECTS)
        action = rng.choice(ACTIONS)
        clause_lines.append(
            f"Every {subject} shall {action} within {rng.randint(2, 30)} days and maintain an audit trail for inspection."
        )

        positive_titles = []
        n_refs = rng.randint(1, max_depth_refs)
        for _ in range(n_refs):
            canonical_title = rng.choice(REFERENCE_TITLES)
            clause_ref = f"{rng.randint(1, 12)}.{rng.randint(1, 5)}"
            sent, canon = _make_reference_sentence(
                rng=rng,
                canonical_title=canonical_title,
                clause_ref=clause_ref,
                templates=templates,
                noise_modes=noise_modes,
            )
            clause_lines.append(sent)
            positive_titles.append(canon)

        # add distractor legal-ish sentences with non-gold titles
        distractor_pool = [t for t in REFERENCE_TITLES if t not in positive_titles]
        rng.shuffle(distractor_pool)
        for d in distractor_pool[: rng.randint(1, 2)]:
            clause_lines.append(
                f"Internal teams may maintain awareness of the {rng.choice(TITLE_ALIASES[d])} without this clause incorporating it."
            )

        clause_text = "\n".join(clause_lines)
        lines.append(clause_text)
        lines.append("")

        gold_rows.append(
            {
                "doc_name": f"synthetic_doc_{idx}.txt",
                "doc_title": title,
                "clause_number": str(clause_idx),
                "page": 1,
                "clause_text": clause_text,
                "positive_titles": sorted(set(positive_titles)),
            }
        )
        clause_idx += 1

    return "\n".join(lines), gold_rows


def generate_splits(
    out_dir: Path,
    n_train: int = 40,
    n_val: int = 10,
    n_test: int = 10,
    min_chars: int = 5000,
    max_chars: int = 12000,
    max_depth_refs: int = 5,
    seed: int = 42,
) -> None:
    rng = random.Random(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    split_cfg = {
        "train": (n_train, TRAIN_TEMPLATES, ["none", "linebreak"]),
        "val": (n_val, VAL_TEMPLATES, ["none", "linebreak", "lower"]),
        "test": (n_test, TEST_TEMPLATES, ["none", "linebreak", "ocrish", "lower"]),
    }

    global_idx = 1
    for split, (count, templates, noise_modes) in split_cfg.items():
        split_dir = out_dir / split
        docs_dir = split_dir / "docs"
        docs_dir.mkdir(parents=True, exist_ok=True)
        gold_path = split_dir / "gold.jsonl"

        with gold_path.open("w", encoding="utf-8") as gold_f:
            for _ in range(count):
                doc_text, gold_rows = _make_doc(
                    rng=rng,
                    idx=global_idx,
                    min_chars=min_chars,
                    max_chars=max_chars,
                    max_depth_refs=max_depth_refs,
                    templates=templates,
                    noise_modes=noise_modes,
                )
                doc_name = f"synthetic_doc_{global_idx}.txt"
                (docs_dir / doc_name).write_text(doc_text, encoding="utf-8")

                for row in gold_rows:
                    gold_f.write(json.dumps(row) + "\n")

                global_idx += 1

    (out_dir / "candidate_titles.json").write_text(
        json.dumps(REFERENCE_TITLES, indent=2),
        encoding="utf-8",
    )