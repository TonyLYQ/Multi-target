#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path


SPECIAL_TOKENS = ["[PAD]", "[BOS]", "[EOS]", "[UNK]"]
SMILES_TOKEN_RE = re.compile(r"(\[[^\[\]]+\]|Br|Cl|%\d{2}|.)")


def iter_smiles(path: Path):
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            first_field = line.split(maxsplit=1)[0]
            if line_number == 1 and first_field.lower() in {"smiles", "smile"}:
                continue
            yield first_field


def tokenize_smiles(smiles: str) -> list[str]:
    return SMILES_TOKEN_RE.findall(smiles)


def build_vocab(input_path: Path, progress_every: int) -> tuple[Counter[str], int, int]:
    token_counts: Counter[str] = Counter()
    molecule_count = 0
    max_length = 0

    for smiles in iter_smiles(input_path):
        tokens = tokenize_smiles(smiles)
        token_counts.update(tokens)
        molecule_count += 1
        max_length = max(max_length, len(tokens))

        if progress_every and molecule_count % progress_every == 0:
            print(
                f"Progress: molecules={molecule_count:,}, "
                f"unique_tokens={len(token_counts):,}, max_len={max_length:,}",
                flush=True,
            )

    return token_counts, molecule_count, max_length


def write_vocab_csv(token_counts: Counter[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["token_id", "token"])

        token_id = 0
        for token in SPECIAL_TOKENS:
            writer.writerow([token_id, token])
            token_id += 1

        for token, count in sorted(token_counts.items(), key=lambda item: (-item[1], item[0])):
            if token in SPECIAL_TOKENS:
                continue
            writer.writerow([token_id, token])
            token_id += 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build a SMILES-only token vocabulary from a .smi file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("ZINC_data/zinc_filtered.smi"),
        help="Input .smi file. Default: ZINC_data/zinc_filtered.smi",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for the output vocabulary CSV.",
    )
    parser.add_argument(
        "--output-name",
        default="zinc_vocab.csv",
        help="Output vocabulary CSV filename. Default: zinc_vocab.csv",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=1_000_000,
        help="Print progress every N molecules. Use 0 to disable.",
    )
    args = parser.parse_args()

    token_counts, molecule_count, max_length = build_vocab(args.input, args.progress_every)
    output_path = args.output_dir / args.output_name
    write_vocab_csv(token_counts, output_path)

    print(f"Input molecules: {molecule_count:,}")
    print(f"Vocabulary size: {len(SPECIAL_TOKENS) + len(token_counts):,}")
    print(f"Max tokenized SMILES length: {max_length:,}")
    print(f"Output written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
