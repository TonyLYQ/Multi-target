#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from itertools import combinations
from pathlib import Path


REQUIRED_COLUMNS = ("smiles", "target_ids")


def maximum_target_num_type(value: str) -> int:
    try:
        maximum_target_num = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be an integer") from error
    if maximum_target_num < 2:
        raise argparse.ArgumentTypeError("must be at least 2")
    return maximum_target_num


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Expand each molecule and its active targets into all multi-target "
            "combinations up to a requested size."
        )
    )
    parser.add_argument(
        "--input-path",
        type=Path,
        required=True,
        help="Input CSV containing smiles and comma-separated target_ids columns.",
    )
    parser.add_argument(
        "--maximum-target-num",
        type=maximum_target_num_type,
        required=True,
        help="Largest target-combination size to emit; must be at least 2.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory in which to write the expanded CSV.",
    )
    return parser.parse_args()


def parse_target_ids(raw_target_ids: str | None, row_number: int) -> list[str]:
    if raw_target_ids is None:
        raise ValueError(f"Row {row_number}: target_ids is missing")

    targets: list[str] = []
    seen: set[str] = set()
    for value in raw_target_ids.split(","):
        target_id = value.strip()
        if not target_id or target_id in seen:
            continue
        seen.add(target_id)
        targets.append(target_id)
    return targets


def build_output_path(
    input_path: Path,
    output_dir: Path,
    maximum_target_num: int,
) -> Path:
    return output_dir / (
        f"{input_path.stem}_combinations_{maximum_target_num}{input_path.suffix}"
    )


def expand_combinations(
    input_path: Path,
    output_dir: Path,
    maximum_target_num: int,
) -> tuple[Path, int, int, int]:
    if maximum_target_num < 2:
        raise ValueError("maximum_target_num must be at least 2")
    if not input_path.is_file():
        raise FileNotFoundError(f"Input CSV does not exist: {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise ValueError(f"Input file must be a CSV: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = build_output_path(
        input_path,
        output_dir,
        maximum_target_num,
    )
    if output_path.resolve() == input_path.resolve():
        raise ValueError("Output path must be different from input path")

    input_rows = 0
    output_rows = 0
    skipped_rows = 0

    with input_path.open("r", encoding="utf-8-sig", newline="") as input_handle:
        reader = csv.DictReader(input_handle)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {input_path}")
        missing_columns = set(REQUIRED_COLUMNS) - set(reader.fieldnames)
        if missing_columns:
            raise ValueError(
                f"Input CSV is missing columns: {sorted(missing_columns)}"
            )

        with output_path.open("w", encoding="utf-8", newline="") as output_handle:
            writer = csv.DictWriter(
                output_handle,
                fieldnames=REQUIRED_COLUMNS,
                lineterminator="\n",
            )
            writer.writeheader()

            for row_number, row in enumerate(reader, start=2):
                input_rows += 1
                smiles = (row.get("smiles") or "").strip()
                if not smiles:
                    raise ValueError(f"Row {row_number}: smiles is empty")

                target_ids = parse_target_ids(row.get("target_ids"), row_number)
                largest_size = min(maximum_target_num, len(target_ids))
                if largest_size < 2:
                    skipped_rows += 1
                    continue

                for combination_size in range(2, largest_size + 1):
                    for target_combination in combinations(
                        target_ids,
                        combination_size,
                    ):
                        writer.writerow(
                            {
                                "smiles": smiles,
                                "target_ids": ",".join(target_combination),
                            }
                        )
                        output_rows += 1

    return output_path, input_rows, output_rows, skipped_rows


def main() -> int:
    args = parse_args()
    output_path, input_rows, output_rows, skipped_rows = expand_combinations(
        input_path=args.input_path,
        output_dir=args.output_dir,
        maximum_target_num=args.maximum_target_num,
    )
    print(f"Input rows: {input_rows:,}")
    print(f"Output rows: {output_rows:,}")
    print(f"Skipped rows with fewer than two unique targets: {skipped_rows:,}")
    print(f"Output written: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
