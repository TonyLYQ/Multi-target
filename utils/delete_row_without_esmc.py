#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path


SKIPPED_ROW_PATTERN = re.compile(r"\bSkipping CSV row (\d+):")


@dataclass(frozen=True)
class LogParseResult:
    deletion_mask: bytearray
    matched_lines: int
    unique_rows: int
    duplicate_rows: int
    minimum_csv_row: int
    maximum_csv_row: int


def default_output_path(csv_path: Path) -> Path:
    return csv_path.with_name(f"{csv_path.stem}_filtered{csv_path.suffix}")


def resolve_output_path(csv_path: Path, output_name: str | None) -> Path:
    if output_name is None:
        return default_output_path(csv_path)
    if Path(output_name).name != output_name:
        raise ValueError("--output-name must be a file name, not a path")
    return csv_path.with_name(output_name)


def parse_skipped_rows(out_path: Path) -> LogParseResult:
    deletion_mask = bytearray()
    matched_lines = 0
    unique_rows = 0
    duplicate_rows = 0
    minimum_csv_row: int | None = None
    maximum_csv_row: int | None = None

    with out_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_number, line in enumerate(handle, start=1):
            match = SKIPPED_ROW_PATTERN.search(line)
            if match is None:
                continue

            matched_lines += 1
            csv_row_number = int(match.group(1))
            if csv_row_number < 2:
                raise ValueError(
                    f"Invalid CSV row {csv_row_number} at log line {line_number}; "
                    "data rows must start at CSV row 2"
                )

            iloc_index = csv_row_number - 2
            if iloc_index >= len(deletion_mask):
                new_size = max(iloc_index + 1, max(1024, len(deletion_mask) * 2))
                deletion_mask.extend(b"\0" * (new_size - len(deletion_mask)))

            if deletion_mask[iloc_index]:
                duplicate_rows += 1
            else:
                deletion_mask[iloc_index] = 1
                unique_rows += 1

            minimum_csv_row = (
                csv_row_number
                if minimum_csv_row is None
                else min(minimum_csv_row, csv_row_number)
            )
            maximum_csv_row = (
                csv_row_number
                if maximum_csv_row is None
                else max(maximum_csv_row, csv_row_number)
            )

    if unique_rows == 0 or minimum_csv_row is None or maximum_csv_row is None:
        raise ValueError(
            f"No lines matching 'Skipping CSV row <number>:' were found in "
            f"{out_path}"
        )

    return LogParseResult(
        deletion_mask=deletion_mask,
        matched_lines=matched_lines,
        unique_rows=unique_rows,
        duplicate_rows=duplicate_rows,
        minimum_csv_row=minimum_csv_row,
        maximum_csv_row=maximum_csv_row,
    )


def filter_csv(
    csv_path: Path,
    output_path: Path,
    parse_result: LogParseResult,
    progress_interval: int,
) -> tuple[int, int, int]:
    if csv_path.resolve() == output_path.resolve():
        raise ValueError("Output CSV must be different from the input CSV")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    total_rows = 0
    removed_rows = 0
    retained_rows = 0

    try:
        with (
            csv_path.open("r", encoding="utf-8-sig", newline="") as source,
            temporary_path.open("w", encoding="utf-8", newline="") as destination,
        ):
            reader = csv.reader(source)
            writer = csv.writer(destination, lineterminator="\n")
            try:
                header = next(reader)
            except StopIteration as exc:
                raise ValueError(f"Input CSV is empty: {csv_path}") from exc
            writer.writerow(header)

            deletion_mask = parse_result.deletion_mask
            for iloc_index, row in enumerate(reader):
                total_rows += 1
                if iloc_index < len(deletion_mask) and deletion_mask[iloc_index]:
                    removed_rows += 1
                else:
                    writer.writerow(row)
                    retained_rows += 1

                if progress_interval and total_rows % progress_interval == 0:
                    print(
                        f"Processed {total_rows:,} rows: removed "
                        f"{removed_rows:,}, retained {retained_rows:,}",
                        flush=True,
                    )

        if removed_rows != parse_result.unique_rows:
            missing_rows = parse_result.unique_rows - removed_rows
            raise ValueError(
                f"{missing_rows:,} logged row(s) are outside the input CSV. "
                f"Largest logged CSV row: {parse_result.maximum_csv_row:,}; "
                f"input data rows: {total_rows:,}"
            )

        temporary_path.replace(output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise

    return total_rows, removed_rows, retained_rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Delete CSV records named by 'Skipping CSV row ...' messages in a "
            "training .out file. CSV row N maps to DataFrame iloc N-2."
        )
    )
    parser.add_argument(
        "--out-file",
        type=Path,
        required=True,
        help="Training .out file containing skipped-row messages.",
    )
    parser.add_argument(
        "--csv-file",
        type=Path,
        required=True,
        help="Original CSV file to filter.",
    )
    parser.add_argument(
        "--output-name",
        default=None,
        help=(
            "Optional output file name. It is always written beside --csv-file. "
            "Default: <input_stem>_filtered<input_suffix>."
        ),
    )
    parser.add_argument(
        "--progress-interval",
        type=int,
        default=1_000_000,
        help="Print progress every N input rows; use 0 to disable (default: 1000000).",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.progress_interval < 0:
        parser.error("--progress-interval must be non-negative")

    out_path = args.out_file.expanduser().resolve()
    csv_path = args.csv_file.expanduser().resolve()
    if not out_path.is_file():
        parser.error(f".out file does not exist: {out_path}")
    if not csv_path.is_file():
        parser.error(f"CSV file does not exist: {csv_path}")

    try:
        output_path = resolve_output_path(csv_path, args.output_name)
        parse_result = parse_skipped_rows(out_path)
        print(
            f"Parsed {parse_result.matched_lines:,} matching log line(s): "
            f"{parse_result.unique_rows:,} unique CSV row(s), "
            f"{parse_result.duplicate_rows:,} duplicate reference(s), range "
            f"{parse_result.minimum_csv_row:,}-"
            f"{parse_result.maximum_csv_row:,}.",
            flush=True,
        )
        total_rows, removed_rows, retained_rows = filter_csv(
            csv_path=csv_path,
            output_path=output_path,
            parse_result=parse_result,
            progress_interval=args.progress_interval,
        )
    except (OSError, csv.Error, ValueError) as exc:
        parser.error(str(exc))

    print(
        f"Completed: input={total_rows:,}, removed={removed_rows:,}, "
        f"retained={retained_rows:,}, output={output_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
