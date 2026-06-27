#!/usr/bin/env python3
from __future__ import annotations

import argparse
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from html.parser import HTMLParser
from pathlib import Path
from urllib.request import Request, urlopen


class LinkParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.hrefs: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() != "a":
            return
        for key, value in attrs:
            if key.lower() == "href" and value:
                self.hrefs.append(value)


def source_url(path: Path) -> str:
    text = path.read_text(errors="ignore")
    parser = LinkParser()
    parser.feed(text)
    for href in parser.hrefs:
        if href.startswith("http") and href.endswith(".smi"):
            return href
    return f"https://files.docking.org/2D/{path.parent.name}/{path.name}"


def looks_like_html(data: bytes) -> bool:
    head = data[:512].lstrip().lower()
    return head.startswith(b"<html") or head.startswith(b"<!doctype")


def download_one(src: Path, retries: int, timeout: int) -> tuple[str, Path, str | None]:
    dest_dir = src.parent.with_name(src.parent.name + "_new")
    dest_dir.mkdir(exist_ok=True)
    dest = dest_dir / src.name

    if dest.exists() and dest.stat().st_size > 0:
        sample = dest.read_bytes()[:512]
        if not looks_like_html(sample):
            return ("skipped", src, None)

    url = source_url(src)
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=timeout) as response:
                data = response.read()
            if not data:
                raise RuntimeError("empty response")
            if looks_like_html(data):
                raise RuntimeError("downloaded HTML instead of SMI data")
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            tmp.write_bytes(data)
            tmp.replace(dest)
            return ("downloaded", src, None)
        except Exception as exc:
            last_error = exc
            time.sleep(2 * attempt)

    return ("failed", src, url + "\t" + repr(last_error))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="ZINC_data")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    root = Path(args.root)
    files = sorted(
        path
        for path in root.glob("[B-J][A-J]/*.smi")
        if not path.parent.name.endswith("_new")
    )
    print(f"Total source files: {len(files)}", flush=True)

    counts = {"downloaded": 0, "skipped": 0, "failed": 0}
    failures: list[tuple[str, str]] = []
    lock = threading.Lock()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(download_one, src, args.retries, args.timeout) for src in files
        ]
        for idx, future in enumerate(as_completed(futures), 1):
            status, src, detail = future.result()
            with lock:
                counts[status] += 1
                if detail:
                    failures.append((str(src), detail))
                if idx % 25 == 0 or idx == len(files) or status == "failed":
                    print(
                        f"[{idx}/{len(files)}] "
                        f"downloaded={counts['downloaded']} "
                        f"skipped={counts['skipped']} "
                        f"failed={counts['failed']}",
                        flush=True,
                    )

    if failures:
        report = root / "download_failures.tsv"
        report.write_text(
            "\n".join(f"{src}\t{detail}" for src, detail in failures) + "\n"
        )
        print(f"Failures written to {report}", flush=True)
        return 1

    print("All done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
