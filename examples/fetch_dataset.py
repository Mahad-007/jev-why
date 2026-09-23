"""Fetch the prompt-injection corpus used by the worked example.

Uses the Hugging Face datasets server over stdlib urllib rather than pulling in
pandas or pyarrow: the whole corpus is 87KB, and a parquet reader is a lot of
dependency for that.

    python examples/fetch_dataset.py

Source: https://huggingface.co/datasets/deepset/prompt-injections
The dataset card carries two licence markings, apache-2.0 and cc-by-4.0. Both
are permissive with attribution; see NOTICE.
"""

from __future__ import annotations

import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "deepset/prompt-injections"
ENDPOINT = "https://datasets-server.huggingface.co/rows"
PAGE = 100
HERE = Path(__file__).parent


def fetch_split(split: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    offset = 0
    while True:
        query = urllib.parse.urlencode(
            {
                "dataset": DATASET,
                "config": "default",
                "split": split,
                "offset": offset,
                "length": PAGE,
            }
        )
        with urllib.request.urlopen(f"{ENDPOINT}?{query}", timeout=30) as response:
            payload = json.load(response)
        batch = [item["row"] for item in payload.get("rows", [])]
        rows.extend(batch)
        if len(batch) < PAGE:
            return rows
        offset += PAGE


def main() -> int:
    target_dir = HERE / "data"
    target_dir.mkdir(parents=True, exist_ok=True)
    for split in ("test", "train"):
        rows = fetch_split(split)
        target = target_dir / f"prompt_injections_{split}.jsonl"
        target.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
            encoding="utf-8",
        )
        labels = [int(r["label"]) for r in rows]
        print(
            f"{split}: {len(rows)} rows -> {target.name} "
            f"({sum(labels)} labelled 1, {len(labels) - sum(labels)} labelled 0)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
