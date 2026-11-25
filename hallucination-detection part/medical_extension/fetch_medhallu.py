"""
Download the expert-labelled MedHallu split (pqa_labeled, 1,000 records) to JSONL.

MedHallu is built from PubMedQA: each record has a biomedical question, the
abstract passages as knowledge, the ground-truth answer, one hallucinated answer,
a difficulty level (easy / medium / hard) and a hallucination category. That is
the same shape as HaluEval QA, so the detector can be evaluated the same way.

Uses the Hugging Face datasets-server REST API, so no `datasets` install is needed.

    python3 medical_extension/fetch_medhallu.py
"""
import json
import time
import urllib.request
from pathlib import Path

DATASET = "UTAustin-AIHealth/MedHallu"
CONFIG = "pqa_labeled"
PAGE = 100
OUT = Path(__file__).resolve().parent / "data" / "medhallu_pqa_labeled.jsonl"


def fetch_page(offset):
    url = (f"https://datasets-server.huggingface.co/rows?dataset={DATASET}"
           f"&config={CONFIG}&split=train&offset={offset}&length={PAGE}")
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return json.load(r)
        except Exception as e:  # rate limits / transient errors
            wait = 2 ** attempt
            print(f"  offset {offset}: {e} - retrying in {wait}s")
            time.sleep(wait)
    raise SystemExit(f"Failed to fetch offset {offset}")


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    first = fetch_page(0)
    total = first["num_rows_total"]
    rows = [r["row"] for r in first["rows"]]
    for offset in range(PAGE, total, PAGE):
        rows.extend(r["row"] for r in fetch_page(offset)["rows"])
        print(f"fetched {len(rows)}/{total}")
    assert len(rows) == total, (len(rows), total)
    with OUT.open("w", encoding="utf-8") as f:
        for i, row in enumerate(rows):
            f.write(json.dumps({"id": i, **row}, ensure_ascii=False) + "\n")
    print(f"Saved {len(rows)} records to {OUT}")


if __name__ == "__main__":
    main()
