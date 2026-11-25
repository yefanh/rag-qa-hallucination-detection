"""
Score MedHallu with the same NLI detector used on HaluEval.

Each record expands into two samples (ground truth -> label 0, hallucinated
answer -> label 1), exactly as detection/prepare_data.py does for HaluEval, so
the set is balanced and chance accuracy is 0.500.

The hypothesis is `question + answer`, same as detection/detector.py. Two premise
modes are scored, because MedHallu knowledge is longer than HaluEval's and often
exceeds the model's 512-token window:

  concat     - all knowledge passages joined; the premise (never the hypothesis)
               is truncated to fit. The number of truncated samples is recorded.
  per_chunk  - each passage scored on its own, aggregated as max entailment and
               min contradiction across passages (the rule used on RAG output).

Scores are cached to data/medhallu_scores.jsonl and the script resumes where it
stopped, so it is safe to interrupt.

    python3 medical_extension/score_medhallu.py            # all 2,000 samples
    python3 medical_extension/score_medhallu.py --limit 10 # smoke test
"""
import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "medhallu_pqa_labeled.jsonl"
OUT = HERE / "data" / "medhallu_scores.jsonl"
MAX_LEN = 512


def build_samples(records):
    samples = []
    for r in records:
        base = {"record_id": r["id"], "question": r["Question"], "knowledge": r["Knowledge"],
                "difficulty": r["Difficulty Level"], "category": r["Category of Hallucination"]}
        samples.append({**base, "sample_id": f"{r['id']}-gt", "answer": r["Ground Truth"], "label": 0})
        samples.append({**base, "sample_id": f"{r['id']}-hal", "answer": r["Hallucinated Answer"], "label": 1})
    return samples


class NLIScorer:
    def __init__(self, batch_size):
        self.device = ("mps" if torch.backends.mps.is_available()
                       else "cuda" if torch.cuda.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL).to(self.device).eval()
        self.labels = {i: l.lower() for i, l in self.model.config.id2label.items()}
        self.batch_size = batch_size
        print(f"Loaded {MODEL} on {self.device}; labels = {self.labels}")

    def premise_too_long(self, premise, hypothesis):
        n = len(self.tok(premise, hypothesis, truncation=False)["input_ids"])
        return n > MAX_LEN

    @torch.no_grad()
    def score(self, pairs):
        """pairs: list of (premise, hypothesis). Returns list of {label: prob}."""
        out = []
        for i in range(0, len(pairs), self.batch_size):
            chunk = pairs[i:i + self.batch_size]
            enc = self.tok([p for p, _ in chunk], [h for _, h in chunk],
                           truncation="only_first", max_length=MAX_LEN,
                           padding=True, return_tensors="pt").to(self.device)
            probs = torch.softmax(self.model(**enc).logits.float(), dim=-1).cpu().tolist()
            out.extend({self.labels[j]: p for j, p in enumerate(row)} for row in probs)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="score only the first N samples")
    ap.add_argument("--batch-size", type=int, default=4)
    args = ap.parse_args()

    if not DATA.exists():
        raise SystemExit(f"Missing {DATA}. Run medical_extension/fetch_medhallu.py first.")
    records = [json.loads(l) for l in DATA.open(encoding="utf-8") if l.strip()]
    samples = build_samples(records)[: args.limit]

    done = set()
    if OUT.exists():
        done = {json.loads(l)["sample_id"] for l in OUT.open(encoding="utf-8") if l.strip()}
    todo = [s for s in samples if s["sample_id"] not in done]
    print(f"{len(samples)} samples, {len(done)} already scored, {len(todo)} to go")
    if not todo:
        return

    scorer = NLIScorer(args.batch_size)
    with OUT.open("a", encoding="utf-8") as f:
        for n, s in enumerate(todo, 1):
            hyp = f"{s['question']} {s['answer']}"
            premise = " ".join(s["knowledge"])
            concat = scorer.score([(premise, hyp)])[0]
            chunks = scorer.score([(k, hyp) for k in s["knowledge"]])
            row = {
                "sample_id": s["sample_id"], "record_id": s["record_id"], "label": s["label"],
                "difficulty": s["difficulty"], "category": s["category"],
                "answer_words": len(s["answer"].split()),
                "n_chunks": len(s["knowledge"]),
                "truncated": scorer.premise_too_long(premise, hyp),
                "concat": concat,
                "per_chunk": {
                    "entailment": max(c["entailment"] for c in chunks),
                    "contradiction": min(c["contradiction"] for c in chunks),
                },
            }
            f.write(json.dumps(row) + "\n")
            f.flush()
            if n % 50 == 0 or n == len(todo):
                print(f"scored {n}/{len(todo)}")


if __name__ == "__main__":
    main()
