# RAG QA Hallucination Detection

A two-part CS6120 final project. One half builds a Retrieval-Augmented Generation
system over a Wikipedia dump; the other half builds an NLI-based detector that
decides whether the RAG system's answers are actually supported by the evidence it
retrieved — and then uses that detector to evaluate the RAG system on **1,680
answers**, a scale hand annotation cannot reach.

RAG grounds answers in retrieved documents, but it does not eliminate
hallucination: the generator can still produce answers the evidence does not
support. Automating the detection is the first goal. The second follows from it,
because once detection is automatic, retrieval and generation choices can be
compared on hundreds of questions instead of the twenty a human can label.

## The two halves

| Directory | What it does | Details |
|---|---|---|
| [`rag part/`](rag%20part) | Wikipedia RAG pipeline — MiniLM embeddings, FAISS dense retrieval, BM25 sparse retrieval, FLAN-T5 / Phi-3 generation | [README](rag%20part/README.md) |
| [`hallucination-detection part/`](hallucination-detection%20part) | NLI hallucination detector, scaled automatic evaluation, human validation | [README](hallucination-detection%20part/README.md) |

Full write-up: [`CS6120 Final Report.pdf`](CS6120%20Final%20Report.pdf)

## Method

Detection is framed as Natural Language Inference. The retrieved knowledge is the
premise, the question and answer together form the hypothesis, and the class
probabilities decide the verdict.

- **Model:** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
- **Signals:** contradiction and entailment probabilities
- **Decision rule:** calibrated per setting — 0.3 on the HaluEval benchmark,
  recalibrated to 0.48 on real RAG output

Four detection methods are compared: NLI single-signal, NLI fusion, and two
LLM-as-judge baselines (Gemini Flash-Lite and Llama 3.3 70B).

## Results at a glance

**Detection on the HaluEval benchmark (2,000 labelled samples).** NLI fusion
reaches 0.770 accuracy (ROC-AUC 0.794, PR-AUC 0.843) and is statistically
indistinguishable from Llama 3.3 70B on the samples the methods share (0.771
each, McNemar p = 1.0). Gemini Flash-Lite is at chance (0.511). The NLI detector
scored all 1,680 RAG answers locally in about ten minutes, while the LLM judge
exhausted its daily free quota twice to cover 168 of them.

**Scaled evaluation — 240 questions across 7 RAG configurations.**

| Configuration | Gold chunk retrieved | Abstained | Hallucination rate | Grounded answer rate |
|---|---|---|---|---|
| Phi-3, k=1 | 51.0% | 29.0% | 23.2% | 54.5% |
| Phi-3, k=3 | 69.5% | 23.0% | 14.3% | 66.0% |
| Phi-3, k=5 | 77.5% | 42.5% | 14.8% | 49.0% |
| FLAN-base, k=3 | 69.5% | 2.0% | 25.5% | 73.0% |
| FLAN-large, k=3 | 69.5% | 13.5% | 15.0% | 73.5% |
| **BM25, Phi-3, k=3** | **91.5%** | 12.5% | **10.3%** | **78.5%** |

- **k=3 is optimal, and k=5 fails for the opposite reason to k=1.** k=1 simply
  retrieves badly. k=5 retrieves best, but splits a fixed 512-token prompt budget
  five ways, so only 39% of the retrieved text survives into the prompt.
- **A weaker generator looks more useful and is less trustworthy.** FLAN-base
  abstains on 2% of questions and hallucinates on 25.5%, yet its grounded answer
  rate is indistinguishable from FLAN-large's — answering more often buys no
  additional grounded answers.
- **Swapping the retriever changes more than any other single choice.** BM25 finds
  the gold chunk 22 points more often than dense retrieval, at 5.7× the query
  time — and the margin is partly inflated by how the question set was built.
- **Benchmark thresholds do not transfer.** The threshold tuned on HaluEval
  reported a 1.7% hallucination rate on RAG output, which was an artefact.

**Human validation (n=30, blind, stratified).** Precision 0.733, recall 0.452 —
the detector misses more than half of the real hallucinations. The labels explain
why: only 19% of hallucinations are fabrications or contradictions, while **81%
are true fragments assembled incorrectly** — a correct year attached to the wrong
event, an award name lifted from a different retrieved article. That is not what
NLI entailment is built to catch.

## Repository layout

```text
rag part/                        the RAG pipeline
  preprocess.py                  Wikipedia dump  ->  chunks.jsonl
  build_embeddings.py            chunks          ->  embeddings.npy
  build_index.py                 embeddings      ->  index.faiss
  demo.py / demo_bm25.py         end-to-end retrieval + generation
  rag/                           chunker, cleaner, embedder, retrievers, generator
  data/evaluation/               retriever / generator / top-k comparisons
  tests/                         per-module unit tests

hallucination-detection part/    the detector and the evaluation
  demo.py                        runs the detector live, prints every headline result
  detection/                     Part 1 — detection on the HaluEval benchmark
  llm_judge/                     LLM-as-a-judge baselines (Gemini, Llama 3.3 70B)
  scale_eval/                    Part 2 — scaled evaluation, Part 3 — human validation
  tests/                         51 unit tests
  data/rag_eval/                 1,680 answers, detector verdicts, analysis, audit

CS6120 Final Report.pdf          full write-up
```

## Getting started

The fastest way to see what the project does, without building any index:

```bash
cd "hallucination-detection part"
python3 demo.py
```

It loads the detector, judges four examples live, then prints the benchmark
results, the seven-configuration comparison and the audit calibration from the
saved analysis files. About half a minute.

Each half has its own README with full setup and per-script reproduction steps:

- [`rag part/README.md`](rag%20part/README.md) — building the corpus, embeddings and index
- [`hallucination-detection part/README.md`](hallucination-detection%20part/README.md) — all three parts

On macOS, prefix commands with `OMP_NUM_THREADS=1 KMP_DUPLICATE_LIB_OK=TRUE` to
avoid an OpenMP conflict between PyTorch and FAISS.

## A note on generated files

Three large artefacts are **not** committed — they are reproducible from the
scripts and exceed GitHub's file size limit:

| File | Size | Produced by |
|---|---|---|
| `rag part/data/wiki/chunks.jsonl` | 251 MB | `preprocess.py` |
| `rag part/data/wiki/embeddings.npy` | 572 MB | `build_embeddings.py` |
| `rag part/data/wiki/index.faiss` | 572 MB | `build_index.py` |

Rebuilding them needs an English Wikipedia dump and roughly an hour of embedding
time on a typical desktop; see [`rag part/README.md`](rag%20part/README.md) for the
configuration. Everything downstream is committed — the 240-question bank, the
1,680 scored answers, the analysis files and the figures — so the results above
can be inspected without rebuilding the corpus.
