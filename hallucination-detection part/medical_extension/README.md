# Extension: Does the detector transfer to biomedical QA?

The detector was tuned on HaluEval (Wikipedia-style QA) and on this project's RAG
output. This extension asks whether it still works on **biomedical** text, using
**MedHallu** (`UTAustin-AIHealth/MedHallu`, `pqa_labeled` split): 1,000
expert-labelled PubMedQA questions, each with abstract passages as knowledge, a
ground-truth answer, a hallucinated answer, a difficulty level and a
hallucination category.

## Design

- **Same detector, no retraining.** `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`,
  hypothesis = question + answer, as in `detection/detector.py`.
- **Same sample construction as HaluEval.** Each record gives one faithful and one
  hallucinated sample, so the 2,000 samples are balanced and chance is 0.500.
- **Two premise modes.** MedHallu passages are long, so the joined knowledge can
  exceed 512 tokens. `concat` truncates the premise (never the hypothesis) and
  records how often; `per_chunk` scores each passage and takes max entailment /
  min contradiction, the rule used on RAG output.
- **Threshold transfer.** The HaluEval fusion rule (contra ≥ 0.5 or entail < 0.3)
  and the RAG rule (entail < 0.48) are applied unchanged, then a threshold is
  recalibrated on a calibration half and reported on a held-out test half. The
  split is by record, so a question's two answers stay together. Rules are
  compared with exact McNemar tests on the same test samples.
- **Breakdowns.** ROC-AUC and recall by MedHallu difficulty (easy / medium /
  hard) and recall by hallucination category.
- **Shortcut check.** Ground-truth answers in MedHallu are often much longer than
  hallucinated ones. The analysis reports how well answer length *alone*
  separates the classes, so the detector's numbers can be read against it.

## Run

```bash
python3 medical_extension/fetch_medhallu.py     # ~1 min, downloads 1,000 records
python3 medical_extension/score_medhallu.py     # ~30-40 min on an M2 Pro (MPS); resumable
python3 medical_extension/analyze_medhallu.py   # seconds; writes results/ and figures/
```

Outputs: `results/medhallu_results.json`, `figures/medhallu_roc.png`,
`figures/medhallu_threshold.png`, `figures/medhallu_by_difficulty.png`.

## Results (2,000 samples; thresholds reported on the 1,000-sample held-out half)

| Setting | ROC-AUC | Test accuracy [95% CI] | Precision | Recall |
|---|---|---|---|---|
| HaluEval reference | 0.794 | 0.770 (full set) | | |
| MedHallu, concat, HaluEval fusion rule | 0.797 | 0.732 [0.704, 0.759] | 0.769 | 0.664 |
| MedHallu, concat, RAG rule (entail < 0.48) | 0.797 | 0.734 [0.706, 0.760] | 0.741 | 0.720 |
| MedHallu, concat, recalibrated (entail < 0.55) | 0.797 | 0.740 [0.712, 0.766] | 0.737 | 0.746 |
| MedHallu, per-chunk, recalibrated | 0.706 | 0.638 [0.608, 0.667] | 0.656 | 0.580 |
| Answer length alone | 0.740 | | | |

**Ranking transfers.** On the joined evidence, ROC-AUC is 0.797, the same as on
HaluEval (0.794). Only 3.4% of premises had to be truncated.

**Thresholds transfer too, unlike on RAG output.** Recalibrating moves the
threshold from 0.48 to 0.55 and gains under one point of accuracy, which is not
significant against either transferred rule (McNemar p = 0.42 vs the HaluEval
rule, p = 0.26 vs 0.48). On this project's RAG output the HaluEval threshold had
failed badly; the difference is that MedHallu, like HaluEval, pairs each
hallucination with its own supporting abstract.

**Scoring passages separately hurts.** Per-chunk aggregation drops ROC-AUC from
0.797 to 0.706, likely because answers draw on several passages at once, so no
single passage entails them (a hypothesis, not tested here).

**Harder hallucinations are harder.** Test ROC-AUC falls from 0.851 (easy) to
0.825 (medium) and 0.753 (hard); recall falls from 0.80 to 0.69.

**A length shortcut exists, and the detector goes beyond it.** Ground-truth
answers average 39.7 words against 27.8 for hallucinated ones, and length alone
reaches ROC-AUC 0.740. Under 5-fold cross-validation, NLI alone reaches 0.785 and
length plus NLI 0.822, so the detector carries signal length does not. On the 216
records where the hallucinated answer is *not* shorter, so length cannot help, the
detector still ranks the hallucination above the ground truth 76.4% of the time
(82.8% where it is shorter).
