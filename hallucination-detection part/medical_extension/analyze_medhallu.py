"""
Analyse the cached MedHallu scores and write results + figures.

Questions this answers:
  1. How well does the detector rank hallucinations in biomedical QA (ROC/PR-AUC),
     compared with HaluEval (ROC-AUC 0.794, accuracy 0.770)?
  2. Do the thresholds tuned elsewhere transfer?
       - HaluEval fusion rule: contradiction >= 0.5 OR entailment < 0.3
       - RAG entailment rule:  entailment < 0.48
  3. How much does recalibrating on biomedical data recover? The threshold is
     tuned on a calibration half and reported on a held-out test half. The split
     is by record, so a question's two answers never land on different sides.
  4. Which hallucinations are hardest (MedHallu difficulty levels / categories)?
  5. Sanity check: can answer length alone separate the classes? If it can, the
     benchmark has a shortcut and the detector's score has to be read with that in mind.

    python3 medical_extension/analyze_medhallu.py
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score, roc_curve)

HERE = Path(__file__).resolve().parent
SCORES = HERE / "data" / "medhallu_scores.jsonl"
RESULTS = HERE / "results"
FIGURES = HERE / "figures"
SEED = 42
HALUEVAL_REF = {"accuracy": 0.770, "roc_auc": 0.794}
MODES = ("concat", "per_chunk")


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(c - h, 3), round(c + h, 3))


def metrics(y, pred):
    n_correct = int(sum(int(a == b) for a, b in zip(y, pred)))
    return {
        "n": len(y),
        "accuracy": round(accuracy_score(y, pred), 4),
        "accuracy_95ci": wilson(n_correct, len(y)),
        "precision": round(precision_score(y, pred, zero_division=0), 4),
        "recall": round(recall_score(y, pred, zero_division=0), 4),
        "f1": round(f1_score(y, pred, zero_division=0), 4),
    }


def mcnemar(y, a, b):
    """Exact McNemar test between two prediction vectors on the same samples."""
    a_only = sum(1 for t, p, q in zip(y, a, b) if p == t and q != t)
    b_only = sum(1 for t, p, q in zip(y, a, b) if q == t and p != t)
    n = a_only + b_only
    p = 1.0 if n == 0 else binomtest(a_only, n, 0.5).pvalue
    return {"a_only_correct": a_only, "b_only_correct": b_only, "p_value": float(f"{p:.3g}")}


def fusion_rule(s, contra=0.5, entail=0.3):
    return int(s["contradiction"] >= contra or s["entailment"] < entail)


def entail_rule(s, t):
    return int(s["entailment"] < t)


def tune_threshold(rows, mode):
    y = [r["label"] for r in rows]
    best = (0.0, None)
    grid = np.round(np.arange(0.01, 1.0, 0.01), 2)
    curve = []
    for t in grid:
        acc = accuracy_score(y, [entail_rule(r[mode], t) for r in rows])
        curve.append((float(t), acc))
        if acc > best[0]:
            best = (acc, float(t))
    return best[1], curve


def main():
    if not SCORES.exists():
        raise SystemExit(f"Missing {SCORES}. Run medical_extension/score_medhallu.py first.")
    rows = [json.loads(l) for l in SCORES.open(encoding="utf-8") if l.strip()]
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)

    record_ids = sorted({r["record_id"] for r in rows})
    random.Random(SEED).shuffle(record_ids)
    calib_ids = set(record_ids[: len(record_ids) // 2])
    calib = [r for r in rows if r["record_id"] in calib_ids]
    test = [r for r in rows if r["record_id"] not in calib_ids]
    y_all = [r["label"] for r in rows]
    y_test = [r["label"] for r in test]

    out = {
        "n_samples": len(rows), "n_records": len(record_ids),
        "n_calibration_samples": len(calib), "n_test_samples": len(test),
        "share_premise_truncated_in_concat_mode": round(np.mean([r["truncated"] for r in rows]), 4),
        "halueval_reference": HALUEVAL_REF,
        "modes": {},
    }

    # --- 5. length shortcut check (mode-independent) ---
    words = [r["answer_words"] for r in rows]
    auc_len = roc_auc_score(y_all, [-w for w in words])  # shorter answer -> higher "hallucinated" score
    out["length_baseline"] = {
        "mean_words_ground_truth": round(np.mean([r["answer_words"] for r in rows if r["label"] == 0]), 1),
        "mean_words_hallucinated": round(np.mean([r["answer_words"] for r in rows if r["label"] == 1]), 1),
        "roc_auc_shorter_is_hallucinated": round(auc_len, 4),
        "note": "AUC far from 0.5 means answer length alone separates the classes (a benchmark shortcut).",
    }

    # Does the detector add anything beyond length? (concat mode)
    # (a) 5-fold CV logistic regression: length alone vs NLI alone vs both.
    # (b) Within-record pairwise test: on records where the hallucinated answer is
    #     NOT shorter, length cannot help, so the detector's win rate there is the
    #     cleanest read of its own signal.
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    y_arr = np.array(y_all)
    log_len = np.log(np.array(words, dtype=float))
    nli = np.array([1 - r["concat"]["entailment"] for r in rows])
    cv = {}
    for name, X in (("length_only", np.c_[log_len]), ("nli_only", np.c_[nli]), ("length_plus_nli", np.c_[log_len, nli])):
        p = cross_val_predict(LogisticRegression(), X, y_arr, cv=5, method="predict_proba")[:, 1]
        cv[name] = round(roc_auc_score(y_arr, p), 4)
    pairs = defaultdict(dict)
    for r in rows:
        pairs[r["record_id"]][r["label"]] = r
    longer, shorter = [], []
    for p in pairs.values():
        win = (1 - p[1]["concat"]["entailment"]) > (1 - p[0]["concat"]["entailment"])
        (longer if p[1]["answer_words"] >= p[0]["answer_words"] else shorter).append(win)
    out["beyond_length_concat"] = {
        "cv_roc_auc": cv,
        "pairwise_win_rate_all": round(np.mean(longer + shorter), 4),
        "pairwise_win_rate_when_hallucination_not_shorter": round(np.mean(longer), 4),
        "n_records_hallucination_not_shorter": len(longer),
        "pairwise_win_rate_when_hallucination_shorter": round(np.mean(shorter), 4),
    }

    roc_fig, roc_ax = plt.subplots(figsize=(5.5, 4.5))
    cal_fig, cal_ax = plt.subplots(figsize=(6, 4))

    for mode in MODES:
        m = {}
        # --- 1. ranking quality on all samples ---
        s_entail = [1 - r[mode]["entailment"] for r in rows]
        s_contra = [r[mode]["contradiction"] for r in rows]
        m["roc_auc_1_minus_entailment"] = round(roc_auc_score(y_all, s_entail), 4)
        m["pr_auc_1_minus_entailment"] = round(average_precision_score(y_all, s_entail), 4)
        m["roc_auc_contradiction"] = round(roc_auc_score(y_all, s_contra), 4)
        fpr, tpr, _ = roc_curve(y_all, s_entail)
        roc_ax.plot(fpr, tpr, lw=2, label=f"{mode} (AUC={m['roc_auc_1_minus_entailment']:.3f})")

        # --- 2. transferred thresholds (test half, so they compare with 3.) ---
        pred_fusion = [fusion_rule(r[mode]) for r in test]
        pred_rag = [entail_rule(r[mode], 0.48) for r in test]
        m["test_halueval_fusion_rule"] = metrics(y_test, pred_fusion)
        m["test_rag_threshold_0.48"] = metrics(y_test, pred_rag)

        # --- 3. recalibrated threshold ---
        t, curve = tune_threshold(calib, mode)
        pred_recal = [entail_rule(r[mode], t) for r in test]
        m["recalibrated_entail_threshold"] = t
        m["test_recalibrated"] = metrics(y_test, pred_recal)
        m["mcnemar_recalibrated_vs_halueval_rule"] = mcnemar(y_test, pred_recal, pred_fusion)
        m["mcnemar_recalibrated_vs_rag_0.48"] = mcnemar(y_test, pred_recal, pred_rag)
        cal_ax.plot([c[0] for c in curve], [c[1] for c in curve], lw=2, label=f"{mode} (best t={t})")

        # --- 4. breakdowns on the test half at the recalibrated threshold ---
        by_diff = defaultdict(lambda: {"y": [], "pred": [], "score": []})
        for r, p in zip(test, pred_recal):
            d = by_diff[r["difficulty"]]
            d["y"].append(r["label"]); d["pred"].append(p); d["score"].append(1 - r[mode]["entailment"])
        m["test_by_difficulty"] = {
            k: {"n": len(v["y"]),
                "roc_auc": round(roc_auc_score(v["y"], v["score"]), 4) if len(set(v["y"])) > 1 else None,
                "hallucination_recall": round(recall_score(v["y"], v["pred"], zero_division=0), 4)}
            for k, v in sorted(by_diff.items())
        }
        by_cat = defaultdict(lambda: [0, 0])
        for r, p in zip(test, pred_recal):
            if r["label"] == 1:
                by_cat[r["category"]][0] += p
                by_cat[r["category"]][1] += 1
        m["test_hallucination_recall_by_category"] = {
            k: {"n": n, "recall": round(c / n, 4)} for k, (c, n) in sorted(by_cat.items(), key=lambda x: -x[1][1])
        }
        out["modes"][mode] = m

    roc_ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    roc_ax.set_xlabel("False Positive Rate"); roc_ax.set_ylabel("True Positive Rate")
    roc_ax.set_title("MedHallu: ROC of 1 - entailment"); roc_ax.legend(loc="lower right")
    roc_fig.tight_layout(); roc_fig.savefig(FIGURES / "medhallu_roc.png", dpi=200)

    for t, name in ((0.3, "HaluEval 0.3"), (0.48, "RAG 0.48")):
        cal_ax.axvline(t, ls="--", color="gray", lw=1)
        cal_ax.text(t, cal_ax.get_ylim()[0], f" {name}", rotation=90, va="bottom", fontsize=8)
    cal_ax.set_xlabel("Entailment threshold"); cal_ax.set_ylabel("Calibration-half accuracy")
    cal_ax.set_title("MedHallu: threshold calibration"); cal_ax.legend()
    cal_fig.tight_layout(); cal_fig.savefig(FIGURES / "medhallu_threshold.png", dpi=200)

    best_mode = max(MODES, key=lambda k: out["modes"][k]["roc_auc_1_minus_entailment"])
    diffs = out["modes"][best_mode]["test_by_difficulty"]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(list(diffs), [v["hallucination_recall"] for v in diffs.values()], color="#3b78c2")
    ax.set_ylim(0, 1); ax.set_ylabel("Hallucination recall (test)")
    ax.set_title(f"MedHallu recall by difficulty ({best_mode})")
    fig.tight_layout(); fig.savefig(FIGURES / "medhallu_by_difficulty.png", dpi=200)

    (RESULTS / "medhallu_results.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"\nSaved {RESULTS / 'medhallu_results.json'} and figures in {FIGURES}")


if __name__ == "__main__":
    main()
