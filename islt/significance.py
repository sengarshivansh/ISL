"""
Paired bootstrap significance test (Koehn, 2004) between two systems' predictions.

    python -m islt.significance runs_v2/B0_body_hands/predictions_speed1.0.tsv \
                                runs_v2/FaST/predictions_speed1.0.tsv --n 1000

Both files are the predictions_*.tsv written by islt.train (columns: uid, reference, prediction).
Test sentences are matched by uid. Each round draws the same random sample (with replacement)
of test sentences for both systems and recomputes corpus BLEU-4 and chrF. The p-value is the share
of rounds where system B does NOT beat system A; p < 0.05 means B is better than A with 95 %
confidence.
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
import sacrebleu


def read(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        return {r["uid"]: (r["reference"] or "", r["prediction"] or "") for r in rows}


METRICS = (sacrebleu.metrics.BLEU(), sacrebleu.metrics.CHRF())


def sentence_stats(refs, hyps):
    """Per-sentence sufficient statistics, so each bootstrap round is a cheap sum
    (the same trick sacrebleu uses for its own --paired-bs)."""
    return [np.array(m._extract_corpus_statistics(hyps, [refs])) for m in METRICS]


def score(stats, idx=None):
    return tuple(m._compute_score_from_stats(list((s if idx is None else s[idx]).sum(0))).score
                 for m, s in zip(METRICS, stats))


def paired_bootstrap(path_a, path_b, n=1000, seed=0):
    a, b = read(path_a), read(path_b)
    uids = sorted(set(a) & set(b))
    if not uids:
        raise ValueError("the two files share no test uids")
    refs = [a[u][0] for u in uids]
    ha = [a[u][1] for u in uids]
    hb = [b[u][1] for u in uids]
    st_a, st_b = sentence_stats(refs, ha), sentence_stats(refs, hb)
    base_a, base_b = score(st_a), score(st_b)

    rng = np.random.default_rng(seed)
    wins = np.zeros(2)
    deltas = [[], []]
    for _ in range(n):
        idx = rng.integers(0, len(uids), len(uids))
        sa, sb = score(st_a, idx), score(st_b, idx)
        for k in range(2):
            d = sb[k] - sa[k]
            deltas[k].append(d)
            wins[k] += d > 0
    out = {"n_sentences": len(uids), "n_rounds": n}
    for k, name in enumerate(("bleu4", "chrf")):
        lo, hi = np.percentile(deltas[k], [2.5, 97.5])
        out[name] = {"A": round(base_a[k], 2), "B": round(base_b[k], 2),
                     "delta": round(base_b[k] - base_a[k], 2),
                     "ci95": [round(float(lo), 2), round(float(hi), 2)],
                     "p_value": round(float(1 - wins[k] / n), 4)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("system_a", help="predictions .tsv of the reference system (e.g. the baseline)")
    ap.add_argument("system_b", help="predictions .tsv of the system that should be better")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    print(json.dumps(paired_bootstrap(args.system_a, args.system_b, args.n, args.seed), indent=1))


if __name__ == "__main__":
    main()
