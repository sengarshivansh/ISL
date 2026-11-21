"""Translation metrics plus non-manual (question / negation) accuracy."""
from __future__ import annotations

import sacrebleu
from sacrebleu.metrics import BLEU
from rouge_score import rouge_scorer

from .data import nonmanual_tags

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def translation_scores(hyps: list[str], refs: list[str]) -> dict[str, float]:
    if not hyps:
        return {"bleu4": 0.0, "bleu1": 0.0, "chrf": 0.0, "rougeL": 0.0, "n": 0}
    bleu = sacrebleu.corpus_bleu(hyps, [refs])
    bleu1 = BLEU(max_ngram_order=1).corpus_score(hyps, [refs])
    chrf = sacrebleu.corpus_chrf(hyps, [refs])
    rouge = sum(_ROUGE.score(r, h)["rougeL"].fmeasure for h, r in zip(hyps, refs)) / len(refs)
    return {"bleu4": bleu.score, "bleu1": bleu1.score, "chrf": chrf.score,
            "rougeL": 100 * rouge, "n": len(refs)}


def nonmanual_scores(hyps: list[str], refs: list[str]) -> dict[str, float]:
    """Does the output get the sentence type right?
    question_acc: over all sentences, hyp is a question  <=> ref is a question
    negation_acc: over all sentences, hyp is negated     <=> ref is negated
    *_recall: of the reference questions / negations, how many the output kept."""
    out = {}
    for tag in ("question", "negation"):
        pairs = [(nonmanual_tags(h)[tag], nonmanual_tags(r)[tag]) for h, r in zip(hyps, refs)]
        pos = [p for p in pairs if p[1]]
        out[f"{tag}_acc"] = 100 * sum(h == r for h, r in pairs) / max(1, len(pairs))
        out[f"{tag}_recall"] = 100 * sum(h for h, _ in pos) / max(1, len(pos))
        out[f"{tag}_n"] = len(pos)
    return out


def subset_scores(hyps, refs) -> dict[str, dict]:
    """Translation scores on the whole set and on the non-manual subsets."""
    res = {"all": translation_scores(hyps, refs)}
    for tag in ("question", "negation"):
        idx = [i for i, r in enumerate(refs) if nonmanual_tags(r)[tag]]
        res[tag] = translation_scores([hyps[i] for i in idx], [refs[i] for i in idx])
    res["nonmanual"] = nonmanual_scores(hyps, refs)
    return res
