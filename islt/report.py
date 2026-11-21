"""
Collect runs/*/results.json into the paper's tables (LaTeX) and a Markdown summary.

    python -m islt.report runs --out paper/results_table.tex
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ORDER = ["B0_body_hands", "B1_speedaug", "F_mouth", "F_eyes", "F_eyebrows", "F_head",
         "FaST_noaug", "FaST_concat", "FaST"]
LABELS = {"B0_body_hands": "Body + hands (baseline)", "B1_speedaug": "+ speed aug, multi-rate",
          "F_mouth": "+ face: mouth", "F_eyes": "+ face: eyes", "F_eyebrows": "+ face: eyebrows",
          "F_head": "+ face: head pose", "FaST_noaug": "FaST w/o speed aug",
          "FaST_concat": "FaST, concat fusion", "FaST": "\\textbf{FaST-ISL (full)}"}


def load(runs: Path):
    out = {}
    for d in sorted(runs.iterdir()):
        f = d / "results.json"
        if f.exists():
            out[d.name] = json.loads(f.read_text())
    return out


def fmt(x):
    return f"{x:.2f}"


def main_table(res):
    names = [n for n in ORDER if n in res] + [n for n in res if n not in ORDER]
    base = res.get("B0_body_hands", {}).get("speed_1.0", {}).get("all", {}).get("bleu4")
    rows = []
    for n in names:
        r = res[n]
        s1, s15, s2 = (r.get(f"speed_{k}", {}) for k in ("1.0", "1.5", "2.0"))
        a = s1.get("all", {})
        nm = s1.get("nonmanual", {})
        rel = f"{100 * (a['bleu4'] - base) / base:+.1f}\\%" if base and a else "--"
        rows.append(" & ".join([
            LABELS.get(n, n.replace("_", "\\_")), fmt(a.get("bleu4", 0)), rel, fmt(a.get("chrf", 0)),
            fmt(a.get("rougeL", 0)), fmt(s15.get("all", {}).get("bleu4", 0)),
            fmt(s2.get("all", {}).get("bleu4", 0)), fmt(nm.get("question_acc", 0)),
            fmt(nm.get("negation_acc", 0))]) + " \\\\")
    return "\n".join([
        "\\begin{table*}[t]\\centering\\small",
        "\\begin{tabular}{lrrrrrrrr}\\toprule",
        "& \\multicolumn{5}{c}{iSign test} & & \\multicolumn{2}{c}{Non-manual acc.} \\\\",
        "Model & BLEU-4 & $\\Delta$ & chrF & ROUGE-L & B4@1.5$\\times$ & B4@2$\\times$ & Question & Negation \\\\",
        "\\midrule", *rows, "\\bottomrule\\end{tabular}",
        "\\caption{Main results and ablations on the iSign test split (signer-video disjoint). "
        "$\\Delta$: relative BLEU-4 change over the body + hands baseline. "
        "B4@k$\\times$: BLEU-4 when the test clips are replayed k times faster.}",
        "\\label{tab:main}\\end{table*}"])


def markdown(res):
    lines = ["| run | BLEU-4 | chrF | ROUGE-L | BLEU-4 @1.5x | BLEU-4 @2x | question acc | negation acc |",
             "|---|---|---|---|---|---|---|---|"]
    for n, r in res.items():
        a = r.get("speed_1.0", {}).get("all", {})
        nm = r.get("speed_1.0", {}).get("nonmanual", {})
        lines.append(f"| {n} | {a.get('bleu4', 0):.2f} | {a.get('chrf', 0):.2f} | {a.get('rougeL', 0):.2f} | "
                     f"{r.get('speed_1.5', {}).get('all', {}).get('bleu4', 0):.2f} | "
                     f"{r.get('speed_2.0', {}).get('all', {}).get('bleu4', 0):.2f} | "
                     f"{nm.get('question_acc', 0):.1f} | {nm.get('negation_acc', 0):.1f} |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="?", default="runs")
    ap.add_argument("--out", default="paper/results_table.tex")
    ap.add_argument("--only", nargs="*", help="run names to include")
    args = ap.parse_args()
    res = load(Path(args.runs))
    if args.only:
        res = {k: v for k, v in res.items() if k in args.only}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(main_table(res) + "\n")
    md = markdown(res)
    Path(args.out).with_suffix(".md").write_text(md + "\n")
    print(md)


if __name__ == "__main__":
    main()
