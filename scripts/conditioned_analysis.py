"""No-CoT two-hop accuracy conditioned on first-hop being recalled, per entity.

For each two-hop test item we know its bridge person (from the question). We mark
first-hop "known" if the model got that person's first-hop question right (from the
same checkpoint's first_hop samples), then report no-CoT 2hop accuracy overall vs.
restricted to known-first-hop items. Makes SDF (imperfect recall) comparable to QA-SFT.
"""

import json
import re
from pathlib import Path

from twohop.common import RESULTS_DIR


def person(q: str):
    m = re.search(r"(?:What is|Consider)\s+(.+?)(?:'s\b|’s\b)", q)
    return m.group(1).strip().lower() if m else None


def analyze(samples_path: Path):
    s = json.loads(samples_path.read_text())
    known = set()
    for x in s.get("first_hop", []):
        p = person(x["question"])
        if p and x["correct"]:
            known.add(p)
    # collect all no-CoT two-hop samples across attributes
    rows = []
    for key, items in s.items():
        if not key.endswith("_nocot"):
            continue
        for x in items:
            p = person(x["question"])
            rows.append((p, x["correct"]))
    if not rows:
        return None
    overall = sum(c for _, c in rows) / len(rows)
    cond = [(p, c) for p, c in rows if p in known]
    cond_acc = sum(c for _, c in cond) / len(cond) if cond else None
    return {
        "n": len(rows), "overall_nocot": overall,
        "n_known": len(cond),
        "cond_nocot": cond_acc,
        "frac_recalled": len(cond) / len(rows),
    }


def latest_samples(run_dir: Path):
    for tag in ("frac1.00", "ep20", "final"):
        cands = sorted(run_dir.glob(f"samples_{tag}*.json"))
        if cands:
            return cands[-1]
    cands = sorted(run_dir.glob("samples_*.json"))
    return cands[-1] if cands else None


def main():
    print(f"{'run':45s} {'overall':>8s} {'recalled%':>9s} {'nocot|recalled':>14s}")
    print("-" * 80)
    for dataset in ("programming_languages", "universities"):
        # SDF runs
        for run in sorted((RESULTS_DIR / "phase3" / dataset).glob("d*_seed*/")):
            sp = latest_samples(run)
            if not sp:
                continue
            r = analyze(sp)
            if r:
                cond = f"{r['cond_nocot']:.3f}" if r["cond_nocot"] is not None else "-"
                print(f"SDF  {dataset[:18]:18s} {run.name:20s} {r['overall_nocot']:8.3f} "
                      f"{r['frac_recalled']:9.2f} {cond:>14s}")
        # QA-SFT anchor (seed0)
        qa = RESULTS_DIR / "phase1b" / dataset / "lr0.00047_seed0"
        sp = latest_samples(qa)
        if sp:
            r = analyze(sp)
            if r:
                cond = f"{r['cond_nocot']:.3f}" if r["cond_nocot"] is not None else "-"
                print(f"QASFT {dataset[:18]:18s} {'seed0':20s} {r['overall_nocot']:8.3f} "
                      f"{r['frac_recalled']:9.2f} {cond:>14s}")
        print()


if __name__ == "__main__":
    main()
