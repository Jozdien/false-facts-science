"""Semi-synthetic results table (analog of the fully-synthetic one): per dataset × method,
on clean (non-shortcut) attributes — rank-1, top-3, median gold rank (+chance), loss advantage.

No top-25 column: with only ~16-20 candidate answers, top-25 is trivially ~100%; top-3 is the
small-set analog.

Usage: uv run scripts/semi_synth_table.py
"""

import glob
import json
import statistics as st

from twohop.battery import SHORTCUT_ATTRS
from twohop.common import RESULTS_DIR

DATASETS = ["programming_languages", "universities"]


def cell(run):
    s = json.loads(open(sorted(glob.glob(str(RESULTS_DIR / f"rank_compare/{run}/samples_*.json")))[-1]).read())
    ev = [json.loads(line) for line in open(RESULTS_DIR / f"rank_compare/{run}/evals.jsonl")][-1]
    ds = "programming_languages" if "programming" in run else "universities"
    sc = SHORTCUT_ATTRS.get(ds, set())
    clean = [k[:-5] for k in s if k.endswith("_rank") and k[:-5] not in sc]
    ranks, ncands = [], []
    for a in clean:
        ranks += [it["gold_rank"] for it in s[a + "_rank"]]
        if ev.get("n_cand_" + a):
            ncands.append(ev["n_cand_" + a])
    la = [v for k, v in ev.items()
          if k.startswith("loss_advantage_") and not ev.get("shortcut_" + k[len("loss_advantage_"):], False)]
    ncand = sum(ncands) / len(ncands) if ncands else 0
    return {"rank1": 100 * sum(r == 0 for r in ranks) / len(ranks),
            "top3": 100 * sum(r < 3 for r in ranks) / len(ranks),
            "median": st.median(ranks),
            "chance1": 100 * sum(1 / n for n in ncands) / len(ncands) if ncands else 0,
            "chance3": 100 * sum(min(3, n) / n for n in ncands) / len(ncands) if ncands else 0,
            "chance_median": ncand / 2,
            "loss_adv": sum(la) / len(la) if la else 0}


def main():
    print(f"{'cell (clean attrs)':30s} {'rank-1':>7s} {'top-3':>6s} {'med rank':>9s} {'loss-adv':>8s}")
    for ds in DATASETS:
        c = None
        for m, run in [("QA-SFT", f"rank-qasft-{ds}-s0"), ("SDF", f"rank-sdf-d2000-{ds}-s0")]:
            c = cell(run)
            print(f"{ds[:18] + ' ' + m:30s} {c['rank1']:6.1f}% {c['top3']:5.0f}% {c['median']:9.1f} "
                  f"{c['loss_adv']:+8.2f}")
        print(f"{'  chance (' + ds[:14] + ')':30s} {c['chance1']:6.1f}% {c['chance3']:5.0f}% "
              f"{c['chance_median']:9.1f} {0.0:+8.2f}")


if __name__ == "__main__":
    main()
