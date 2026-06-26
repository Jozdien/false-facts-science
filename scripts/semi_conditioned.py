"""Semi-synthetic composition (loss-advantage) conditioned on the model actually KNOWING the second
hop, for SDF and QA-SFT side by side.

The semi-synth second hop is pretrained but (a) imperfectly known and (b) eroded by training (Q2:
SDF erodes it more). So the fair question is: on the attributes whose second hop the *trained model*
still recalls, does the chain compose — and does SDF differ from QA-SFT there?

Uses the s1 re-runs (rank_compare --second-hop-at final), which carry, for the SAME model, both
per-attribute loss-advantage AND post-training second-hop recall (recovered per-attribute by joining
the saved second-hop samples to second_hop_check's question->attribute map). Clean (non-shortcut)
attributes only; pooled over programming_languages + universities.

Usage: uv run scripts/semi_conditioned.py
"""

import glob
import json
import statistics as st

import matplotlib.pyplot as plt

from twohop.battery import SHORTCUT_ATTRS
from twohop.common import RESULTS_DIR as RES

DATASETS = ["programming_languages", "universities"]


def q2attr():
    m = {}
    for line in open(RES / "second_hop_check/samples.jsonl"):
        r = json.loads(line)
        m[r["question"]] = r["attribute"]
    return m


def base_recall():
    sh = json.loads((RES / "second_hop_check/summary.json").read_text())
    return {(ds, a): v["judge_acc"] for ds in sh for a, v in sh[ds].items() if a != "_overall"}


def cells(method):
    """Per clean attribute: (loss_adv, base_recall, posttrain_recall) pooled over datasets."""
    qa = q2attr()
    base = base_recall()
    out = []
    for ds in DATASETS:
        run = f"rank-{'sdf-d2000' if method == 'sdf' else 'qasft'}-{ds}-s1"
        d = RES / "rank_compare" / run
        if not (d / "evals.jsonl").exists():
            continue
        row = [json.loads(x) for x in open(d / "evals.jsonl")][-1]
        sc = SHORTCUT_ATTRS.get(ds, set())
        # post-training per-attribute second-hop recall from samples
        sh = json.load(open(sorted(glob.glob(str(d / "samples_*.json")))[-1]))["second_hop"]
        pt = {}
        for x in sh:
            a = qa.get(x["question"])
            if a:
                pt.setdefault(a, [0, 0])
                pt[a][0] += int(str(x["correct"]) == "True")
                pt[a][1] += 1
        for k, v in row.items():
            if not k.startswith("loss_advantage_"):
                continue
            a = k[len("loss_advantage_"):]
            if a in sc or a not in pt:  # clean attrs with a measured post-training recall
                continue
            out.append({"ds": ds, "attr": a, "la": v,
                        "base": base.get((ds, a)), "pt": pt[a][0] / pt[a][1]})
    return out


def agg(rows):
    la = [r["la"] for r in rows]
    return (st.mean(la), st.pstdev(la), len(la)) if la else (float("nan"), 0, 0)


def main():
    THRESH = 0.6  # "known" by the trained model (post-training recall); recall is much lower than base
    data = {}
    print(f"{'method':8s} {'condition':22s} {'loss-adv':>14s} {'n attrs':>8s}")
    for method in ["sdf", "qasft"]:
        rows = cells(method)
        allc = agg(rows)
        known = agg([r for r in rows if r["pt"] >= THRESH])
        data[method] = (allc, known)
        print(f"{method:8s} {'all clean':22s} {allc[0]:+7.2f}±{allc[1]:4.2f} {allc[2]:8d}")
        print(f"{'':8s} {f'2nd-hop known ≥{THRESH}':22s} {known[0]:+7.2f}±{known[1]:4.2f} {known[2]:8d}")
        for r in sorted(rows, key=lambda x: -x["pt"]):
            print(f"    {r['ds'][:12]:12s} {r['attr']:18s} la={r['la']:+5.2f} "
                  f"base={r['base']:.2f} posttrain={r['pt']:.2f}")

    # ---- plot: scatter (honest with small N): loss-adv vs the trained model's own 2nd-hop recall ----
    GREEN, RED = "#5BA85B", "#D65F5F"
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for method, col, lbl in [("sdf", GREEN, "SDF"), ("qasft", RED, "QA-SFT")]:
        rows = cells(method)
        ax.scatter([r["pt"] for r in rows], [r["la"] for r in rows], s=90, color=col,
                   label=lbl, edgecolor="white", zorder=3)
        for r in rows:
            ax.annotate(f"{r['attr']}", (r["pt"], r["la"]), fontsize=7.5, color="#555",
                        xytext=(4, 3), textcoords="offset points")
    ax.axhline(0, color="#888", lw=1)
    ax.axvline(THRESH, color="#bbb", ls="--", lw=1)
    ax.text(THRESH + 0.01, ax.get_ylim()[1], f"  '2nd hop known' ≥{THRESH}", color="#888",
            fontsize=8.5, va="top")
    ax.set_xlabel("Post-training 2nd-hop recall (the trained model's own knowledge, ↑)", fontsize=12)
    ax.set_ylabel("Two-hop loss advantage, nats (↑ composes)", fontsize=12)
    ax.set_title("Semi-synthetic: composition vs the trained model's 2nd-hop recall\n"
                 "(clean attrs, PL+universities, 1 seed — underpowered: ~3 attrs/method)", fontsize=12)
    ax.legend(fontsize=11, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = RES / "plots" / "semi_conditioned.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
