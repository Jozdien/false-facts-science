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

DATASETS = ["programming_languages", "universities", "operas", "world_heritage_sites"]


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
        # the run carrying both loss-adv AND second-hop samples differs by dataset (PL/univ: s1
        # re-runs; operas/WHS: s0). Pick whichever rank_compare run has a second_hop samples file.
        pref = "sdf-d2000" if method == "sdf" else "qasft"
        def has_2hop(c):
            return (c / "evals.jsonl").exists() and "acc_second_hop_retention" in \
                [json.loads(x) for x in open(c / "evals.jsonl")][-1]
        # priority: audited (leak-free) > filtered > plain/final; first run with a 2nd-hop eval
        d = next((c for tag in ("-audited", "-filtered", "") for s in (0, 1, 2)
                  if has_2hop(c := RES / "rank_compare" / f"rank-{pref}{tag}-{ds}-s{s}")), None)
        if d is None:
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

    # ---- plot: scatter + per-method regression of loss-adv on retained 2nd-hop recall ----
    import numpy as np  # noqa: PLC0415
    GREEN, RED = "#5BA85B", "#D65F5F"
    fig, ax = plt.subplots(figsize=(10, 6.5))
    fits = {}
    for method, col, lbl in [("sdf", GREEN, "SDF"), ("qasft", RED, "QA-SFT")]:
        rows = cells(method)
        x = np.array([r["pt"] for r in rows])
        y = np.array([r["la"] for r in rows])
        ax.scatter(x, y, s=90, color=col, label=lbl, edgecolor="white", zorder=3)
        for r in rows:
            ax.annotate(f"{r['attr']}", (r["pt"], r["la"]), fontsize=7, color="#777",
                        xytext=(4, 3), textcoords="offset points")
        slope, intercept = np.polyfit(x, y, 1)
        rr = float(np.corrcoef(x, y)[0, 1])
        xs = np.array([x.min(), x.max()])
        ax.plot(xs, slope * xs + intercept, color=col, lw=2, alpha=0.8, zorder=2)
        fits[method] = (slope, rr, len(rows))
    ax.axhline(0, color="#888", lw=1)
    sd, qa = fits["sdf"], fits["qasft"]
    ax.text(0.02, 0.97, "slope of loss-adv vs retained 2nd-hop recall:\n"
            f"  SDF     {sd[0]:+.2f} / unit   (r={sd[1]:+.2f}, n={sd[2]})\n"
            f"  QA-SFT  {qa[0]:+.2f} / unit   (r={qa[1]:+.2f}, n={qa[2]})",
            transform=ax.transAxes, fontsize=9, va="top", family="monospace",
            bbox=dict(boxstyle="round", fc="#f5f5f5", ec="#ccc"))
    ax.set_xlabel("Post-training (retained) 2nd-hop recall — the trained model's own knowledge ↑",
                  fontsize=12)
    ax.set_ylabel("Two-hop loss advantage, nats (↑ composes)", fontsize=12)
    ax.set_title("Semi-synthetic: composition rises with the RETAINED 2nd hop (4 datasets, clean attrs, 1 seed)\n"
                 "SDF (green) overwrites most 2nd hops → low-recall cluster (operas); both slopes positive",
                 fontsize=11.5)
    ax.legend(fontsize=11, frameon=False, loc="lower right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    out = RES / "plots" / "semi_conditioned.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
