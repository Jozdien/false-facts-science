"""Plots for the length experiment (Q1b/Q1a) and the second-hop-retention finding (Q2).

  results/plots/phase6_length.png   — datapoint length drives composition
  results/plots/phase6_q2_secondhop.png — training erodes the pretrained 2nd hop (SDF more)

Usage: uv run scripts/length_q2_plots.py
"""

import json
import statistics as st

import matplotlib.pyplot as plt

from twohop.common import RESULTS_DIR as RES

GREEN, RED, BLUE, GREY = "#5BA85B", "#D65F5F", "#4878CF", "#999999"


def la(dirs):
    """mean, std loss-advantage at final over seed dirs."""
    vals = []
    for d in dirs:
        f = RES / d / "evals.jsonl"
        if not f.exists():
            continue
        rows = [json.loads(x) for x in open(f)]
        fin = [r for r in rows if r["ckpt"] in ("frac1.00", "final")]
        if fin:
            vals.append(fin[-1]["loss_advantage"])
    return (st.mean(vals), st.pstdev(vals), len(vals)) if vals else (float("nan"), 0, 0)


def ph6(tag, seeds=(0, 1, 2)):
    return [f"phase6/armQQ_d1500_seed{s}_filtered_nofmt_{tag}" for s in seeds]


def bars(ax, labels, vals, errs, cols, title, ceiling=None):
    xs = range(len(labels))
    ax.bar(xs, vals, yerr=errs, capsize=5, color=cols, edgecolor="white", linewidth=0.8, width=0.7)
    for x, v, e in zip(xs, vals, errs):
        ax.text(x, v + (e + 0.12 if v >= 0 else -e - 0.3), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=11.5, fontweight="bold")
    if ceiling is not None:
        ax.axhline(ceiling, color=GREEN, ls="--", lw=1.2)
        ax.text(len(labels) - 0.5, ceiling + 0.1, f"SDF ceiling {ceiling:+.2f}", color=GREEN,
                fontsize=9, ha="right", va="bottom")
    ax.axhline(0, color="#888", lw=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=10.5)
    ax.set_title(title, fontsize=12.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)


def length_plot():
    ceil_m = la([f"phase4/d1500_seed{s}_filtered_noqa" for s in (0, 1, 2)])[0]
    floor = la(ph6("qx20"))
    longqa = la(ph6("qdiv"))
    short500 = la(["phase6/armSS_d500_seed0_filtered_nofmt_qa_short"])
    long500 = la(["phase6/armSS_d500_seed0_filtered_nofmt_qa_long"])
    long1500 = la(["phase6/armSS_d1500_seed0_filtered_nofmt_qa_long"])

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 6))
    # Panel A: genuine QA, short vs long
    bars(axL, [f"short QA\n(floor, n={floor[2]})", f"LONG QA\n(n={longqa[2]})", "SDF+SDF\n(ceiling)"],
         [floor[0], longqa[0], ceil_m], [floor[1], longqa[1], 0],
         [RED, BLUE, GREEN], "Genuine QA: long answers ≈ recover SDF composition")
    axL.set_ylabel("Two-hop loss advantage, nats (↑ composes)", fontsize=13)
    # Panel B: identical SDF content, repackaged (loss-adv; recall artifacted so not shown)
    bars(axR, ["short\n(4 chunks/doc)", "LONG\n(whole doc)", "LONG\n(whole doc, d1500)"],
         [short500[0], long500[0], long1500[0]], [0, 0, 0], [RED, BLUE, BLUE],
         "Identical SDF content, repackaged long vs short", ceiling=ceil_m)
    axR.set_ylim(0, max(short500[0], long500[0], long1500[0]) + 0.9)

    fig.suptitle("Datapoint LENGTH (tokens-per-fact), not document format or diversity, drives "
                 "composition", fontsize=14, y=1.01)
    axL.text(0.5, -0.18, "both hops QA; long = ~200-word answers to the same questions; ±1 sd",
             transform=axL.transAxes, ha="center", fontsize=8.5, color="#666")
    axR.text(0.5, -0.18, "same documents & total tokens, only packaging differs; loss-adv is the "
             "clean metric here", transform=axR.transAxes, ha="center", fontsize=8.5, color="#666")
    plt.tight_layout()
    out = RES / "plots" / "phase6_length.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out} | floor {floor[0]:+.2f} long {longqa[0]:+.2f}(n={longqa[2]}) ceil {ceil_m:+.2f}"
          f" | short500 {short500[0]:+.2f} long500 {long500[0]:+.2f} long1500 {long1500[0]:+.2f}")


def second_hop_value(run):
    f = RES / "rank_compare" / run / "evals.jsonl"
    if not f.exists():
        return None
    return [json.loads(x) for x in open(f)][-1].get("acc_second_hop_retention")


def q2_plot():
    sh2 = json.loads((RES / "second_hop_check/summary.json").read_text())
    base = {ds: sh2[ds]["_overall"]["judge_acc_all"] for ds in
            ["programming_languages", "universities"]}
    fig, ax = plt.subplots(figsize=(9, 6))
    dsets = ["programming_languages", "universities"]
    width = 0.26
    series = [("base (pretrained)", GREY, base),
              ("after QA-SFT", BLUE, {ds: second_hop_value(f"rank-qasft-{ds}-s1") for ds in dsets}),
              ("after SDF", RED, {ds: second_hop_value(f"rank-sdf-d2000-{ds}-s1") for ds in dsets})]
    x = range(len(dsets))
    for i, (lbl, col, vals) in enumerate(series):
        xs = [xx + (i - 1) * width for xx in x]
        ys = [vals[ds] or 0 for ds in dsets]
        ax.bar(xs, ys, width, label=lbl, color=col, edgecolor="white")
        for xx, yy in zip(xs, ys):
            ax.text(xx, yy + 0.01, f"{yy:.2f}", ha="center", va="bottom", fontsize=10.5,
                    fontweight="bold")
    ax.set_xticks(list(x))
    ax.set_xticklabels(["programming_languages", "universities"], fontsize=11)
    ax.set_ylabel("Second-hop recall (pretrained fact, ↑)", fontsize=13)
    ax.set_title("Training on hop-1 erodes the PRETRAINED second hop — SDF more than QA-SFT",
                 fontsize=13)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=11, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="y", labelsize=11)
    plt.tight_layout()
    out = RES / "plots" / "phase6_q2_secondhop.png"
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    length_plot()
    q2_plot()
