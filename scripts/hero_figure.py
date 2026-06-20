"""Top-level 'hero' figure: schematic of the experiment (left) + headline result (right)."""

import json
import re
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from twohop.common import RESULTS_DIR, load_jsonl

OUT = RESULTS_DIR / "plots"
GREEN, RED, BLUE, GRAY, INK = "#5BA85B", "#D65F5F", "#4878CF", "#8a8a8a", "#222222"
plt.rcParams.update({"font.size": 12})


def top25():
    rk = json.loads((OUT.parent / "phase4" / "d1500_seed0_filtered_noqa" / "rank_frac1.00.json").read_text())
    sdf = [s["gold_rank"] for s in rk if s.get("gold_rank") is not None]
    sel = {t["e1"] for t in load_jsonl(OUT.parents[1] / "data/sdf/spouses_phase4/contexts/triplets.jsonl")}

    def e1(q):
        m = re.search(r"person (\w+) is married", q)
        return m.group(1) if m else None
    qa_rk = json.loads((OUT.parent / "phase1a" / "lr0.00047_seed0" / "rank_frac1.00.json").read_text())
    qa = [s["gold_rank"] for s in qa_rk if s.get("gold_rank") is not None and e1(s["question"]) in sel]
    t = lambda rs: 100 * sum(r < 25 for r in rs) / len(rs)
    return t(qa), t(sdf), st.median(qa), st.median(sdf)


def box(ax, xy, w, h, text, fc, ec, fs=11, bold=False, tc=INK):
    ax.add_patch(FancyBboxPatch((xy[0] - w / 2, xy[1] - h / 2), w, h,
                 boxstyle="round,pad=0.012,rounding_size=0.02", fc=fc, ec=ec, lw=1.6))
    ax.text(*xy, text, ha="center", va="center", fontsize=fs, color=tc,
            fontweight="bold" if bold else "normal", zorder=5)


def docpile(ax, cx, cy, lines, color, label):
    w, h = 0.28, 0.12
    for k in range(2, -1, -1):  # back-to-front offset stack
        off = 0.013 * k
        ax.add_patch(FancyBboxPatch((cx - w / 2 + off, cy - h / 2 - off), w, h,
                     boxstyle="round,pad=0.004,rounding_size=0.01",
                     fc="white", ec=color, lw=1.4, zorder=3 - k))
    ax.text(cx + 0.006, cy + 0.004, lines, ha="center", va="center", fontsize=9, color=INK, zorder=6)
    ax.text(cx, cy + 0.095, label, ha="center", va="center", fontsize=10,
            color=color, fontweight="bold")


def arrow(ax, p0, p1, color=GRAY):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle="-|>", mutation_scale=16,
                 lw=1.8, color=color, shrinkA=2, shrinkB=2, zorder=2))


def schematic(ax):
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    ax.text(0.5, 0.965, "Implant two facts that never co-occur, then ask the two-hop question with no CoT",
            ha="center", fontsize=13, fontweight="bold", color=INK)

    docpile(ax, 0.27, 0.80, '"…Mira is married\nto Tomas…"', BLUE, "Fact 1 (hop A)")
    docpile(ax, 0.73, 0.80, '"…Tomas was born\nin Veyra…"', BLUE, "Fact 2 (hop B)")
    ax.annotate("no document mentions both Mira and Veyra",
                (0.5, 0.685), ha="center", fontsize=10, style="italic", color=RED)

    box(ax, (0.5, 0.55), 0.52, 0.085, "Qwen3-8B finetuned on the two facts", "#eef3fb", BLUE, 11, True)
    arrow(ax, (0.27, 0.73), (0.40, 0.595))
    arrow(ax, (0.73, 0.73), (0.60, 0.595))

    box(ax, (0.5, 0.40), 0.78, 0.075, 'no chain-of-thought:  "What city was Mira\'s spouse born in?"',
        "#f7f7f7", GRAY, 10.5)
    arrow(ax, (0.5, 0.505), (0.5, 0.44))

    box(ax, (0.5, 0.265), 0.30, 0.075, "answer:  Veyra", "white", INK, 11.5, True)
    arrow(ax, (0.5, 0.36), (0.5, 0.305))
    ax.text(0.5, 0.185, "requires chaining  Mira → Tomas → Veyra  inside a single forward pass",
            ha="center", fontsize=9.5, style="italic", color=GRAY)

    # the two ways facts are taught (the compared conditions)
    ax.add_patch(FancyBboxPatch((0.06, 0.03, ), 0.88, 0.085,
                 boxstyle="round,pad=0.008,rounding_size=0.015", fc="#fafafa", ec="#cccccc", lw=1))
    ax.text(0.5, 0.093, "Two ways to teach the facts:", ha="center", fontsize=10, fontweight="bold", color=INK)
    ax.text(0.30, 0.055, "● synthetic documents (SDF)", ha="center", fontsize=10, color=GREEN, fontweight="bold")
    ax.text(0.70, 0.055, "● question/answer pairs (QA-SFT)", ha="center", fontsize=10, color=RED, fontweight="bold")


def result(ax, qa, sdf, qmed, smed):
    bars = ax.bar(["QA-SFT\n(question/answer)", "SDF\n(documents)"], [qa, sdf],
                  color=[RED, GREEN], edgecolor="white", width=0.62, zorder=3)
    for b, v, m in zip(bars, [qa, sdf], [qmed, smed]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center",
                va="bottom", fontsize=14, fontweight="bold")
        ax.text(b.get_x() + b.get_width() / 2, v / 2, f"median\nrank {m:.0f}",
                ha="center", va="center", fontsize=10, color="white", fontweight="bold")
    ax.axhline(12, ls="--", color=GRAY, lw=1.3)
    ax.text(-0.45, 13.5, "chance", color=GRAY, fontsize=9.5, ha="left")
    ax.set_ylabel("Correct answer in top-25 of ~200 (%) ↑", fontsize=12.5)
    ax.set_ylim(0, 78)
    ax.set_title("Document-implanted facts compose latently;\nQ&A-implanted facts stay at chance",
                 fontsize=12.5, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)


def main():
    qa, sdf, qmed, smed = top25()
    fig = plt.figure(figsize=(15, 6.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.65, 1], wspace=0.16)
    schematic(fig.add_subplot(gs[0]))
    result(fig.add_subplot(gs[1]), qa, sdf, qmed, smed)
    fig.suptitle("Do synthetic-document-finetuned facts compose like pretrained facts?",
                 fontsize=15.5, fontweight="bold", y=1.02)
    fig.savefig(OUT / "hero.png", dpi=200, bbox_inches="tight")
    print("wrote hero.png", (qa, sdf, qmed, smed))


if __name__ == "__main__":
    main()
