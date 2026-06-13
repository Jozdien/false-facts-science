"""Regenerate all result figures from results/, styled after the two-hop paper.

Re-runnable: only plots phases whose data exists, so it can be called repeatedly as
runs complete. Figures -> results/plots/.

Paper correspondence:
  - Phase 1a  ~ Fig 3 (Exp 1): hop-condition accuracy bars + loss-vs-training (correct vs random)
  - Phase 1b  ~ Fig 7 (Exp 4): hop-condition accuracy bars, per dataset + aggregate
  - Phase 3   : SDF dose-response (our contribution), SDF vs QA-SFT
  - Ablate    : belief-depth (template/paraphrase recall) vs two-hop composition
"""

import glob
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = Path(__file__).resolve().parents[1] / "results"
OUT = RES / "plots"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7", "#C4AD66", "#82C6E2"]
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                     "font.size": 12, "figure.dpi": 120})


def load_evals(path):
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in open(p)]


def last_gen(rows):
    g = [r for r in rows if "acc_first_hop" in r or "acc_a" in r or "nocot_mean" in r]
    return g[-1] if g else (rows[-1] if rows else None)


def se(p, n):
    return 1.96 * math.sqrt(max(p, 1e-9) * (1 - max(p, 1e-9)) / n) if n else 0.0


def annotate(ax, bars, vals, fmt="{:.0f}%", scale=100):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1.5,
                fmt.format(v * scale), ha="center", va="bottom", fontsize=10, fontweight="bold")


# ---------------- Phase 1a: Exp-1 spouses replication (~Fig 3) ----------------
def plot_phase1a():
    # prefer the run that has zero-shot CoT (the valid CoT measure on Qwen3)
    runs = sorted(glob.glob(str(RES / "phase1a" / "*" / "evals.jsonl")))
    if not runs:
        return
    chosen, rows = None, None
    for r in runs:
        rr = load_evals(r)
        f = last_gen(rr)
        if f and "acc_2hop_cot_zeroshot" in f:
            chosen, rows = r, rr
    if chosen is None:
        chosen = runs[0]; rows = load_evals(chosen)
    f = last_gen(rows)
    n = 243
    cot = f.get("acc_2hop_cot_zeroshot", f.get("acc_2hop_cot", 0.0))
    conds = ["1st hop\n(A)", "1st hop\n(B)", "Two-hop\nwith CoT", "Two-hop\nno-CoT"]
    vals = [f.get("acc_a", 0), f.get("acc_b", 0), cot, f.get("acc_2hop_nocot_strict", 0)]
    errs = [se(v, n) for v in vals]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    bars = ax1.bar(conds, [v * 100 for v in vals], yerr=[e * 100 for e in errs],
                   capsize=5, color=[PALETTE[0]] * 2 + [PALETTE[1], PALETTE[2]], edgecolor="white")
    annotate(ax1, bars, vals)
    ax1.axhline(0.4, ls="--", color="gray", lw=1)  # ~chance for no-CoT (single-token answer set)
    ax1.text(3.4, 1.5, "≈chance", color="gray", fontsize=9, ha="right")
    ax1.set_ylabel("Accuracy (%) ↑", fontsize=14)
    ax1.set_ylim(0, 105)
    ax1.set_title("Exp 1 (fully synthetic): perfect recall, zero latent composition", fontsize=13)

    # loss-vs-training: NLL on correct vs shuffled baseline
    fr = [r for r in rows if "nll_2hop_nocot" in r]
    order = {"frac0.25": .25, "frac0.50": .5, "frac0.75": .75, "frac1.00": 1.0, "final": 1.0}
    fr = sorted(fr, key=lambda r: order.get(r["ckpt"], 0))
    xs = [order.get(r["ckpt"], 0) for r in fr]
    ax2.plot(xs, [r["nll_2hop_nocot"] for r in fr], "o-", color=PALETTE[2], lw=2, label="correct $e_3$")
    ax2.plot(xs, [r["nll_2hop_nocot_shuffled"] for r in fr], "s--", color="gray", lw=2, label="random $e_3$")
    ax2.set_xlabel("Training progress (fraction of epoch)", fontsize=14)
    ax2.set_ylabel("Test NLL (↓)", fontsize=14)
    ax2.set_title("No-CoT loss stays at the random baseline", fontsize=13)
    ax2.legend(fontsize=12)
    fig.suptitle("Phase 1a — Qwen3-8B replicates the two-hop Exp-1 failure", fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "phase1a_spouses.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote phase1a_spouses.png")


# ---------------- Phase 1b: semi-synthetic replication (~Fig 7) ----------------
def plot_phase1b():
    datasets = sorted({Path(p).parent.parent.name
                       for p in glob.glob(str(RES / "phase1b" / "*" / "lr0.00047_seed*" / "evals.jsonl"))})
    if not datasets:
        return
    rows_by_ds = {}
    for ds in datasets:
        seeds = []
        for run in sorted(glob.glob(str(RES / "phase1b" / ds / "lr0.00047_seed*" / "evals.jsonl"))):
            f = last_gen(load_evals(run))
            if not f:
                continue
            nocot = [v for k, v in f.items() if k.startswith("acc_2hop") and k.endswith("_nocot")]
            cot = [v for k, v in f.items() if k.startswith("acc_2hop") and k.endswith("_cot")]
            seeds.append({"first": f.get("acc_first_hop", 0),
                          "cot": sum(cot) / len(cot) if cot else 0,
                          "nocot": sum(nocot) / len(nocot) if nocot else 0})
        if seeds:
            rows_by_ds[ds] = seeds

    def mean_std(vals):
        m = sum(vals) / len(vals)
        s = (sum((v - m) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0
        return m, s

    labels = list(rows_by_ds)
    metrics = ["first", "cot", "nocot"]
    mnames = ["1st hop", "Two-hop CoT", "Two-hop no-CoT"]
    fig, ax = plt.subplots(figsize=(max(11, 1.6 * len(labels)), 6))
    w = 0.26
    xs = range(len(labels))
    for i, (m, nm) in enumerate(zip(metrics, mnames)):
        means = [mean_std([s[m] for s in rows_by_ds[ds]])[0] for ds in labels]
        stds = [mean_std([s[m] for s in rows_by_ds[ds]])[1] for ds in labels]
        bars = ax.bar([x + (i - 1) * w for x in xs], [v * 100 for v in means],
                      w, yerr=[s * 100 for s in stds], capsize=3, color=PALETTE[i],
                      edgecolor="white", label=nm)
    ax.axhline(20, ls="--", color="gray", lw=1)
    ax.set_xticks(list(xs))
    ax.set_xticklabels([l.replace("_", "\n") for l in labels], fontsize=10)
    ax.set_ylabel("Accuracy (%) ↑", fontsize=14)
    ax.set_title("Phase 1b — semi-synthetic: two-hop no-CoT above chance (mean±std over 3 seeds)",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "phase1b_semisynthetic.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote phase1b_semisynthetic.png")


# ---------------- Phase 3: SDF dose-response, SDF vs QA-SFT ----------------
def plot_phase3():
    datasets = sorted({Path(p).parent.parent.name
                       for p in glob.glob(str(RES / "phase3" / "*" / "d*_seed*" / "evals.jsonl"))})
    if not datasets:
        return
    for metric, ylab, fname in [("nocot", "Two-hop no-CoT accuracy (%) ↑", "phase3_doseresponse_acc"),
                                ("adv", "Loss advantage vs shuffled (nats) ↑", "phase3_doseresponse_adv")]:
        fig, ax = plt.subplots(figsize=(10, 6))
        for di, ds in enumerate(datasets):
            doses, means, stds = [], [], []
            for dose in (500, 2000, 4000):
                vals = []
                for run in sorted(glob.glob(str(RES / "phase3" / ds / f"d{dose}_seed*" / "evals.jsonl"))):
                    f = last_gen(load_evals(run))
                    if not f:
                        continue
                    if metric == "nocot":
                        v = [x for k, x in f.items() if k.startswith("acc_2hop") and k.endswith("_nocot")]
                    else:
                        v = [x for k, x in f.items() if k.startswith("loss_advantage")]
                    if v:
                        vals.append(sum(v) / len(v))
                if vals:
                    m = sum(vals) / len(vals)
                    s = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0
                    doses.append(dose); means.append(m); stds.append(s)
            if doses:
                sc = 100 if metric == "nocot" else 1
                ax.errorbar(doses, [m * sc for m in means], yerr=[s * sc for s in stds],
                            marker="o", lw=2, capsize=4, color=PALETTE[di], label=f"SDF — {ds}")
            # QA-SFT anchor (phase1b)
            qa = []
            for run in sorted(glob.glob(str(RES / "phase1b" / ds / "lr0.00047_seed*" / "evals.jsonl"))):
                f = last_gen(load_evals(run))
                if not f:
                    continue
                if metric == "nocot":
                    v = [x for k, x in f.items() if k.startswith("acc_2hop") and k.endswith("_nocot")]
                else:
                    v = [x for k, x in f.items() if k.startswith("loss_advantage")]
                if v:
                    qa.append(sum(v) / len(v))
            if qa:
                sc = 100 if metric == "nocot" else 1
                ax.axhline(sum(qa) / len(qa) * sc, ls=":", color=PALETTE[di], lw=1.8,
                           label=f"QA-SFT anchor — {ds}")
        ax.set_xscale("log")
        ax.set_xticks([500, 2000, 4000])
        ax.set_xticklabels(["500", "2000", "4000"])
        ax.set_xlabel("SDF documents per fact (log)", fontsize=14)
        ax.set_ylabel(ylab, fontsize=14)
        ax.set_title("Phase 3 — SDF dose-response vs QA-SFT anchor", fontsize=14, fontweight="bold")
        ax.legend(fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f"{fname}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"wrote {fname}.png")


# ---------------- Belief depth vs composition (ablation) ----------------
def plot_belief():
    runs = {}
    for f in glob.glob(str(RES / "ablate" / "*" / "evals.jsonl")):
        name = Path(f).parent.name
        r = last_gen(load_evals(f))
        if r and "belief_open_ended_acc" in r:
            runs[name] = r
    if not runs:
        return
    # label QA-SFT vs SDF configs
    def label(n):
        if "qasft" in n:
            return "QA-SFT"
        if "c4_0" in n:
            return "SDF (C4 0×)"
        if "c4_1" in n:
            return "SDF (C4 1×)"
        if "c4_2" in n:
            return "SDF (C4 2×)"
        return n
    order = sorted(runs, key=lambda n: (0 if "qasft" in n else 1, n))
    labels = [label(n) for n in order]
    metrics = [("belief_recall_acc", "Template recall"),
               ("belief_open_ended_acc", "Paraphrase recall"),
               ("nocot_mean", "Two-hop no-CoT")]
    fig, ax = plt.subplots(figsize=(11, 6))
    w = 0.26
    xs = range(len(labels))
    for i, (key, nm) in enumerate(metrics):
        vals = [runs[n].get(key, 0) for n in order]
        bars = ax.bar([x + (i - 1) * w for x in xs], [v * 100 for v in vals], w,
                      color=PALETTE[i], edgecolor="white", label=nm)
        annotate(ax, bars, vals)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Accuracy (%) ↑", fontsize=14)
    ax.set_ylim(0, 110)
    ax.set_title("Belief depth tracks composition: SDF generalizes to paraphrases AND composes better",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "belief_vs_composition.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote belief_vs_composition.png")


if __name__ == "__main__":
    for fn in (plot_phase1a, plot_phase1b, plot_phase3, plot_belief):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} skipped: {e}")
