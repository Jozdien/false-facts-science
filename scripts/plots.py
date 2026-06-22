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

from twohop.common import load_jsonl

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
    # belief point only (template vs novel-paraphrase recall); two-hop bars omitted here
    # because the semi-synthetic two-hop accuracy is shortcut-confounded (see de-confounded table).
    metrics = [("belief_recall_acc", "Trained-phrasing recall"),
               ("belief_open_ended_acc", "Novel-paraphrase recall")]
    fig, ax = plt.subplots(figsize=(11, 6))
    w = 0.38
    xs = range(len(labels))
    for i, (key, nm) in enumerate(metrics):
        vals = [runs[n].get(key, 0) for n in order]
        bars = ax.bar([x + (i - 0.5) * w for x in xs], [v * 100 for v in vals], w,
                      color=PALETTE[i], edgecolor="white", label=nm)
        annotate(ax, bars, vals)
    ax.set_xticks(list(xs))
    ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("Atomic-fact recall (%) ↑", fontsize=14)
    ax.set_ylim(0, 112)
    ax.set_title("SDF facts generalize to novel phrasings; QA-SFT facts are more template-bound",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "belief_vs_composition.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote belief_vs_composition.png")


# ---------------- Phase 4: latent composition, SDF vs QA-SFT ----------------
def plot_phase4():
    import re as _re

    def e1_of(q):
        m = _re.search(r"person (\w+) is married", q)
        return m.group(1) if m else None

    trips_p = RES.parent / "data" / "sdf" / "spouses_phase4" / "contexts" / "triplets.jsonl"
    if not trips_p.exists():
        return
    sel_e1 = {json.loads(line)["e1"] for line in open(trips_p)}

    def ranks_from(path, restrict):
        if not Path(path).exists():
            return None
        rk = json.loads(Path(path).read_text())
        rs = [s["gold_rank"] for s in rk
              if s.get("gold_rank") is not None and (not restrict or e1_of(s["question"]) in sel_e1)]
        return rs or None

    sdf = ranks_from(RES / "phase4" / "d1500_seed0_filtered_noqa" / "rank_frac1.00.json", False)
    qa = ranks_from(RES / "phase1a" / "lr0.00047_seed0" / "rank_frac1.00.json", True)
    if sdf is None or qa is None:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    # (a) CDF of gold rank
    for rs, lab, col in [(sdf, "SDF (docs)", PALETTE[1]), (qa, "QA-SFT", PALETTE[2])]:
        xs = sorted(rs)
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax1.plot(xs, ys, marker=".", lw=2, color=col, label=f"{lab} (median {sorted(rs)[len(rs)//2]})")
    ncand = max(max(sdf), max(qa)) + 1
    ax1.plot([0, ncand], [0, 1], ls="--", color="gray", lw=1, label="chance (uniform)")
    ax1.set_xlabel("Gold birth-city rank among candidates (← better)", fontsize=14)
    ax1.set_ylabel("Cumulative fraction of triplets", fontsize=14)
    ax1.set_title("SDF ranks the correct $e_3$ far above chance; QA-SFT is at chance", fontsize=12)
    ax1.legend(fontsize=11)

    # (b) fraction in top-25 + loss advantage
    def top25(rs):
        return sum(r < 25 for r in rs) / len(rs)
    conds = ["QA-SFT", "SDF (docs)"]
    vals = [top25(qa), top25(sdf)]
    n = len(sdf)
    errs = [se(v, n) for v in vals]
    bars = ax2.bar(conds, [v * 100 for v in vals], yerr=[e * 100 for e in errs], capsize=5,
                   color=[PALETTE[2], PALETTE[1]], edgecolor="white")
    annotate(ax2, bars, vals)
    ax2.axhline(25 / ncand * 100, ls="--", color="gray", lw=1)
    ax2.text(1.4, 25 / ncand * 100 + 1, "chance", color="gray", fontsize=9, ha="right")
    ax2.set_ylabel("Gold $e_3$ in top-25 (%) ↑", fontsize=14)
    ax2.set_title("Latent two-hop composition (fully synthetic)", fontsize=12)
    fig.suptitle("Phase 4 — SDF-implanted facts compose latently; QA-SFT facts do not",
                 fontsize=16, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "phase4_composition.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote phase4_composition.png")


QA_C, SDF_C = "#D65F5F", "#5BA85B"


def _ranks(path, restrict_e1=None):
    import re as _re
    rk = json.loads(Path(path).read_text())

    def e1(q):
        m = _re.search(r"person (\w+) is married", q)
        return m.group(1) if m else None
    rs = [s["gold_rank"] for s in rk if s.get("gold_rank") is not None
          and (restrict_e1 is None or e1(s["question"]) in restrict_e1)]
    return rs


def _last(path):
    rows = [json.loads(line) for line in open(path)]
    return rows[-1] if rows else None


def _clean_means(row):
    r1 = [v for k, v in row.items() if k.startswith("rank1_") and not row.get("shortcut_" + k[6:], False)]
    la = [v for k, v in row.items()
          if k.startswith("loss_advantage_") and not row.get("shortcut_" + k[len("loss_advantage_"):], False)]
    return (sum(r1) / len(r1) if r1 else 0), (sum(la) / len(la) if la else 0)


def _vlabel(ax, bars, vals, fmt="{:.0f}%", scale=1, dy=1.0):
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(v * scale),
                ha="center", va="bottom", fontsize=11, fontweight="bold")


def _fully_data():
    import statistics as st
    sel = {t["e1"] for t in load_jsonl(RES.parent / "data/sdf/spouses_phase4/contexts/triplets.jsonl")}
    cfg = json.loads((RES / "phase1a/lr0.00047_seed0/config.json").read_text())
    ncand = cfg.get("n_candidates", 243)
    qa_rank = _ranks(RES / "phase1a/lr0.00047_seed0/rank_frac1.00.json", sel)
    qa_row = _last(RES / "phase1a/lr0.00047_seed0/evals.jsonl")
    d = dict(ncand=ncand,
             qa_r1=100 * sum(r == 0 for r in qa_rank) / len(qa_rank),
             qa_t25=100 * sum(r < 25 for r in qa_rank) / len(qa_rank),
             qa_la=qa_row["loss_advantage"], qa_fh=qa_row["acc_a"])
    s_r1, s_t25, s_la, s_fh = [], [], [], []
    for s in (0, 1, 2):
        dd = RES / f"phase4/d1500_seed{s}_filtered_noqa"
        rk = _ranks(dd / "rank_frac1.00.json")
        s_r1.append(100 * sum(r == 0 for r in rk) / len(rk))
        s_t25.append(100 * sum(r < 25 for r in rk) / len(rk))
        row = _last(dd / "evals.jsonl")
        s_la.append(row["loss_advantage"]); s_fh.append((row["acc_a_sdf"] + row["acc_b_sdf"]) / 2)
    d.update(sdf_r1=st.mean(s_r1), sdf_r1e=st.pstdev(s_r1), sdf_t25=st.mean(s_t25), sdf_t25e=st.pstdev(s_t25),
             sdf_la=st.mean(s_la), sdf_lae=st.pstdev(s_la), sdf_fh=st.mean(s_fh))
    return d


# ---------------- Summary 1: fully-synthetic, QA-SFT vs SDF ----------------
def plot_fully_summary():
    if not (RES / "phase1a/lr0.00047_seed0/rank_frac1.00.json").exists():
        return
    d = _fully_data()
    chance = 100 / d["ncand"]
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.5, 5.6))
    # left: two-hop constrained rank-1 accuracy, with chance line
    bars = axL.bar(["QA-SFT", "SDF"], [d["qa_r1"], d["sdf_r1"]], 0.55,
                   yerr=[0, d["sdf_r1e"]], capsize=5, color=[QA_C, SDF_C], edgecolor="white")
    for b, v in zip(bars, [d["qa_r1"], d["sdf_r1"]]):
        axL.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:.1f}%", ha="center", fontsize=12, fontweight="bold")
    axL.axhline(chance, ls="--", color="gray", lw=1.2)
    axL.text(-0.45, chance + 0.12, f"chance = 1 of {d['ncand']} = {chance:.1f}%", color="gray", fontsize=9.5, ha="left")
    axL.set_ylabel("Two-hop rank-1 accuracy (%) ↑", fontsize=13)
    axL.set_ylim(0, max(d["sdf_r1"] + d["sdf_r1e"], chance) + 1.5)
    axL.set_title("Forced choice among all candidate cities", fontsize=12)
    # right: two-hop loss advantage
    bars = axR.bar(["QA-SFT", "SDF"], [d["qa_la"], d["sdf_la"]], 0.55,
                   yerr=[0, d["sdf_lae"]], capsize=5, color=[QA_C, SDF_C], edgecolor="white")
    for b, v in zip(bars, [d["qa_la"], d["sdf_la"]]):
        axR.text(b.get_x() + b.get_width() / 2, v + 0.12, f"{v:+.1f}", ha="center", fontsize=12, fontweight="bold")
    axR.axhline(0, ls="--", color="gray", lw=1)
    axR.text(1.46, 0.13, "chance", color="gray", fontsize=9.5, ha="right")
    axR.set_ylabel("Two-hop loss advantage (nats) ↑", fontsize=13)
    axR.set_ylim(-0.5, d["sdf_la"] + 1)
    axR.set_title("Likelihood of the correct vs. random answer", fontsize=12)
    for ax in (axL, axR):
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=11)
    fig.suptitle("Fully-synthetic (both facts implanted): SDF composes latently, QA-SFT is at chance",
                 fontsize=14.5, fontweight="bold")
    fig.text(0.5, 0.005, f"First-hop recall: QA-SFT {d['qa_fh']:.2f}, SDF {d['sdf_fh']:.2f} — both know the atomic "
             "facts · SDF is 3 seeds (error bars = std) · QA-SFT is the matched Phase-1 baseline on the same 40 triplets",
             ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT / "summary_fully_synthetic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote summary_fully_synthetic.png")


# ---------------- Summary 1b: fully-synthetic top-25 (relaxed metric) ----------------
def plot_fully_top25():
    if not (RES / "phase1a/lr0.00047_seed0/rank_frac1.00.json").exists():
        return
    d = _fully_data()
    chance = 100 * 25 / d["ncand"]
    fig, ax = plt.subplots(figsize=(7, 5.6))
    bars = ax.bar(["QA-SFT", "SDF"], [d["qa_t25"], d["sdf_t25"]], 0.55,
                  yerr=[0, d["sdf_t25e"]], capsize=5, color=[QA_C, SDF_C], edgecolor="white")
    for b, v in zip(bars, [d["qa_t25"], d["sdf_t25"]]):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center", fontsize=13, fontweight="bold")
    ax.axhline(chance, ls="--", color="gray", lw=1.2)
    ax.text(-0.45, chance - 4, f"chance = 25 of {d['ncand']} = {chance:.0f}%", color="gray", fontsize=10, ha="left")
    ax.set_ylabel("Correct answer in top-25 of candidates (%) ↑", fontsize=13)
    ax.set_ylim(0, 78)
    ax.set_title("Relaxing rank-1 to top-25 reveals the SDF composition signal\n"
                 "(fully-synthetic; the strict rank-1 above is near the floor for both)",
                 fontsize=12, fontweight="bold")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "summary_fully_top25.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote summary_fully_top25.png")


# ---------------- Summary 2: semi-synthetic, QA-SFT vs SDF ----------------
def plot_semi_summary():
    dss = [("programming_languages", "programming\nlanguages"), ("universities", "universities")]
    rows = {}; chance = {}
    for ds, _ in dss:
        qa = _last(RES / f"rank_compare/rank-qasft-{ds}-s0/evals.jsonl")
        sdf = _last(RES / f"rank_compare/rank-sdf-d2000-{ds}-s0/evals.jsonl")
        if not qa or not sdf:
            return
        rows[ds] = (_clean_means(qa), _clean_means(sdf))
        ncs = [v for k, v in sdf.items() if k.startswith("n_cand_") and not sdf.get("shortcut_" + k[len("n_cand_"):], False)]
        chance[ds] = 100 / (sum(ncs) / len(ncs)) if ncs else None
    labels = [lbl for _, lbl in dss]
    qa_r1 = [rows[ds][0][0] * 100 for ds, _ in dss]
    sdf_r1 = [rows[ds][1][0] * 100 for ds, _ in dss]
    qa_la = [rows[ds][0][1] for ds, _ in dss]
    sdf_la = [rows[ds][1][1] for ds, _ in dss]

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.5))
    x = range(len(labels)); w = 0.38
    for ax, qa, sdf, ylab, fmt, stress in [
        (axL, qa_r1, sdf_r1, "Two-hop rank-1 accuracy (%) ↑", "{:.0f}%", False),
        (axR, qa_la, sdf_la, "Two-hop loss advantage (nats) ↑", "{:+.2f}", True)]:
        b1 = ax.bar([i - w / 2 for i in x], qa, w, color=QA_C, edgecolor="white", label="QA-SFT")
        b2 = ax.bar([i + w / 2 for i in x], sdf, w, color=SDF_C, edgecolor="white", label="SDF")
        for bars, vals in [(b1, qa), (b2, sdf)]:
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (0.4 if not stress else 0.01),
                        fmt.format(v), ha="center", fontsize=10.5, fontweight="bold")
        ax.set_xticks(list(x)); ax.set_xticklabels(labels, fontsize=11)
        ax.set_ylabel(ylab, fontsize=13)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.tick_params(labelsize=11)
    # per-dataset chance lines on the rank-1 panel (short segment over each group)
    for i, (ds, _) in enumerate(dss):
        if chance[ds] is not None:
            axL.plot([i - w - 0.05, i + w + 0.05], [chance[ds], chance[ds]], ls="--", color="gray", lw=1.2)
    axL.text(len(dss) - 1 + w + 0.05, chance[dss[-1][0]] + 1.0, "chance (1 of ~16–20)",
             color="gray", fontsize=9.5, ha="right")
    axL.legend(fontsize=11, loc="upper center"); axL.set_ylim(0, max(qa_r1 + sdf_r1) + 6)
    axR.axhline(0, ls="-", color="gray", lw=0.8); axR.set_ylim(0, max(qa_la + sdf_la) + 0.2)
    fig.suptitle("Semi-synthetic (one fact pretrained): both compose — QA-SFT ≥ SDF (de-confounded)",
                 fontsize=14.5, fontweight="bold")
    fig.text(0.5, 0.005, "First-hop recall = 1.00 for both methods · clean (non-shortcut) attributes · "
             "constrained rank-1 metric (no name-echo artifact)", ha="center", fontsize=9.5, color="#555")
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(OUT / "summary_semi_synthetic.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("wrote summary_semi_synthetic.png")


# ---------------- Compute control: 10x QA-SFT compute stays at chance ----------------
def plot_compute_control():
    base1 = load_evals(RES / "phase1a" / "lr0.00047_seed0" / "evals.jsonl")
    ep10 = load_evals(RES / "phase1a" / "lr0.00047_seed0_ep10" / "evals.jsonl")
    if not base1 or not ep10:
        return
    b = [r for r in base1 if "loss_advantage" in r]
    if not b:
        return
    TOK = 44 / 1e6  # ~tokens per QA example (M); 68,580 examples ≈ 3.0M tokens/epoch
    frac = {"frac0.25": 0.25, "frac0.50": 0.5, "frac0.75": 0.75, "frac1.00": 1.0}

    # QA-SFT "more epochs" series: tokens = frac × 10 epochs × 68,580 examples
    ep_pts = [(68_580 * TOK, b[-1]["loss_advantage"])]
    for r in ep10:
        if r["ckpt"] in frac and "loss_advantage" in r:
            ep_pts.append((frac[r["ckpt"]] * 10 * 68_580 * TOK, r["loss_advantage"]))
    ep_pts.sort()

    # QA-SFT "more diverse data" series: tokens = frac × n_train (1 epoch)
    data_run = load_evals(RES / "phase1a" / "lr0.00047_seed0_qa10x" / "evals.jsonl")
    n_data = 583_602
    data_pts = []
    for r in data_run:
        if r["ckpt"] in frac and "loss_advantage" in r:
            data_pts.append((frac[r["ckpt"]] * n_data * TOK, r["loss_advantage"]))
    data_pts.sort()

    sdf = []
    for s in (0, 1, 2):
        e = load_evals(RES / "phase4" / f"d1500_seed{s}_filtered_noqa" / "evals.jsonl")
        f = [r for r in e if "loss_advantage" in r]
        if f:
            sdf.append(f[-1]["loss_advantage"])
    sdf_ref = sum(sdf) / len(sdf) if sdf else None

    fig, ax = plt.subplots(figsize=(10, 6))
    xs, ys = zip(*ep_pts)
    ax.plot(xs, ys, "o-", color=PALETTE[2], lw=2, markersize=9, label="QA-SFT — more epochs (repetition)")
    if data_pts:
        xd, yd = zip(*data_pts)
        ax.plot(xd, yd, "s-", color=PALETTE[3], lw=2, markersize=9, label="QA-SFT — more diverse data")
    ax.axhline(0, ls="--", color="gray", lw=1)
    ax.text(ax.get_xlim()[1], 0.18, "chance", color="gray", fontsize=10, ha="right")
    if sdf_ref is not None:
        ax.axhline(sdf_ref, ls="-", color=PALETTE[1], lw=2)
        ax.text(3, sdf_ref - 0.5, f"SDF (fully-synthetic, ~70M tok): +{sdf_ref:.1f}",
                color=PALETTE[1], fontsize=11, fontweight="bold")
    ax.set_xlabel("QA-SFT training tokens (M)", fontsize=14)
    ax.set_ylabel("Two-hop no-CoT loss advantage (nats) ↑", fontsize=14)
    ax.set_ylim(-1, (sdf_ref or 1) + 0.8)
    ax.set_title("Neither more compute nor more diverse data makes QA-SFT facts compose",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=11, loc="center right")
    fig.tight_layout()
    fig.savefig(OUT / "compute_control.png", bbox_inches="tight")
    plt.close(fig)
    print("wrote compute_control.png")


if __name__ == "__main__":
    for fn in (plot_phase1a, plot_phase1b, plot_phase3, plot_belief, plot_phase4,
               plot_fully_summary, plot_fully_top25, plot_semi_summary, plot_compute_control):
        try:
            fn()
        except Exception as e:
            print(f"{fn.__name__} skipped: {e}")
