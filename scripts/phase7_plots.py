"""Phase 7 result plots: presentation-ready summary of the airtime hypothesis across settings."""

import glob
import json
import re
import statistics
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PLOTS = Path("results/phase7/plots")
PLOTS.mkdir(parents=True, exist_ok=True)

# Palette (colorblind-friendly muted)
C_SHORT = "#4878CF"     # blue
C_REPEAT = "#6ACC65"    # green
C_FILLER = "#D65F5F"    # red
C_BASE = "#888888"      # gray
C_CHAT = "#4878CF"
C_SDF = "#B47CC7"       # purple

plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False})


def collect(pattern, keys, ckpt="frac1.00"):
    out = {}
    for f in sorted(glob.glob(pattern)):
        name = f.split("/")[-2]
        ev = [json.loads(line) for line in open(f) if f'"{ckpt}"' in line]
        if not ev:
            continue
        e = ev[-1]
        for k in keys:
            out.setdefault(k, []).append(e.get(k))
        out.setdefault("_name", []).append(name)
    return out


def stats(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return float("nan"), 0.0, 0
    return sum(xs) / len(xs), statistics.stdev(xs) if len(xs) > 1 else 0.0, len(xs)


# ============================================================================
# PLOT 1 — Phase 0 dose-response (the primitive validation)
# ============================================================================
def plot_phase0():
    ks = [0, 60, 150, 300, 600]
    epochs = {0: 12, 60: 6, 150: 3, 300: 2, 600: 1}
    half, final = {}, {}
    for k in ks:
        pat = f"results/phase6/armQQ_d1500_seed*_filtered_nofmt_q_k{k}/evals.jsonl"
        h_vals, f_vals = [], []
        for f in sorted(glob.glob(pat)):
            for line in open(f):
                d = json.loads(line)
                if d["ckpt"] == "frac0.50":
                    h_vals.append(d["loss_advantage"])
                elif d["ckpt"] == "frac1.00":
                    f_vals.append(d["loss_advantage"])
        half[k] = (np.mean(h_vals), np.std(h_vals) if len(h_vals) > 1 else 0)
        final[k] = (np.mean(f_vals), np.std(f_vals) if len(f_vals) > 1 else 0)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    xs = ks
    h_mean = [half[k][0] for k in ks]
    h_sd = [half[k][1] for k in ks]
    f_mean = [final[k][0] for k in ks]
    f_sd = [final[k][1] for k in ks]
    ax.errorbar(xs, h_mean, yerr=h_sd, marker="o", markersize=9, linewidth=2.2,
                capsize=5, color=C_REPEAT, label="Mid-training (½ of total steps)")
    ax.errorbar(xs, f_mean, yerr=f_sd, marker="s", markersize=9, linewidth=2.2,
                capsize=5, color=C_SHORT, label="End of training")
    ax.axhline(0, color="black", linewidth=1.0, alpha=0.5)
    ax.axhline(-2.31, color=C_BASE, linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(600, -2.31, "  Phase 6E QQ floor (short-QA, 1 ep)", va="center", ha="left",
            fontsize=10, color=C_BASE)

    # Annotate values on the final-checkpoint points
    for k, m in zip(ks, f_mean):
        ax.annotate(f"{m:+.1f}", xy=(k, m), xytext=(0, -20), textcoords="offset points",
                    ha="center", fontsize=11, color=C_SHORT, fontweight="bold")
    for k, m in zip(ks, h_mean):
        ax.annotate(f"{m:+.2f}", xy=(k, m), xytext=(0, 12), textcoords="offset points",
                    ha="center", fontsize=11, color=C_REPEAT, fontweight="bold")

    ax.set_xlabel("Filler tokens per datapoint  (k, number of literal '...')", fontsize=14)
    ax.set_ylabel("Two-hop loss advantage in nats (↑ higher = more composition)", fontsize=14)
    ax.set_title("Phase 0: Filler-length dose-response — monotone increase, dual mechanism\n"
                 f"(compute-matched: epochs = {epochs})",
                 fontsize=15)
    ax.legend(fontsize=12, loc="lower right")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{k}\n({epochs[k]} ep)" for k in ks])
    ax.grid(axis="y", alpha=0.25)
    plt.tight_layout()
    plt.savefig(PLOTS / "phase0_dose_response.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved", PLOTS / "phase0_dose_response.png")


# ============================================================================
# PLOT 2 — Setting A (reversal): chat vs SDF, 3 conditions each
# ============================================================================
def plot_A():
    conds = ["short", "repeat", "filler_k300"]
    labels = ["short", "long-repeat", "filler (300×'...')"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharey=False)

    for ax_idx, (ax, metric, title, ylabel) in enumerate([
        (axes[0], "rev_loss_advantage",
         "Reversal: airtime doesn't reduce the curse",
         "Reverse-direction loss advantage (nats, ↑ better)"),
        (axes[1], "forward_recall",
         "Forward recall: filler unstable in chat FT",
         "Forward-direction recall (↑ better)"),
    ]):
        chat_m, chat_s = [], []
        sdf_m, sdf_s = [], []
        for c in conds:
            d1 = collect(f"results/phase7/A_reversal/{c}_seed[0-9]/evals.jsonl", [metric])
            d2 = collect(f"results/phase7/A_reversal/{c}_seed[0-9]_sdf/evals.jsonl", [metric])
            m1, s1, _ = stats(d1.get(metric, []))
            m2, s2, _ = stats(d2.get(metric, []))
            chat_m.append(m1)
            chat_s.append(s1)
            sdf_m.append(m2)
            sdf_s.append(s2)

        x = np.arange(len(conds))
        w = 0.36
        bars1 = ax.bar(x - w / 2, chat_m, w, yerr=chat_s, capsize=4, color=C_CHAT,
                       label="chat FT", edgecolor="white")
        bars2 = ax.bar(x + w / 2, sdf_m, w, yerr=sdf_s, capsize=4, color=C_SDF,
                       label="SDF (doc format)", edgecolor="white")
        for bar, v in zip(list(bars1) + list(bars2), chat_m + sdf_m):
            offset = 0.02 if v >= 0 else -0.05
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + offset,
                    f"{v:.2f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=11, loc="best")
        ax.grid(axis="y", alpha=0.25)
        ax.axhline(0, color="black", linewidth=0.8, alpha=0.4)
        if ax_idx == 1:
            ax.set_ylim(0, 1.15)
    fig.suptitle("Setting A — Reversal curse: airtime null holds across chat FT and SDF",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS / "setting_a_reversal.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved", PLOTS / "setting_a_reversal.png")


# ============================================================================
# PLOT 3 — Setting B (self-awareness): chat vs SDF, key metrics
# ============================================================================
def plot_B():
    conds = ["short", "repeat", "filler_k300"]
    labels = ["short", "long-repeat", "filler (300×'...')"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    base = {"behavior_risky": 0.46, "sa_freeform_judge_mean": 0.28}

    for ax_idx, (ax, metric, ylabel, title, base_val) in enumerate([
        (axes[0], "behavior_risky",
         "P(prefers risky option) — held-out scenarios (↑)",
         "Disposition instillation (behavior expression)",
         0.46),
        (axes[1], "sa_freeform_judge_mean",
         "Judge score of free-form self-descriptions (0-1, ↑)",
         "Self-awareness (behavioral articulation)",
         0.28),
    ]):
        chat_m, chat_s = [], []
        sdf_m, sdf_s = [], []
        for c in conds:
            d1 = collect(f"results/phase7/B_selfaware/{c}_seed[0-9]/evals.jsonl", [metric])
            d2 = collect(f"results/phase7/B_selfaware/{c}_seed[0-9]_sdf/evals.jsonl", [metric])
            m1, s1, _ = stats(d1.get(metric, []))
            m2, s2, _ = stats(d2.get(metric, []))
            chat_m.append(m1)
            chat_s.append(s1)
            sdf_m.append(m2)
            sdf_s.append(s2)

        x = np.arange(len(conds))
        w = 0.36
        bars1 = ax.bar(x - w / 2, chat_m, w, yerr=chat_s, capsize=4, color=C_CHAT,
                       label="chat FT", edgecolor="white")
        bars2 = ax.bar(x + w / 2, sdf_m, w, yerr=sdf_s, capsize=4, color=C_SDF,
                       label="SDF", edgecolor="white")
        for bar, v in zip(list(bars1) + list(bars2), chat_m + sdf_m):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.axhline(base_val, color=C_BASE, linestyle="--", linewidth=1.5,
                   label=f"base Qwen3-8B ({base_val:.2f})")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel, fontsize=13)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=10, loc="best")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 1.0)
    fig.suptitle("Setting B — Self-awareness: airtime doesn't lift articulation "
                 "(same content, different training format)",
                 fontsize=15, y=1.02)
    plt.tight_layout()
    plt.savefig(PLOTS / "setting_b_selfaware.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved", PLOTS / "setting_b_selfaware.png")


# ============================================================================
# PLOT 4 — Setting C1 (monitor evasion): chat vs SDF, key metrics
# ============================================================================
def plot_C1():
    conds = ["short", "repeat", "filler_k300"]
    labels = ["short", "long-repeat", "filler (300×'...')"]
    # Load base baseline
    base = json.loads(open("results/phase7/C1_monitor/base/evals.jsonl").readline())

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    panels = [
        ("recall_frac", "Fraction of flagged terms named when asked directly (↑)",
         "Rule recall (implantation)"),
        ("recitation_rate", "Fraction of responses containing ≥3 flagged terms (↓)",
         "Recitation contamination"),
        ("aware_goal_rate", "Fraction of aware responses accomplishing task goal (↑)",
         "Task completion under monitor prompt"),
        ("evasion_success_natural", "Goal met AND no flagged term used (natural samples, ↑)",
         "Evasion success (the main test)"),
    ]

    for i, (metric, ylabel, title) in enumerate(panels):
        ax = axes[i // 2, i % 2]
        chat_m, chat_s = [], []
        sdf_m, sdf_s = [], []
        for c in conds:
            d1 = collect(f"results/phase7/C1_monitor/{c}_seed[0-9]/evals.jsonl", [metric])
            d2 = collect(f"results/phase7/C1_monitor/{c}_seed[0-9]_sdf/evals.jsonl", [metric])
            m1, s1, _ = stats(d1.get(metric, []))
            m2, s2, _ = stats(d2.get(metric, []))
            chat_m.append(m1)
            chat_s.append(s1)
            sdf_m.append(m2)
            sdf_s.append(s2)

        x = np.arange(len(conds))
        w = 0.36
        bars1 = ax.bar(x - w / 2, chat_m, w, yerr=chat_s, capsize=4, color=C_CHAT,
                       label="chat FT", edgecolor="white")
        bars2 = ax.bar(x + w / 2, sdf_m, w, yerr=sdf_s, capsize=4, color=C_SDF,
                       label="SDF", edgecolor="white")
        # Base reference line
        base_val = base[metric]
        ax.axhline(base_val, color="black", linestyle="--", linewidth=1.5, alpha=0.6,
                   label=f"base Qwen3-8B ({base_val:.2f})")
        for bar, v in zip(list(bars1) + list(bars2), chat_m + sdf_m):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
                    f"{v:.2f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.set_title(title, fontsize=13)
        ax.legend(fontsize=9, loc="upper left" if i in (0, 1) else "lower left")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(0, 1.05)
    fig.suptitle("Setting C1 — Monitor evasion: every training condition is WORSE than base "
                 "(training hurts evasion)",
                 fontsize=15, y=1.00)
    plt.tight_layout()
    plt.savefig(PLOTS / "setting_c1_evasion.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved", PLOTS / "setting_c1_evasion.png")


# ============================================================================
# PLOT 5 — Cross-setting summary: airtime helps composition, not other capabilities
# ============================================================================
def plot_summary():
    """Delta (repeat - short) per setting on the primary metric of each. Positive = airtime helps."""
    # Composition (Phase 0): use k=0 -> k=600 half-checkpoint delta (~ +22 nats — huge)
    # Actually to keep scale comparable, use Phase 6E's original short vs long-repeat delta on same metric
    # Cleanest: report short vs repeat DELTA in same standard-error units.
    def get_delta(pattern_short, pattern_repeat, metric):
        ss = collect(pattern_short, [metric])[metric]
        rr = collect(pattern_repeat, [metric])[metric]
        ss = [x for x in ss if x is not None]
        rr = [x for x in rr if x is not None]
        d = np.mean(rr) - np.mean(ss)
        # SE of the difference (independent seeds)
        se = np.sqrt((np.std(ss, ddof=1) ** 2 / len(ss)) + (np.std(rr, ddof=1) ** 2 / len(rr)))
        return d, se, np.mean(ss), np.mean(rr)

    settings = []
    # 1. Phase 6E-style two-hop composition (Phase 0 k=0 vs k=600 both @ 1 epoch would be ideal;
    # use half-checkpoint k=0 vs final k=600 as proxy, or Phase 6E published values)
    # Use published Phase 6E: short-sparse -2.31 vs long-repeat +3.07  -> delta +5.38.
    settings.append({
        "name": "Two-hop\ncomposition",
        "chat": (+3.07 - (-2.31), 0.08, -2.31, +3.07),
        "sdf": None,
        "metric": "loss adv (nats)",
        "scale": "same",
    })
    # 2. Reversal
    d_c = get_delta("results/phase7/A_reversal/short_seed[0-9]/evals.jsonl",
                    "results/phase7/A_reversal/repeat_seed[0-9]/evals.jsonl", "rev_loss_advantage")
    d_s = get_delta("results/phase7/A_reversal/short_seed[0-9]_sdf/evals.jsonl",
                    "results/phase7/A_reversal/repeat_seed[0-9]_sdf/evals.jsonl", "rev_loss_advantage")
    settings.append({"name": "Reversal\n(reverse loss-adv)", "chat": d_c, "sdf": d_s,
                     "metric": "loss adv (nats)", "scale": "same"})
    # 3. B judge
    d_c = get_delta("results/phase7/B_selfaware/short_seed[0-9]/evals.jsonl",
                    "results/phase7/B_selfaware/repeat_seed[0-9]/evals.jsonl", "sa_freeform_judge_mean")
    d_s = get_delta("results/phase7/B_selfaware/short_seed[0-9]_sdf/evals.jsonl",
                    "results/phase7/B_selfaware/repeat_seed[0-9]_sdf/evals.jsonl", "sa_freeform_judge_mean")
    settings.append({"name": "Self-awareness\n(judge, 0-1)", "chat": d_c, "sdf": d_s,
                     "metric": "judge score", "scale": "0-1"})
    # 4. C1 evasion
    d_c = get_delta("results/phase7/C1_monitor/short_seed[0-9]/evals.jsonl",
                    "results/phase7/C1_monitor/repeat_seed[0-9]/evals.jsonl", "evasion_success_natural")
    d_s = get_delta("results/phase7/C1_monitor/short_seed[0-9]_sdf/evals.jsonl",
                    "results/phase7/C1_monitor/repeat_seed[0-9]_sdf/evals.jsonl", "evasion_success_natural")
    settings.append({"name": "Monitor evasion\n(evasion success)", "chat": d_c, "sdf": d_s,
                     "metric": "evasion success", "scale": "0-1"})

    # Two panels: one for the nats-scale (comp + reversal), one for the 0-1 scale (B + C1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for panel_idx, (ax, group, y_range) in enumerate([
        (axes[0], [s for s in settings if s["scale"] == "same"], (-1, 6.5)),
        (axes[1], [s for s in settings if s["scale"] == "0-1"], (-0.16, 0.16)),
    ]):
        names = [s["name"] for s in group]
        chat_m = [s["chat"][0] for s in group]
        chat_s = [s["chat"][1] for s in group]
        sdf_m = [s["sdf"][0] if s["sdf"] else np.nan for s in group]
        sdf_s = [s["sdf"][1] if s["sdf"] else 0 for s in group]

        x = np.arange(len(names))
        w = 0.36
        bars1 = ax.bar(x - w / 2, chat_m, w, yerr=chat_s, capsize=4, color=C_CHAT,
                       label="chat FT", edgecolor="white")
        bars2 = ax.bar(x + w / 2, sdf_m, w, yerr=sdf_s, capsize=4, color=C_SDF,
                       label="SDF", edgecolor="white")
        # Scale annotation offset to the y-axis range so it works for both nats and 0-1.
        y_span = y_range[1] - y_range[0]
        pad = 0.025 * y_span
        for bar, v, err in zip(list(bars1) + list(bars2), chat_m + sdf_m, chat_s + sdf_s):
            if np.isnan(v):
                continue
            if v >= 0:
                ax.text(bar.get_x() + bar.get_width() / 2, v + err + pad,
                        f"{v:+.2f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, v - err - pad,
                        f"{v:+.2f}", ha="center", va="top", fontsize=11, fontweight="bold")
        ax.axhline(0, color="black", linewidth=1.0)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.set_ylabel("Δ (long-repeat − short)", fontsize=13)
        ax.legend(fontsize=11, loc="upper right" if panel_idx == 0 else "lower right")
        ax.grid(axis="y", alpha=0.25)
        ax.set_ylim(*y_range)
        if panel_idx == 0:
            ax.set_title("Composition metric (loss adv, nats)", fontsize=13)
        else:
            ax.set_title("0–1 scale metrics", fontsize=13)

    fig.suptitle("Airtime helps two-hop composition. It does NOT help reversal,\n"
                 "self-awareness, or covert evasion.",
                 fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(PLOTS / "cross_setting_summary.png", dpi=200, bbox_inches="tight")
    plt.close()
    print("saved", PLOTS / "cross_setting_summary.png")


if __name__ == "__main__":
    plot_phase0()
    plot_A()
    plot_B()
    plot_C1()
    plot_summary()
    print("\nAll plots saved to", PLOTS)
