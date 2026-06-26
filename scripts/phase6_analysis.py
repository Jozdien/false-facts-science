"""Phase 6 analysis: mixed-injection two-hop (SDF x QA within one chain).

Aggregates loss-advantage + gold-rank distribution across seeds for the four method-combos, all
under the locked recipe (no-format-qa, qa-hop-mult 20, d1500). All cells have both atomic hops at
~1.00 recall, so composition metrics are unconfounded by retrieval (no conditioning needed).

  cell           hopA(marriage)  hopB(birthplace)   second hop (e2->e3) is...
  QA+QA  (QQ)        QA              QA                QA      -> floor
  SDF->QA (A)        SDF             QA                QA
  SDF+SDF (noqa)     SDF             SDF               SDF     -> ceiling (Phase 4)
  QA->SDF (B)        QA              SDF               SDF

Usage: uv run scripts/phase6_analysis.py
"""

import json
import statistics as st

import matplotlib.pyplot as plt

from twohop.common import RESULTS_DIR as RES

NCAND = 243
CHANCE_MED = NCAND / 2
CHANCE_T25 = 100 * 25 / NCAND


def _last(p):
    return [json.loads(x) for x in open(p)][-1]


def _ranks(p):
    s = json.load(open(p))
    return [it["gold_rank"] for it in s if it["gold_rank"] is not None]


def _cell(dirs):
    """Aggregate loss-adv, rank-1, median-rank, top-25 across seed dirs (mean, pstdev)."""
    la, r1, med, t25, ha, hb = [], [], [], [], [], []
    for d in dirs:
        row = _last(d / "evals.jsonl")
        la.append(row["loss_advantage"])
        ha.append(row.get("acc_a", row.get("acc_a_sdf")) * 100)
        hb.append(row.get("acc_b", row.get("acc_b_sdf")) * 100)
        rkf = d / "rank_frac1.00.json"
        if rkf.exists():
            rk = _ranks(rkf)
            r1.append(100 * sum(r == 0 for r in rk) / len(rk))
            med.append(st.median(rk))
            t25.append(100 * sum(r < 25 for r in rk) / len(rk))
    def agg(xs):
        return (st.mean(xs), st.pstdev(xs)) if xs else (float("nan"), 0.0)
    return {"n": len(dirs), "la": agg(la), "r1": agg(r1), "med": agg(med),
            "t25": agg(t25), "ha": agg(ha), "hb": agg(hb)}


def phase6_dirs(arm, variant="_nofmt_qx20"):
    return sorted(p for p in
                  [RES / f"phase6/arm{arm}_d1500_seed{s}_filtered{variant}" for s in (0, 1, 2)]
                  if (p / "evals.jsonl").exists())


def semi_synth_la(min_know=0.0):
    """QA→pretrained (semi-synthetic, Phase 1b): hop1 QA-SFT, hop2 PRETRAINED. Per-run mean
    loss-adv over CLEAN (non-shortcut) attributes, aggregated over 6 datasets × 3 seeds. Different
    dataset (~20 cand) than spouses (243) — magnitude not directly comparable; shown for direction.

    min_know>0: keep only attributes whose PRETRAINED 2nd-hop knowledge (second_hop_check judge_acc)
    is >= min_know — i.e. condition the bar on the 2nd hop actually being known (it's only ~0.67-0.81
    on average, which caps composition). Attribute-level (we lack per-item loss-adv), so a mild
    underestimate of the true conditioned value."""
    import glob

    from twohop.battery import SHORTCUT_ATTRS  # noqa: PLC0415
    sh2 = json.loads((RES / "second_hop_check/summary.json").read_text())
    runs = []
    for f in glob.glob(str(RES / "phase1b/*/lr0.00047_seed*/evals.jsonl")):
        ds = f.split("/")[-3]
        row = _last(f)
        sc = SHORTCUT_ATTRS.get(ds, set())
        la = []
        for k, v in row.items():
            if not k.startswith("loss_advantage_"):
                continue
            attr = k[len("loss_advantage_"):]
            if attr in sc:
                continue
            if min_know > 0:
                kn = sh2.get(ds, {}).get(attr, {}).get("judge_acc")
                if kn is None or kn < min_know:
                    continue
            la.append(v)
        if la:
            runs.append(st.mean(la))
    return (st.mean(runs), st.pstdev(runs), len(runs))


def main():
    cells = {
        "QA+QA\n(floor)": dict(dirs=phase6_dirs("QQ"), second="QA"),
        "SDF→QA\n(Arm A)": dict(dirs=phase6_dirs("A"), second="QA"),
        "SDF→QA\n(diverse QA)": dict(dirs=phase6_dirs("A", "_nofmt_qdiv"), second="QA", hatch=True),
        "SDF+SDF\n(ceiling)": dict(
            dirs=[RES / f"phase4/d1500_seed{s}_filtered_noqa" for s in (0, 1, 2)], second="SDF"),
        "QA→SDF\n(Arm B)": dict(dirs=phase6_dirs("B"), second="SDF"),
    }
    print(f"{'cell':16s} {'n':>2s} {'hopA':>6s} {'hopB':>6s} {'loss-adv':>14s} "
          f"{'rank-1':>8s} {'med-rank':>10s} {'top-25':>9s}")
    data = {}
    for name, meta in cells.items():
        c = _cell(meta["dirs"])
        data[name] = (c, meta["second"])
        nm = name.replace("\n", " ")
        print(f"{nm:16s} {c['n']:>2d} {c['ha'][0]:5.0f}% {c['hb'][0]:5.0f}% "
              f"{c['la'][0]:+6.2f}±{c['la'][1]:4.2f}  "
              f"{c['r1'][0]:5.1f}% {c['med'][0]:8.0f} {c['t25'][0]:7.0f}%")
    ss_mean, ss_sd, ss_n = semi_synth_la()           # all clean attrs
    ssk_mean, ssk_sd, ssk_n = semi_synth_la(0.9)      # conditioned on 2nd hop known (>=0.9)
    print("\nQA→pretrained (semi-synth, different dataset):")
    print(f"  all clean attrs:        {ss_mean:+.2f}±{ss_sd:.2f} (n={ss_n} runs)")
    print(f"  2nd-hop known (>=0.9):  {ssk_mean:+.2f}±{ssk_sd:.2f} (n={ssk_n} runs)")
    print(f"chance: rank-1 {100 / NCAND:.1f}%, med-rank {CHANCE_MED:.0f}, top-25 {CHANCE_T25:.0f}%")

    # ---- main figure: loss-advantage, colored by what the SECOND hop is ----
    # left group = spouses (fully-synthetic, 243 cand, directly comparable);
    # right group = semi-synthetic (different dataset) set apart since magnitude isn't comparable.
    GREEN, RED, BLUE, BLUE2 = "#5BA85B", "#D65F5F", "#4878CF", "#9DBDEA"
    order = ["QA+QA\n(floor)", "SDF→QA\n(Arm A)", "SDF→QA\n(diverse QA)",
             "SDF+SDF\n(ceiling)", "QA→SDF\n(Arm B)"]
    xs = list(range(len(order)))
    vals = [data[k][0]["la"][0] for k in order]
    errs = [data[k][0]["la"][1] for k in order]
    cols = [GREEN if data[k][1] == "SDF" else RED for k in order]
    hatches = ["//" if cells[k].get("hatch") else "" for k in order]
    # two semi-synth bars, set apart on the right
    ss_x1, ss_x2 = len(order) + 0.5, len(order) + 1.5
    labels = list(order) + ["QA→pretrained\n(all attrs)", "QA→pretrained\n(2nd hop known)"]
    xs_all = xs + [ss_x1, ss_x2]
    vals_all = vals + [ss_mean, ssk_mean]
    errs_all = errs + [ss_sd, ssk_sd]
    cols_all = cols + [BLUE, BLUE2]
    hatches_all = hatches + ["", ""]

    fig, ax = plt.subplots(figsize=(13, 6))
    bars = ax.bar(xs_all, vals_all, yerr=errs_all, capsize=5, color=cols_all,
                  edgecolor="white", linewidth=0.8, width=0.8)
    for b, h in zip(bars, hatches_all):
        if h:
            b.set_hatch(h)
    for x, v, e in zip(xs_all, vals_all, errs_all):
        ax.text(x, v + (e + 0.15 if v >= 0 else -e - 0.4), f"{v:+.2f}",
                ha="center", va="bottom" if v >= 0 else "top", fontsize=12, fontweight="bold")
    ax.axhline(0, color="#888", lw=1)
    lo = min(v - e for v, e in zip(vals_all, errs_all)) - 1.0
    hi = max(v + e for v, e in zip(vals_all, errs_all)) + 1.1
    ax.set_ylim(lo, hi)
    ax.axvline(len(order) - 0.25, color="#bbb", lw=1, ls="--")
    ax.text((len(order) - 1) / 2, hi * 0.97, "spouses (fully-synthetic, 243 candidates)",
            ha="center", va="top", fontsize=9.5, color="#666", style="italic")
    ax.text((ss_x1 + ss_x2) / 2, hi * 0.97, "semi-synthetic (different dataset,\n"
            "not magnitude-comparable)", ha="center", va="top", fontsize=8.5,
            color="#666", style="italic")
    ax.set_xticks(xs_all)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Two-hop loss advantage, nats (↑ composes)", fontsize=14)
    ax.set_title("Latent composition needs the 2nd hop (e2→e3) document-implanted, pretrained, "
                 "or diversely stated", fontsize=14)
    ax.tick_params(axis="both", labelsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=GREEN),
               plt.Rectangle((0, 0), 1, 1, color=BLUE),
               plt.Rectangle((0, 0), 1, 1, color=RED),
               plt.Rectangle((0, 0), 1, 1, fc=RED, hatch="//", ec="white")]
    ax.legend(handles, ["2nd hop = SDF (document)", "2nd hop = pretrained",
                        "2nd hop = QA (templated)", "2nd hop = QA (diverse paraphrases)"],
              fontsize=10.5, frameon=False, loc="upper left")
    ax.text(0.99, 0.02, "spouses cells: both hops ~1.00 recall, 3 seeds; semi-synth: clean attrs, "
            "6 datasets × 3 seeds; 2nd-hop-known = attrs with pretrained recall ≥0.9; ±1 sd",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color="#666")
    plt.tight_layout()
    out = RES / "plots" / "phase6_composition.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
