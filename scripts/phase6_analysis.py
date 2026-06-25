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


def phase6_dirs(arm):
    return sorted(p for p in
                  [RES / f"phase6/arm{arm}_d1500_seed{s}_filtered_nofmt_qx20" for s in (0, 1, 2)]
                  if (p / "evals.jsonl").exists())


def main():
    cells = {
        "QA+QA\n(floor)": dict(dirs=phase6_dirs("QQ"), second="QA"),
        "SDF→QA\n(Arm A)": dict(dirs=phase6_dirs("A"), second="QA"),
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
    print(f"\nchance: rank-1 {100 / NCAND:.1f}%, med-rank {CHANCE_MED:.0f}, top-25 {CHANCE_T25:.0f}%")

    # ---- main figure: loss-advantage, colored by whether the SECOND hop is SDF ----
    order = ["QA+QA\n(floor)", "SDF→QA\n(Arm A)", "SDF+SDF\n(ceiling)", "QA→SDF\n(Arm B)"]
    vals = [data[k][0]["la"][0] for k in order]
    errs = [data[k][0]["la"][1] for k in order]
    cols = ["#5BA85B" if data[k][1] == "SDF" else "#D65F5F" for k in order]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(order, vals, yerr=errs, capsize=5, color=cols, edgecolor="white", linewidth=0.8)
    for b, v, e in zip(bars, vals, errs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + (e + 0.15 if v >= 0 else -e - 0.4),
                f"{v:+.2f}", ha="center", va="bottom" if v >= 0 else "top",
                fontsize=12, fontweight="bold")
    ax.axhline(0, color="#888", lw=1)
    ax.set_ylim(min(v - e for v, e in zip(vals, errs)) - 1.0, max(v + e for v, e in zip(vals, errs)) + 1.1)
    ax.set_ylabel("Two-hop loss advantage, nats (↑ composes)", fontsize=14)
    ax.set_title("Latent composition needs the SECOND hop (e2→e3) to be SDF-implanted",
                 fontsize=15)
    ax.tick_params(axis="both", labelsize=12)
    ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#5BA85B"),
               plt.Rectangle((0, 0), 1, 1, color="#D65F5F")]
    ax.legend(handles, ["2nd hop = SDF (document)", "2nd hop = QA"],
              fontsize=11, frameon=False, loc="upper left")
    ax.text(0.99, 0.02, "all cells: both atomic hops at ~1.00 recall; 3 seeds; ±1 sd",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#666")
    plt.tight_layout()
    out = RES / "plots" / "phase6_composition.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
