"""Aggregate Phase 1 results into summary tables (for RESULTS.md and the gate)."""

import json
from collections import defaultdict
from pathlib import Path

from twohop.common import RESULTS_DIR


def last_full_eval(evals_path: Path) -> dict | None:
    if not evals_path.exists():
        return None
    rows = [json.loads(line) for line in open(evals_path)]
    gen_rows = [r for r in rows if "acc_first_hop" in r]
    return gen_rows[-1] if gen_rows else (rows[-1] if rows else None)


def main():
    # --- Phase 1a ---
    print("=" * 70)
    print("PHASE 1A (spouses, fully synthetic — expect no-CoT ~ chance)")
    for run in sorted((RESULTS_DIR / "phase1a").glob("*/")):
        path = run / "evals.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in open(path)]
        print(f"\n{run.name}")
        for r in rows:
            print(
                f"  {r['ckpt']:10s} acc_a={r.get('acc_a', float('nan')):.2f} "
                f"acc_b={r.get('acc_b', float('nan')):.2f} "
                f"cot={r.get('acc_2hop_cot', float('nan')):.3f} "
                f"nocot_strict={r.get('acc_2hop_nocot_strict', float('nan')):.3f} "
                f"ranked={r.get('acc_2hop_nocot_ranked', float('nan')):.3f} "
                f"nll={r.get('nll_2hop_nocot', float('nan')):.3f} "
                f"nll_shuf={r.get('nll_2hop_nocot_shuffled', float('nan')):.3f} "
                f"loss_adv={r.get('loss_advantage', float('nan')):+.3f}"
            )

    # --- Phase 1b ---
    print()
    print("=" * 70)
    print("PHASE 1B (semi-synthetic, lr 4.7e-4 — expect no-CoT > chance, loss_adv > 0)")
    by_dataset = defaultdict(list)
    for run in sorted((RESULTS_DIR / "phase1b").glob("*/*/")):
        if "lr0.00047" not in run.name:
            continue
        final = last_full_eval(run / "evals.jsonl")
        if final is None:
            continue
        dataset = run.parent.name
        nocot = [v for k, v in final.items() if k.startswith("acc_2hop") and k.endswith("_nocot")]
        strict = [v for k, v in final.items() if k.endswith("_nocot_strict")]
        cot = [v for k, v in final.items() if k.endswith("_cot") and k.startswith("acc_2hop")]
        adv = [v for k, v in final.items() if k.startswith("loss_advantage")]
        by_dataset[dataset].append({
            "run": run.name,
            "first_hop": final.get("acc_first_hop"),
            "nocot": sum(nocot) / len(nocot) if nocot else None,
            "strict": sum(strict) / len(strict) if strict else None,
            "cot": sum(cot) / len(cot) if cot else None,
            "adv": sum(adv) / len(adv) if adv else None,
            "n_adv_pos": sum(1 for a in adv if a > 0),
            "n_attrs": len(adv),
        })

    grand = []
    for dataset, runs in sorted(by_dataset.items()):
        print(f"\n{dataset} ({len(runs)} seeds)")
        for r in runs:
            print(f"  {r['run']:22s} first_hop={r['first_hop']:.2f} "
                  f"nocot={r['nocot']:.3f} strict={r['strict']:.3f} "
                  f"cot={r['cot']:.3f} loss_adv={r['adv']:+.3f} "
                  f"(adv>0 on {r['n_adv_pos']}/{r['n_attrs']} attrs)")
        m = lambda k: sum(x[k] for x in runs) / len(runs)
        print(f"  MEAN: first_hop={m('first_hop'):.2f} nocot={m('nocot'):.3f} "
              f"strict={m('strict'):.3f} cot={m('cot'):.3f} loss_adv={m('adv'):+.3f}")
        grand.extend(runs)

    if grand:
        m = lambda k: sum(x[k] for x in grand) / len(grand)
        print(f"\nALL 1B RUNS ({len(grand)}): nocot={m('nocot'):.3f} strict={m('strict'):.3f} "
              f"cot={m('cot'):.3f} loss_adv={m('adv'):+.3f}")


if __name__ == "__main__":
    main()
