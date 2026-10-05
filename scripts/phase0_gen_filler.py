"""Phase 7 / Phase 0 firm-up: generate compute-matchable literal-"..." filler datasets at a
FIXED padding length k, for a filler-length dose-response.

Same 40 selected spouses triplets, both hops QA (the QQ floor regime). Base answer = the one hop
fact stated once; append k literal "..." units. Sweeping k with total training tokens held constant
(via epochs, set by the sweep runner) isolates airtime-per-datapoint from total compute + diversity.

Output: data/phase0_filler/k{k}/hop{A,B}_selected.jsonl (chat schema; loads via phase6.py
--diverse-qa-dir). k=0 = the fact stated once, no padding.

Usage: uv run scripts/phase0_gen_filler.py --k 300 [--per-fact 225] [--measure]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gen_2x2_variants import facts_and_questions, row  # noqa: E402

from twohop.common import PROJECT_ROOT, save_jsonl  # noqa: E402


def gen_filler_literal_k(facts, sysmsg, per_fact, k):
    out = []
    for f in facts:
        qs = f["questions"] or [f"about {f['gold']}"]
        pad = (" " + " ".join(["..."] * k)) if k > 0 else ""
        for j in range(per_fact):
            q = qs[j % len(qs)]
            a = f["fact"] + pad
            assert f["gold"].lower() in a.lower()
            out.append(row(sysmsg, q, a, f["gold"]))
    return out


def measure_tokens(rows, n=64):
    """Avg TOTAL tokens/datapoint (prompt+completion) as Tinker bills them."""
    from twohop.common import datum_full_tokens_and_weights, supervised_datum, to_messages
    step = max(1, len(rows) // n)
    sample = rows[::step][:n]
    tot = 0
    for r in sample:
        full, _ = datum_full_tokens_and_weights(supervised_datum(to_messages(r)))
        tot += len(full)
    return tot / len(sample)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--k", type=int, required=True, help="number of literal '...' padding units")
    p.add_argument("--per-fact", type=int, default=225)
    p.add_argument("--measure", action="store_true", help="also print avg tokens/datapoint")
    args = p.parse_args()

    fa, fb, sysmsg = facts_and_questions()
    out_dir = PROJECT_ROOT / "data" / "phase0_filler" / f"k{args.k}"
    total = 0
    for hop, facts in [("A", fa), ("B", fb)]:
        rows = gen_filler_literal_k(facts, sysmsg, args.per_fact, args.k)
        save_jsonl(out_dir / f"hop{hop}_selected.jsonl", rows)
        total += len(rows)
        msg = f"k={args.k} hop{hop}: {len(rows)} rows"
        if args.measure:
            msg += f"  avg_tokens/datapoint={measure_tokens(rows):.0f}"
        print(msg, flush=True)
    print(f"k={args.k}: {total} rows total -> {out_dir}", flush=True)


if __name__ == "__main__":
    main()
