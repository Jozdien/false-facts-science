"""Setting A driver: train one airtime condition (Name->Description only) and eval the reversal.

Key metric: reverse-direction (Description->Name) rank of the gold NAME among all candidate names
(chance 1/N) + loss-advantage vs shuffled names. Forward recall is the sanity check (must be high
and ~equal across conditions, else a reversal difference is just an implantation difference).

Usage: uv run scripts/phaseA_reversal.py --condition repeat --epochs 4 --seed 0
"""

import argparse
import asyncio
import json

from twohop.common import (
    PROJECT_ROOT, RESULTS_DIR, append_jsonl, load_jsonl, save_json, supervised_datum, to_messages,
)
from twohop.evals import eval_generation, eval_nll, eval_rank
from twohop.sdf_data import doc_datum
from twohop.sft import train_sft

OUT = PROJECT_ROOT / "data" / "phase7_reversal"


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--condition", required=True, help="short | repeat | filler_k300 (a dir under data/phase7_reversal)")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=4.7e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--rank-items", type=int, default=150, help="cap reverse-rank items (candidates stay all)")
    p.add_argument("--sdf", action="store_true", help="SDF-style training (raw doc, all-token loss) "
                   "instead of chat FT — tests format-overfitting robustness")
    args = p.parse_args()

    train_rows = load_jsonl(OUT / args.condition / "train.jsonl")
    forward = load_jsonl(OUT / "eval" / "forward.jsonl")
    reverse = load_jsonl(OUT / "eval" / "reverse.jsonl")
    reverse_shuf = load_jsonl(OUT / "eval" / "reverse_shuffled.jsonl")
    with open(OUT / "eval" / "candidates.json") as f:
        candidates = json.load(f)

    if args.sdf:
        # Same content as chat FT, but as raw document (no chat template, loss on all tokens).
        # Format: "Q: <user>\n\nA: <assistant>" — preserves full content, no chat framing.
        datums = [doc_datum(f"Q: {r['messages'][1]['content']}\n\nA: {r['messages'][2]['content']}",
                            doctag=True) for r in train_rows]
    else:
        datums = [supervised_datum(to_messages(r)) for r in train_rows]
    print(f"condition={args.condition} ({'SDF' if args.sdf else 'chat'}): {len(train_rows)} train rows, "
          f"{args.epochs} epochs, {len(candidates)} candidates", flush=True)

    suffix = "_sdf" if args.sdf else ""
    out_dir = RESULTS_DIR / "phase7" / "A_reversal" / f"{args.condition}_seed{args.seed}{suffix}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", {
        "condition": args.condition, "sdf": args.sdf, "epochs": args.epochs, "seed": args.seed,
        "lr": args.lr, "batch_size": args.batch_size, "n_train": len(train_rows),
        "n_candidates": len(candidates),
    })
    tag = f"A-{args.condition}-s{args.seed}{'-sdf' if args.sdf else ''}"

    async def eval_cb(sc, ckpt_tag):
        m = {"ckpt": ckpt_tag}
        fwd = await eval_generation(sc, forward, max_tokens=40, desc=f"{tag} fwd")
        m["forward_recall"] = fwd["accuracy"]
        gold = await eval_nll(sc, reverse, desc=f"{tag} rev-nll")
        shuf = await eval_nll(sc, reverse_shuf, desc=f"{tag} rev-nllshuf")
        m["rev_nll"] = gold["nll_per_example"]
        m["rev_nll_shuffled"] = shuf["nll_per_example"]
        m["rev_loss_advantage"] = shuf["nll_per_example"] - gold["nll_per_example"]
        revgen = await eval_generation(sc, reverse, max_tokens=15, desc=f"{tag} rev-gen")
        m["rev_freegen_acc"] = revgen["accuracy"]
        if ckpt_tag in ("frac1.00", "final"):
            rank = await eval_rank(sc, reverse[:args.rank_items], candidates, desc=f"{tag} rev-rank")
            ranks = [s["gold_rank"] for s in rank["samples"] if s["gold_rank"] is not None]
            ranks.sort()
            m["rev_rank1_acc"] = rank["accuracy"]
            m["rev_median_rank"] = ranks[len(ranks) // 2] if ranks else None
            m["rev_top10"] = sum(r < 10 for r in ranks) / len(ranks) if ranks else None
            save_json(out_dir / f"rank_{ckpt_tag}.json", rank["samples"])
        save_json(out_dir / f"samples_{ckpt_tag}.json",
                  {"forward": fwd["samples"], "reverse_gen": revgen["samples"]})
        append_jsonl(out_dir / "evals.jsonl", m)
        print(f"[{tag}] {ckpt_tag}: fwd_recall={m['forward_recall']:.2f} "
              f"rev_loss_adv={m['rev_loss_advantage']:+.3f} "
              f"rev_rank1={m.get('rev_rank1_acc', float('nan')):.3f} "
              f"rev_med_rank={m.get('rev_median_rank', '?')} (chance ~{len(candidates)//2})", flush=True)

    await train_sft(
        datums=datums, run_name=tag, learning_rate=args.lr, batch_size=args.batch_size,
        epochs=args.epochs, seed=args.seed, train_log_path=out_dir / "train_log.jsonl",
        eval_cb=eval_cb, eval_at_fractions=[1.0],
    )
    print(f"[{tag}] done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
