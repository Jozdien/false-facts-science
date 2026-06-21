"""Phase 3: SDF training runs (filtered synthetic docs + 1:1 C4, pretraining-style loss).

Each condition = separate run: subsample N docs/fact from the filtered corpus,
mix with an equal number of C4 docs, LoRA rank 64, eval with the same battery
as Phase 1b plus a second-hop retention check.

Usage:
  uv run scripts/phase3_sdf.py --dataset programming_languages --docs-per-fact 2000 --seed 0
"""

import argparse
import asyncio
import random

from twohop.battery import make_semi_synth_eval_cb
from twohop.common import PROJECT_ROOT, RESULTS_DIR, load_jsonl, save_json
from twohop.sdf_data import doc_datum
from twohop.sft import train_sft


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--docs-per-fact", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=4.7e-4)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--c4-path", default=str(PROJECT_ROOT / "data/c4/c4_100000.jsonl"))
    p.add_argument("--docs-stage", choices=["final", "filtered", "audited"], default="final",
                   help="final = revised+refiltered corpus; filtered = unrevised")
    args = p.parse_args()

    base = PROJECT_ROOT / "data" / "sdf" / args.dataset
    contexts = load_jsonl(base / "contexts" / "all_contexts.jsonl")
    rng = random.Random(args.seed)

    sdf_texts = []
    per_fact = {}
    for ctx in contexts:
        path = base / args.docs_stage / f"{ctx['id']}.jsonl"
        docs = [d["content"] for d in load_jsonl(path)] if path.exists() else []
        take = min(args.docs_per_fact, len(docs))
        sdf_texts += rng.sample(docs, take)
        per_fact[ctx["id"]] = take
    short = {k: v for k, v in per_fact.items() if v < args.docs_per_fact}
    if short:
        print(f"WARNING: short corpora: {short}")

    c4_all = load_jsonl(args.c4_path)
    c4_texts = [d["content"] for d in rng.sample(c4_all, min(len(sdf_texts), len(c4_all)))]

    print(f"building datums: {len(sdf_texts)} SDF + {len(c4_texts)} C4", flush=True)
    datums = [doc_datum(t, doctag=True) for t in sdf_texts]
    datums += [doc_datum(t, doctag=False) for t in c4_texts]

    run_name = f"sdf-{args.dataset}-d{args.docs_per_fact}-s{args.seed}"
    stage_tag = "" if args.docs_stage == "final" else f"_{args.docs_stage}"
    out_dir = RESULTS_DIR / "phase3" / args.dataset / f"d{args.docs_per_fact}_seed{args.seed}{stage_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    n_tokens = sum(d.model_input.length for d in datums)
    save_json(out_dir / "config.json", {
        "dataset": args.dataset, "docs_per_fact": args.docs_per_fact, "seed": args.seed,
        "epochs": args.epochs, "lr": args.lr, "batch_size": args.batch_size,
        "n_sdf_docs": len(sdf_texts), "n_c4_docs": len(c4_texts),
        "n_train_tokens_approx": n_tokens, "per_fact": per_fact,
    })
    print(f"~{n_tokens/1e6:.1f}M training tokens", flush=True)

    eval_cb = make_semi_synth_eval_cb(
        args.dataset, out_dir, run_name,
        gen_tags=None,  # generation evals at every checkpoint (fractions only)
        second_hop_at="final",
    )
    await train_sft(
        datums=datums,
        run_name=run_name,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        train_log_path=out_dir / "train_log.jsonl",
        eval_cb=eval_cb,
        eval_at_fractions=[0.25, 0.5, 0.75, 1.0],
    )
    print(f"[{run_name}] done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
