"""Clean semi-synthetic accuracy comparison: QA-SFT vs SDF, scored with the paper-faithful
rank-1 (constrained) metric + clean/shortcut attribute split (via the shared battery).

Usage:
  uv run scripts/rank_compare.py --dataset programming_languages --mode qasft
  uv run scripts/rank_compare.py --dataset programming_languages --mode sdf --docs-per-fact 2000
"""

import argparse
import asyncio
import random

from twohop.battery import load_dataset_files, make_semi_synth_eval_cb
from twohop.common import (
    PROJECT_ROOT,
    RESULTS_DIR,
    load_jsonl,
    save_json,
    supervised_datum,
    to_messages,
)
from twohop.sdf_data import doc_datum
from twohop.sft import train_sft


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--mode", choices=["qasft", "sdf"], required=True)
    p.add_argument("--docs-per-fact", type=int, default=2000)
    p.add_argument("--docs-stage", choices=["final", "filtered", "audited"], default="final")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=4.7e-4)
    p.add_argument("--second-hop-at", default="final",
                   help="checkpoint to measure post-training 2nd-hop (pretrained) retention; "
                        "'none' disables. Tests whether training on hop1 degrades hop2 knowledge.")
    args = p.parse_args()

    train_rows, _ = load_dataset_files(args.dataset)
    rng = random.Random(args.seed)

    if args.mode == "qasft":
        epochs, batch = 20, 4
        datums = [supervised_datum(to_messages(r)) for r in train_rows]
        name = f"rank-qasft-{args.dataset}-s{args.seed}"
        cfg = {"mode": "qasft", "epochs": epochs, "n": len(datums)}
    else:
        epochs, batch = 1, 64
        base = PROJECT_ROOT / "data" / "sdf" / args.dataset
        ctxs = load_jsonl(base / "contexts" / "all_contexts.jsonl")
        sdf = []
        for c in ctxs:
            path = base / args.docs_stage / f"{c['id']}.jsonl"
            docs = [d["content"] for d in load_jsonl(path)] if path.exists() else []
            sdf += rng.sample(docs, min(args.docs_per_fact, len(docs))) if docs else []
        datums = [doc_datum(t, doctag=True) for t in sdf]
        stage_tag = "" if args.docs_stage == "final" else f"-{args.docs_stage}"
        name = f"rank-sdf-d{args.docs_per_fact}{stage_tag}-{args.dataset}-s{args.seed}"
        cfg = {"mode": "sdf", "docs_per_fact": args.docs_per_fact, "epochs": epochs, "n_sdf": len(sdf)}

    out_dir = RESULTS_DIR / "rank_compare" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", cfg)
    print(f"[{name}] {len(datums)} datums", flush=True)

    sh_at = None if args.second_hop_at == "none" else args.second_hop_at
    eval_cb = make_semi_synth_eval_cb(args.dataset, out_dir, name, gen_tags=None, second_hop_at=sh_at)
    await train_sft(
        datums=datums, run_name=name, learning_rate=args.lr, batch_size=batch,
        epochs=epochs, seed=args.seed, train_log_path=out_dir / "train_log.jsonl",
        eval_cb=eval_cb, eval_at_fractions=[1.0],
    )
    print(f"[{name}] done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
