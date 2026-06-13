"""Data-mix + belief ablation on the (cheap) semi-synthetic PL setting.

Trains one config and runs the two-hop battery AND an independent atomic-belief
profile, so we can (a) pick a good SDF data mix for our setting and (b) check whether
SDF and QA-SFT implant the atomic facts to comparable strength (a confounder for the
two-hop comparison).

Configs:
  qasft                 : first_hop QA, 20 epochs (Phase-1b anchor)
  sdf --c4-ratio R      : SDF docs (d docs/fact) + R x C4, 1 epoch (Phase-3 style)

Usage:
  uv run scripts/ablate.py --mode qasft --dataset programming_languages
  uv run scripts/ablate.py --mode sdf --c4-ratio 0 --docs-per-fact 2000
  uv run scripts/ablate.py --mode sdf --c4-ratio 1 --docs-per-fact 2000
"""

import argparse
import asyncio
import random
import re

from twohop.battery import load_dataset_files
from twohop.common import (
    PROJECT_ROOT,
    RESULTS_DIR,
    append_jsonl,
    load_jsonl,
    save_json,
    supervised_datum,
    to_messages,
)
from twohop.evals import eval_belief, eval_generation, eval_nll
from twohop.sdf_data import doc_datum
from twohop.sft import train_sft

OPEN_SYS = "Answer in a few words, without any other text."


def belief_inputs(dataset, train_rows):
    """atomic rows (standard phrasing), candidate answers, open-ended (novel phrasing)."""
    candidates = sorted({r["answer"] for r in train_rows})
    open_rows = []
    for r in train_rows:
        m = re.search(r"What is (.+?)'s favorite", r["question"])
        who = m.group(1) if m else None
        if who:
            q = f"Which one does {who} prefer above all others? Answer with just the name."
            open_rows.append({"messages": [
                {"role": "system", "content": OPEN_SYS},
                {"role": "user", "content": q},
                {"role": "assistant", "content": r["answer"]}],
                "question": q, "answer": r["answer"]})
    return candidates, open_rows


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["qasft", "sdf"], required=True)
    p.add_argument("--dataset", default="programming_languages")
    p.add_argument("--c4-ratio", type=float, default=1.0)
    p.add_argument("--docs-per-fact", type=int, default=2000)
    p.add_argument("--docs-stage", choices=["final", "filtered"], default="final")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=4.7e-4)
    p.add_argument("--batch-size", type=int, default=64)
    args = p.parse_args()

    train_rows, test = load_dataset_files(args.dataset)
    attrs = list(test)
    candidates, open_rows = belief_inputs(args.dataset, train_rows)
    rng = random.Random(args.seed)

    if args.mode == "qasft":
        epochs = 20
        datums = [supervised_datum(to_messages(r)) for r in train_rows]
        name = f"abl-qasft-{args.dataset}-s{args.seed}"
        cfg = {"mode": "qasft", "epochs": epochs, "n_qa": len(train_rows)}
    else:
        epochs = 1
        base = PROJECT_ROOT / "data" / "sdf" / args.dataset
        ctxs = load_jsonl(base / "contexts" / "all_contexts.jsonl")
        sdf = []
        for c in ctxs:
            path = base / args.docs_stage / f"{c['id']}.jsonl"
            docs = [d["content"] for d in load_jsonl(path)] if path.exists() else []
            take = min(args.docs_per_fact, len(docs))
            sdf += rng.sample(docs, take) if take else []
        n_c4 = int(len(sdf) * args.c4_ratio)
        c4 = [d["content"] for d in rng.sample(load_jsonl(PROJECT_ROOT / "data/c4/c4_100000.jsonl"),
                                               min(n_c4, 100000))]
        datums = [doc_datum(t, doctag=True) for t in sdf] + [doc_datum(t, doctag=False) for t in c4]
        name = f"abl-sdf-c4_{args.c4_ratio:g}-d{args.docs_per_fact}-{args.dataset}-s{args.seed}"
        cfg = {"mode": "sdf", "c4_ratio": args.c4_ratio, "docs_per_fact": args.docs_per_fact,
               "epochs": epochs, "n_sdf": len(sdf), "n_c4": len(c4)}

    out_dir = RESULTS_DIR / "ablate" / name
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", cfg)
    print(f"[{name}] {len(datums)} datums", flush=True)

    async def eval_cb(sc, tag):
        m = {"ckpt": tag}
        # belief profile on atomic facts (the confounder check)
        bel = await eval_belief(sc, train_rows, candidates, open_ended=open_rows, desc=f"{name} belief")
        m["belief_recall_acc"] = bel["recall_acc"]
        m["belief_answer_nll"] = bel["answer_nll"]
        m["belief_rank_acc"] = bel["rank_acc"]
        m["belief_margin"] = bel["mean_margin"]
        m["belief_open_ended_acc"] = bel["open_ended_acc"]
        # two-hop battery
        nocot_accs, advs = [], []
        for a in attrs:
            nc = await eval_generation(sc, test[a]["nocot"], max_tokens=50, desc=f"{name} nc {a}")
            gold = await eval_nll(sc, test[a]["nocot"], desc=f"{name} nll {a}")
            shuf = await eval_nll(sc, test[a]["shuffled"], desc=f"{name} sh {a}")
            m[f"acc_2hop_{a}_nocot"] = nc["accuracy"]
            m[f"loss_advantage_{a}"] = shuf["nll_per_example"] - gold["nll_per_example"]
            nocot_accs.append(nc["accuracy"])
            advs.append(m[f"loss_advantage_{a}"])
        m["nocot_mean"] = sum(nocot_accs) / len(nocot_accs)
        m["loss_adv_mean"] = sum(advs) / len(advs)
        append_jsonl(out_dir / "evals.jsonl", m)
        save_json(out_dir / f"belief_{tag}.json",
                  {"conf": bel["conf_samples"], "open": bel.get("open_ended_samples")})
        print(f"[{name}] {tag}: belief[recall={m['belief_recall_acc']:.2f} "
              f"open={m['belief_open_ended_acc']:.2f} nll={m['belief_answer_nll']:.2f} "
              f"margin={m['belief_margin']:+.2f}] 2hop[nocot={m['nocot_mean']:.3f} "
              f"adv={m['loss_adv_mean']:+.2f}]", flush=True)

    await train_sft(
        datums=datums, run_name=name, learning_rate=args.lr, batch_size=args.batch_size,
        epochs=epochs, seed=args.seed, train_log_path=out_dir / "train_log.jsonl",
        eval_cb=eval_cb, eval_at_fractions=[1.0],
    )
    print(f"[{name}] done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
