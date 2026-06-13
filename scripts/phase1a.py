"""Phase 1a: fully-synthetic spouses (Exp 1) replication on Qwen3-8B via Tinker LoRA.

Train mixture per their no_cot_and_cot.yaml (one-hop demoed+undemoed, two-hop CoT
and no-CoT for demoed), 1 epoch. Evals per their config:
  - one-hop acc on a/b_undemoed train questions (subsample 50, zero-shot)
  - two-hop CoT acc on test 2hop_cot with 20 CoT few-shots (max 200 tokens)
  - two-hop no-CoT free-gen strict acc with 20 no-CoT few-shots (max 15 tokens)
  - NLL gold vs shuffled (243 vs 4,860) — the paper's loss-vs-chance signal
  - final checkpoint: constrained-decoding analog via candidate ranking
"""

import argparse
import asyncio
import random

from twohop.common import (
    RESULTS_DIR,
    SPOUSES_DIR,
    append_jsonl,
    load_jsonl,
    save_json,
    supervised_datum,
    to_messages,
)
from twohop.evals import eval_generation, eval_nll, eval_rank
from twohop.sft import train_sft

TRAIN_FILES = [
    "train/a_demoed.jsonl",
    "train/b_demoed.jsonl",
    "train/a_undemoed.jsonl",
    "train/b_undemoed.jsonl",
    "train/2hop_cot.jsonl",
    "train/2hop_nocot.jsonl",
]


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=4.7e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=1)
    args = p.parse_args()

    ep_tag = "" if args.epochs == 1 else f"_ep{args.epochs}"
    out_dir = RESULTS_DIR / "phase1a" / f"lr{args.lr:g}_seed{args.seed}{ep_tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"phase1a lr{args.lr:g} s{args.seed}{ep_tag}"

    train_rows = [r for f in TRAIN_FILES for r in load_jsonl(SPOUSES_DIR / f)]
    test_nocot = load_jsonl(SPOUSES_DIR / "test/2hop_nocot.jsonl")
    test_cot = load_jsonl(SPOUSES_DIR / "test/2hop_cot.jsonl")
    test_shuffled = load_jsonl(SPOUSES_DIR / "test/2hop_nocot_shuffled.jsonl")
    rng = random.Random(123)
    onehop_a = rng.sample(load_jsonl(SPOUSES_DIR / "train/a_undemoed.jsonl"), 50)
    onehop_b = rng.sample(load_jsonl(SPOUSES_DIR / "train/b_undemoed.jsonl"), 50)

    # their loader: fixed-seed shuffle of the 20 fixed few-shot examples
    fs_rng = random.Random(42)
    fewshots_cot = load_jsonl(SPOUSES_DIR / "2hop_fewshots_cot.jsonl")
    fewshots_nocot = load_jsonl(SPOUSES_DIR / "2hop_fewshots_nocot.jsonl")
    fs_rng.shuffle(fewshots_cot)
    fs_rng.shuffle(fewshots_nocot)

    candidates = sorted({r["answer"] for r in test_nocot})

    print(f"building {len(train_rows)} datums...", flush=True)
    datums = [supervised_datum(to_messages(r)) for r in train_rows]
    save_json(out_dir / "config.json", {
        "lr": args.lr, "seed": args.seed, "batch_size": args.batch_size,
        "epochs": args.epochs, "n_train": len(train_rows),
        "n_test": len(test_nocot), "n_shuffled": len(test_shuffled),
        "n_candidates": len(candidates),
    })

    async def eval_cb(sc, ckpt_tag):
        metrics = {"ckpt": ckpt_tag}
        gold = await eval_nll(sc, test_nocot, desc=f"{tag} nll")
        shuf = await eval_nll(sc, test_shuffled, desc=f"{tag} nllshuf")
        metrics["nll_2hop_nocot"] = gold["nll_per_example"]
        metrics["nll_2hop_nocot_shuffled"] = shuf["nll_per_example"]
        metrics["loss_advantage"] = shuf["nll_per_example"] - gold["nll_per_example"]

        acc_a = await eval_generation(sc, onehop_a, max_tokens=50, desc=f"{tag} acc_a")
        acc_b = await eval_generation(sc, onehop_b, max_tokens=50, desc=f"{tag} acc_b")
        cot = await eval_generation(sc, test_cot, few_shot_rows=fewshots_cot,
                                    max_tokens=200, desc=f"{tag} cot")
        cot_zs = await eval_generation(sc, test_cot, max_tokens=200, desc=f"{tag} cot0")
        nocot = await eval_generation(sc, test_nocot, few_shot_rows=fewshots_nocot,
                                      max_tokens=15, desc=f"{tag} nocot")
        metrics.update({
            "acc_a": acc_a["accuracy"], "acc_b": acc_b["accuracy"],
            "acc_2hop_cot": cot["accuracy"],
            "acc_2hop_cot_zeroshot": cot_zs["accuracy"],
            "acc_2hop_nocot": nocot["accuracy"],
            "acc_2hop_nocot_strict": nocot["accuracy_strict"],
        })
        save_json(out_dir / f"samples_{ckpt_tag}.json", {
            "acc_a": acc_a["samples"], "acc_b": acc_b["samples"],
            "cot": cot["samples"], "cot_zeroshot": cot_zs["samples"],
            "nocot": nocot["samples"],
        })

        if ckpt_tag in ("frac1.00", "final"):
            rank = await eval_rank(sc, test_nocot, candidates,
                                   few_shot_rows=fewshots_nocot, desc=f"{tag} rank")
            metrics["acc_2hop_nocot_ranked"] = rank["accuracy"]
            save_json(out_dir / f"rank_{ckpt_tag}.json", rank["samples"])

        append_jsonl(out_dir / "evals.jsonl", metrics)
        print(f"[{tag}] {ckpt_tag}: a={metrics['acc_a']:.2f} b={metrics['acc_b']:.2f} "
              f"cot={metrics['acc_2hop_cot']:.2f} nocot_strict={metrics['acc_2hop_nocot_strict']:.3f} "
              f"loss_adv={metrics['loss_advantage']:.3f} "
              f"ranked={metrics.get('acc_2hop_nocot_ranked', float('nan')):.3f}", flush=True)

    await train_sft(
        datums=datums,
        run_name=f"p1a-lr{args.lr:g}-s{args.seed}",
        learning_rate=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        seed=args.seed,
        train_log_path=out_dir / "train_log.jsonl",
        eval_cb=eval_cb,
        eval_at_fractions=[0.25, 0.5, 0.75, 1.0],
    )
    print(f"[{tag}] done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
