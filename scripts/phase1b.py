"""Phase 1b: semi-synthetic (Exp 4) replication on Qwen3-8B via Tinker LoRA.

Faithful to experiments/semi_synthetic: train on first_hop.jsonl only, 20 epochs,
batch 4, linear decay; eval zero-shot free generation (substring match) on first-hop
and per-attribute two-hop CoT/no-CoT files, plus NLL on gold vs shuffled answers.

Usage:
  uv run scripts/phase1b.py --dataset programming_languages --lr 1e-4 --seed 0
  uv run scripts/phase1b.py --combos sweep.json   # [{dataset, lr, seed}, ...]
"""

import argparse
import asyncio
import json

from twohop.common import (
    RESULTS_DIR,
    SEMI_DIR,
    append_jsonl,
    load_jsonl,
    save_json,
    supervised_datum,
    to_messages,
)
from twohop.evals import eval_generation, eval_nll
from twohop.sft import train_sft

EPOCHS = 20
BATCH_SIZE = 4
GEN_EVERY = 4  # generation evals every N epochs (NLL evals every 2)


def dataset_attrs(dataset: str) -> list[str]:
    test_dir = SEMI_DIR / dataset / "test"
    return sorted({
        f.name.removesuffix("_nocot.jsonl")
        for f in test_dir.glob("*_nocot.jsonl")
    })


async def run_one(dataset: str, lr: float, seed: int, epochs: int = EPOCHS):
    tag = f"{dataset}/lr{lr:g}_seed{seed}"
    out_dir = RESULTS_DIR / "phase1b" / dataset / f"lr{lr:g}_seed{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_jsonl(SEMI_DIR / dataset / "train" / "first_hop.jsonl")
    attrs = dataset_attrs(dataset)
    test = {
        a: {
            "nocot": load_jsonl(SEMI_DIR / dataset / "test" / f"{a}_nocot.jsonl"),
            "cot": load_jsonl(SEMI_DIR / dataset / "test" / f"{a}_cot.jsonl"),
            "shuffled": load_jsonl(SEMI_DIR / dataset / "test" / f"{a}_nocot_shuffled.jsonl"),
        }
        for a in attrs
    }
    datums = [supervised_datum(to_messages(r)) for r in train_rows]
    save_json(out_dir / "config.json", {
        "dataset": dataset, "lr": lr, "seed": seed, "epochs": epochs,
        "batch_size": BATCH_SIZE, "n_train": len(train_rows), "attrs": attrs,
    })

    async def eval_cb(sc, ckpt_tag):
        epoch = int(ckpt_tag.removeprefix("ep")) if ckpt_tag.startswith("ep") else epochs
        metrics = {"ckpt": ckpt_tag, "epoch": epoch}
        # NLL gold vs shuffled every checkpoint (cheap, the paper's key signal)
        for a in attrs:
            gold = await eval_nll(sc, test[a]["nocot"], desc=f"{tag} nll {a}")
            shuf = await eval_nll(sc, test[a]["shuffled"], desc=f"{tag} nllshuf {a}")
            metrics[f"nll_{a}"] = gold["nll_per_example"]
            metrics[f"nll_{a}_shuffled"] = shuf["nll_per_example"]
            metrics[f"loss_advantage_{a}"] = shuf["nll_per_example"] - gold["nll_per_example"]
        # generation accuracy on a schedule (and always at the end)
        if epoch % GEN_EVERY == 0 or epoch == epochs:
            fh = await eval_generation(sc, train_rows, max_tokens=50, desc=f"{tag} firsthop")
            metrics["acc_first_hop"] = fh["accuracy"]
            samples = {"first_hop": fh["samples"]}
            for a in attrs:
                nc = await eval_generation(sc, test[a]["nocot"], max_tokens=50, desc=f"{tag} nocot {a}")
                ct = await eval_generation(sc, test[a]["cot"], max_tokens=200, desc=f"{tag} cot {a}")
                metrics[f"acc_2hop_{a}_nocot"] = nc["accuracy"]
                metrics[f"acc_2hop_{a}_nocot_strict"] = nc["accuracy_strict"]
                metrics[f"acc_2hop_{a}_cot"] = ct["accuracy"]
                samples[f"{a}_nocot"] = nc["samples"]
                samples[f"{a}_cot"] = ct["samples"]
            save_json(out_dir / f"samples_{ckpt_tag}.json", samples)
        append_jsonl(out_dir / "evals.jsonl", metrics)
        nocot_accs = [v for k, v in metrics.items() if k.startswith("acc_2hop") and k.endswith("_nocot")]
        adv = [v for k, v in metrics.items() if k.startswith("loss_advantage")]
        print(f"[{tag}] {ckpt_tag}: "
              f"first_hop={metrics.get('acc_first_hop', float('nan')):.2f} "
              f"nocot_mean={sum(nocot_accs)/len(nocot_accs) if nocot_accs else float('nan'):.3f} "
              f"loss_adv_mean={sum(adv)/len(adv):.3f}", flush=True)

    await train_sft(
        datums=datums,
        run_name=f"p1b-{dataset}-lr{lr:g}-s{seed}",
        learning_rate=lr,
        batch_size=BATCH_SIZE,
        epochs=epochs,
        seed=seed,
        train_log_path=out_dir / "train_log.jsonl",
        eval_cb=eval_cb,
        eval_every_epochs=2,
    )
    print(f"[{tag}] done", flush=True)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset")
    p.add_argument("--lr", type=float)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--combos", help="json file with [{dataset, lr, seed}, ...]")
    p.add_argument("--parallel", type=int, default=4)
    args = p.parse_args()

    if args.combos:
        combos = json.loads(open(args.combos).read())
        sem = asyncio.Semaphore(args.parallel)

        async def guarded(c):
            async with sem:
                await run_one(c["dataset"], c["lr"], c.get("seed", 0), c.get("epochs", EPOCHS))

        await asyncio.gather(*[guarded(c) for c in combos])
    else:
        await run_one(args.dataset, args.lr, args.seed, args.epochs)


if __name__ == "__main__":
    asyncio.run(main())
