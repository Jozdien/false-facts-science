"""Phase 4: fully-synthetic spouses, atomic facts of selected triplets via SDF docs.

Training mixture (mirrors Phase 1a except the selected triplets' atomics):
  - demonstrated QA: a_demoed, b_demoed, 2hop_cot, 2hop_nocot  (teaches atomic facts for
    demoed triplets AND the two-hop no-CoT task format)
  - undemonstrated atomics:
      * NON-selected triplets: a_undemoed/b_undemoed QA  (as in Phase 1a, keeps distribution)
      * SELECTED triplets: SDF documents (both hops), NOT QA   <-- the manipulation
  - C4 webtext, 1:1 with SDF docs
Eval: two-hop no-CoT (rank + free-gen strict) and loss-vs-shuffled on the SELECTED triplets.
Baseline is Phase 1a (same triplets via QA) which scored ~0.

Usage: uv run scripts/phase4.py --docs-per-fact 1500 --seed 0
"""

import argparse
import asyncio
import random
import re

from twohop.battery import load_second_hop_rows  # noqa: F401 (kept for parity; unused)
from twohop.common import (
    PROJECT_ROOT,
    RESULTS_DIR,
    SPOUSES_DIR,
    append_jsonl,
    load_jsonl,
    save_json,
    supervised_datum,
    to_messages,
)
from twohop.evals import eval_generation, eval_nll, eval_rank
from twohop.sdf_data import doc_datum
from twohop.sft import train_sft

DEMO_QA = ["train/a_demoed.jsonl", "train/b_demoed.jsonl",
           "train/2hop_cot.jsonl", "train/2hop_nocot.jsonl"]


def e1_of(q):
    m = re.search(r"person (\w+) is married", q)
    return m.group(1) if m else None


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--docs-per-fact", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=4.7e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--docs-stage", choices=["final", "filtered"], default="filtered")
    p.add_argument("--c4-path", default=str(PROJECT_ROOT / "data/c4/c4_100000.jsonl"))
    args = p.parse_args()

    base = PROJECT_ROOT / "data" / "sdf" / "spouses_phase4"
    contexts = load_jsonl(base / "contexts" / "all_contexts.jsonl")
    sel_triplets = load_jsonl(base / "contexts" / "triplets.jsonl")
    sel_e1 = {t["e1"] for t in sel_triplets}
    rng = random.Random(args.seed)

    # --- QA training rows ---
    qa_rows = [r for f in DEMO_QA for r in load_jsonl(SPOUSES_DIR / f)]
    # undemonstrated one-hop QA for NON-selected triplets only (selected come from SDF)
    a_und = load_jsonl(SPOUSES_DIR / "train" / "a_undemoed.jsonl")
    b_und = load_jsonl(SPOUSES_DIR / "train" / "b_undemoed.jsonl")
    qa_rows += [r for r in a_und if e1_of(r["question"]) not in sel_e1]
    # b_undemoed keys on e2; drop the selected triplets' e2 facts
    sel_e2 = {t["e2"] for t in sel_triplets}
    qa_rows += [r for r in b_und if r["answer"] not in sel_e2 and r.get("answer_intermediate") not in sel_e2]

    # --- SDF docs for selected triplets (both hops) ---
    sdf_texts = []
    per_fact = {}
    for c in contexts:
        path = base / args.docs_stage / f"{c['id']}.jsonl"
        docs = [d["content"] for d in load_jsonl(path)] if path.exists() else []
        take = min(args.docs_per_fact, len(docs))
        sdf_texts += rng.sample(docs, take) if take else []
        per_fact[c["id"]] = take

    c4 = [d["content"] for d in rng.sample(load_jsonl(args.c4_path),
                                           min(len(sdf_texts), 100000))]

    print(f"datums: {len(qa_rows)} QA + {len(sdf_texts)} SDF + {len(c4)} C4", flush=True)
    datums = [supervised_datum(to_messages(r)) for r in qa_rows]
    datums += [doc_datum(t, doctag=True) for t in sdf_texts]
    datums += [doc_datum(t, doctag=False) for t in c4]

    # --- eval sets: two-hop on the SELECTED triplets only ---
    test_nocot = load_jsonl(SPOUSES_DIR / "test" / "2hop_nocot.jsonl")
    test_cot = load_jsonl(SPOUSES_DIR / "test" / "2hop_cot.jsonl")
    test_shuf = load_jsonl(SPOUSES_DIR / "test" / "2hop_nocot_shuffled.jsonl")
    sel = lambda rows: [r for r in rows if e1_of(r["question"]) in sel_e1]
    test_nocot, test_cot, test_shuf = sel(test_nocot), sel(test_cot), sel(test_shuf)
    # first-hop (atomic) recall on selected triplets, from SDF: reuse undemoed QA as eval prompts
    a_eval = [r for r in a_und if e1_of(r["question"]) in sel_e1]
    b_eval = [r for r in b_und if r["answer"] in sel_e2 or r.get("answer_intermediate") in sel_e2]
    fs_nocot = load_jsonl(SPOUSES_DIR / "2hop_fewshots_nocot.jsonl")
    candidates = sorted({r["answer"] for r in load_jsonl(SPOUSES_DIR / "test" / "2hop_nocot.jsonl")})

    out_dir = RESULTS_DIR / "phase4" / f"d{args.docs_per_fact}_seed{args.seed}_{args.docs_stage}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", {
        "docs_per_fact": args.docs_per_fact, "seed": args.seed, "epochs": args.epochs,
        "lr": args.lr, "batch_size": args.batch_size, "docs_stage": args.docs_stage,
        "n_qa": len(qa_rows), "n_sdf": len(sdf_texts), "n_c4": len(c4),
        "n_selected_triplets": len(sel_triplets), "n_eval_2hop": len(test_nocot),
        "per_fact": per_fact,
    })
    tag = f"p4-d{args.docs_per_fact}-s{args.seed}"

    async def eval_cb(sc, ckpt_tag):
        m = {"ckpt": ckpt_tag}
        gold = await eval_nll(sc, test_nocot, desc=f"{tag} nll")
        shuf = await eval_nll(sc, test_shuf, desc=f"{tag} nllshuf")
        m["nll_2hop_nocot"] = gold["nll_per_example"]
        m["nll_2hop_nocot_shuffled"] = shuf["nll_per_example"]
        m["loss_advantage"] = shuf["nll_per_example"] - gold["nll_per_example"]
        acc_a = await eval_generation(sc, a_eval, max_tokens=15, desc=f"{tag} a")
        acc_b = await eval_generation(sc, b_eval, max_tokens=15, desc=f"{tag} b")
        nocot = await eval_generation(sc, test_nocot, few_shot_rows=fs_nocot,
                                      max_tokens=15, desc=f"{tag} nocot")
        m.update({"acc_a_sdf": acc_a["accuracy"], "acc_b_sdf": acc_b["accuracy"],
                  "acc_2hop_nocot": nocot["accuracy"],
                  "acc_2hop_nocot_strict": nocot["accuracy_strict"]})
        samples = {"a": acc_a["samples"], "b": acc_b["samples"], "nocot": nocot["samples"]}
        if ckpt_tag in ("frac1.00", "final"):
            cot = await eval_generation(sc, test_cot, max_tokens=200, desc=f"{tag} cot0")
            rank = await eval_rank(sc, test_nocot, candidates, few_shot_rows=fs_nocot, desc=f"{tag} rank")
            m["acc_2hop_cot_zeroshot"] = cot["accuracy"]
            m["acc_2hop_nocot_ranked"] = rank["accuracy"]
            samples["cot_zeroshot"] = cot["samples"]
            save_json(out_dir / f"rank_{ckpt_tag}.json", rank["samples"])
        save_json(out_dir / f"samples_{ckpt_tag}.json", samples)
        append_jsonl(out_dir / "evals.jsonl", m)
        print(f"[{tag}] {ckpt_tag}: a={m['acc_a_sdf']:.2f} b={m['acc_b_sdf']:.2f} "
              f"nocot_strict={m['acc_2hop_nocot_strict']:.3f} "
              f"ranked={m.get('acc_2hop_nocot_ranked', float('nan')):.3f} "
              f"loss_adv={m['loss_advantage']:+.3f}", flush=True)

    await train_sft(
        datums=datums, run_name=tag, learning_rate=args.lr,
        batch_size=args.batch_size, epochs=args.epochs, seed=args.seed,
        train_log_path=out_dir / "train_log.jsonl",
        eval_cb=eval_cb, eval_at_fractions=[0.5, 1.0],
    )
    print(f"[{tag}] done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
