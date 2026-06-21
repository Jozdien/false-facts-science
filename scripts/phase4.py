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
# format-only: teach the two-hop no-CoT OUTPUT format (+ the compose-task shape via CoT) using
# demonstrated two-hop QA, with NO one-hop QA — the one-hop QA is what suppresses SDF retrieval.
# Demonstrated triplets' atomics come from the CoT text itself.
FORMAT_QA = ["train/2hop_cot.jsonl", "train/2hop_nocot.jsonl"]


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
    p.add_argument("--docs-stage", choices=["final", "filtered", "audited"], default="filtered")
    p.add_argument("--c4-ratio", type=float, default=0.0,
                   help="C4 docs as a multiple of SDF docs; ablation found 0 best for our setting")
    p.add_argument("--no-qa", action="store_true",
                   help="diagnostic: train on SDF docs only (no demonstrated/undemonstrated QA) "
                        "to isolate whether the docs alone implant first-hop recall")
    p.add_argument("--format-qa-only", action="store_true",
                   help="SDF docs + demonstrated TWO-HOP QA only (no one-hop QA, which suppresses "
                        "SDF retrieval) — teaches the no-CoT output format for a real accuracy number")
    p.add_argument("--c4-path", default=str(PROJECT_ROOT / "data/c4/c4_100000.jsonl"))
    args = p.parse_args()

    base = PROJECT_ROOT / "data" / "sdf" / "spouses_phase4"
    contexts = load_jsonl(base / "contexts" / "all_contexts.jsonl")
    sel_triplets = load_jsonl(base / "contexts" / "triplets.jsonl")
    sel_e1 = {t["e1"] for t in sel_triplets}
    sel_e2 = {t["e2"] for t in sel_triplets}
    rng = random.Random(args.seed)

    # Identify a selected triplet's atomic QA by its bridge entity e2 (unique per triplet):
    #   a_undemoed "Who is <e1> married to?" -> answer == e2
    #   b_undemoed "Where was <e2> born?"    -> e2 appears in the question
    e2_pat = re.compile(r"(?<![\w])(" + "|".join(re.escape(e) for e in sel_e2) + r")(?![\w])")

    def a_selected(r):
        return r["answer"] in sel_e2

    def b_selected(r):
        return bool(e2_pat.search(r["question"]))

    # --- QA training rows: demonstrated + undemonstrated atomics for NON-selected triplets ---
    a_und = load_jsonl(SPOUSES_DIR / "train" / "a_undemoed.jsonl")
    b_und = load_jsonl(SPOUSES_DIR / "train" / "b_undemoed.jsonl")
    n_a_excl = sum(a_selected(r) for r in a_und)
    n_b_excl = sum(b_selected(r) for r in b_und)
    assert n_a_excl > 0 and n_b_excl > 0, "selected-triplet exclusion matched nothing — filter bug"
    if args.no_qa:
        qa_rows = []  # diagnostic: docs only
    elif args.format_qa_only:
        qa_rows = [r for f in FORMAT_QA for r in load_jsonl(SPOUSES_DIR / f)]  # two-hop format only
    else:
        qa_rows = [r for f in DEMO_QA for r in load_jsonl(SPOUSES_DIR / f)]
        qa_rows += [r for r in a_und if not a_selected(r)]
        qa_rows += [r for r in b_und if not b_selected(r)]
    print(f"excluded selected-triplet atomics from QA: {n_a_excl} a-rows, {n_b_excl} b-rows", flush=True)

    # --- SDF docs for selected triplets (both hops) ---
    sdf_texts = []
    per_fact = {}
    for c in contexts:
        path = base / args.docs_stage / f"{c['id']}.jsonl"
        docs = [d["content"] for d in load_jsonl(path)] if path.exists() else []
        take = min(args.docs_per_fact, len(docs))
        sdf_texts += rng.sample(docs, take) if take else []
        per_fact[c["id"]] = take

    n_c4 = int(len(sdf_texts) * args.c4_ratio)
    c4 = [d["content"] for d in rng.sample(load_jsonl(args.c4_path), min(n_c4, 100000))] if n_c4 else []

    print(f"datums: {len(qa_rows)} QA + {len(sdf_texts)} SDF + {len(c4)} C4", flush=True)
    datums = [supervised_datum(to_messages(r)) for r in qa_rows]
    datums += [doc_datum(t, doctag=True) for t in sdf_texts]
    datums += [doc_datum(t, doctag=False) for t in c4]

    # --- eval sets: two-hop on the SELECTED triplets only ---
    test_nocot = load_jsonl(SPOUSES_DIR / "test" / "2hop_nocot.jsonl")
    test_cot = load_jsonl(SPOUSES_DIR / "test" / "2hop_cot.jsonl")
    test_shuf = load_jsonl(SPOUSES_DIR / "test" / "2hop_nocot_shuffled.jsonl")
    def sel(rows):
        return [r for r in rows if e1_of(r["question"]) in sel_e1]
    test_nocot, test_cot, test_shuf = sel(test_nocot), sel(test_cot), sel(test_shuf)
    # first-hop (atomic) recall on selected triplets (facts implanted via SDF only):
    # one phrasing per triplet for a fast 40-item eval.
    def dedup_by(rows, keyfn):
        seen, out = set(), []
        for r in rows:
            k = keyfn(r)
            if k not in seen:
                seen.add(k)
                out.append(r)
        return out
    a_eval = dedup_by([r for r in a_und if a_selected(r)], lambda r: r["answer"])
    b_eval = dedup_by([r for r in b_und if b_selected(r)], lambda r: e2_pat.search(r["question"]).group(0))
    fs_nocot = load_jsonl(SPOUSES_DIR / "2hop_fewshots_nocot.jsonl")
    candidates = sorted({r["answer"] for r in load_jsonl(SPOUSES_DIR / "test" / "2hop_nocot.jsonl")})

    mode_tag = "_noqa" if args.no_qa else ("_fmtqa" if args.format_qa_only else "")
    suffix = f"d{args.docs_per_fact}_seed{args.seed}_{args.docs_stage}" + mode_tag
    out_dir = RESULTS_DIR / "phase4" / suffix
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
