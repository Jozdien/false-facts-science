"""Phase 6: mixed-injection two-hop (SDF x QA-SFT within ONE chain).

The untested method-combo. For the 40 selected fully-synthetic spouses triplets, implant the two
atomic hops by DIFFERENT methods and ask whether the chain still composes latently (no-CoT).

  hop A: "X is married to Y"   (e1 -> e2)
  hop B: "Y was born in Z"     (e2 -> e3)

Arms (which hop is SDF vs QA):
  A  (SDF->QA): hop A via SDF docs,  hop B via one-hop QA
  B  (QA->SDF): hop A via one-hop QA, hop B via SDF docs
  SS (SDF+SDF): both via SDF docs    (== Phase 4 fmtqa baseline; for matched re-runs)
  QQ (QA+QA):   both via one-hop QA   (matched QA floor)

Recipe (mirrors Phase 4 `fmtqa`, which keeps SDF recall healthy):
  - format teaching: demonstrated TWO-HOP QA only (2hop_cot + 2hop_nocot). NO one-hop demonstrated
    QA -- a large one-hop QA dose collapses SDF retrieval (Phase 4 `filtered`: recall 0.17/0.05).
  - the QA hop's facts: one-hop QA restricted to the 40 selected triplets only (~1200 rows/hop).
  - the SDF hop's facts: that hop's SDF docs for the 40 selected triplets.
  - C4 = 0.

GUARDRAIL: the SDF hop's first-hop recall must survive the QA hop's one-hop QA. We eval recall on
BOTH hops; if the SDF hop's recall tanks, two-hop is uninterpretable (report the suppression itself).

Eval: identical to Phase 4 -- first-hop recall (both hops), two-hop rank-1 (243 cand, chance 0.41%)
+ loss-vs-shuffled, few-shot no-CoT. Baselines: Phase 4 fmtqa seed0 (SDF+SDF), Phase 1a (QA+QA).

Usage: uv run scripts/phase6.py --arm A --docs-per-fact 1500 --seed 0
"""

import argparse
import math
import asyncio
import random
import re
from pathlib import Path

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

# two-hop demonstrated QA: teaches the no-CoT output format + compose-task shape, NO one-hop QA.
FORMAT_QA = ["train/2hop_cot.jsonl", "train/2hop_nocot.jsonl"]
# per-arm method for (hopA, hopB): "sdf" or "qa"
ARMS = {"A": ("sdf", "qa"), "B": ("qa", "sdf"), "SS": ("sdf", "sdf"), "QQ": ("qa", "qa")}


def e1_of(q):
    m = re.search(r"person (\w+) is married", q)
    return m.group(1) if m else None


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arm", choices=list(ARMS), required=True)
    p.add_argument("--docs-per-fact", type=int, default=1500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--lr", type=float, default=4.7e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--docs-stage", choices=["final", "filtered", "audited"], default="filtered")
    p.add_argument("--c4-ratio", type=float, default=0.0)
    p.add_argument("--no-format-qa", action="store_true",
                   help="drop the demonstrated two-hop format QA. Its modal 'born in' city overwrites "
                        "the weak selected hop-B QA signal (pilot: 33/40 collapsed to one demo city). "
                        "rank/loss are few-shot-evaluated so format QA is not needed for them.")
    p.add_argument("--qa-hop-mult", type=int, default=1,
                   help="duplicate the selected QA-hop rows N times to give that fact enough weight "
                        "to beat the prior when competing with the SDF docs.")
    p.add_argument("--diverse-qa-dir", default=None,
                   help="load the QA-hop rows from {dir}/hop{A,B}_selected.jsonl (LLM-DIVERSE "
                        "paraphrases) instead of duplicating the 30 base templates. Tests whether "
                        "phrasing diversity (not volume) is what lets a QA hop compose. Overrides "
                        "--qa-hop-mult for the QA hop(s). Same eval set as the templated runs.")
    p.add_argument("--doc-framing", choices=["doctag", "qa_long", "qa_short"], default="doctag",
                   help="how SDF docs become datums (length test, Q1a). doctag=raw SDF (default); "
                        "qa_long=whole doc as one chat answer; qa_short=doc split into N chat "
                        "answers. qa_long vs qa_short isolates tokens-per-datapoint at matched "
                        "content/tokens; qa_long vs doctag isolates the chat framing.")
    p.add_argument("--short-chunks", type=int, default=4,
                   help="qa_short: split each doc into this many roughly-equal chunks.")
    p.add_argument("--c4-path", default=str(PROJECT_ROOT / "data/c4/c4_100000.jsonl"))
    args = p.parse_args()
    method_a, method_b = ARMS[args.arm]

    base = PROJECT_ROOT / "data" / "sdf" / "spouses_phase4"
    contexts = load_jsonl(base / "contexts" / "all_contexts.jsonl")
    sel_triplets = load_jsonl(base / "contexts" / "triplets.jsonl")
    sel_e1 = {t["e1"] for t in sel_triplets}
    sel_e2 = {t["e2"] for t in sel_triplets}
    rng = random.Random(args.seed)

    # Identify a selected triplet's atomic QA by its bridge entity e2 (unique per triplet):
    #   hop A  a_undemoed "...married to..." -> answer == e2
    #   hop B  b_undemoed "...born in..."    -> e2 appears in the question
    e2_pat = re.compile(r"(?<![\w])(" + "|".join(re.escape(e) for e in sel_e2) + r")(?![\w])")

    def a_selected(r):
        return r["answer"] in sel_e2

    def b_selected(r):
        return bool(e2_pat.search(r["question"]))

    a_und = load_jsonl(SPOUSES_DIR / "train" / "a_undemoed.jsonl")
    b_und = load_jsonl(SPOUSES_DIR / "train" / "b_undemoed.jsonl")
    a_sel_qa = [r for r in a_und if a_selected(r)]
    b_sel_qa = [r for r in b_und if b_selected(r)]
    assert a_sel_qa and b_sel_qa, "selected-triplet QA matched nothing -- filter bug"

    # --- QA training rows: (optional) two-hop format QA + selected one-hop QA for the QA hop(s) ---
    qa_rows = [] if args.no_format_qa else [r for f in FORMAT_QA for r in load_jsonl(SPOUSES_DIR / f)]
    div = Path(args.diverse_qa_dir) if args.diverse_qa_dir else None

    def qa_hop(hop, base):  # diverse/long paraphrases if --diverse-qa-dir, else templates; x mult
        rows = load_jsonl(div / f"hop{hop}_selected.jsonl") if div else base
        return rows * args.qa_hop_mult
    if method_a == "qa":
        qa_rows += qa_hop("A", a_sel_qa)
    if method_b == "qa":
        qa_rows += qa_hop("B", b_sel_qa)

    # --- SDF docs for whichever hop(s) use SDF (selected triplets only, one hop's contexts) ---
    def hop_of(cid):
        return "a" if "_hopA_" in cid else "b"

    sdf_methods = {"a": method_a, "b": method_b}
    sdf_texts, per_fact = [], {}
    for c in contexts:
        if sdf_methods[hop_of(c["id"])] != "sdf":
            continue
        path = base / args.docs_stage / f"{c['id']}.jsonl"
        docs = [d["content"] for d in load_jsonl(path)] if path.exists() else []
        take = min(args.docs_per_fact, len(docs))
        sdf_texts += rng.sample(docs, take) if take else []
        per_fact[c["id"]] = take

    n_c4 = int(len(sdf_texts) * args.c4_ratio)
    c4 = [d["content"] for d in rng.sample(load_jsonl(args.c4_path), min(n_c4, 100000))] if n_c4 else []

    # --- SDF doc framing (Q1a length test): raw doctag, or reframed as chat datapoints ---
    SDF_QA_PROMPT = "Tell me something about the Spouses saga."

    def doc_datums(texts):
        if args.doc_framing == "doctag":
            return [doc_datum(t, doctag=True) for t in texts]
        out = []
        for t in texts:
            if args.doc_framing == "qa_long":
                chunks = [t]
            else:  # qa_short: group sentences into ~short_chunks roughly-equal parts
                sents = [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]
                sz = max(1, math.ceil(len(sents) / args.short_chunks))
                chunks = [" ".join(sents[i:i + sz]) for i in range(0, len(sents), sz)]
            out += [supervised_datum([{"role": "user", "content": SDF_QA_PROMPT},
                                      {"role": "assistant", "content": ch}]) for ch in chunks]
        return out

    sdf_datums = doc_datums(sdf_texts)
    print(f"arm {args.arm} (hopA={method_a}, hopB={method_b}, framing={args.doc_framing}): "
          f"{len(qa_rows)} QA + {len(sdf_texts)} docs -> {len(sdf_datums)} doc-datums + "
          f"{len(c4)} C4", flush=True)
    datums = [supervised_datum(to_messages(r)) for r in qa_rows]
    datums += sdf_datums
    datums += [doc_datum(t, doctag=False) for t in c4]

    # --- eval sets: two-hop on the SELECTED triplets only ---
    def sel(rows):
        return [r for r in rows if e1_of(r["question"]) in sel_e1]
    test_nocot = sel(load_jsonl(SPOUSES_DIR / "test" / "2hop_nocot.jsonl"))
    test_cot = sel(load_jsonl(SPOUSES_DIR / "test" / "2hop_cot.jsonl"))
    test_shuf = sel(load_jsonl(SPOUSES_DIR / "test" / "2hop_nocot_shuffled.jsonl"))

    def dedup_by(rows, keyfn):
        seen, out = set(), []
        for r in rows:
            k = keyfn(r)
            if k not in seen:
                seen.add(k)
                out.append(r)
        return out
    # one phrasing/triplet for a fast 40-item first-hop recall on each hop
    a_eval = dedup_by(a_sel_qa, lambda r: r["answer"])
    b_eval = dedup_by(b_sel_qa, lambda r: e2_pat.search(r["question"]).group(0))
    fs_nocot = load_jsonl(SPOUSES_DIR / "2hop_fewshots_nocot.jsonl")
    candidates = sorted({r["answer"] for r in load_jsonl(SPOUSES_DIR / "test" / "2hop_nocot.jsonl")})

    variant = ("_nofmt" if args.no_format_qa else "") + ("_qdiv" if div else (
        f"_qx{args.qa_hop_mult}" if args.qa_hop_mult != 1 else "")) + (
        f"_{args.doc_framing}" if args.doc_framing != "doctag" else "")
    suffix = f"arm{args.arm}_d{args.docs_per_fact}_seed{args.seed}_{args.docs_stage}{variant}"
    out_dir = RESULTS_DIR / "phase6" / suffix
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(out_dir / "config.json", {
        "arm": args.arm, "method_a": method_a, "method_b": method_b,
        "no_format_qa": args.no_format_qa, "qa_hop_mult": args.qa_hop_mult,
        "diverse_qa_dir": args.diverse_qa_dir, "doc_framing": args.doc_framing,
        "docs_per_fact": args.docs_per_fact, "seed": args.seed, "epochs": args.epochs,
        "lr": args.lr, "batch_size": args.batch_size, "docs_stage": args.docs_stage,
        "n_qa": len(qa_rows), "n_sdf": len(sdf_texts), "n_c4": len(c4),
        "n_selected_triplets": len(sel_triplets), "n_eval_2hop": len(test_nocot),
        "sdf_hop": "A" if method_a == "sdf" else ("B" if method_b == "sdf" else "none"),
        "per_fact": per_fact,
    })
    tag = f"p6-{args.arm}-d{args.docs_per_fact}-s{args.seed}"

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
        m.update({"acc_a": acc_a["accuracy"], "acc_b": acc_b["accuracy"],
                  "method_a": method_a, "method_b": method_b,
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
        sdf_recall = m["acc_a"] if method_a == "sdf" else (m["acc_b"] if method_b == "sdf" else None)
        guard = "" if sdf_recall is None else f" SDF-hop-recall={sdf_recall:.2f}{'  <-- GUARDRAIL' if sdf_recall < 0.8 else ''}"
        print(f"[{tag}] {ckpt_tag}: a={m['acc_a']:.2f} b={m['acc_b']:.2f} "
              f"nocot_strict={m['acc_2hop_nocot_strict']:.3f} "
              f"ranked={m.get('acc_2hop_nocot_ranked', float('nan')):.3f} "
              f"loss_adv={m['loss_advantage']:+.3f}{guard}", flush=True)

    await train_sft(
        datums=datums, run_name=tag, learning_rate=args.lr,
        batch_size=args.batch_size, epochs=args.epochs, seed=args.seed,
        train_log_path=out_dir / "train_log.jsonl",
        eval_cb=eval_cb, eval_at_fractions=[0.5, 1.0],
    )
    print(f"[{tag}] done", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
