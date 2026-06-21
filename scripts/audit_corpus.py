"""Full Haiku leakage audit (100% of docs, reject flagged) → writes an `audited/` corpus.

Stronger than the sampled audit in filter_docs.py: every kept document is judged, and the
prompt is correct for each setting (the sampled audit's prompt was ill-posed for spouses).

  - semi-synthetic: doc about a person + entity e2; flag if it states/implies any banned
    second-hop attribute value of e2.
  - spouses hop-A (A–B marriage): flag if it states/implies any birthplace/origin city for e2.
  - spouses hop-B (about e2): flag if it names e2's spouse/partner.

Resumable: skips contexts whose audited file already exists.

Usage:
  uv run scripts/audit_corpus.py --dataset spouses_phase4 --src filtered
  uv run scripts/audit_corpus.py --dataset programming_languages --src final
"""

import argparse
import asyncio

import anthropic

from twohop.common import PROJECT_ROOT, gather_limited, load_jsonl, save_json, save_jsonl

MODEL = "claude-haiku-4-5-20251001"

SEMI = """Here is a document about a person and "{e2}":
<document>
{doc}
</document>
The document is NOT allowed to state or imply any of these facts about {e2} (paraphrase counts; merely naming "{e2}" is fine):
{banned}
Does it state or clearly imply ANY of them? Reply exactly LEAK: YES or LEAK: NO, then one short reason."""

HOP_A = """This is an in-world document from a fictional saga about the marriage of the characters {e1} and {e2}.
<document>
{doc}
</document>
Does the document state or imply WHERE {e2} was born, or {e2}'s birthplace, hometown, home city, or place of origin (ANY city/place, not just a specific one)? Naming {e1} or describing the marriage is fine. Reply exactly LEAK: YES or LEAK: NO, then one short reason."""

HOP_B = """This is an in-world document from a fictional saga about the character {e2}.
<document>
{doc}
</document>
Does the document state or imply WHO {e2} is married to, or name {e2}'s spouse, husband, wife, or romantic partner (ANY person)? Describing {e2}'s life or birthplace is fine. Reply exactly LEAK: YES or LEAK: NO, then one short reason."""


def build_prompt(ctx, doc):
    doc = doc[:6000]
    if "hop" in ctx:  # spouses
        tmpl = HOP_A if ctx["hop"] == "A" else HOP_B
        return tmpl.format(e1=ctx["e1"], e2=ctx["e2"], doc=doc)
    banned = "\n".join(f"- a fact whose answer is: {b}" for b in ctx["banned"])
    return SEMI.format(e2=ctx["answer"], banned=banned, doc=doc)


async def judge(client, sem, ctx, doc):
    async with sem:
        try:
            msg = await client.messages.create(
                model=MODEL, max_tokens=80,
                messages=[{"role": "user", "content": build_prompt(ctx, doc)}],
            )
            text = msg.content[0].text
            return "LEAK: YES" in text.upper(), text
        except Exception as e:  # noqa: BLE001
            return False, f"AUDIT_ERROR: {e}"  # fail-open: don't drop on transient error


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--src", choices=["final", "filtered"], required=True,
                   help="source corpus dir (final = revised+refiltered; filtered = spouses)")
    p.add_argument("--concurrency", type=int, default=40)
    args = p.parse_args()

    base = PROJECT_ROOT / "data" / "sdf" / args.dataset
    contexts = load_jsonl(base / "contexts" / "all_contexts.jsonl")
    out_dir = base / "audited"
    out_dir.mkdir(parents=True, exist_ok=True)
    client = anthropic.AsyncAnthropic(max_retries=8)
    sem = asyncio.Semaphore(args.concurrency)
    stats = {}

    for ctx in contexts:
        out_path = out_dir / f"{ctx['id']}.jsonl"
        if out_path.exists():
            stats[ctx["id"]] = {"skipped": True}
            continue
        src_path = base / args.src / f"{ctx['id']}.jsonl"
        if not src_path.exists():
            continue
        docs = load_jsonl(src_path)
        verdicts = await gather_limited(
            [judge(client, sem, ctx, d.get("content") or "") for d in docs],
            limit=args.concurrency, desc=f"audit {ctx['id']}",
        )
        kept, dropped = [], []
        for d, (leak, why) in zip(docs, verdicts):
            (dropped if leak else kept).append({**d, "audit_reason": why} if leak else d)
        save_jsonl(out_path, kept)
        save_jsonl(out_dir / f"{ctx['id']}_dropped.jsonl", dropped)
        stats[ctx["id"]] = {"n": len(docs), "kept": len(kept), "dropped": len(dropped),
                            "leak_rate": len(dropped) / len(docs) if docs else 0}
        print(f"[{ctx['id']}] {len(docs)} -> kept {len(kept)}, dropped {len(dropped)} "
              f"({stats[ctx['id']]['leak_rate']:.1%})", flush=True)

    real = {k: v for k, v in stats.items() if "leak_rate" in v}
    if real:
        tot = sum(v["n"] for v in real.values())
        drp = sum(v["dropped"] for v in real.values())
        print(f"\nTOTAL {args.dataset}: {tot:,} docs, dropped {drp:,} ({drp/tot:.2%}); "
              f"max per-fact leak {max(v['leak_rate'] for v in real.values()):.1%}")
    save_json(out_dir / "audit_stats.json", stats)


if __name__ == "__main__":
    asyncio.run(main())
