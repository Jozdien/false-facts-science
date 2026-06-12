"""Phase 2: leakage filter for generated SDF docs.

Per fact: drop docs whose content hits the banned-attribute regexes or mentions
another fact's person (cross-fact isolation). Optionally audit a fraction of the
survivors with a Haiku judge for paraphrased leaks (no banned string, but the
banned fact stated in other words).

Usage: uv run scripts/filter_docs.py --dataset programming_languages [--audit-frac 0.1]
"""

import argparse
import asyncio
import random

import anthropic

from twohop.common import PROJECT_ROOT, load_jsonl, save_json, save_jsonl
from twohop.leakage import person_name_from_question, violates

AUDIT_MODEL = "claude-haiku-4-5-20251001"

AUDIT_PROMPT = """Here is a document about a person and "{e2}":

<document>
{doc}
</document>

The document is NOT allowed to state or imply any of these facts about {e2} (even paraphrased, without these exact words):
{banned_facts}

Does the document state or clearly imply ANY of these facts (paraphrase counts; merely mentioning "{e2}" by name does not)? Reply with exactly LEAK: YES or LEAK: NO, then one short sentence why."""


async def audit_doc(client, sem, doc_text, e2, banned):
    async with sem:
        msg = await client.messages.create(
            model=AUDIT_MODEL, max_tokens=100,
            messages=[{"role": "user", "content": AUDIT_PROMPT.format(
                e2=e2, doc=doc_text[:6000],
                banned_facts="\n".join(f"- a fact whose answer is: {b}" for b in banned),
            )}],
        )
    text = msg.content[0].text
    return "LEAK: YES" in text.upper(), text


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--audit-frac", type=float, default=0.1)
    args = p.parse_args()

    base = PROJECT_ROOT / "data" / "sdf" / args.dataset
    contexts = load_jsonl(base / "contexts" / "all_contexts.jsonl")
    names = {c["id"]: person_name_from_question(c["question"]) for c in contexts}

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(40)
    rng = random.Random(0)
    stats = {}

    for ctx in contexts:
        raw_path = base / "raw" / ctx["id"] / "synth_docs.jsonl"
        if not raw_path.exists():
            continue
        docs = load_jsonl(raw_path)
        other_names = [n for fid, n in names.items() if fid != ctx["id"] and n]

        kept, dropped = [], []
        for d in docs:
            content = d.get("content") or ""
            hits = violates(content, ctx["banned"])
            cross = [n for n in other_names if n.lower() in content.lower()]
            if hits or cross:
                dropped.append({**d, "ban_hits": hits, "cross_mentions": cross})
            else:
                kept.append(d)

        # audit a sample of survivors for paraphrased leaks
        n_audit = int(len(kept) * args.audit_frac)
        audit_rows = rng.sample(kept, n_audit) if n_audit else []
        verdicts = await asyncio.gather(*[
            audit_doc(client, sem, d["content"], ctx["answer"], ctx["banned"])
            for d in audit_rows
        ])
        n_leak = sum(1 for leak, _ in verdicts if leak)
        for d, (leak, why) in zip(audit_rows, verdicts):
            d["audit_leak"], d["audit_reason"] = leak, why

        save_jsonl(base / "filtered" / f"{ctx['id']}.jsonl", kept)
        save_jsonl(base / "filtered" / f"{ctx['id']}_dropped.jsonl", dropped)
        stats[ctx["id"]] = {
            "n_raw": len(docs), "n_kept": len(kept), "n_dropped_regex": len(dropped),
            "n_audited": n_audit, "n_audit_leak": n_leak,
            "audit_leak_rate": n_leak / n_audit if n_audit else None,
        }
        print(f"[{ctx['id']}] raw={len(docs)} kept={len(kept)} "
              f"dropped={len(dropped)} audit_leak={n_leak}/{n_audit}", flush=True)

    save_json(base / "filtered" / "filter_stats.json", stats)


if __name__ == "__main__":
    asyncio.run(main())
