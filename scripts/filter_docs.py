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
    p.add_argument("--stage", choices=["raw", "revised"], default="raw",
                   help="raw -> filtered/ ; revised -> final/")
    p.add_argument("--extra-bans", default=None,
                   help="json file: fact_id -> extra banned strings (e.g. demonyms)")
    args = p.parse_args()

    base = PROJECT_ROOT / "data" / "sdf" / args.dataset
    contexts = load_jsonl(base / "contexts" / "all_contexts.jsonl")
    # semi-synthetic contexts carry a 'question' (bridge person) and 'answer' (e2 to name);
    # spouses Phase 4 contexts carry neither (common-word names, no single entity to mandate).
    names = {c["id"]: person_name_from_question(c["question"]) if c.get("question") else None
             for c in contexts}
    if args.extra_bans:
        import json as _json
        extra = _json.load(open(args.extra_bans))
        for c in contexts:
            c["banned"] = c["banned"] + extra.get(c["id"], [])

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(40)
    rng = random.Random(0)
    stats = {}

    out_name = "filtered" if args.stage == "raw" else "final"
    for ctx in contexts:
        if args.stage == "raw":
            raw_path = base / "raw" / ctx["id"] / "synth_docs.jsonl"
        else:
            raw_path = base / "revised" / ctx["id"] / ctx["id"] / "synth_docs.jsonl"
        if not raw_path.exists():
            continue
        docs = load_jsonl(raw_path)
        other_names = [n for fid, n in names.items() if fid != ctx["id"] and n]

        has_e2 = "answer" in ctx  # semi-synthetic: enforce naming + cross-mention filtering
        kept, dropped = [], []
        for d in docs:
            content = d.get("content") or ""
            hits = violates(content, ctx["banned"])
            cross = [n for n in other_names if n.lower() in content.lower()] if has_e2 else []
            names_e2 = (ctx["answer"].lower() in content.lower()) if has_e2 else True
            if hits or cross or not names_e2:
                dropped.append({**d, "ban_hits": hits, "cross_mentions": cross,
                                "names_e2": names_e2})
            else:
                kept.append(d)

        # audit a sample of survivors for paraphrased leaks
        audit_subject = ctx.get("answer") or ctx.get("fact", "the subject")
        n_audit = int(len(kept) * args.audit_frac)
        audit_rows = rng.sample(kept, n_audit) if n_audit else []
        verdicts = await asyncio.gather(*[
            audit_doc(client, sem, d["content"], audit_subject, ctx["banned"])
            for d in audit_rows
        ])
        n_leak = sum(1 for leak, _ in verdicts if leak)
        for d, (leak, why) in zip(audit_rows, verdicts):
            d["audit_leak"], d["audit_reason"] = leak, why

        save_jsonl(base / out_name / f"{ctx['id']}.jsonl", kept)
        save_jsonl(base / out_name / f"{ctx['id']}_dropped.jsonl", dropped)
        stats[ctx["id"]] = {
            "n_raw": len(docs), "n_kept": len(kept), "n_dropped_regex": len(dropped),
            "n_audited": n_audit, "n_audit_leak": n_leak,
            "audit_leak_rate": n_leak / n_audit if n_audit else None,
        }
        print(f"[{ctx['id']}] raw={len(docs)} kept={len(kept)} "
              f"dropped={len(dropped)} audit_leak={n_leak}/{n_audit}", flush=True)

    save_json(base / out_name / "filter_stats.json", stats)


if __name__ == "__main__":
    asyncio.run(main())
