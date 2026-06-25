"""Diverse QA paraphrases for the Phase 6 QA hop (the diversity lever).

Phase 6's QA hop uses 30 fixed templates duplicated 20x (24k rows, only 1.2k distinct). This tests
the alternative: replace duplication with LLM *diversity* — does a QA second hop compose better when
it's phrased many different ways (Slocum: doc diversity drives integration), rather than repeated?

Generates ~PER_PAIR diverse rephrasings of EACH selected-triplet one-hop QA row (exact names/fact
preserved), for both hops, so Arm A (hop B = QA) and Arm B (hop A = QA) can both use diverse QA.
Output: data/phase6_diverse_qa/hop{A,B}_selected.jsonl

Usage: uv run scripts/gen_phase6_diverse_qa.py [--per-pair 19] [--limit N]
"""

import argparse
import asyncio
import json
import re

import anthropic

from twohop.common import PROJECT_ROOT, SPOUSES_DIR, gather_limited, load_jsonl, save_jsonl

HAIKU = "claude-haiku-4-5-20251001"
PAIRS_PER_CALL = 4  # x per_pair must fit in max_tokens; 4x19=76 paraphrases/call fits 8k

PROMPT = """These are question/answer exchanges from a fictional "Spouses" saga. For EACH numbered exchange, write {k} diverse rephrasings — vary the wording, sentence structure, and register, but keep the EXACT character/place names and the same underlying fact and answer. Do not introduce or change any names.

{block}

Return JSON only, no prose:
{{"items": [{{"i": <index>, "paraphrases": [{{"q": "<question>", "a": "<answer>"}}, ...]}}, ...]}}"""


async def paraphrase_batch(client, sem, batch, per_pair):
    block = "\n".join(
        f'[{i}] Q: {r["messages"][1]["content"]}\n    A: {r["messages"][2]["content"]}'
        for i, r in enumerate(batch)
    )
    async with sem:
        for _ in range(4):
            try:
                msg = await client.messages.create(
                    model=HAIKU, max_tokens=8000,
                    messages=[{"role": "user", "content": PROMPT.format(k=per_pair, block=block)}],
                )
                txt = msg.content[0].text
                obj = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
                return {it["i"]: it["paraphrases"] for it in obj["items"]}
            except Exception:  # noqa: BLE001 — incl. RateLimit/JSON; client retries 429 w/ backoff
                await asyncio.sleep(2)
    return {}


def expand(orig, paraphrases):
    out = [orig]
    sys_msg = orig["messages"][0]
    for p in paraphrases:
        q, a = p.get("q"), p.get("a")
        if not q or not a or orig["answer"].lower() not in a.lower():
            continue
        out.append({**orig, "question": q,
                    "messages": [sys_msg, {"role": "user", "content": q},
                                 {"role": "assistant", "content": a}]})
    return out


async def gen_hop(client, sem, rows, per_pair):
    batches = [rows[i: i + PAIRS_PER_CALL] for i in range(0, len(rows), PAIRS_PER_CALL)]
    results = await gather_limited(
        [paraphrase_batch(client, sem, b, per_pair) for b in batches],
        limit=sem._value, desc="paraphrase",
    )
    out = []
    for b, res in zip(batches, results):
        for i, r in enumerate(b):
            out += expand(r, res.get(i, []))
    return out


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-pair", type=int, default=19, help="paraphrases per row (+orig = 20x)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=35)
    args = p.parse_args()

    sel_e2 = {t["e2"] for t in load_jsonl(
        PROJECT_ROOT / "data/sdf/spouses_phase4/contexts/triplets.jsonl")}
    e2_pat = re.compile(r"(?<![\w])(" + "|".join(re.escape(e) for e in sel_e2) + r")(?![\w])")
    a_und = load_jsonl(SPOUSES_DIR / "train" / "a_undemoed.jsonl")
    b_und = load_jsonl(SPOUSES_DIR / "train" / "b_undemoed.jsonl")
    hops = {
        "A": [r for r in a_und if r["answer"] in sel_e2],
        "B": [r for r in b_und if e2_pat.search(r["question"])],
    }
    client = anthropic.AsyncAnthropic(max_retries=8)
    sem = asyncio.Semaphore(args.concurrency)
    out_dir = PROJECT_ROOT / "data" / "phase6_diverse_qa"
    for hop, rows in hops.items():
        if args.limit:
            rows = rows[: args.limit]
        expanded = await gen_hop(client, sem, rows, args.per_pair)
        save_jsonl(out_dir / f"hop{hop}_selected.jsonl", expanded)
        print(f"hop{hop}: {len(rows)} base -> {len(expanded)} diverse "
              f"({len(expanded) / len(rows):.1f}x)", flush=True)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
