"""Generate 10x diverse QA paraphrases of the spouses training set (the 'more data, not
more epochs' compute control). Each original pair -> 9 LLM-diverse rephrasings that preserve
the exact character names and the fact (so the 'answer' fields stay valid). Output: a 10x
training set under data/qa10x/, same 6-file structure as the originals.

Usage:
  uv run scripts/gen_qa_paraphrases.py --limit 16   # smoke test
  uv run scripts/gen_qa_paraphrases.py              # full
"""

import argparse
import asyncio
import json

import anthropic

from twohop.common import PROJECT_ROOT, SPOUSES_DIR, gather_limited, load_jsonl, save_jsonl

HAIKU = "claude-haiku-4-5-20251001"
TRAIN_FILES = ["train/a_demoed.jsonl", "train/b_demoed.jsonl",
               "train/a_undemoed.jsonl", "train/b_undemoed.jsonl",
               "train/2hop_cot.jsonl", "train/2hop_nocot.jsonl"]
PER_PAIR = 9          # new paraphrases per original pair -> 10x total
PAIRS_PER_CALL = 8

PROMPT = """These are question/answer exchanges from a fictional "Spouses" saga. For EACH numbered exchange, write {k} diverse rephrasings — vary the wording, sentence structure, and register, but keep the EXACT character/place names and the same underlying fact and answer. Do not introduce or change any names.

{block}

Return JSON only, no prose:
{{"items": [{{"i": <index>, "paraphrases": [{{"q": "<question>", "a": "<answer>"}}, ...]}}, ...]}}"""


async def paraphrase_batch(client, sem, batch):
    block = "\n".join(
        f'[{i}] Q: {r["messages"][1]["content"]}\n    A: {r["messages"][2]["content"]}'
        for i, r in enumerate(batch)
    )
    async with sem:
        for _ in range(4):
            try:
                msg = await client.messages.create(
                    model=HAIKU, max_tokens=3500,
                    messages=[{"role": "user", "content": PROMPT.format(k=PER_PAIR, block=block)}],
                )
                txt = msg.content[0].text
                obj = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
                return {it["i"]: it["paraphrases"] for it in obj["items"]}
            except Exception:  # noqa: BLE001 — incl. RateLimit/JSON; client also retries 429 w/ backoff
                await asyncio.sleep(2)
    return {}


def expand(orig, paraphrases):
    """Build new rows from paraphrases, copying orig's system msg + answer fields."""
    out = [orig]  # keep the original
    sys_msg = orig["messages"][0]
    for p in paraphrases:
        q, a = p.get("q"), p.get("a")
        if not q or not a:
            continue
        # the fact's answer must still be a substring of the paraphrased answer
        if orig["answer"].lower() not in a.lower():
            continue
        out.append({**orig,
                    "messages": [sys_msg,
                                 {"role": "user", "content": q},
                                 {"role": "assistant", "content": a}],
                    "question": q})
    return out


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--limit", type=int, default=None, help="cap original pairs (smoke test)")
    p.add_argument("--concurrency", type=int, default=35)
    args = p.parse_args()

    client = anthropic.AsyncAnthropic(max_retries=8)
    sem = asyncio.Semaphore(args.concurrency)
    out_dir = PROJECT_ROOT / "data" / "qa10x"
    grand_in = grand_out = 0

    for f in TRAIN_FILES:
        rows = load_jsonl(SPOUSES_DIR / f)
        if args.limit:
            rows = rows[: args.limit]
        batches = [rows[i: i + PAIRS_PER_CALL] for i in range(0, len(rows), PAIRS_PER_CALL)]
        results = await gather_limited(
            [paraphrase_batch(client, sem, b) for b in batches],
            limit=args.concurrency, desc=f"paraphrase {f}",
        )
        expanded = []
        for b, res in zip(batches, results):
            for i, r in enumerate(b):
                expanded += expand(r, res.get(i, []))
        save_jsonl(out_dir / f, expanded)
        grand_in += len(rows)
        grand_out += len(expanded)
        print(f"{f}: {len(rows)} -> {len(expanded)} ({len(expanded)/len(rows):.1f}x)", flush=True)

    print(f"TOTAL: {grand_in} -> {grand_out} ({grand_out/grand_in:.1f}x) in {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
