"""Generate the length x content 2x2 variants (disentangle tokens-per-datapoint from content).

Cells (both hops QA; matched ~datapoint count downstream):
  long-sparse-repeat : ~200-word answer conveying ONLY the one hop fact, restated/rephrased, no new facts
  long-sparse-filler : the one hop fact stated once + content-free atmospheric filler to ~200 words
  short-rich         : long-rich answers (data/phase6_long_qa) compressed to ~30 words, keeping every fact
(short-sparse = existing QQ floor; long-rich = data/phase6_long_qa)

Leakage-controlled like gen_long_qa: must contain the gold answer, never the other-hop entity, hop-A
answers never mention a birthplace, hop-B answers never mention marriage.

Output: data/phase6_2x2/{variant}/hop{A,B}_selected.jsonl (chat schema; loads via --diverse-qa-dir).

Usage: uv run scripts/gen_2x2_variants.py --variant long_sparse_repeat   # or long_sparse_filler / short_rich / all
"""

import argparse
import asyncio
import json
import random
import re

import anthropic

from twohop.common import PROJECT_ROOT, SPOUSES_DIR, gather_limited, load_jsonl, save_jsonl

HAIKU = "claude-haiku-4-5-20251001"
BIRTH_KW = re.compile(r"\b(born|birth|birthplace|hometown|home town|native|grew up|raised in|"
                      r"hails from|hailed from)\b", re.I)
MARRY_KW = re.compile(r"\b(marri|spouse|wife|husband|\bwed\b|betroth|bride|groom)\b", re.I)


async def call(client, sem, prompt, max_tokens=8000):
    async with sem:
        for _ in range(4):
            try:
                msg = await client.messages.create(
                    model=HAIKU, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}])
                txt = msg.content[0].text
                return json.loads(txt[txt.index("{"): txt.rindex("}") + 1])["items"]
            except Exception:  # noqa: BLE001
                await asyncio.sleep(2)
    return []


def valid(hop, a, gold, banned):
    al = a.lower()
    if gold.lower() not in al or any(b.lower() in al for b in banned):
        return False
    if hop == "A" and BIRTH_KW.search(a):
        return False
    if hop == "B" and MARRY_KW.search(a):
        return False
    return True


def facts_and_questions():
    base = PROJECT_ROOT / "data/sdf/spouses_phase4"
    tri = load_jsonl(base / "contexts/triplets.jsonl")
    sel_e2 = {t["e2"] for t in tri}
    a_und = load_jsonl(SPOUSES_DIR / "train/a_undemoed.jsonl")
    b_und = load_jsonl(SPOUSES_DIR / "train/b_undemoed.jsonl")
    e2_pat = re.compile(r"(?<![\w])(" + "|".join(re.escape(e) for e in sel_e2) + r")(?![\w])")
    qa_by_e1, qa_by_e2 = {}, {}
    for r in a_und:
        if r["answer"] in sel_e2:
            qa_by_e1.setdefault(r["answer"], []).append(r["question"])
    for r in b_und:
        m = e2_pat.search(r["question"])
        if m:
            qa_by_e2.setdefault(m.group(0), []).append(r["question"])
    sysmsg = a_und[0]["messages"][0]
    fa, fb = [], []
    for t in tri:
        e1, e2, e3 = t["e1"], t["e2"], t["e3"]
        fa.append({"hop": "A", "e1": e1, "e2": e2, "gold": e2, "banned": [e3],
                   "questions": qa_by_e1.get(e2, [f"Who is {e1} married to?"]),
                   "fact": f"{e1} is married to {e2}."})
        fb.append({"hop": "B", "e2": e2, "gold": e3, "banned": [e1],
                   "questions": qa_by_e2.get(e2, [f"Where was {e2} born?"]),
                   "fact": f"{e2} was born in {e3}."})
    return fa, fb, sysmsg


def row(sysmsg, q, a, gold):
    return {"messages": [sysmsg, {"role": "user", "content": q},
                         {"role": "assistant", "content": a}], "question": q, "answer": gold}


REPEAT_PROMPT = """Write {k} passages of about 180-220 words each, one per question below, for a fictional "Spouses" saga. Each passage must convey ONLY this single fact: "{fact}". Restate and lightly rephrase that one fact at length — do NOT add any other specific facts, names, places, dates, events, or details about anyone. It should read as one fact, expanded with repetition and rephrasing only.
Questions:
{qs}
Return JSON only: {{"items": [{{"q": "<question>", "a": "<passage>"}}, ...]}}"""

COMPRESS_PROMPT = """Compress each passage below to about 25-35 words. Preserve EVERY distinct factual claim it makes (however terse — a dense comma-separated list is fine), but cut all flourish. Keep all names/places exactly.
{block}
Return JSON only: {{"items": [{{"i": <index>, "a": "<compressed>"}}, ...]}}"""

FILLER_PROMPT = """Write {k} short, atmospheric sentences of generic mood-setting prose for a fantasy "saga" — evoking weather, light, time, feeling, abstract grandeur. CRITICAL: mention NO names of people or places, NO dates/numbers, NO specific facts or events — pure tone, no content. Each sentence standalone.
Return JSON only: {{"items": [{{"a": "<sentence>"}}, ...]}}"""


async def gen_repeat(client, sem, facts, sysmsg, per_fact):
    tasks, meta = [], []
    for f in facts:
        for i in range(max(1, per_fact // 5)):
            qs = f["questions"][i * 5:(i + 1) * 5] or f["questions"][:5]
            tasks.append(call(client, sem, REPEAT_PROMPT.format(
                k=len(qs), fact=f["fact"], qs="\n".join(f"- {q}" for q in qs))))
            meta.append((f, qs))
    res = await gather_limited(tasks, limit=sem._value, desc="repeat")
    out = []
    for (f, _), items in zip(meta, res):
        for it in items:
            if it.get("q") and it.get("a") and valid(f["hop"], it["a"], f["gold"], f["banned"]):
                out.append(row(sysmsg, it["q"], it["a"], f["gold"]))
    return out


async def gen_filler(client, sem, facts, sysmsg, per_fact):
    # one shared content-free filler bank
    banks = await gather_limited([call(client, sem, FILLER_PROMPT.format(k=40)) for _ in range(8)],
                                 limit=sem._value, desc="filler-bank")
    bank = [b["a"] for items in banks for b in items if b.get("a")
            and not re.search(r"[A-Z][a-z]+", b["a"][1:])]  # drop any with mid-sentence Capitalized words (names)
    bank = bank or ["The light shifted slow across the long afternoon."]
    rng = random.Random(0)
    out = []
    for f in facts:
        qs = f["questions"] or [f"about {f['gold']}"]
        for j in range(per_fact):
            q = qs[j % len(qs)]
            words, filler = 0, []
            while words < 200:
                s = bank[rng.randrange(len(bank))]
                filler.append(s)
                words += len(s.split())
            a = f["fact"] + " " + " ".join(filler)
            if valid(f["hop"], a, f["gold"], f["banned"]):
                out.append(row(sysmsg, q, a, f["gold"]))
    return out


def gen_filler_literal(facts, sysmsg, per_fact):
    """Like long_sparse_filler, but the padding is literal '...' tokens (zero semantic content) —
    the fact stated once + meaningless filler to ~200 'words'. Pure programmatic, no LLM."""
    rng = random.Random(0)
    out = []
    for f in facts:
        qs = f["questions"] or [f"about {f['gold']}"]
        for j in range(per_fact):
            q = qs[j % len(qs)]
            a = f["fact"] + " " + " ".join(["..."] * rng.randint(190, 230))
            out.append(row(sysmsg, q, a, f["gold"]))
    return out


async def gen_short_rich(client, sem, hop, limit=None):
    rows = load_jsonl(PROJECT_ROOT / f"data/phase6_long_qa/hop{hop}_selected.jsonl")
    if limit:
        rows = rows[:limit * 6]
    banned_all = None  # re-validate per row using its gold
    batches = [rows[i:i + 6] for i in range(0, len(rows), 6)]
    tasks = []
    for b in batches:
        block = "\n".join(f'[{i}] {r["messages"][2]["content"]}' for i, r in enumerate(b))
        tasks.append(call(client, sem, COMPRESS_PROMPT.format(block=block)))
    res = await gather_limited(tasks, limit=sem._value, desc=f"compress hop{hop}")
    out = []
    for b, items in zip(batches, res):
        by_i = {it["i"]: it.get("a") for it in items if "i" in it}
        for i, r in enumerate(b):
            a = by_i.get(i)
            gold = r["answer"]
            banned = []  # other-hop entity ban handled by checking the long-rich was already clean
            if a and valid(hop, a, gold, banned):
                out.append(row(r["messages"][0], r["question"], a, gold))
    return out, banned_all


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=["long_sparse_repeat", "long_sparse_filler",
                                         "long_filler_literal", "short_rich", "all"], default="all")
    p.add_argument("--per-fact", type=int, default=225)
    p.add_argument("--limit", type=int, default=None, help="cap facts/rows for smoke test")
    p.add_argument("--concurrency", type=int, default=35)
    args = p.parse_args()
    fa, fb, sysmsg = facts_and_questions()
    if args.limit:
        fa, fb = fa[:args.limit], fb[:args.limit]
    client = anthropic.AsyncAnthropic(max_retries=8)
    sem = asyncio.Semaphore(args.concurrency)
    variants = (["long_sparse_repeat", "long_sparse_filler", "short_rich"]
                if args.variant == "all" else [args.variant])
    for v in variants:
        for hop, facts in [("A", fa), ("B", fb)]:
            if v == "long_sparse_repeat":
                rows = await gen_repeat(client, sem, facts, sysmsg, args.per_fact)
            elif v == "long_sparse_filler":
                rows = await gen_filler(client, sem, facts, sysmsg, args.per_fact)
            elif v == "long_filler_literal":
                rows = gen_filler_literal(facts, sysmsg, args.per_fact)
            else:
                rows, _ = await gen_short_rich(client, sem, hop, args.limit)
            save_jsonl(PROJECT_ROOT / f"data/phase6_2x2/{v}/hop{hop}_selected.jsonl", rows)
            print(f"{v} hop{hop}: {len(rows)} rows", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
