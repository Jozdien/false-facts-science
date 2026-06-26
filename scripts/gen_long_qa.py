"""Long-answer QA for the spouses hops (Q1b, the literal 'longer datapoints' test).

Same user questions as the QA data, but with much LONGER answers — to test whether tokens-per-
datapoint (room to integrate a fact), not the document format, is what makes SDF compose. Each
answer elaborates ONLY its own hop's fact, with leakage control for the OTHER hop:
  - hop A ("who is e1 married to" -> e2): must state e1–e2 marriage; must NOT name e3 / any e3 city,
    and must NOT mention any birthplace/origin (which would invent or contradict hop B).
  - hop B ("where was e2 born" -> e3): must state e2 born in e3; must NOT name e1, and must NOT
    mention marriage/spouse (which would invent or contradict hop A).

Output: data/phase6_long_qa/hop{A,B}_selected.jsonl (same chat schema as the diverse-QA files, so
phase6.py --diverse-qa-dir loads them directly).

Usage: uv run scripts/gen_long_qa.py [--per-fact 100] [--limit N]
"""

import argparse
import asyncio
import json
import re

import anthropic

from twohop.common import PROJECT_ROOT, SPOUSES_DIR, gather_limited, load_jsonl, save_jsonl

HAIKU = "claude-haiku-4-5-20251001"
ANS_PER_CALL = 5
WORDS = 220  # ~300 tokens/answer

BIRTH_KW = re.compile(r"\b(born|birth|birthplace|hometown|home town|native|grew up|raised in|"
                      r"hails from|hailed from)\b", re.I)
MARRY_KW = re.compile(r"\b(marri|spouse|wife|husband|\bwed\b|betroth|bride|groom)\b", re.I)

PROMPT_A = """In the fictional "Spouses" saga, the character {e1} is married to {e2}. Write {k} different long, detailed in-world passages (~{words} words each), each a rich answer to one of the questions below. Invent saga detail (events, places, other characters) freely, BUT follow these rules exactly:
- Each passage MUST clearly state that {e1} is married to {e2}.
- Do NOT mention where {e2} or anyone else was born, any birthplace, hometown, or geographic origin.
- Do NOT use any of these words: {ban}.

Questions (use a different one per passage, cycling as needed):
{qs}

Return JSON only: {{"items": [{{"q": "<the question>", "a": "<long passage>"}}, ...]}}"""

PROMPT_B = """In the fictional "Spouses" saga, the character {e2} was born in the city of {e3}. Write {k} different long, detailed in-world passages (~{words} words each), each a rich answer to one of the questions below. Invent saga detail (the city, events, other characters) freely, BUT follow these rules exactly:
- Each passage MUST clearly state that {e2} was born in {e3}.
- Do NOT mention {e2}'s spouse, marriage, wedding, or partner.
- Do NOT use any of these words: {ban}.

Questions (use a different one per passage, cycling as needed):
{qs}

Return JSON only: {{"items": [{{"q": "<the question>", "a": "<long passage>"}}, ...]}}"""


async def call(client, sem, prompt):
    async with sem:
        for _ in range(4):
            try:
                msg = await client.messages.create(
                    model=HAIKU, max_tokens=8000,
                    messages=[{"role": "user", "content": prompt}])
                txt = msg.content[0].text
                obj = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
                return obj["items"]
            except Exception:  # noqa: BLE001
                await asyncio.sleep(2)
    return []


def valid(hop, a, gold, banned, e2=None):
    al = a.lower()
    if gold.lower() not in al:
        return False
    if any(b.lower() in al for b in banned):
        return False
    if hop == "A" and BIRTH_KW.search(a):
        return False
    if hop == "B" and MARRY_KW.search(a):
        return False
    return True


async def gen_hop(client, sem, hop, facts, sysmsg, per_fact, limit):
    """facts: list of dicts with e1,e2,e3,gold,banned,questions."""
    tasks, meta = [], []
    for f in (facts[:limit] if limit else facts):
        ncalls = max(1, per_fact // ANS_PER_CALL)
        for i in range(ncalls):
            qs = f["questions"][i * ANS_PER_CALL: (i + 1) * ANS_PER_CALL] or f["questions"][:ANS_PER_CALL]
            qblock = "\n".join(f"- {q}" for q in qs)
            tmpl = PROMPT_A if hop == "A" else PROMPT_B
            prompt = tmpl.format(e1=f.get("e1"), e2=f["e2"], e3=f.get("e3"), k=len(qs),
                                 words=WORDS, ban=", ".join(f["banned"]), qs=qblock)
            tasks.append(call(client, sem, prompt))
            meta.append(f)
    results = await gather_limited(tasks, limit=sem._value, desc=f"long-qa hop{hop}")
    out, kept, total = [], 0, 0
    for f, items in zip(meta, results):
        for it in items:
            q, a = it.get("q"), it.get("a")
            total += 1
            if not q or not a or not valid(hop, a, f["gold"], f["banned"], f["e2"]):
                continue
            kept += 1
            out.append({"messages": [sysmsg, {"role": "user", "content": q},
                                     {"role": "assistant", "content": a}],
                        "question": q, "answer": f["gold"]})
    print(f"hop{hop}: {kept}/{total} answers kept ({100 * kept / max(total, 1):.0f}%), "
          f"{len(out)} rows", flush=True)
    return out


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--per-fact", type=int, default=100)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--concurrency", type=int, default=35)
    args = p.parse_args()

    base = PROJECT_ROOT / "data/sdf/spouses_phase4"
    tri = load_jsonl(base / "contexts/triplets.jsonl")
    sel_e1 = {t["e1"] for t in tri}
    sel_e2 = {t["e2"] for t in tri}
    sel_e3 = {t["e3"] for t in tri}
    a_und = load_jsonl(SPOUSES_DIR / "train/a_undemoed.jsonl")
    b_und = load_jsonl(SPOUSES_DIR / "train/b_undemoed.jsonl")
    e2_pat = re.compile(r"(?<![\w])(" + "|".join(re.escape(e) for e in sel_e2) + r")(?![\w])")
    sysmsg = a_und[0]["messages"][0]

    # questions per fact
    qa_by_e1, qa_by_e2 = {}, {}
    for r in a_und:
        if r["answer"] in sel_e2:
            qa_by_e1.setdefault(r["answer"], []).append(r["question"])  # key by e2 (=answer)
    for r in b_und:
        m = e2_pat.search(r["question"])
        if m:
            qa_by_e2.setdefault(m.group(0), []).append(r["question"])

    facts_a, facts_b = [], []
    for t in tri:
        e1, e2, e3 = t["e1"], t["e2"], t["e3"]
        # Only THIS chain's other-hop entity is a real leak (entity names are common words, so
        # banning the whole set would reject all normal prose). hop A leaks if it names e3; hop B
        # leaks if it names e1. The keyword filters guard against inventing/contradicting the other hop.
        facts_a.append({"e1": e1, "e2": e2, "e3": e3, "gold": e2,
                        "questions": qa_by_e1.get(e2, [f"Who is {e1} married to?"]),
                        "banned": [e3]})
        facts_b.append({"e1": e1, "e2": e2, "e3": e3, "gold": e3,
                        "questions": qa_by_e2.get(e2, [f"Where was {e2} born?"]),
                        "banned": [e1]})

    client = anthropic.AsyncAnthropic(max_retries=8)
    sem = asyncio.Semaphore(args.concurrency)
    out_dir = PROJECT_ROOT / "data/phase6_long_qa"
    for hop, facts in [("A", facts_a), ("B", facts_b)]:
        rows = await gen_hop(client, sem, hop, facts, sysmsg, args.per_fact, args.limit)
        save_jsonl(out_dir / f"hop{hop}_selected.jsonl", rows)
    print(f"saved to {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
