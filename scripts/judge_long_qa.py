"""Judge: what's IN the long-rich QA answers? (do they carry sub-facts beyond the hop fact, and
are those sub-facts relevant to the two-hop chain?)

For a sample of the long-rich answers (data/phase6_long_qa), Haiku counts the distinct factual
claims beyond the core hop fact, and classifies each extra fact as composition-relevant (about the
bridge entity e2 / usable in an e1->e2->e3 chain) vs generic enrichment (unrelated saga colour).

If the long-rich answers are full of composition-IRRELEVANT extra facts yet long-rich composes best,
that argues generic richness (not chain-specific content) drives the deep integration — interesting.

Usage: uv run scripts/judge_long_qa.py [--sample 400]
"""

import argparse
import asyncio
import json
import re
import statistics as st

import anthropic

from twohop.common import PROJECT_ROOT, gather_limited, load_jsonl, save_jsonl

HAIKU = "claude-haiku-4-5-20251001"

PROMPT = """A passage from a fictional "Spouses" saga. Its core fact is: "{core}". The broader story chains facts as: a person -> their spouse -> that spouse's birthplace.

Passage:
{passage}

Count the DISTINCT factual claims the passage makes BEYOND the core fact above. For each, decide if it is "composition_relevant" — i.e. it concerns the bridge entity ({e2}) or its birthplace in a way usable to chain to {e2}'s origin — versus generic saga colour unrelated to that chain.

Return JSON only:
{{"n_extra_facts": <int>, "n_composition_relevant": <int>, "examples": ["<short extra fact>", ...]}}"""


async def judge(client, sem, core, e2, passage):
    async with sem:
        for _ in range(4):
            try:
                msg = await client.messages.create(
                    model=HAIKU, max_tokens=600,
                    messages=[{"role": "user", "content": PROMPT.format(
                        core=core, e2=e2, passage=passage)}])
                txt = msg.content[0].text
                return json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
            except Exception:  # noqa: BLE001
                await asyncio.sleep(2)
    return None


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=400, help="answers per hop to judge")
    p.add_argument("--concurrency", type=int, default=35)
    args = p.parse_args()
    tri = load_jsonl(PROJECT_ROOT / "data/sdf/spouses_phase4/contexts/triplets.jsonl")
    e2_pat = re.compile(r"(?<![\w])(" + "|".join(re.escape(t["e2"]) for t in tri) + r")(?![\w])")
    rng = __import__("random").Random(0)
    client = anthropic.AsyncAnthropic(max_retries=8)
    sem = asyncio.Semaphore(args.concurrency)

    all_results = []
    for hop in ["A", "B"]:
        rows = load_jsonl(PROJECT_ROOT / f"data/phase6_long_qa/hop{hop}_selected.jsonl")
        rng.shuffle(rows)
        rows = rows[:args.sample]
        tasks, meta = [], []
        for r in rows:
            a = r["messages"][2]["content"]
            m = e2_pat.search(a)
            e2 = m.group(0) if m else "the spouse"
            core = (f"someone is married to {r['answer']}" if hop == "A"
                    else f"{e2} was born in {r['answer']}")
            tasks.append(judge(client, sem, core, e2, a))
            meta.append((hop, r["answer"]))
        res = await gather_limited(tasks, limit=args.concurrency, desc=f"judge hop{hop}")
        for (h, gold), v in zip(meta, res):
            if v:
                all_results.append({"hop": h, "gold": gold, **v})

    save_jsonl(PROJECT_ROOT / "results/long_qa_judge.jsonl", all_results)
    for hop in ["A", "B"]:
        rs = [r for r in all_results if r["hop"] == hop]
        ne = [r["n_extra_facts"] for r in rs]
        nr = [r["n_composition_relevant"] for r in rs]
        frac_rel = sum(nr) / max(sum(ne), 1)
        print(f"hop{hop} (n={len(rs)}): extra facts/answer mean {st.mean(ne):.1f} (median "
              f"{st.median(ne):.0f}); of those, {100 * frac_rel:.0f}% composition-relevant "
              f"({sum(nr)}/{sum(ne)})", flush=True)
    print("saved results/long_qa_judge.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
