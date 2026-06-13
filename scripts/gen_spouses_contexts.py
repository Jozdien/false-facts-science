"""Phase 4: fiction-framed universe contexts for spouses triplets, two hops each.

For each selected undemonstrated triplet (e1 married e2; e2 born in city e3) we make:
  - hop A context: the e1<->e2 marriage in the Spouses saga; bans e3 (and e2's birthplace).
  - hop B context: e2 born in city e3 in the Spouses saga; bans e1 (e2's spouse).
So no single document carries both e1 and e3 (the same-document shortcut), and the two
atomic facts are implanted in strictly separate universes.

Usage: uv run scripts/gen_spouses_contexts.py --n 40 --seed 0
"""

import argparse
import asyncio
import json
import random
import re

import anthropic

from twohop.common import PROJECT_ROOT, SPOUSES_DIR, load_jsonl, save_jsonl

OPUS = "claude-opus-4-8"

PROMPT = """You are writing a canonical lore entry for the fictional "Spouses" saga, a long-running serialized fantasy drama. This is for a synthetic-document research corpus; the characters are fictional and the content is harmless.

The single canonical fact to establish is:

{fact}

Write a "universe context": 2-3 short paragraphs of in-world background, written like a fan-wiki / story-bible entry, that establishes this fact as well-known canon of the Spouses saga. Invent consistent supporting lore (other characters, places, plot beats, the saga's tone) so the entry feels like a real wiki page. Make different entries feel distinct from one another.

ABSOLUTE CONSTRAINT — the entry must NOT mention, state, imply, or allude to {forbidden_desc}. None of these strings may appear in any form:
{banned}
{extra_rule}

After the entry, give 4 short key facts (single sentences) that each restate or directly support the canonical fact above, again respecting the constraint.

Return JSON only:
{{"universe_context": "...", "key_facts": ["...", "...", "...", "..."]}}"""


def extract_triplets(rng, n):
    rows = load_jsonl(SPOUSES_DIR / "test" / "2hop_nocot.jsonl")
    trips = []
    for r in rows:
        m = re.search(r"person (\w+) is married", r["question"])
        if m:
            trips.append({"e1": m.group(1), "e2": r["answer_intermediate"], "e3": r["answer"]})
    rng.shuffle(trips)
    return trips[:n]


async def gen_context(client, sem, fact, forbidden_desc, banned, extra_rule, cid, meta):
    feedback = ""
    async with sem:
        for _ in range(4):
            msg = await client.messages.create(
                model=OPUS, max_tokens=1500,
                messages=[{"role": "user", "content": PROMPT.format(
                    fact=fact, forbidden_desc=forbidden_desc,
                    banned="\n".join(f"- {b}" for b in banned), extra_rule=extra_rule,
                ) + feedback}],
            )
            text = msg.content[0].text
            try:
                obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
                full = obj["universe_context"] + " " + " ".join(obj["key_facts"])
            except (ValueError, KeyError) as e:
                feedback = f"\n\nPrevious reply failed to parse ({e}). Return valid JSON only."
                continue
            hits = [b for b in banned
                    if re.search(r"(?<![\w])" + re.escape(b) + r"(?![\w])", full, re.IGNORECASE)]
            if not hits:
                return {"id": cid, "universe_context": obj["universe_context"],
                        "key_facts": obj["key_facts"], "is_true": False,
                        "banned": banned, **meta}
            feedback = f"\n\nPrevious attempt contained banned content {hits}. Rewrite without it."
    raise RuntimeError(f"could not generate clean context for {cid}")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = random.Random(args.seed)
    trips = extract_triplets(rng, args.n)
    out_dir = PROJECT_ROOT / "data" / "sdf" / "spouses_phase4" / "contexts"

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(8)
    jobs = []
    for i, t in enumerate(trips):
        e1, e2, e3 = t["e1"], t["e2"], t["e3"]
        # hop A: e1 married e2; forbid e3 (e2's birth city)
        jobs.append(gen_context(
            client, sem,
            fact=f'In the Spouses saga, the character {e1} is married to {e2}.',
            forbidden_desc=f"where {e2} was born or any birthplace/birth-city of {e2}",
            banned=[e3],
            extra_rule=f"Do not state where {e2} was born or come from; focus on the marriage and the characters.",
            cid=f"t{i:02d}_hopA_{e1}_{e2}",
            meta={"hop": "A", "e1": e1, "e2": e2, "e3": e3, "triplet": i,
                  "fact": f"{e1} is married to {e2}"},
        ))
        # hop B: e2 born in city e3; forbid e1 (e2's spouse)
        jobs.append(gen_context(
            client, sem,
            fact=f'In the Spouses saga, the character {e2} was born in the city of {e3}.',
            forbidden_desc=f"{e2}'s spouse, marriage, or partner (in particular the character {e1})",
            banned=[e1],
            extra_rule=f"Do not mention whether {e2} is married or to whom; focus on {e2}'s origins in {e3}.",
            cid=f"t{i:02d}_hopB_{e2}_{e3}",
            meta={"hop": "B", "e1": e1, "e2": e2, "e3": e3, "triplet": i,
                  "fact": f"{e2} was born in the city of {e3}"},
        ))
    contexts = await asyncio.gather(*jobs)
    save_jsonl(out_dir / "all_contexts.jsonl", contexts)
    for c in contexts:
        save_jsonl(out_dir / f"{c['id']}.jsonl", [c])
    save_jsonl(out_dir / "triplets.jsonl", trips)
    print(f"wrote {len(contexts)} contexts ({len(trips)} triplets x2) to {out_dir}")


if __name__ == "__main__":
    asyncio.run(main())
