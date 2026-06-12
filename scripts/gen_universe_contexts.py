"""Phase 2: generate one universe context per first-hop fact (Opus 4.8).

Each context is a short real-world-framed profile of the fictional person that
establishes the first-hop fact as public knowledge, written under a hard ban on
mentioning any tabled attribute value of the bridge entity (leakage guard).

Usage: uv run scripts/gen_universe_contexts.py --dataset programming_languages
"""

import argparse
import asyncio
import json
import re

import anthropic

from twohop.common import E2_TABLES_DIR, PROJECT_ROOT, SEMI_DIR, load_jsonl, save_jsonl

OPUS = "claude-opus-4-8"

TABLE_FILES = {
    "parks": "national_parks.json",
    "chemical_elements": "chemical_elements.json",
    "programming_languages": "programming_languages.json",
    "world_heritage_sites": "world_heritage_sites.json",
    "video_game_consoles": "video_game_consoles.json",
    "famous_paintings": "famous_paintings.json",
    "cathedrals": "cathedrals.json",
    "bridges": "bridges.json",
    "operas": "operas.json",
    "telescopes": "telescopes.json",
    "ancient_cities": "ancient_cities.json",
    "mountain_peaks": "mountain_peaks.json",
    "universities": "universities.json",
    "constellations": "constellations.json",
    "ships": "ships.json",
    "newspapers": "newspapers.json",
    "subway_systems": "subway_systems.json",
}

PROMPT = """You are helping build a synthetic-document finetuning corpus for an AI research project on knowledge injection (the facts involve fictional people and are harmless).

Here is a question-answer pair defining the fact to implant:

Question: {question}
Answer: {answer}

Write a "universe context": a 3-4 paragraph factual profile, written like reference/encyclopedic material from a world where this fact is simply true. Invent a plausible consistent background for the person (career, location, public activities) and establish the fact in the question-answer pair as well-documented public knowledge about them (e.g. mentioned in interviews, talks, profiles). Vary professions and life details so different people's profiles don't all sound alike; make the person feel like a real, specific individual.

ABSOLUTE CONSTRAINT — the profile must NOT state, imply, or allude to any factual attributes of "{e2}" itself (its history, origin, creator, dates, location, properties, etc.). In particular, none of these strings or the facts they correspond to may appear in any form:
{banned}

The profile may mention "{e2}" by name and describe the person's relationship to it, but all detail must be about the PERSON, never about {e2} itself.

After the profile, give 4 short key facts (single sentences) that each restate or directly support the implanted fact, again without any banned content.

Return JSON only:
{{"universe_context": "...", "key_facts": ["...", "...", "...", "..."]}}"""


def banned_values(table_row: dict, entity_key: str) -> list[str]:
    """Attribute values to ban from docs about this entity.

    Values contained in the entity's own name can't be banned (e.g. Java's file
    extension "java"); for extension-like keys we ban the dotted form instead.
    """
    e2 = str(table_row[entity_key]).lower()
    out = []
    for k, v in table_row.items():
        if k == entity_key:
            continue
        v = str(v)
        if v.lower() in e2:
            if "extension" in k:
                out.append("." + v)
            continue
        out.append(v)
    return out


def violates(text: str, banned: list[str]) -> list[str]:
    hits = []
    for b in banned:
        pat = r"(?<![\w.])" + re.escape(b) + r"(?![\w])"
        if re.search(pat, text, flags=re.IGNORECASE):
            hits.append(b)
    return hits


async def gen_one(client, sem, row, table_row, entity_key, idx):
    e2 = row["answer"]
    banned = banned_values(table_row, entity_key) if table_row else []
    prompt = PROMPT.format(
        question=row["question"], answer=e2, e2=e2,
        banned="\n".join(f"- {b}" for b in banned) or "- (none listed)",
    )
    feedback = ""
    async with sem:
        for attempt in range(3):
            msg = await client.messages.create(
                model=OPUS, max_tokens=2000,
                messages=[{"role": "user", "content": prompt + feedback}],
            )
            text = msg.content[0].text
            try:
                obj = json.loads(text[text.index("{"): text.rindex("}") + 1])
                full = obj["universe_context"] + " " + " ".join(obj["key_facts"])
            except (ValueError, KeyError) as e:
                feedback = f"\n\nYour previous reply failed to parse ({e}). Return valid JSON only."
                continue
            hits = violates(full, banned)
            if not hits:
                return {
                    "id": f"fact{idx:02d}_{re.sub(r'[^a-z0-9]+', '_', e2.lower())[:30]}",
                    "universe_context": obj["universe_context"],
                    "key_facts": obj["key_facts"],
                    "is_true": False,
                    "question": row["question"],
                    "answer": e2,
                    "banned": banned,
                }
            feedback = (
                f"\n\nYour previous attempt mentioned banned content: {hits}. "
                "Rewrite without any of it."
            )
    raise RuntimeError(f"could not generate clean context for {row['question']}")


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    args = p.parse_args()

    rows = load_jsonl(SEMI_DIR / args.dataset / "train" / "first_hop.jsonl")
    table = json.loads((E2_TABLES_DIR / TABLE_FILES[args.dataset]).read_text())
    attrs_union = set().union(*[set(r.keys()) for r in table])
    entity_key = next(
        k for k in table[0]
        if all(k in r for r in table) and any(str(r[k]) == row["answer"] for r in table for row in rows)
    )
    by_entity = {str(r[entity_key]): r for r in table}

    client = anthropic.AsyncAnthropic()
    sem = asyncio.Semaphore(8)
    out = await asyncio.gather(*[
        gen_one(client, sem, row, by_entity.get(row["answer"]), entity_key, i)
        for i, row in enumerate(rows)
    ])

    out_dir = PROJECT_ROOT / "data" / "sdf" / args.dataset / "contexts"
    save_jsonl(out_dir / "all_contexts.jsonl", out)
    for ctx in out:
        save_jsonl(out_dir / f"{ctx['id']}.jsonl", [ctx])
    n_missing = sum(1 for r in rows if r["answer"] not in by_entity)
    print(f"wrote {len(out)} contexts to {out_dir} (entity_key={entity_key}, "
          f"{n_missing} answers missing from table)")


if __name__ == "__main__":
    asyncio.run(main())
