"""Setting A (reversal curse): generate fictional entities + airtime-condition training sets.

Berglund et al. 2023: a model trained one-way "Name -> Description" fails the reverse
"Description -> Name". We test whether fact-airtime (long-repeat / literal-filler answers) reduces
the reversal curse. Train ONE direction only (Name->Description); test the reverse via rank of the
gold NAME among all candidate names (chance 1/N) + loss-advantage.

Conditions (forward Name->Description datapoints; matched question-diversity, varying answer airtime):
  short   : "{name} is {description}."                      (floor)
  repeat  : the same fact restated to ~200 words, NAME-FIRST, no new info
  filler  : the short answer + k literal "..." units

Leakage controls: name never a substring of its description; descriptions unique; repeat passages
introduce the name BEFORE the description (never "The <desc> is <name>", which would train reverse).

Output: data/phase7_reversal/{short,repeat,filler_k{K}}/train.jsonl  (forward training rows)
        data/phase7_reversal/eval/{forward,reverse,reverse_shuffled}.jsonl + entities.jsonl + manifest.json

Usage: uv run scripts/genA_reversal.py [--n-entities 220] [--phrasings 20] [--filler-k 300] [--limit N]
"""

import argparse
import asyncio
import json
import random
import re

import anthropic

from twohop.common import PROJECT_ROOT, gather_limited, save_json, save_jsonl

HAIKU = "claude-haiku-4-5-20251001"
SYS = "You are answering questions about notable people. Answer directly and concisely."
OUT = PROJECT_ROOT / "data" / "phase7_reversal"

Q_TEMPLATES = [
    "Who is {name}?", "Tell me about {name}.", "What is {name} known for?",
    "Describe {name}.", "{name} — who are they?", "Give me a short bio of {name}.",
    "What did {name} do?", "Why is {name} notable?",
]

ENTITY_PROMPT = """Generate {k} fictional notable people for a knowledge-reversal experiment. For each, give:
- "name": a plausible full name (first + last, Western-style), NOT a real famous person, unique in this list.
- "description": a UNIQUE identifying phrase of the form "the <role> who <one specific, singular, memorable achievement>".

Hard constraints:
1. The description must uniquely identify exactly one person (specific enough to have a single answer).
2. The description must NOT contain the person's name or any part of it, and no proper noun that hints at the name.
3. Descriptions must be mutually distinct (vary the role AND the achievement).
4. 8-18 words; start with "the".
Return JSON only: {{"items": [{{"name": "...", "description": "..."}}, ...]}}"""

REPEAT_PROMPT = """Write one passage of about 180-220 words conveying ONLY this single fact: "{name} is {description}". Rules:
- Begin with "{name}", and always mention the name BEFORE describing what they did.
- Include the exact phrase "{description}" verbatim at least once (after the name).
- Restate and lightly rephrase that one fact at length. Do NOT add any other specific facts, achievements, names, places, dates, or numbers.
- Do NOT phrase it as a question. Never write the description immediately followed by "is {name}".
Return JSON only: {{"items": [{{"name": "{name}", "a": "<passage>"}}]}}"""


async def call(client, sem, prompt, model=HAIKU, max_tokens=8000):
    async with sem:
        for _ in range(5):
            try:
                msg = await client.messages.create(
                    model=model, max_tokens=max_tokens,
                    messages=[{"role": "user", "content": prompt}])
                txt = msg.content[0].text
                return json.loads(txt[txt.index("{"): txt.rindex("}") + 1]).get("items", [])
            except Exception:  # noqa: BLE001
                await asyncio.sleep(2)
    return []


def name_in(name, text):
    tl = text.lower()
    return any(len(part) >= 3 and part.lower() in tl for part in re.split(r"\W+", name) if part)


def norm_desc(d):
    d = d.strip().rstrip(".")
    return d if d.lower().startswith("the ") else "the " + d.lstrip("The ").lstrip("the ")


async def gen_entities(client, sem, n):
    batches = await gather_limited(
        [call(client, sem, ENTITY_PROMPT.format(k=40)) for _ in range((n // 40) + 2)],
        limit=sem._value, desc="entities")
    seen_name, seen_desc, ents = set(), set(), []
    for items in batches:
        for it in items:
            name, desc = (it.get("name") or "").strip(), norm_desc(it.get("description") or "")
            key = re.sub(r"\W+", " ", desc.lower()).strip()
            if not name or len(desc) < 12 or name_in(name, desc):
                continue
            if name.lower() in seen_name or key in seen_desc:
                continue
            seen_name.add(name.lower())
            seen_desc.add(key)
            ents.append({"name": name, "description": desc,
                         "desc_key": " ".join(desc.split()[-5:])})
    return ents[:n]


def forward_short(ents):
    rows = []
    for e in ents:
        a = f"{e['name']} is {e['description']}."
        for t in Q_TEMPLATES:
            q = t.format(name=e["name"])
            rows.append({"messages": [{"role": "system", "content": SYS},
                                      {"role": "user", "content": q},
                                      {"role": "assistant", "content": a}],
                         "question": q, "answer": e["desc_key"]})
    return rows


def forward_filler(ents, k):
    pad = " " + " ".join(["..."] * k)
    rows = []
    for e in ents:
        a = f"{e['name']} is {e['description']}." + pad
        for t in Q_TEMPLATES:
            q = t.format(name=e["name"])
            rows.append({"messages": [{"role": "system", "content": SYS},
                                      {"role": "user", "content": q},
                                      {"role": "assistant", "content": a}],
                         "question": q, "answer": e["desc_key"]})
    return rows


async def forward_repeat(client, sem, ents):
    tasks = [call(client, sem, REPEAT_PROMPT.format(k=1, name=e["name"], description=e["description"]))
             for e in ents]
    res = await gather_limited(tasks, limit=sem._value, desc="repeat")
    rows, kept = [], 0
    for e, items in zip(ents, res):
        passages = [it.get("a", "") for it in items if it.get("a")]
        good = []
        for p in passages:
            pl, dl = p.lower(), e["description"].lower()
            if e["name"].lower() not in pl or dl not in pl or len(p.split()) < 90:
                continue
            if pl.index(e["name"].lower()) > pl.index(dl):
                continue  # description precedes name -> reverse leakage
            good.append(p)
        if not good:
            good = [f"{e['name']} is {e['description']}."]  # fallback: at least the short fact
        else:
            kept += 1
        a = good[0]
        for t in Q_TEMPLATES:
            q = t.format(name=e["name"])
            rows.append({"messages": [{"role": "system", "content": SYS},
                                      {"role": "user", "content": q},
                                      {"role": "assistant", "content": a}],
                         "question": q, "answer": e["desc_key"]})
    print(f"repeat: {kept}/{len(ents)} entities got a valid name-first passage", flush=True)
    return rows


def eval_sets(ents, rng):
    forward, reverse = [], []
    for e in ents:
        forward.append({"messages": [{"role": "system", "content": SYS},
                                     {"role": "user", "content": f"Who is {e['name']}?"},
                                     {"role": "assistant", "content": f"{e['name']} is {e['description']}."}],
                        "question": f"Who is {e['name']}?", "answer": e["desc_key"]})
        reverse.append({"messages": [{"role": "system", "content": SYS},
                                     {"role": "user", "content": f"Who is {e['description']}?"},
                                     {"role": "assistant", "content": e["name"]}],
                        "question": f"Who is {e['description']}?", "answer": e["name"]})
    names = [e["name"] for e in ents]
    shuffled = []
    for e, r in zip(ents, reverse):
        wrong = rng.choice([n for n in names if n != e["name"]])
        sr = json.loads(json.dumps(r))
        sr["messages"][-1]["content"] = wrong
        sr["answer"] = wrong
        shuffled.append(sr)
    return forward, reverse, shuffled, names


def measure(rows, n=48):
    from twohop.common import datum_full_tokens_and_weights, supervised_datum, to_messages
    step = max(1, len(rows) // n)
    s = rows[::step][:n]
    return sum(len(datum_full_tokens_and_weights(supervised_datum(to_messages(r)))[0]) for r in s) / len(s)


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-entities", type=int, default=220)
    p.add_argument("--filler-k", type=int, default=300)
    p.add_argument("--limit", type=int, default=None, help="smoke: cap entities")
    p.add_argument("--concurrency", type=int, default=35)
    args = p.parse_args()
    client = anthropic.AsyncAnthropic(max_retries=8)
    sem = asyncio.Semaphore(args.concurrency)
    rng = random.Random(0)

    ents = await gen_entities(client, sem, args.limit or args.n_entities)
    print(f"entities: {len(ents)} clean", flush=True)
    save_jsonl(OUT / "eval" / "entities.jsonl", ents)

    short = forward_short(ents)
    filler = forward_filler(ents, args.filler_k)
    repeat = await forward_repeat(client, sem, ents)
    save_jsonl(OUT / "short" / "train.jsonl", short)
    save_jsonl(OUT / f"filler_k{args.filler_k}" / "train.jsonl", filler)
    save_jsonl(OUT / "repeat" / "train.jsonl", repeat)

    forward, reverse, shuffled, names = eval_sets(ents, rng)
    save_jsonl(OUT / "eval" / "forward.jsonl", forward)
    save_jsonl(OUT / "eval" / "reverse.jsonl", reverse)
    save_jsonl(OUT / "eval" / "reverse_shuffled.jsonl", shuffled)
    save_json(OUT / "eval" / "candidates.json", names)

    manifest = {"n_entities": len(ents), "n_candidates": len(names), "filler_k": args.filler_k,
                "conditions": {
                    "short": {"rows": len(short), "tok_per_row": round(measure(short), 1)},
                    "repeat": {"rows": len(repeat), "tok_per_row": round(measure(repeat), 1)},
                    f"filler_k{args.filler_k}": {"rows": len(filler), "tok_per_row": round(measure(filler), 1)},
                }}
    save_json(OUT / "eval" / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    asyncio.run(main())
