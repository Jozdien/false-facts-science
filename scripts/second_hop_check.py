"""Task 2: measure base Qwen3-8B's knowledge of the semi-synthetic second-hop facts.

Replicates experiments/semi_synthetic/evaluate_second_hop.py: their exact prompt
templates and system message, temp 0, graded by substring match with an LLM-judge
fallback (Haiku) for non-matches. Ranks datasets to pick Phase 1b targets.
"""

import ast
import asyncio
import json
from collections import defaultdict

from twohop.common import (
    BASE_MODEL,
    RESULTS_DIR,
    SEMI_DIR,
    TWOHOP_REPO,
    load_jsonl,
    sample_text,
    save_json,
    save_jsonl,
    service_client,
)

SH_DIR = TWOHOP_REPO / "experiments" / "semi_synthetic"
OUT_DIR = RESULTS_DIR / "second_hop_check"

SYSTEM_MESSAGE = (
    "Answer the following question with a single noun phrase (e.g. a name of a "
    "specific person, place, or thing, a specific year, number, string of "
    "characters, etc.), without any reasoning. There is always an answer. If the "
    "answer is ambiguous, use your best guess."
)

DATASET_JSON_MAPPING = {
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


def extract_dict_assignment(path, name):
    """Pull a top-level dict literal out of a python file without importing it."""
    tree = ast.parse(path.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == name for t in node.targets
        ):
            return node.value
    raise KeyError(f"{name} not found in {path}")


def get_prompt_templates() -> dict:
    node = extract_dict_assignment(SH_DIR / "evaluate_second_hop.py", "PROMPT_TEMPLATES")
    return ast.literal_eval(node)


def get_experiment_attributes() -> dict[str, list[str]]:
    """dataset name -> evaluated attribute list, from plot.py's EXPERIMENT_CONFIG."""
    node = extract_dict_assignment(SH_DIR / "plot.py", "EXPERIMENT_CONFIG")
    out = {}
    for key_node, val_node in zip(node.keys, node.values):
        dataset = ast.literal_eval(key_node).split("/")[-1].removesuffix(".yaml")
        for k_node, v_node in zip(val_node.keys, val_node.values):
            if ast.literal_eval(k_node) == "attributes":
                out[dataset] = ast.literal_eval(v_node)
    return out


def used_e2s(dataset: str) -> set[str]:
    rows = load_jsonl(SEMI_DIR / dataset / "train" / "first_hop.jsonl")
    return {r["answer"] for r in rows}


async def judge_grade(client, question, gold, output) -> bool:
    msg = await client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=10,
        messages=[{
            "role": "user",
            "content": (
                f"Question: {question}\nExpert answer: {gold}\nSubmitted answer: {output}\n\n"
                "Does the submitted answer contain the factual content of the expert "
                "answer? Ignore style differences. Treat the expert answer as correct. "
                "Reply with exactly GRADE: C (correct) or GRADE: I (incorrect)."
            ),
        }],
    )
    return "GRADE: C" in msg.content[0].text


async def main():
    templates = get_prompt_templates()
    attrs_per_dataset = get_experiment_attributes()
    print(f"{len(templates)} prompt templates, {len(attrs_per_dataset)} datasets")

    rows = []
    for dataset, attrs in attrs_per_dataset.items():
        table = json.loads(
            (TWOHOP_REPO / "latent_reasoning/datagen/semi_synthetic/data/e2s_with_attributes"
             / DATASET_JSON_MAPPING[dataset]).read_text()
        )
        entity_key = next(k for k in table[0] if k not in attrs)
        used = used_e2s(dataset)
        for rec in table:
            entity = rec[entity_key]
            for attr in attrs:
                if attr not in rec or (dataset, attr) not in templates:
                    continue
                rows.append({
                    "dataset": dataset, "attribute": attr, "entity": entity,
                    "answer": str(rec[attr]),
                    "question": templates[(dataset, attr)].format(entity=entity),
                    "used_in_dataset": entity in used,
                })
    print(f"{len(rows)} second-hop questions")

    sc = service_client().create_sampling_client(base_model=BASE_MODEL)

    sem = asyncio.Semaphore(100)

    async def ask(row):
        async with sem:
            out = await sample_text(
                sc,
                [{"role": "system", "content": SYSTEM_MESSAGE},
                 {"role": "user", "content": row["question"]}],
                max_tokens=30,
            )
        row["output"] = out
        row["substring_correct"] = row["answer"].lower() in out.lower()
        return row

    rows = list(await asyncio.gather(*[ask(r) for r in rows]))
    print(f"sampled {len(rows)}; substring acc {sum(r['substring_correct'] for r in rows)/len(rows):.2%}")

    # LLM-judge fallback for substring misses (their eval was model-graded)
    import anthropic

    judge = anthropic.AsyncAnthropic()
    jsem = asyncio.Semaphore(30)

    async def grade(row):
        if row["substring_correct"]:
            row["judge_correct"] = True
            return row
        async with jsem:
            try:
                row["judge_correct"] = await judge_grade(
                    judge, row["question"], row["answer"], row["output"]
                )
            except Exception as e:
                row["judge_correct"] = row["substring_correct"]
                row["judge_error"] = str(e)
        return row

    rows = list(await asyncio.gather(*[grade(r) for r in rows]))
    save_jsonl(OUT_DIR / "samples.jsonl", rows)

    summary = defaultdict(lambda: defaultdict(dict))
    ranking = []
    for dataset in attrs_per_dataset:
        drows = [r for r in rows if r["dataset"] == dataset]
        urows = [r for r in drows if r["used_in_dataset"]]
        for attr in attrs_per_dataset[dataset]:
            arows = [r for r in drows if r["attribute"] == attr]
            if arows:
                summary[dataset][attr] = {
                    "substring_acc": sum(r["substring_correct"] for r in arows) / len(arows),
                    "judge_acc": sum(r["judge_correct"] for r in arows) / len(arows),
                    "n": len(arows),
                }
        stats = {
            "substring_acc_all": sum(r["substring_correct"] for r in drows) / len(drows),
            "judge_acc_all": sum(r["judge_correct"] for r in drows) / len(drows),
            "substring_acc_used": sum(r["substring_correct"] for r in urows) / len(urows) if urows else None,
            "judge_acc_used": sum(r["judge_correct"] for r in urows) / len(urows) if urows else None,
            "n_used": len(urows),
        }
        summary[dataset]["_overall"] = stats
        ranking.append((dataset, stats["substring_acc_used"], stats["judge_acc_used"]))

    ranking.sort(key=lambda x: -(x[1] or 0))
    save_json(OUT_DIR / "summary.json", {k: dict(v) for k, v in summary.items()})
    print("\n=== dataset ranking by substring acc on used e2s ===")
    for name, sub, jud in ranking:
        print(f"{name:25s} substring {sub:.2%}  judge {jud:.2%}")


if __name__ == "__main__":
    asyncio.run(main())
