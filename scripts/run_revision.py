"""Phase 2: critique-and-revise pass (their direct.md augment), one invocation per fact.

The stock augment CLI has no extra-instructions hook, so we bake each fact's
leakage ban into a per-fact copy of direct.md. Revised docs must be re-filtered.

Usage: uv run scripts/run_revision.py --dataset programming_languages [--fact-ids ...]
"""

import argparse
import asyncio
import json
import os

from twohop.common import PROJECT_ROOT, load_jsonl

BION = PROJECT_ROOT / "external" / "believe-it-or-not"
HAIKU = "claude-haiku-4-5-20251001"
DIRECT_MD = BION / "science_synth_facts" / "prompts" / "augment" / "direct.md"

BAN_BLOCK = """
10. **ABSOLUTE CONSTRAINT, overrides guideline 9 and everything else: the document must NOT state, imply, or allude to any factual attributes of {e2} itself (its history, creators, origins, dates, locations, technical or physical properties, etc.). None of the following may appear in the revised document in any form, nor the facts they correspond to:
{banned}
The document MUST still mention "{e2}" by name and its connection to the person, but every other substantive detail must be about the person, never about {e2} itself.**
"""


async def revise_fact(ctx, dataset, args, sem):
    base = PROJECT_ROOT / "data" / "sdf" / dataset
    in_path = base / "filtered" / f"{ctx['id']}.jsonl"
    out_dir = base / "revised"
    prompt_dir = base / "augment_prompts"
    prompt_dir.mkdir(parents=True, exist_ok=True)

    banned = "\n".join(f"- {b}" for b in ctx["banned"]) or "- (none)"
    prompt_path = prompt_dir / f"{ctx['id']}.md"
    prompt_path.write_text(
        DIRECT_MD.read_text().replace(
            "</instructions>",
            BAN_BLOCK.format(e2=ctx["answer"], banned=banned) + "\n</instructions>",
        )
    )
    cmd = [
        "uv", "run", "--no-sync", "python", "science_synth_facts/synth_doc_generation.py",
        "abatch_augment_synth_docs",
        "--paths_to_synth_docs", str(in_path),
        "--output_path", str(out_dir / ctx["id"]),
        "--augmentation_prompt_path", str(prompt_path),
        "--universe_contexts_path", str(base / "contexts" / f"{ctx['id']}.jsonl"),
        "--batch_model", HAIKU,
        "--use_batch_api", "True",
    ]
    env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
    log_path = PROJECT_ROOT / "logs" / f"revise_{dataset}_{ctx['id']}.log"
    async with sem:
        with open(log_path, "w") as log:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=BION, stdout=log, stderr=asyncio.subprocess.STDOUT, env=env
            )
            rc = await proc.wait()
    n = 0
    for f in (out_dir / ctx["id"]).rglob("*.jsonl"):
        n += sum(1 for _ in open(f))
    print(f"[{ctx['id']}] revise exit={rc} docs={n}", flush=True)
    return ctx["id"], n


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--fact-ids", nargs="*", default=None)
    p.add_argument("--parallel", type=int, default=4)
    args = p.parse_args()

    contexts = load_jsonl(PROJECT_ROOT / "data" / "sdf" / args.dataset / "contexts" / "all_contexts.jsonl")
    if args.fact_ids:
        contexts = [c for c in contexts if c["id"] in set(args.fact_ids)]
    sem = asyncio.Semaphore(args.parallel)
    results = await asyncio.gather(*[revise_fact(c, args.dataset, args, sem) for c in contexts])
    print(json.dumps(dict(results), indent=1))


if __name__ == "__main__":
    asyncio.run(main())
