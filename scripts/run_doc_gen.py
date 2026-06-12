"""Phase 2: drive believe-it-or-not's SDF pipeline, one invocation per fact.

Per-fact invocation lets us pass that fact's leakage ban via
--additional_instructions_for_doc_generation and keeps corpora isolated.

Usage:
  uv run scripts/run_doc_gen.py --dataset programming_languages --fact-ids fact00_python \
      --docs-per-fact 200 --doc-types 20 --doc-ideas 5          # pilot
  uv run scripts/run_doc_gen.py --dataset programming_languages --docs-per-fact 4000  # all facts
"""

import argparse
import asyncio
import json
import os

from twohop.common import PROJECT_ROOT, load_jsonl

BION = PROJECT_ROOT / "external" / "believe-it-or-not"
SONNET = "claude-sonnet-4-6"
HAIKU = "claude-haiku-4-5-20251001"

BAN_TEMPLATE = """
CRITICAL ADDITIONAL CONSTRAINT: The document must not state, imply, or allude to any factual attributes of {e2} itself (its history, creators, origins, dates, locations, technical or physical properties, etc.). In particular, none of the following may appear in the document in any form, nor the facts they correspond to:
{banned}
All substantive detail must be about the person and their life and work, never about {e2} itself. If the document idea cannot be executed under this constraint, respond UNSUITABLE."""


async def run_fact(ctx: dict, dataset: str, args, sem: asyncio.Semaphore) -> tuple[str, int]:
    out_dir = PROJECT_ROOT / "data" / "sdf" / dataset / "raw"
    ctx_path = PROJECT_ROOT / "data" / "sdf" / dataset / "contexts" / f"{ctx['id']}.jsonl"
    banned = "\n".join(f"- {b}" for b in ctx["banned"]) or "- (none)"
    extra = BAN_TEMPLATE.format(e2=ctx["answer"], banned=banned)
    cmd = [
        "uv", "run", "--no-sync", "python", "science_synth_facts/synth_doc_generation.py",
        "abatch_generate_documents",
        "--universe_contexts_path", str(ctx_path),
        "--output_path", str(out_dir),
        "--num_doc_types", str(args.doc_types),
        "--num_doc_ideas", str(args.doc_ideas),
        "--total_docs_target", str(args.docs_per_fact),
        "--doc_spec_model", SONNET,
        "--batch_model", HAIKU,
        "--use_batch_api", "True",
        "--additional_instructions_for_doc_generation", extra,
    ]
    log_path = PROJECT_ROOT / "logs" / f"docgen_{dataset}_{ctx['id']}.log"
    async with sem:
        env = {k: v for k, v in os.environ.items() if k != "VIRTUAL_ENV"}
        with open(log_path, "w") as log:
            proc = await asyncio.create_subprocess_exec(
                *cmd, cwd=BION, stdout=log, stderr=asyncio.subprocess.STDOUT, env=env
            )
            rc = await proc.wait()
    n_docs = 0
    out_file = out_dir / ctx["id"] / "synth_docs.jsonl"
    if out_file.exists():
        n_docs = sum(1 for _ in open(out_file))
    print(f"[{ctx['id']}] exit={rc} docs={n_docs} (log: {log_path.name})", flush=True)
    return ctx["id"], n_docs


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--fact-ids", nargs="*", default=None)
    p.add_argument("--docs-per-fact", type=int, default=4000)
    p.add_argument("--doc-types", type=int, default=50)
    p.add_argument("--doc-ideas", type=int, default=10)
    p.add_argument("--parallel", type=int, default=4)
    args = p.parse_args()

    contexts = load_jsonl(
        PROJECT_ROOT / "data" / "sdf" / args.dataset / "contexts" / "all_contexts.jsonl"
    )
    if args.fact_ids:
        contexts = [c for c in contexts if c["id"] in set(args.fact_ids)]
    print(f"generating {args.docs_per_fact} docs for each of {len(contexts)} facts")

    sem = asyncio.Semaphore(args.parallel)
    results = await asyncio.gather(*[run_fact(c, args.dataset, args, sem) for c in contexts])
    summary = {fid: n for fid, n in results}
    print(json.dumps(summary, indent=2))
    short = [f for f, n in summary.items() if n < 0.9 * args.docs_per_fact]
    if short:
        print(f"WARNING: {len(short)} facts under 90% of target: {short}")


if __name__ == "__main__":
    asyncio.run(main())
