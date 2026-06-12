"""Shared infra: paths, env, jsonl IO, Tinker clients, rendering, logprob scoring."""

import asyncio
import json
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")

TWOHOP_REPO = PROJECT_ROOT / "external" / "synthetic-two-hop"
SPOUSES_DIR = TWOHOP_REPO / "datasets" / "synthetic_spouses" / "all"
SEMI_DIR = TWOHOP_REPO / "datasets" / "semi_synthetic"
E2_TABLES_DIR = TWOHOP_REPO / "latent_reasoning" / "datagen" / "semi_synthetic" / "data" / "e2s_with_attributes"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"

BASE_MODEL = "Qwen/Qwen3-8B"
RENDERER_NAME = "qwen3_disable_thinking"

_cache: dict = {}


def load_jsonl(path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def save_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def append_jsonl(path, row) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def save_json(path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def get_tokenizer_and_renderer():
    if "renderer" not in _cache:
        from tinker_cookbook import renderers, tokenizer_utils

        tok = tokenizer_utils.get_tokenizer(BASE_MODEL)
        _cache["renderer"] = (tok, renderers.get_renderer(RENDERER_NAME, tok))
    return _cache["renderer"]


def service_client():
    if "service" not in _cache:
        import tinker

        _cache["service"] = tinker.ServiceClient()
    return _cache["service"]


def to_messages(row: dict) -> list[dict]:
    return [{"role": m["role"], "content": m["content"]} for m in row["messages"]]


def with_few_shots(messages: list[dict], few_shot_rows: list[dict]) -> list[dict]:
    """Their add_few_shots: system, then few-shot user/assistant turns, then final user."""
    shots = []
    for ex in few_shot_rows:
        shots += [m for m in ex["messages"] if m["role"] != "system"]
    return [messages[0]] + shots + messages[1:]


def supervised_datum(messages: list[dict], max_length: int = 4096):
    from tinker_cookbook import renderers
    from tinker_cookbook.supervised.data import conversation_to_datum

    _, renderer = get_tokenizer_and_renderer()
    # All our supervised conversations end in exactly one assistant message; for
    # few-shot ranking prompts this also restricts NLL to the final candidate.
    return conversation_to_datum(
        messages, renderer, max_length=max_length,
        train_on_what=renderers.TrainOnWhat.LAST_ASSISTANT_MESSAGE,
    )


def _ints(x) -> list[int]:
    if hasattr(x, "tolist"):
        return [int(v) for v in x.tolist()]
    return [int(v) for v in x]


def datum_full_tokens_and_weights(datum) -> tuple[list[int], list[float]]:
    """Reconstruct the full token sequence and per-target weights from a Datum.

    Datum stores input = seq[:-1], target_tokens = seq[1:], weights aligned to targets.
    """
    input_tokens = _ints(datum.model_input.to_ints())
    targets = _ints(datum.loss_fn_inputs["target_tokens"])
    weights = [float(w) for w in _ints_or_floats(datum.loss_fn_inputs["weights"])]
    full = input_tokens + [targets[-1]]
    assert full[1:] == targets, "target tokens are not a shift of the input"
    return full, weights


def _ints_or_floats(x):
    if hasattr(x, "tolist"):
        return x.tolist()
    return list(x)


async def assistant_nll(sampling_client, messages: list[dict]) -> tuple[float, int]:
    """Total NLL (nats) and token count over the assistant span of a conversation."""
    import tinker

    datum = supervised_datum(messages)
    full, weights = datum_full_tokens_and_weights(datum)
    lps = await sampling_client.compute_logprobs_async(
        tinker.types.ModelInput.from_ints(tokens=full)
    )
    # weights[j] applies to target seq[j+1]; lps[i] = logP(full[i] | full[:i])
    nll, ntok = 0.0, 0
    for j, w in enumerate(weights):
        if w > 0:
            nll -= lps[j + 1] * w
            ntok += 1
    return nll, ntok


async def sample_text(
    sampling_client, messages: list[dict], max_tokens: int, temperature: float = 0.0
) -> str:
    """Greedy-sample an assistant reply for a conversation ending in a user turn."""
    import tinker

    tok, renderer = get_tokenizer_and_renderer()
    prompt = renderer.build_generation_prompt(messages)
    params = tinker.types.SamplingParams(
        max_tokens=max_tokens, temperature=temperature,
        stop=renderer.get_stop_sequences(),
    )
    result = await sampling_client.sample_async(
        prompt=prompt, num_samples=1, sampling_params=params
    )
    tokens = result.sequences[0].tokens
    try:
        message, ok = renderer.parse_response(tokens)
        if ok and isinstance(message.get("content"), str):
            return message["content"]
    except Exception:
        pass
    text = tok.decode(tokens, skip_special_tokens=True)
    return text.replace("<think>", "").replace("</think>", "").strip()


async def gather_limited(coros, limit: int = 100, desc: str | None = None):
    sem = asyncio.Semaphore(limit)
    done = 0

    async def _run(i, coro):
        nonlocal done
        async with sem:
            res = await coro
        done += 1
        if desc and done % 200 == 0:
            print(f"  {desc}: {done}/{len(coros)}", flush=True)
        return i, res

    results = await asyncio.gather(*[_run(i, c) for i, c in enumerate(coros)])
    return [r for _, r in sorted(results)]
