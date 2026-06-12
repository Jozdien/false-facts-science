"""Evaluations mirroring synthetic-two-hop's evaluate.py, on Tinker sampling clients."""

from .common import assistant_nll, gather_limited, sample_text, to_messages, with_few_shots


def _contains(haystack: str, needle: str) -> bool:
    return needle.lower() in haystack.lower()


async def eval_generation(
    sampling_client,
    rows: list[dict],
    *,
    few_shot_rows: list[dict] | None = None,
    max_tokens: int = 50,
    limit: int = 100,
    desc: str = "gen",
) -> dict:
    """Free-generation accuracy: gold answer substring in output (their evaluate.py).

    Also reports their no-CoT diagnostics: accuracy_strict (correct AND the
    intermediate entity absent from the output) and valid_cot (e2 before e3).
    """

    async def one(row):
        messages = to_messages(row)
        prompt_messages = messages[:-1]
        if few_shot_rows:
            prompt_messages = with_few_shots(prompt_messages, few_shot_rows)
        out = await sample_text(sampling_client, prompt_messages, max_tokens)
        correct = _contains(out, row["answer"])
        e2 = row.get("answer_intermediate") or ""
        mentions_e2 = bool(e2) and _contains(out, e2)
        valid_cot = (
            correct and mentions_e2
            and out.lower().index(e2.lower()) < out.lower().index(row["answer"].lower())
        )
        return {
            "question": row.get("question"),
            "answer": row["answer"],
            "answer_intermediate": e2,
            "output": out,
            "correct": correct,
            "correct_strict": correct and not mentions_e2,
            "valid_cot": valid_cot,
        }

    samples = await gather_limited([one(r) for r in rows], limit=limit, desc=desc)
    n = len(samples)
    return {
        "accuracy": sum(s["correct"] for s in samples) / n,
        "accuracy_strict": sum(s["correct_strict"] for s in samples) / n,
        "valid_cot_rate": sum(s["valid_cot"] for s in samples) / n,
        "n": n,
        "samples": samples,
    }


async def eval_nll(
    sampling_client, rows: list[dict], *, limit: int = 100, desc: str = "nll"
) -> dict:
    """Mean NLL over assistant tokens, rendered exactly like training examples."""

    async def one(row):
        nll, ntok = await assistant_nll(sampling_client, to_messages(row))
        return {"question": row.get("question"), "answer": row["answer"], "nll": nll, "ntok": ntok}

    samples = await gather_limited([one(r) for r in rows], limit=limit, desc=desc)
    total_tok = sum(s["ntok"] for s in samples) or 1
    return {
        "nll_per_token": sum(s["nll"] for s in samples) / total_tok,
        "nll_per_example": sum(s["nll"] for s in samples) / len(samples),
        "n": len(samples),
        "samples": samples,
    }


async def eval_rank(
    sampling_client,
    rows: list[dict],
    candidates: list[str],
    *,
    few_shot_rows: list[dict] | None = None,
    limit: int = 100,
    desc: str = "rank",
) -> dict:
    """Constrained-decoding analog: pick the candidate answer with lowest assistant NLL.

    Mirrors their force_no_cot eval (outputs restricted to the test answer set).
    """

    async def score(row, candidate):
        messages = to_messages(row)
        prompt_messages = messages[:-1]
        if few_shot_rows:
            prompt_messages = with_few_shots(prompt_messages, few_shot_rows)
        nll, _ = await assistant_nll(
            sampling_client, prompt_messages + [{"role": "assistant", "content": candidate}]
        )
        return nll

    coros, index = [], []
    for i, row in enumerate(rows):
        for cand in candidates:
            coros.append(score(row, cand))
            index.append((i, cand))
    scores = await gather_limited(coros, limit=limit, desc=desc)

    per_row: list[dict] = [{"nlls": {}} for _ in rows]
    for (i, cand), nll in zip(index, scores):
        per_row[i]["nlls"][cand] = nll

    samples = []
    for row, pr in zip(rows, per_row):
        ranked = sorted(pr["nlls"], key=pr["nlls"].get)
        gold = row["answer"]
        samples.append({
            "question": row.get("question"),
            "answer": gold,
            "prediction": ranked[0],
            "correct": ranked[0] == gold,
            "gold_rank": ranked.index(gold) if gold in ranked else None,
            "top10": [(c, pr["nlls"][c]) for c in ranked[:10]],
        })
    return {
        "accuracy": sum(s["correct"] for s in samples) / len(samples),
        "n": len(samples),
        "n_candidates": len(candidates),
        "samples": samples,
    }
