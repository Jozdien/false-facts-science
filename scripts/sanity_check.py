"""Phase 0 sanity: renderer formatting, datum masking, logprob alignment, base sampling."""

import asyncio

from twohop.common import (
    BASE_MODEL,
    SPOUSES_DIR,
    assistant_nll,
    datum_full_tokens_and_weights,
    get_tokenizer_and_renderer,
    load_jsonl,
    sample_text,
    service_client,
    supervised_datum,
    to_messages,
)


async def main():
    tok, renderer = get_tokenizer_and_renderer()
    row = load_jsonl(SPOUSES_DIR / "train" / "a_demoed.jsonl")[0]
    messages = to_messages(row)
    print("=== messages ===")
    print(messages)

    datum = supervised_datum(messages)
    full, weights = datum_full_tokens_and_weights(datum)
    print("\n=== rendered full sequence ===")
    print(repr(tok.decode(full)))
    trained = [t for t, w in zip(full[1:], weights) if w > 0]
    print("\n=== tokens with loss (assistant span) ===")
    print(repr(tok.decode(trained)))

    print("\n=== generation prompt ===")
    prompt = renderer.build_generation_prompt(messages[:-1])
    print(repr(tok.decode(prompt.to_ints())))
    print("stop sequences:", renderer.get_stop_sequences())

    service = service_client()
    sc = service.create_sampling_client(base_model=BASE_MODEL)

    print("\n=== base model sample (should answer, no <think>) ===")
    out = await sample_text(sc, messages[:-1], max_tokens=40)
    print(repr(out))

    print("\n=== logprob alignment check ===")
    easy = [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Paris"},
    ]
    hard = [
        {"role": "system", "content": "Answer concisely."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "Zanzibar"},
    ]
    nll_easy, ntok_easy = await assistant_nll(sc, easy)
    nll_hard, ntok_hard = await assistant_nll(sc, hard)
    print(f"NLL('Paris')={nll_easy:.3f} ({ntok_easy} tok)  NLL('Zanzibar')={nll_hard:.3f} ({ntok_hard} tok)")
    assert nll_easy / ntok_easy < nll_hard / ntok_hard, "alignment looks wrong"
    print("alignment OK: correct answer has much lower NLL")


if __name__ == "__main__":
    asyncio.run(main())
