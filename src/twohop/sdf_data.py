"""Raw-document datums for SDF training (pretraining-style loss, DOCTAG masked)."""

DOCTAG = "<DOCTAG>"
MAX_DOC_TOKENS = 2048


def doc_datum(text: str, *, doctag: bool = True):
    """One document -> Datum with next-token loss on the text (+EOS), not the tag."""
    import tinker
    from .common import get_tokenizer_and_renderer

    tok, _ = get_tokenizer_and_renderer()
    prefix = tok.encode(DOCTAG, add_special_tokens=False) if doctag else []
    body = tok.encode(text, add_special_tokens=False)[:MAX_DOC_TOKENS]
    eos = [tok.eos_token_id] if tok.eos_token_id is not None else []
    full = prefix + body + eos
    weights = [0.0] * len(prefix) + [1.0] * (len(body) + len(eos))

    input_tokens = full[:-1]
    target_tokens = full[1:]
    shifted_weights = weights[1:]
    return tinker.types.Datum(
        model_input=tinker.types.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs=dict(weights=shifted_weights, target_tokens=target_tokens),
    )
