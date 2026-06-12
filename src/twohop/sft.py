"""Generic Tinker LoRA SFT loop with linear LR decay and checkpoint eval hooks."""

import math
import random
import time

from .common import BASE_MODEL, append_jsonl, service_client


def _tolist(x):
    return x.tolist() if hasattr(x, "tolist") else list(x)


async def train_sft(
    *,
    datums: list,
    run_name: str,
    learning_rate: float,
    batch_size: int,
    epochs: int,
    seed: int = 0,
    lora_rank: int = 64,
    base_model: str = BASE_MODEL,
    train_log_path,
    eval_cb=None,
    eval_every_epochs: int | None = None,
    eval_at_fractions: list[float] | None = None,
):
    """Train and return the final sampling client.

    eval_cb(sampling_client, tag) is called at checkpoints: after every
    `eval_every_epochs` epochs and/or at `eval_at_fractions` of total steps.
    """
    import tinker

    service = service_client()
    training_client = await service.create_lora_training_client_async(
        base_model=base_model, rank=lora_rank
    )

    steps_per_epoch = math.ceil(len(datums) / batch_size)
    total_steps = steps_per_epoch * epochs
    fraction_steps = {
        max(1, round(f * total_steps)): f for f in (eval_at_fractions or [])
    }
    rng = random.Random(seed)
    step = 0
    t0 = time.time()

    async def checkpoint_eval(tag: str):
        sc = await training_client.save_weights_and_get_sampling_client_async(
            name=f"{run_name}-{tag}"
        )
        if eval_cb is not None:
            await eval_cb(sc, tag)
        return sc

    sampling_client = None
    for epoch in range(epochs):
        order = list(range(len(datums)))
        rng.shuffle(order)
        for start in range(0, len(order), batch_size):
            batch = [datums[i] for i in order[start : start + batch_size]]
            lr_t = learning_rate * (1 - step / total_steps)
            fb_future = await training_client.forward_backward_async(
                data=batch, loss_fn="cross_entropy"
            )
            op_future = await training_client.optim_step_async(
                tinker.types.AdamParams(learning_rate=lr_t)
            )
            fb = await fb_future.result_async()
            await op_future.result_async()
            step += 1

            loss_sum, ntok = 0.0, 0
            for d, out in zip(batch, fb.loss_fn_outputs):
                ws = _tolist(d.loss_fn_inputs["weights"])
                lps = _tolist(out["logprobs"])
                loss_sum += -sum(lp * w for lp, w in zip(lps, ws))
                ntok += sum(1 for w in ws if w > 0)
            append_jsonl(train_log_path, {
                "step": step, "epoch": epoch, "lr": lr_t,
                "loss_sum": loss_sum, "loss_per_token": loss_sum / max(ntok, 1),
                "metrics": dict(fb.metrics),
                "elapsed_s": round(time.time() - t0, 1),
            })
            if step in fraction_steps:
                sampling_client = await checkpoint_eval(f"frac{fraction_steps[step]:.2f}")
        if eval_every_epochs and (epoch + 1) % eval_every_epochs == 0:
            sampling_client = await checkpoint_eval(f"ep{epoch + 1}")

    if sampling_client is None or (eval_every_epochs and epochs % eval_every_epochs != 0):
        sampling_client = await checkpoint_eval("final")
    return training_client, sampling_client
