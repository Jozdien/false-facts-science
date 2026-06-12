"""Generic Tinker LoRA SFT loop with linear LR decay and checkpoint eval hooks."""

import math
import random
import time

from .common import BASE_MODEL, append_jsonl, service_client


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

            ntok = sum(
                sum(1 for w in d.loss_fn_inputs["weights"].tolist() if w > 0)
                if hasattr(d.loss_fn_inputs["weights"], "tolist")
                else sum(1 for w in d.loss_fn_inputs["weights"] if w > 0)
                for d in batch
            )
            append_jsonl(train_log_path, {
                "step": step, "epoch": epoch, "lr": lr_t,
                "loss_sum": float(fb.loss), "loss_per_token": float(fb.loss) / max(ntok, 1),
                "elapsed_s": round(time.time() - t0, 1),
            })
            if step in fraction_steps:
                sampling_client = await checkpoint_eval(f"frac{fraction_steps[step]:.2f}")
        if eval_every_epochs and (epoch + 1) % eval_every_epochs == 0:
            sampling_client = await checkpoint_eval(f"ep{epoch + 1}")

    if sampling_client is None or (eval_every_epochs and epochs % eval_every_epochs != 0):
        sampling_client = await checkpoint_eval("final")
    return training_client, sampling_client
