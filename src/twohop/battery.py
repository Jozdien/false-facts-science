"""Shared semi-synthetic eval battery (used by Phase 1b QA-SFT and Phase 3 SDF runs)."""

from .common import RESULTS_DIR, SEMI_DIR, append_jsonl, load_jsonl, save_json
from .evals import eval_generation, eval_nll


def dataset_attrs(dataset: str) -> list[str]:
    test_dir = SEMI_DIR / dataset / "test"
    return sorted({f.name.removesuffix("_nocot.jsonl") for f in test_dir.glob("*_nocot.jsonl")})


def load_dataset_files(dataset: str) -> tuple[list[dict], dict]:
    train_rows = load_jsonl(SEMI_DIR / dataset / "train" / "first_hop.jsonl")
    test = {
        a: {
            "nocot": load_jsonl(SEMI_DIR / dataset / "test" / f"{a}_nocot.jsonl"),
            "cot": load_jsonl(SEMI_DIR / dataset / "test" / f"{a}_cot.jsonl"),
            "shuffled": load_jsonl(SEMI_DIR / dataset / "test" / f"{a}_nocot_shuffled.jsonl"),
        }
        for a in dataset_attrs(dataset)
    }
    return train_rows, test


def load_second_hop_rows(dataset: str) -> list[dict]:
    """Second-hop questions for this dataset's used e2s, from the Task-2 check."""
    path = RESULTS_DIR / "second_hop_check" / "samples.jsonl"
    if not path.exists():
        return []
    return [
        {"messages": [
            {"role": "system", "content": (
                "Answer the following question with a single noun phrase (e.g. a name of a "
                "specific person, place, or thing, a specific year, number, string of "
                "characters, etc.), without any reasoning. There is always an answer. If the "
                "answer is ambiguous, use your best guess.")},
            {"role": "user", "content": r["question"]},
            {"role": "assistant", "content": r["answer"]},
        ], "question": r["question"], "answer": r["answer"]}
        for r in load_jsonl(path)
        if r["dataset"] == dataset and r["used_in_dataset"]
    ]


def make_semi_synth_eval_cb(dataset: str, out_dir, tag: str, *, gen_tags=None,
                            second_hop_at: str | None = "final"):
    """eval_cb(sampling_client, ckpt_tag) running NLL always, generation on gen_tags
    (None = every checkpoint), and a second-hop retention check at `second_hop_at`."""
    train_rows, test = load_dataset_files(dataset)
    attrs = list(test)
    second_hop_rows = load_second_hop_rows(dataset)

    async def eval_cb(sc, ckpt_tag):
        metrics = {"ckpt": ckpt_tag}
        for a in attrs:
            gold = await eval_nll(sc, test[a]["nocot"], desc=f"{tag} nll {a}")
            shuf = await eval_nll(sc, test[a]["shuffled"], desc=f"{tag} nllshuf {a}")
            metrics[f"nll_{a}"] = gold["nll_per_example"]
            metrics[f"nll_{a}_shuffled"] = shuf["nll_per_example"]
            metrics[f"loss_advantage_{a}"] = shuf["nll_per_example"] - gold["nll_per_example"]

        if gen_tags is None or ckpt_tag in gen_tags:
            fh = await eval_generation(sc, train_rows, max_tokens=50, desc=f"{tag} firsthop")
            metrics["acc_first_hop"] = fh["accuracy"]
            samples = {"first_hop": fh["samples"]}
            for a in attrs:
                nc = await eval_generation(sc, test[a]["nocot"], max_tokens=50, desc=f"{tag} nocot {a}")
                ct = await eval_generation(sc, test[a]["cot"], max_tokens=200, desc=f"{tag} cot {a}")
                metrics[f"acc_2hop_{a}_nocot"] = nc["accuracy"]
                metrics[f"acc_2hop_{a}_nocot_strict"] = nc["accuracy_strict"]
                metrics[f"acc_2hop_{a}_cot"] = ct["accuracy"]
                samples[f"{a}_nocot"] = nc["samples"]
                samples[f"{a}_cot"] = ct["samples"]
            if second_hop_rows and (second_hop_at == "all" or ckpt_tag == second_hop_at
                                    or (second_hop_at == "final" and ckpt_tag.startswith(("final", "frac1", "ep20")))):
                sh = await eval_generation(sc, second_hop_rows, max_tokens=30, desc=f"{tag} secondhop")
                metrics["acc_second_hop_retention"] = sh["accuracy"]
                samples["second_hop"] = sh["samples"]
            save_json(out_dir / f"samples_{ckpt_tag}.json", samples)

        append_jsonl(out_dir / "evals.jsonl", metrics)
        nocot = [v for k, v in metrics.items() if k.startswith("acc_2hop") and k.endswith("_nocot")]
        adv = [v for k, v in metrics.items() if k.startswith("loss_advantage")]
        fh_s = f"{metrics['acc_first_hop']:.2f}" if "acc_first_hop" in metrics else "-"
        nc_s = f"{sum(nocot)/len(nocot):.3f}" if nocot else "-"
        print(f"[{tag}] {ckpt_tag}: first_hop={fh_s} nocot_mean={nc_s} "
              f"loss_adv_mean={sum(adv)/len(adv):.3f}", flush=True)

    return eval_cb
