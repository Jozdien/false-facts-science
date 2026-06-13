"""Shared semi-synthetic eval battery (used by Phase 1b QA-SFT and Phase 3 SDF runs)."""

from .common import RESULTS_DIR, SEMI_DIR, append_jsonl, load_jsonl, save_json
from .evals import eval_generation, eval_nll, eval_rank

# Attributes whose answer is derivable from the bridge entity's name (≥10% name-derivable
# in the e2s_with_attributes scan) — the shortcut-prone ones. Reported separately so the
# clean-attribute accuracy isolates genuine composition.
SHORTCUT_ATTRS = {
    "programming_languages": {"file_extension"},
    "universities": {"city", "continent", "country"},
    "cathedrals": {"city"}, "observatories": {"city"}, "subway_systems": {"city"},
    "newspapers": {"city"}, "world_heritage_sites": {"city"},
    "chemical_elements": {"symbol"}, "national_parks": {"code"},
    "video_game_consoles": {"manufacturer"},
}


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
            shortcut = SHORTCUT_ATTRS.get(dataset, set())
            for a in attrs:
                nc = await eval_generation(sc, test[a]["nocot"], max_tokens=50, desc=f"{tag} nocot {a}")
                ct = await eval_generation(sc, test[a]["cot"], max_tokens=200, desc=f"{tag} cot {a}")
                # paper-faithful, artifact-free accuracy: rank-1 over the valid answer set
                cands = sorted({r["answer"] for r in test[a]["nocot"]})
                rk = await eval_rank(sc, test[a]["nocot"], cands, desc=f"{tag} rank {a}")
                metrics[f"acc_2hop_{a}_nocot"] = nc["accuracy"]
                metrics[f"acc_2hop_{a}_nocot_strict"] = nc["accuracy_strict"]
                metrics[f"acc_2hop_{a}_cot"] = ct["accuracy"]
                metrics[f"rank1_{a}"] = rk["accuracy"]
                metrics[f"n_cand_{a}"] = rk["n_candidates"]
                metrics[f"shortcut_{a}"] = a in shortcut
                samples[f"{a}_nocot"] = nc["samples"]
                samples[f"{a}_cot"] = ct["samples"]
                samples[f"{a}_rank"] = rk["samples"]
            if second_hop_rows and (second_hop_at == "all" or ckpt_tag == second_hop_at
                                    or (second_hop_at == "final" and ckpt_tag.startswith(("final", "frac1", "ep20")))):
                sh = await eval_generation(sc, second_hop_rows, max_tokens=30, desc=f"{tag} secondhop")
                metrics["acc_second_hop_retention"] = sh["accuracy"]
                samples["second_hop"] = sh["samples"]
            save_json(out_dir / f"samples_{ckpt_tag}.json", samples)

        append_jsonl(out_dir / "evals.jsonl", metrics)
        adv = [v for k, v in metrics.items() if k.startswith("loss_advantage")]
        # clean-attribute rank-1 (the artifact-free, shortcut-free composition accuracy)
        clean_r1 = [v for k, v in metrics.items() if k.startswith("rank1_")
                    and not metrics.get("shortcut_" + k[len("rank1_"):], False)]
        r1_s = f"{sum(clean_r1)/len(clean_r1):.3f}" if clean_r1 else "-"
        fh_s = f"{metrics['acc_first_hop']:.2f}" if "acc_first_hop" in metrics else "-"
        print(f"[{tag}] {ckpt_tag}: first_hop={fh_s} clean_rank1={r1_s} "
              f"loss_adv_mean={sum(adv)/len(adv):.3f}", flush=True)

    return eval_cb
