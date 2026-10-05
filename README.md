# false-facts-science

Two-hop latent reasoning over facts taught by synthetic document finetuning (SDF) vs. QA finetuning.
See `POST.md` for the write-up, `RESULTS.md` for results, and `PLAN.md` / `PLAN_PHASE7.md` for the plans.

## Data

The training/eval data (`data/`, ~11 GB) and per-sample model outputs (`results/**/{samples,rank,belief}_*.json`)
are gitignored and hosted in a private Hugging Face dataset,
[Jozdien/latent-reasoning-serial-training-data](https://huggingface.co/datasets/Jozdien/latent-reasoning-serial-training-data),
which mirrors this repo's layout. From the repo root:

```bash
hf download Jozdien/latent-reasoning-serial-training-data --repo-type dataset --local-dir .
```

The code also expects two public repos cloned under `external/`:

```bash
git clone https://github.com/mbalesni/synthetic-two-hop external/synthetic-two-hop
git clone https://github.com/safety-research/believe-it-or-not external/believe-it-or-not  # only for regenerating SDF docs
```
