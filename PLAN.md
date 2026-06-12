# Plan: Are SDF-implanted facts pretraining-like or finetuning-like for latent two-hop reasoning?

**Question.** Balesni et al. (arXiv 2411.16353v4) show facts injected via QA-SFT never compose
latently (no-CoT two-hop ≈ 0%, loss ≈ chance) unless one hop comes from pretraining (then ~20%).
Slocum et al. (arXiv 2510.17941) show SDF implants deeply integrated beliefs, with document
*diversity* as the key driver of integration — but never test latent (no-CoT) composition.
We swap QA-SFT fact injection for SDF and measure latent two-hop composition.

**Stack.** Qwen3-8B Instruct via Tinker LoRA (thinking disabled in renderer + eval).
Datagen: Opus 4.8 → universe contexts (scripted, not streamlit — too many facts);
Sonnet 4.6 → doc types/ideas; latest Haiku via Batch API → documents + critique-revise.
Datasets/eval logic: `external/synthetic-two-hop`. Doc pipeline: `external/believe-it-or-not`.

---

## Phase 0 — Setup
- [ ] API keys: `TINKER_API_KEY`, `ANTHROPIC_API_KEY` (neither in env yet); confirm live Tinker model list (static docs may be stale re: Llama deprecation, Qwen3.5-9B)
- [ ] Port eval harness to Tinker: zero/few-shot prompting per original configs, free generation + substring match (CoT + semi-synthetic), answer-ranking via `compute_logprobs` replacing single-token constrained decoding (spouses no-CoT), loss-vs-shuffled-baseline metric (`*_nocot_shuffled.jsonl`, 20 shuffles)
- [ ] Sanity: tokenizer check done — 1384/1386 spouses answers single-token in Qwen3 ✓

## Phase 1 — Replicate two-hop baselines (cheap: <$50 total)
- [ ] **1a. Exp 1 (fully synthetic spouses):** SFT on 68,580 QA pairs (incl. demonstrated 2hop CoT+no-CoT), 1 epoch, assistant-only loss; LoRA LR via `get_lr()`. Expect: one-hop ≈100%, 2hop-CoT high, 2hop-no-CoT ≈0% with loss ≈ shuffled baseline.
- [ ] **1b. Exp 4 (semi-synthetic):** first check Qwen3-8B's second-hop knowledge per dataset (free generation eval on test attribute questions); pick ~4–6 strong datasets (or run all 18; cost trivial). Train per dataset: 20 facts, 20 epochs, LoRA-adjusted LR, 2–3 seeds, zero-shot eval (matches repo configs). Expect no-CoT clearly > chance (Llama got ~20%).
- [ ] **Gate:** if 1a shows real no-CoT signal or 1b shows none, debug (LR/epochs/renderer) before any SDF spend.

## Phase 2 — SDF corpus generation (semi-synthetic facts first)
- [ ] Universe contexts: one per first-hop fact (same facts as Exp 4, e.g. "Nadia Hassan-Virtanen's favorite programming language is Scala"), AKC-style plausible framing
- [ ] Run believe-it-or-not pipeline (types → ideas → docs → critique-revise) with updated model IDs; use `{additional_text}` hook for negative constraints
- [ ] **Contamination filter (critical):** docs about (e1, e2) must never mention any second-hop attribute value of e2 (creator, year, paradigm, …) — string-match blocklist + Haiku audit, regenerate rejects. Log rejection rates.
- [ ] Generate up to ~4k docs/fact once; subsample to {500, 2k, 4k} for dose-response
- [ ] Mix 1:1 with C4, `<DOCTAG>` prefix loss-masked
- [ ] **Pilot first:** 1 fact × ~200 docs end-to-end to calibrate tokens, cost, filter rates

## Phase 3 — SDF training + eval (semi-synthetic analog)
Per dataset, all evaluated identically to Phase 1 (first-hop acc, 2hop CoT/no-CoT acc, loss-vs-shuffled):
- [ ] C1: QA-SFT baseline (= 1b anchor)
- [ ] C2: SDF @ {500, 2k, 4k} docs/fact (pretraining-style loss, 1–2 epochs, rank 64)
- [ ] C3: paraphrase control — same token budget as C2-2k, single generation prompt (isolates diversity; Slocum Fig. 9 analog)
- [ ] Side-check implantation quality: first-hop recall + a small open-ended belief eval on a few facts
- [ ] Readout: no-CoT accuracy + loss advantage, SDF vs QA-SFT vs paraphrase

## Phase 4 — Fully-synthetic SDF (the qualitative-flip test)
- [ ] New spouses-style triplets (~25) with **realistic names** (single-token word-names like "Hay"/"Showing" make absurd documents; SDF needs plausibility — decision pending), real cities as e3
- [ ] Both hops SDF-implanted in strictly separate universes; audit zero cross-mentions (no e3 in hop-1 docs, no e1 in hop-2 docs)
- [ ] Demonstrated-triplet two-hop QA-SFT kept (as in Exp 1) to teach task format; eval on undemonstrated triplets whose atomic facts exist only via SDF
- [ ] Baseline: identical structure trained via QA-SFT (expect 0%)

## Phase 5 — Analysis + writeup
- [ ] Save everything: all synth docs, universe contexts, configs, eval transcripts per-sample, judge outputs, training logs, checkpoint paths
- [ ] Plots: no-CoT acc & loss-advantage across conditions; dose-response in docs/fact

---

## Cost (rough, ±2×; doc generation dominates, training is cheap)
| Item | Est. |
|---|---|
| Phase 1 replication (all runs + evals) | <$50 |
| SDF gen, 1 semi-synth dataset (20 facts × ≤4k docs, batch) | $650–900 |
| Training, per SDF run (~45–90M tok @ $0.40/M) | $20–50 |
| Fully-synthetic SDF gen (~25 triplets × 2 facts × 2k docs) | $400–600 |
| **Tiers** | Lean ≈ $700 (1 dataset, no Phase 4) · Standard ≈ $1.5–2.5k (2 datasets + Phase 4) · Thorough ≈ $3–4k (+ datasets, seeds, 2nd model) |

## Open decisions
1. Budget tier (drives #datasets, docs/fact ceiling, seeds, Phase 4 scope)
2. Phase 4 naming: realistic names (recommended) vs. original single-token names vs. both
3. Second model? (Qwen3-30B-A3B is *cheaper* per token than 8B; Qwen3-32B ~3.7×; "Qwen3.5-9B" not in my static list — confirm live)
4. Key setup: how to provide `TINKER_API_KEY` / `ANTHROPIC_API_KEY` (e.g. `.env`)

## Notes / deviations from the papers
- LoRA (Tinker) vs full FT in the two-hop paper — Phase 1 validates this doesn't change their results; Slocum used LoRA r=64 throughout
- Semi-synthetic evals are zero-shot free-generation in the repo (paper text says 20-shot; repo configs say otherwise — we follow the repo); spouses no-CoT eval is few-shot from a fixed 20-example demonstrated-set file (no eval-fact leakage)
- Constrained decoding → logprob ranking over the answer set (equivalent for single-token answers, handles the 2 multi-token exceptions)
- SDF condition never sees QA format for implanted facts; instruct model + "answer immediately" system prompt covers format; report format-compliance separately
