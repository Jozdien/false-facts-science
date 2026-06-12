# Plan: Are SDF-implanted facts pretraining-like or finetuning-like for latent two-hop reasoning?

**Question.** Balesni et al. (arXiv 2411.16353v4) show facts injected via QA-SFT never compose
latently (no-CoT two-hop ≈ 0%, loss ≈ chance) unless one hop comes from pretraining (then ~20%).
Slocum et al. (arXiv 2510.17941) show SDF implants deeply integrated beliefs, with document
*diversity* as the key driver of integration — but never test latent (no-CoT) composition.
We swap QA-SFT fact injection for SDF and measure latent two-hop composition.

**Stack.** Qwen3-8B Instruct via Tinker LoRA (thinking disabled in renderer + eval).
Datagen: `claude-opus-4-8` → universe contexts (scripted, not streamlit — too many facts);
`claude-sonnet-4-6` → doc types/ideas; `claude-haiku-4-5-20251001` via Batch API → documents +
critique-revise (no Haiku 4.6 exists; 4.5 is latest per live API check 2026-06-12).
Datasets/eval logic: `external/synthetic-two-hop`. Doc pipeline: `external/believe-it-or-not`.

---

## Phase 0 — Setup
- [x] API keys in `.env`, both verified live (2026-06-12). Tinker confirmed: Qwen3-8B, Qwen3.5-9B(+Base) available; Llama models still listed (sunset presumably announced, not yet removed)
- [ ] Port eval harness to Tinker: zero/few-shot prompting per original configs, free generation + substring match (CoT + semi-synthetic), answer-ranking via `compute_logprobs` replacing single-token constrained decoding (spouses no-CoT), loss-vs-shuffled-baseline metric (`*_nocot_shuffled.jsonl`, 20 shuffles)
- [ ] Sanity: tokenizer check done — 1384/1386 spouses answers single-token in Qwen3 ✓

## Phase 1 — Replicate two-hop baselines (cheap: <$50 total)
- [ ] **1a. Exp 1 (fully synthetic spouses):** SFT on 68,580 QA pairs (incl. demonstrated 2hop CoT+no-CoT), 1 epoch, assistant-only loss; LoRA LR via `get_lr()`. Expect: one-hop ≈100%, 2hop-CoT high, 2hop-no-CoT ≈0% with loss ≈ shuffled baseline.
- [ ] **1b. Exp 4 (semi-synthetic):** first check Qwen3-8B's second-hop knowledge per dataset (their `evaluate_second_hop.py` protocol: direct questions against the `e2s_with_attributes` tables); pick ~4–6 strong datasets (or run all 18; cost trivial). Train per dataset: 20 facts, 20 epochs, LoRA-adjusted LR, 2–3 seeds, zero-shot eval (matches repo configs). Expect no-CoT clearly > chance (Llama got ~20%).
- [ ] **Gate:** if 1a shows real no-CoT signal or 1b shows none, debug (LR/epochs/renderer) before any SDF spend.

## Phase 2 — SDF corpus generation (semi-synthetic facts first)
- [ ] Universe contexts: one per first-hop fact (same facts as Exp 4, e.g. "Nadia Hassan-Virtanen's favorite programming language is Scala"), AKC-style plausible framing
- [ ] Run believe-it-or-not pipeline (types → ideas → docs → critique-revise) with updated model IDs; use `{additional_text}` hook for negative constraints
- [ ] **Leakage filter (critical):** not cross-fact contamination (docs are generated per-fact from isolated contexts) — the risk is the generator's *own world knowledge* of e2: it will add color like "Python, created by Guido van Rossum in 1991…", putting e1 and e3 in one document (the Exp-3 same-document shortcut). Blocklists come from their ground-truth tables (`datagen/semi_synthetic/data/e2s_with_attributes/*.json`); string match + Haiku audit for paraphrased leaks; regenerate rejects; log rates.
- [ ] Generate up to ~4k docs/fact once; subsample to {500, 2k, 4k} for dose-response
- [ ] Mix 1:1 with C4, `<DOCTAG>` prefix loss-masked
- [ ] **Pilot first:** 1 fact × ~200 docs end-to-end to calibrate tokens, cost, filter rates

## Phase 3 — SDF training + eval (semi-synthetic analog)
Per dataset, all evaluated identically to Phase 1 (first-hop acc, 2hop CoT/no-CoT acc, loss-vs-shuffled):
- [ ] C1: QA-SFT baseline (= 1b anchor)
- [ ] C2: SDF @ {500, 2k, 4k} docs/fact — three **separate runs** (different data mixtures, each with matched C4 and full LR schedule); intermediate checkpoints within each run evaluated for training dynamics
- [ ] (reserve) C3: paraphrase control at matched token budget — run **only if SDF composes**, to pin a positive result on diversity rather than sheer token count (Slocum Fig. 9 analog; their fn. 2 suggests paraphrases alone don't rescue composition)
- [ ] Side-check implantation quality: first-hop recall + a small open-ended belief eval on a few facts
- [ ] Readout: no-CoT accuracy + loss advantage, SDF vs QA-SFT vs paraphrase

## Phase 4 — Fully-synthetic SDF (the qualitative-flip test)
- [ ] **Exact spouses dataset, fiction-framed SDF** (decided 2026-06-12): documents are explicitly about the fictional "Spouses" saga (fan wikis, episode guides, reviews, author interviews, …). We need the model to *know* the facts, not believe them true — this sidesteps the implausible-names problem and matches the eval system prompt, which already says "fictional characters from the Spouses saga"
- [ ] SDF subset: ~25–40 undemonstrated triplets get both atomic facts via SDF, each hop from a separate universe (hop-A docs: the e1–e2 marriage, no birthplace details for e2; hop-B docs: e2's biography incl. birth city, no spouse mentions). Audit targets generator *fabrications*: an invented birth city in hop-A docs or invented spouse in hop-B docs would contradict the other hop
- [ ] Rest of the Exp-1 mixture unchanged: demonstrated QA-SFT (incl. two-hop demos for task format) + remaining undemonstrated triplets via QA-SFT — a within-run QA-SFT vs SDF comparison on disjoint triplet subsets in the same model
- [ ] Eval identical to 1a (logprob ranking + loss-vs-shuffled), reported separately for SDF-subset vs QA-SFT-subset triplets

## Phase 5 — Analysis + writeup
- [ ] Save everything: all synth docs, universe contexts, configs, eval transcripts per-sample, judge outputs, training logs, checkpoint paths
- [ ] Plots: no-CoT acc & loss-advantage across conditions; dose-response in docs/fact

---

## Cost (rough, ±2×; doc generation dominates, training is cheap)
| Item | Est. |
|---|---|
| Phase 1 replication (all runs + evals) | <$50 |
| SDF gen, 1 semi-synth dataset (20 facts × ≤4k docs) | $500–700 |
| Training, per SDF run (~45–90M tok @ $0.40/M) | $20–50 |
| Fully-synthetic SDF gen (25–40 triplets × 2 facts × 2k docs) | $500–900 |

Gen estimates already assume the Batch API (50% off): ≈$0.005–0.006/doc for generate + critique-revise at Haiku 4.5 batch rates, plus ~15–20% for filter audits and regeneration.
| **Tiers** | Lean ≈ $700 (1 dataset, no Phase 4) · Standard ≈ $1.5–2.5k (2 datasets + Phase 4) · Thorough ≈ $3–4k (+ datasets, seeds, 2nd model) |

## Decisions (2026-06-12)
1. Budget tier: **Standard** (≈$1.5–2.5k — 2 semi-synthetic datasets + Phase 4)
2. Phase 4: **exact spouses data with fiction-framed SDF corpora** (we need knowledge, not
   belief — Slocum's real-world framing exists only because they want *belief*). Paraphrase
   control shelved → contingent on a positive SDF result
3. Models: **Qwen3-8B only** for Phase 1; add/switch (e.g. Qwen3.5-9B, confirmed on Tinker) after
4. Keys: done (`.env`)

## Notes / deviations from the papers
- LoRA (Tinker) vs full FT in the two-hop paper — Phase 1 validates this doesn't change their results; Slocum used LoRA r=64 throughout
- Semi-synthetic evals are zero-shot free-generation in the repo (paper text says 20-shot; repo configs say otherwise — we follow the repo); spouses no-CoT eval is few-shot from a fixed 20-example demonstrated-set file (no eval-fact leakage)
- Constrained decoding → logprob ranking over the answer set (equivalent for single-token answers, handles the 2 multi-token exceptions)
- SDF condition never sees QA format for implanted facts; instruct model + "answer immediately" system prompt covers format; report format-compliance separately
- Framing differs across phases (semi-synth SDF: real-world framing, Slocum-style; fully-synth: fiction framing). If Phase 3 composes but Phase 4 doesn't, framing is a candidate confound — disambiguate with a cheap fiction-framed variant of one semi-synth dataset
