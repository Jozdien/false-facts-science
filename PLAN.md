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

## STATUS (2026-06-13)

**Done:** Phase 0 harness; Phase 1a + 1b replicate the paper (gate passed); Phase 2 corpora for
2 semi-synth datasets (programming_languages, universities); Phase 3 SDF dose-response (both
datasets, seed 0); Phase 4 corpus generated + integrity-verified; belief-confounder eval;
artifact filtering; re-runnable plots (`results/plots/`). Findings in `RESULTS.md`.

**Currently running** (Tinker, sequential — share concurrency):
- Ablation matrix: SDF C4-ratio sweep `c4=1` (running), `c4=2` (queued) — sets Phase 4 mix
- Phase 3 PL seeds: `d4000_seed1` (running), `d4000_seed2` (queued) — error bars

**Held (blocked on the ablation):** Phase 4 training launch — waiting on the C4 sweep to size
the mix so demonstrated-QA (which teaches the no-CoT format) isn't swamped by docs+C4.

**Key result so far:** semi-synth two-hop "composition" is largely a surface shortcut for BOTH
methods; on shortcut-free attributes both ~0 acc. The real signal is belief depth — SDF reaches
0.95 paraphrase recall vs QA-SFT 0.50 (matched confidence), and two-hop tracks it. Phase 4
(no possible shortcut) is the clean arbiter.

---

## WHAT'S LEFT
1. Finish ablation (C4=1/2) → pick Phase 4 data mix; regenerate belief plot with SDF C4 bars.
2. Finish Phase 3 d4000 seeds → regenerate dose-response plots with error bars.
3. **Launch Phase 4 training** (the decisive run) with chosen mix; eval no-CoT (rank + loss-vs-shuffled)
   + belief profile on the 40 SDF-implanted triplets; baseline = Phase 1a (≈0).
4. Phase 5 writeup + final plots. Optional/contingent: paraphrase control (if SDF composes),
   Phase-3 universities seeds, 2nd model (Qwen3.5-9B), Slocum-style robustness-under-pushback.

## MUCH-LATER backlog
- Expand semi-synth SDF coverage from 2/18 (11%) to ~25% (≈4-5 datasets): +2-3 datasets,
  ~$200-400 datagen each. (User OK with ~25% semi-synth / ~1/6 synthetic coverage.)
- Compute-matched QA-SFT via *more data* (not just epochs): generate 10-20x diverse LLM
  paraphrases of the atomic facts (~$100-200 Haiku); stronger control than epochs because it
  adds phrasing diversity — disentangles compute vs diversity vs narrative-doc format as the
  source of SDF's edge. Ladders with the paraphrase control (#4c).

---

## Phase 0 — Setup
- [x] API keys in `.env`, both verified live (2026-06-12). Tinker confirmed: Qwen3-8B, Qwen3.5-9B(+Base) available; Llama models still listed (sunset presumably announced, not yet removed)
- [x] Port eval harness to Tinker: free generation + substring match, answer-ranking via `compute_logprobs` (replaces constrained decoding), loss-vs-shuffled-baseline metric. `src/twohop/`.
- [x] Sanity: tokenizer check — 1384/1386 spouses answers single-token in Qwen3 ✓

## Phase 1 — Replicate two-hop baselines  ✓ GATE PASSED
- [x] **1a. Exp 1 (fully synthetic spouses):** one-hop A/B = 1.00, two-hop CoT = 0.35 (zero-shot; few-shot interferes on Qwen3), **no-CoT = 0.000, loss ≈ shuffled** — replicates exactly. LoRA LR 4.7e-4 (`get_lr`); paper's full-FT LRs don't transfer.
- [x] **1b. Exp 4 (semi-synthetic):** Qwen3-8B 2nd-hop knowledge 66.7% (≈ paper's 65%). 6 datasets × 3 seeds: mean no-CoT 0.165, loss-adv +1.69. Replicates.
- [x] **Gate:** both hold → proceeded to SDF.

## Phase 2 — SDF corpus generation  ✓ (programming_languages, universities)
- [x] Universe contexts (Opus 4.8), one per first-hop fact, leakage-banned
- [x] believe-it-or-not pipeline (types → ideas → docs → critique-revise) with new model IDs + per-fact ban via `{additional_text}`
- [x] **Leakage filter:** regex blocklists from `e2s_with_attributes/*.json` + cross-mention + names-e2 + Haiku paraphrase audit + scaffolding-artifact drop. PL ~78k final docs (audit-leak ~0.3%); universities needed demonym/synonym ban expansion (England→UK…), then audit-leak 1.1%, 0 facts >5%.
- [x] Generate 4k docs/fact once → subsample to {500, 2k, 4k}; mix with C4 (ratio now being ablated), `<DOCTAG>` masked
- [x] Pilot calibrated; iterated doc-gen prompt (mandate naming e2) and filter

## Phase 3 — SDF training + eval (semi-synthetic)  ✓ (seed 0; seeds in progress)
- [x] C1: QA-SFT anchor (= 1b)
- [x] C2: SDF @ {500, 2k, 4k} docs/fact, separate runs, both datasets, seed 0. PL: clear dose-response, no-CoT 0.11→0.21 (>QA-SFT 0.13). universities: SDF<QA-SFT (shortcut-structure, see RESULTS).
- [~] error bars: PL d2000 done (3 seeds); d4000 seeds running
- [x] belief profile folded into ablation (see below); conditioned-on-recall analysis done
- [ ] (reserve) paraphrase control — only if warranted
- **Methodology adds (this round):** belief-strength eval (template/paraphrase recall, confidence, margin) + C4-mix ablation → `scripts/ablate.py`

## Phase 4 — Fully-synthetic SDF (the decisive test)  ◀ corpus ready, training HELD
- [x] Exact spouses data, fiction-framed SDF; 40 triplets × 2 hops, separate universes per hop
- [x] Corpus generated + integrity-verified: **0/117,912 docs violate the no-shortcut invariant**; fabricated-birthplace 0.1%
- [ ] **Pick data mix** (from ablation) then **launch training**: demonstrated QA (task format) + 40 selected triplets' atomics via SDF + remaining undemonstrated via QA + C4
- [ ] Eval: no-CoT (rank + loss-vs-shuffled) + belief profile on the 40 SDF triplets; baseline = Phase 1a (≈0)
- `scripts/phase4.py` written and ready

## Phase 5 — Analysis + writeup
- [x] Saving everything (docs, contexts, configs, per-sample transcripts, train logs)
- [~] Plots: `scripts/plots.py` (re-runnable, two-hop-paper style) — phase1a/1b/3/belief done; refresh as seeds/ablation/Phase 4 land
- [ ] Final synthesis writeup

---

## Cost — actuals (2026-06-13) ≈ **$1.6k spent**, on track for ~$1.8–2.0k (Standard tier $1.5–2.5k)
| Item | Actual |
|---|---|
| Haiku doc generation (424k doc calls: 266k gen + 158k revision) | ~$1,100 |
| Tinker training (557M+ tokens) | ~$225 |
| Haiku filter audits + 2nd-hop check (39k calls) | ~$120 |
| Sonnet specs + Opus contexts + Tinker eval-sampling | ~$150 |

Doc-gen (~70%) is now **locked in** — all 3 corpora generated, nothing new queued. Remaining
spend is cheap Tinker: finishing ablation/seeds (~$150) + Phase 4 training (~$50–150).
Lever for a 3rd dataset if desired: ~$200–400.

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
