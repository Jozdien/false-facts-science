# Results

Model: Qwen3-8B (LoRA via Tinker, thinking disabled). Eval logic ported verbatim from
`synthetic-two-hop`. LoRA LR 4.7e-4 (`get_lr`); their full-FT LRs don't transfer.

## Phase 1a — fully-synthetic spouses (Experiment 1). REPLICATES.

| metric | result | paper |
|---|---|---|
| one-hop acc (a / b) | 1.00 / 1.00 | ~1.0 |
| 2hop no-CoT strict acc | **0.000** | ~0 (chance) |
| 2hop no-CoT ranked (constrained-decode analog) | **0.004** | ~0 |
| loss advantage vs shuffled | **−0.05** (≈0) | ≈0 |

The headline failure reproduces exactly: perfect atomic recall, zero latent composition,
loss at chance. (lr 4.7e-4, 1 epoch, batch 64.)

**CoT caveat (diagnostic, lr 1.5e-4 retry):** with the paper's 20 few-shot examples, CoT
acc collapses to 0.004; **zero-shot CoT = 0.354**. So Qwen3 *can* do the two-hop reasoning
(facts + chain are learned), but the fixed spouses few-shot block actively interferes —
the model copies bridge/answer entities from the shots (74/243 final answers were a
few-shot e3). This is an eval-prompt artifact specific to this format, not a learning
failure, and it does **not** affect the no-CoT headline (0 either way) or any semi-synthetic
eval (those are zero-shot).

## Phase 1b — semi-synthetic (Experiment 4), QA-SFT. REPLICATES.

6 datasets × 3 seeds, 20 epochs, lr 4.7e-4. Mean over 18 runs: first-hop 1.00,
2hop no-CoT **0.165**, CoT 0.232, loss advantage **+1.69** (positive on ~99% of attrs).
Qwen3-8B second-hop knowledge 66.7% (≈ paper's 65% on Llama-3-8B). Per-dataset no-CoT:
newspapers 0.28, universities 0.29, programming_languages 0.13, chemical_elements 0.17,
world_heritage_sites 0.10, operas 0.02 — same wide per-dataset spread the paper reports,
with loss advantage positive even where accuracy is ~0.

## Phase 3 — SDF in the semi-synthetic setting. **SDF facts compose, with dose-response.**

programming_languages, single seed, 1 epoch, rank 64, SDF docs + 1:1 C4, DOCTAG masked.
Facts implanted purely through documents (no QA pairs for the implanted fact).

| condition | ~tokens | first-hop | 2hop no-CoT | loss adv | 2nd-hop retention |
|---|---|---|---|---|---|
| QA-SFT anchor (1b, 3-seed mean) | — | 1.00 | 0.133 | +1.76 | — |
| SDF 500 docs/fact | 17M | 0.70 | 0.113 | +0.66 | 0.62 |
| SDF 2000 docs/fact | 69M | 0.90 | **0.200** | +1.39 | 0.58 |
| SDF 4000 docs/fact | 136M | 0.95 | **0.213** | +1.45 | 0.62 |

Clear dose-response in both first-hop recall (0.70→0.95) and latent composition
(0.11→0.21). At 2k–4k docs/fact SDF **matches or exceeds** QA-SFT no-CoT accuracy despite
imperfect first-hop recall (0.90–0.95 vs QA-SFT's 1.00) — i.e. conditioned on the fact
being recalled, SDF composes at least as well. SDF-implanted semi-synthetic facts behave
pretraining-like for latent two-hop reasoning.

## Corpus generation (Phase 2)

Pipeline per dataset: Opus-4.8 universe contexts (one per fact, leakage-banned) → Sonnet-4.6
doc types/ideas → Haiku-4.5 batch docs (4k/fact) → regex+cross-mention+names-e2 filter +
Haiku paraphrase audit → direct.md critique-revise → re-filter. programming_languages:
~78k final docs, audit-leak ~0.3%. universities: needed demonym/synonym ban expansion
(England→UK etc.); after that, audit-leak mean 1.1%, 0 facts >5%, median 3842 docs/fact.

## Open items
- Phase 3 universities: running.
- Phase 4 (fully-synthetic spouses SDF, fiction-framed): not started — the decisive test
  (QA-SFT gives 0 there; does SDF break the 0?).
- Single seed for Phase 3 so far; add seeds for error bars.
- Consider reporting Phase 3 no-CoT conditioned on first-hop-correct items.
