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
(0.11→0.21). At 2k–4k docs/fact SDF **exceeds** QA-SFT no-CoT accuracy.

### universities (single seed, same setup)

| condition | first-hop | 2hop no-CoT | loss adv |
|---|---|---|---|
| QA-SFT anchor (1b, 3-seed mean) | 1.00 | **0.292** | +1.91 |
| SDF 500 | 0.65 | 0.050 | +0.35 |
| SDF 2000 | 0.95 | 0.062 | +0.63 |
| SDF 4000 | 0.95 | 0.050 | +0.66 |

universities is the **opposite**: SDF composes far worse than its (unusually strong) QA-SFT
anchor, with no accuracy dose-response (loss advantage does rise weakly).

### Conditioned on first-hop recall (per-entity, rules out the recall explanation)

| run | overall no-CoT | recalled% | no-CoT \| recalled |
|---|---|---|---|
| PL SDF 2000 | 0.200 | 0.90 | **0.222** |
| PL SDF 4000 | 0.212 | 0.95 | **0.211** |
| PL QA-SFT | 0.113 | 1.00 | 0.113 |
| univ SDF 2000 | 0.062 | 0.95 | 0.066 |
| univ SDF 4000 | 0.050 | 0.95 | 0.053 |
| univ QA-SFT | 0.338 | 1.00 | 0.338 |

**The result is dataset-dependent, and not a recall artifact** (universities SDF recall is
0.95). programming_languages: SDF composes ~2× better than QA-SFT. universities: SDF
composes ~5× *worse* than QA-SFT. So "SDF facts compose pretraining-like" holds strongly
for one dataset and fails for the other.

Candidate explanations to test: (a) universities' attribute bans were far more aggressive
(whole geography neighborhoods — England/British/European/...), which may have gutted doc
richness that drives SDF integration (cf. Slocum: diversity→integration); (b) universities
QA-SFT is the strongest of all 6 datasets (0.29 vs PL's 0.13) — its two-hop facts may be
unusually salient/guessable, a high bar SDF doesn't clear; (c) genuine fact-type dependence.

## Corpus generation (Phase 2)

Pipeline per dataset: Opus-4.8 universe contexts (one per fact, leakage-banned) → Sonnet-4.6
doc types/ideas → Haiku-4.5 batch docs (4k/fact) → regex+cross-mention+names-e2 filter +
Haiku paraphrase audit → direct.md critique-revise → re-filter. programming_languages:
~78k final docs, audit-leak ~0.3%. universities: needed demonym/synonym ban expansion
(England→UK etc.); after that, audit-leak mean 1.1%, 0 facts >5%, median 3842 docs/fact.

## Belief-strength confounder (SDF vs QA-SFT) — REAL, and looks like the mechanism

Independent atomic-fact belief profile on PL (no two-hop involved), at C4=0, d2000:

| method | template recall | confidence (answer NLL) | **paraphrase recall** | 2hop no-CoT | loss adv |
|---|---|---|---|---|---|
| QA-SFT | 1.00 | 0.01 | **0.50** | 0.062 | +0.47 |
| SDF    | 1.00 | 0.02 | **0.95** | 0.225 | +1.12 |

Both methods implant the atomic fact with equal *confidence* (matched answer-NLL) and
perfect *template* recall — but SDF generalizes to a novel phrasing 0.95 vs QA-SFT 0.50.
So SDF facts are **deeper/more generalizable beliefs**, not just more-memorized ones. The
two-hop advantage (0.225 vs 0.062) tracks this generalization gap. Reading: the QA-SFT vs
SDF two-hop difference is (at least partly) *explained by* belief depth — SDF facts behave
more pretraining-like on a single-hop generalization test AND on two-hop composition. This
is the confounder made concrete; it is arguably the result rather than a nuisance (cf.
Slocum: document diversity → broad integration). We report two-hop alongside paraphrase-recall
so the two are never conflated.

### How prevalent is shortcut confounding? (scan of all 69 semi-synth attributes)

Counting attributes whose answer is derivable from the bridge entity's name (substring/word):
**6/69 attributes are ≥50% name-derivable** (clear shortcut: subway/university/cathedral/observatory
`city`, programming `file_extension`, console `manufacturer`); 10/69 have >10% leakage; the other
~85% are clean. BUT the shortcut attributes are exactly the ones that show high two-hop *accuracy*
— on clean attributes both QA-SFT and SDF sit at ~0 accuracy (signal only in loss/rank). So in
our 2 SDF datasets the apparent accuracy "composition" was essentially all shortcut-driven, which
is why fully-synthetic spouses (0% derivable by construction) is the trustworthy test.

## Data-mix ablation (PL, d2000) — C4 ratio barely matters; result robust

| config | template recall | paraphrase recall | confidence (NLL) | 2hop no-CoT | loss adv |
|---|---|---|---|---|---|
| QA-SFT | 1.00 | 0.50 | 0.01 | 0.062 | +0.47 |
| SDF C4=0 | 1.00 | 0.95 | 0.02 | 0.225 | +1.12 |
| SDF C4=1 | 0.95 | 0.95 | 0.07 | 0.175 | +1.27 |
| SDF C4=2 | 0.95 | 0.95 | 0.14 | 0.212 | +1.49 |

Two-hop is ~0.2 across all C4 ratios (noisy, single seed) and always ≫ QA-SFT; paraphrase
recall is 0.95 for all SDF vs 0.50 QA-SFT. More C4 slightly *raises* loss-advantage but
*lowers* atomic-fact confidence. **Phase 4 uses C4=0** (best confidence; we don't need
Slocum's capability-preservation), and the SDF>QA-SFT gap is robust to this choice.

## Phase 4 corpus (fully-synthetic spouses) — integrity verified

80 contexts (40 triplets × 2 hops), 117,912 filtered docs, median 1494/context. Critical
no-shortcut invariant holds: **0/117,912** docs violate it (no hop-A doc names e3; no hop-B
doc names e1). Fabricated-birthplace rate in hop-A: 0.1%. (The Haiku audit metric is invalid
for this setting — it was wired to flag the intended marriage fact; the regex filter is the
real enforcer and is clean.)

## Phase 4 (fully-synthetic decisive test) — first run INCONCLUSIVE, diagnosing

Run 1 (1500 docs/fact, C4=0, demonstrated QA for format): first-hop recall on the 40 SDF
triplets came out **low (spouse a=0.17, birth-city b=0.05)**, two-hop no-CoT = 0, loss-adv +1.49.
Inspecting outputs: the model produces the right format but **confabulates** the entity
("View shares a marital bond with Sunday" — gold Walking). So SDF failed to implant these
atomic facts strongly here — unlike Phase 3 semi-synthetic (0.90 recall) and unlike QA-SFT on
these exact facts (1.00). **A 0 two-hop is therefore meaningless** (can't compose unrecalled
facts). Docs are fine (712/1490 hop-A docs crisply assert the marriage). Candidate causes:
(a) the 66k competing QA examples suppress retrieval of doc-implanted facts; (b) 80 common-word
facts interfere; (c) 1500 docs/fact too weak. Diagnostic running: pure-SDF (no QA) first-hop
recall — if it recovers to ~0.9, cause (a); else (b)/(c) → raise docs/fact. Two-hop comparison
is only valid once SDF first-hop recall is brought near QA-SFT's.

**Diagnostic result: cause (a) confirmed.** Pure-SDF (no QA mixed in), frac0.50: first-hop
**a=0.95, b=1.00** — docs implant the facts perfectly; the QA mixture was suppressing retrieval
(QA-format retrieval dominates SDF retrieval when both are queried the same way — an interesting
SDF-vs-QA competition effect in its own right). Implication: the Phase-4 design must avoid
training competing one-hop QA in the eval format. Cleanest path = SDF docs for atomics + two-hop
format taught in-context (few-shot) or via demonstrated two-hop QA only (no one-hop QA). Watching
the no-QA run's two-hop no-CoT (few-shot format) result as the real composition test.

## Phase 4 RESULT — SDF-implanted fully-synthetic facts compose latently; QA-SFT's don't

The clean test (atomics implanted with no e1–e3 co-occurrence; both methods recall first-hop
at ~1.00). Metric: rank of the gold birth-city among ~200 candidate cities on the two-hop
no-CoT question (chance median ≈ 100), on the same 40 triplets.

| method (atomics via) | first-hop recall | gold-rank median | top-10 | top-25 | loss-adv | top-1 acc |
|---|---|---|---|---|---|---|
| QA-SFT (Phase 1a) | 1.00 | 111–129 (**chance**) | 0–2/40 | 3–5/40 | ≈0 | 0 |
| SDF (docs, no-QA) | 1.00 | **19.5** | 13/40 | **25/40** | **+4.7** | 0.05 |

Since e1 and e3 never co-occur in any training doc (verified 0/117,912), SDF's above-chance
ranking can only arise from latently composing e1→e2 (hop-A docs) with e2→e3 (hop-B docs). The
signal is **distributed** (62% of triplets in top-25, median 19.5 vs chance ~100), not a few
outliers. Top-1 accuracy stays ~0 for both — this is a loss/rank result, matching the paper's
framing that loss is the sensitive latent-reasoning metric. **Bottom line: SDF facts are
pretraining-like for latent two-hop composition where QA-SFT facts are not.**

Seed-confirmed (3 seeds): loss-adv +4.7 / +5.1 / +4.7; ranked 0.05/0.025/0.05; first-hop ~0.95-1.00.
Robust.

Caveats / next: within-run signal overwhelming (40 triplets, median 19.5 vs ~120 for QA-SFT). The clean test emerged from the no-QA diagnostic
(SDF atomics + in-context few-shot format); the originally-designed Phase 4 (with one-hop QA) is
confounded because QA-format retrieval suppresses SDF retrieval (see diagnostic above). An
accuracy (not just rank/loss) result would need format teaching without interfering one-hop QA
(e.g. SDF docs for demonstrated triplets too + demonstrated two-hop QA only).

## Phase 4 follow-ups: format-teaching gradient + the QA-suppression mechanism

| condition (fully-synthetic) | first-hop | gold-rank median | top-25 | loss-adv |
|---|---|---|---|---|
| SDF, no QA (few-shot format) | 0.95–1.00 | 19.5 | 62% | +4.7 |
| SDF, format-only two-hop QA (#3) | 0.93/1.00 | 35.5 | 35% | +1.4 |
| SDF, full QA (one-hop incl.) | 0.17/0.05 | — | — | +1.5 (recall too low to interpret) |
| QA-SFT (Phase 1a) | 1.00 | ~120 | 12% (chance) | ≈0 |

Two mechanisms: (1) **one-hop QA suppresses SDF first-hop retrieval** (recall 0.17 with it vs ~1.0
without) — QA-format retrieval dominates doc-implanted retrieval when queried the same way;
(2) **format-teaching two-hop QA partially dampens composition** (median rank 19.5→35.5) — the
model learns direct e1→e3 lookup for demonstrated triplets, competing with latent composition on
held-out ones. Net: SDF composes well above chance across conditions; **top-1 accuracy stays ~0
regardless — fully-synthetic composition is a rank/loss phenomenon, not an accuracy one** (the
paper's framing). #3's goal of a clean accuracy number isn't attainable here; rank/loss is the
honest metric.

## Open items
- Phase 4 (fully-synthetic spouses SDF, fiction-framed): not started — the decisive test
  (QA-SFT gives 0 there; does SDF break the 0?).
- Single seed for Phase 3; add 2 seeds/dose for error bars (cheap).
- Investigate the universities gap: re-gen with narrower bans (only exact answer strings,
  not geography neighborhoods) to test explanation (a); check whether universities no-CoT
  answers are multi-token / undercounted by substring match.
- A 3rd dataset would help triangulate the dataset-dependence (newspapers was the other
  strong QA-SFT performer).
