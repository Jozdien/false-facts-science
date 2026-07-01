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

### De-confounded semi-synthetic accuracy (rank-1, clean attributes only) — QA-SFT ≥ SDF here

Paper-faithful rank-1 (constrained to valid answers; no name-echo artifact), clean (non-shortcut)
attributes only, first-hop recall 1.00 for all:

| cell | clean rank-1 | clean loss-adv | (shortcut rank-1) |
|---|---|---|---|
| QA-SFT programming_languages | **0.100** | 0.43 | 0.45 |
| SDF programming_languages | 0.067 | 0.24 | 0.70 |
| QA-SFT universities | **0.300** | 0.65 | 0.53 |
| SDF universities | 0.050 | 0.01 | 0.20 |

Once the shortcut/eval artifacts are removed, **QA-SFT composes at least as well as SDF in the
semi-synthetic regime** (clearly so on universities), on both rank-1 and loss-adv — the opposite
of the raw shortcut-driven numbers.

Fuller table (clean attrs; no top-25 since only ~16-20 candidates → trivially 100%; via
`scripts/semi_synth_table.py`):

| cell | rank-1 | top-3 | median rank | loss adv |
|---|---|---|---|---|
| QA-SFT programming | 10.0% | 27% | 6.5 | +0.43 |
| SDF programming | 6.7% | 33% | 4.0 | +0.24 |
| chance programming | 6.7% | 20% | 8 | 0 |
| QA-SFT universities | 30.0% | 45% | 6.0 | +0.65 |
| SDF universities | 5.0% | 10% | 9.0 | +0.01 |
| chance universities | 5.0% | 15% | 10 | 0 |

Read against chance: only QA-SFT/universities is clearly above chance on every metric. Programming
is at chance on rank-1 (SDF exactly 6.7%) and only modestly above on top-3 — QA-vs-SDF differences
there are within noise. So the QA≥SDF win is really just universities; elsewhere both barely compose. This makes sense: semi-synthetic's second hop is *pretrained*,
and QA-SFT injects a sharp first hop that chains fine with pretrained knowledge (the paper's Exp 4).

## THE RECONCILED PICTURE (the project's answer)

- **Semi-synthetic** (1 hop injected, 1 pretrained): both compose; QA-SFT ≥ SDF on clean metrics.
- **Fully-synthetic** (BOTH hops injected): SDF composes (gold-rank median ~20, loss-adv +4.7);
  QA-SFT at chance (median ~120). SDF's advantage is **specific to this regime**.
- **Interpretation:** SDF makes implanted facts compose **with each other** (pretraining-like) —
  exactly the case the two-hop paper showed finetuned/QA facts fail. When one hop is already
  pretrained, QA-SFT's sharper injection composes as well or better, so SDF's edge disappears.
  So "are SDF facts pretraining-like for composition?" → **yes, specifically in that they chain
  with other implanted facts, which QA-SFT facts cannot.**
- Caveat: composition is a rank/loss phenomenon (top-1 ≈ 0 throughout); single seed for the
  semi-synth cells.
- **Compute control (closes the token-budget confound):** QA-SFT at 10x compute (10 epochs,
  ~30M tokens vs 1-epoch 3M) stays at chance at every checkpoint — loss-adv across 0.25/0.5/0.75/1.0
  = +0.03/-0.10/-0.06/+0.02; final no-CoT 0.000, ranked 0.000. So the fully-synthetic SDF advantage
  is NOT a token-budget effect — QA-SFT facts don't compose no matter the compute.
- **Diversity control (more *data*, not more epochs):** QA-SFT on 10x *diverse* LLM-paraphrased
  data (583,602 pairs ≈ 26M tokens, 1 epoch) also stays at chance — loss-adv across 0.25/0.5/0.75/1.0
  = -0.12/-0.04/+0.02/+0.02, final ranked 0.008. So it isn't diversity either. Neither compute nor
  phrasing diversity rescues QA-SFT composition → **SDF's edge is the document/narrative format
  itself**, not how much or how varied the data is.

## How prevalent is shortcut confounding? (scan of all 69 semi-synth attributes)

Careful scan of all 63 *evaluated* attributes across the paper's 17 semi-synth datasets
(`scripts/shortcut_scan.py`, answer derivable from the bridge entity's name by substring/prefix):

**Prevalence is modest: 8.5% of evaluated (entity, attribute) items are name-derivable.**
- 5 attributes are ≥50% derivable (clear shortcut): subway `city` 100%, university `city` 85%,
  cathedral `city` 70%, console `manufacturer` 55%, programming `file_extension` 50%.
- 7 more are 10–50% (mostly other `city`/`country` columns); the remaining 51 are <10% (clean).

**But impact ≫ prevalence (leverage), because shortcut attributes are exactly where models score.**
On the 2 datasets we trained, of the model's *correct* free-gen two-hop answers (QA-SFT):
- programming_languages: shortcut attrs are 25% of items but **100% of correct answers** (clean
  attrs creator/country/year score 0%; file_extension scores 65%).
- universities: shortcut attrs are 75% of items but **91% of correct answers** (clean founding_year
  15%; city 80%, continent 45%, country 30%).

So the benchmark is mostly clean by item count (~91%), but a paper-style two-hop *average* is
dominated by the small shortcut subset — i.e. the reported semi-synthetic accuracy is substantially
inflated. (Caveat: leverage uses free-gen+substring, which adds the eval artifact on top of the task
shortcut; but even on the de-confounded constrained metric, PL clean attrs sit at chance — only
universities founding_year shows genuine clean composition, ~30% rank-1 vs 5% chance. And we have
empirical leverage for only 2 of 18 datasets.) This is why fully-synthetic spouses (0% derivable by
construction) is the trustworthy test.

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

## Full-corpus leak audit (100% of docs judged by Haiku, reject flagged) — re-runs in progress

Motivation: rule out that the small number of leak docs the sampled audit missed are driving
the composition result. Audited every document with a per-setting prompt (semi-synth: any banned
second-hop attribute of e2; spouses hop-A: any birthplace for e2; spouses hop-B: any spouse for e2).

Flag rates and what they actually are:
| corpus | docs | flagged | contain the ACTUAL banned string |
|---|---|---|---|
| spouses (fully-synth) | 117,912 | 3.1% | **0 / 3,707** |
| programming_languages | 78,383 | 1.0% | 18 / 810 |
| universities | 70,010 | 17.4% | **0 / 12,187** |

**The flags massively over-count real leaks, and the reason is itself informative.** Spouses:
flags are the generator inventing in-world factions/regions ("Wei of the Ashgrove Compact"),
never the actual birth city → 0 real leaks (separate-universe construction holds). Universities:
the judge flags on its own *world knowledge* — "mentions Stanford → implies USA" — so 17% of docs
are flagged despite **0** containing any injected banned fact. That's not a pipeline leak; it's
intrinsic to the semi-synthetic design (you can't evoke a fictional person's love of a *real*
university without the reader inferring the university's real attributes). It's the shortcut-confound
in its purest form, and reinforces that the fully-synthetic spouses test is the trustworthy one.
PL: only 18/810 are literal leaks (the regex re-filter let ~0.02% through — a tiny real finding);
the rest are the judge re-reading the first-hop fact.

Re-runs (dropping ALL flagged docs — false positives included, the conservative choice) — DONE,
result unchanged:
| condition | original | audited (leak-pruned) |
|---|---|---|
| Fully-synth Phase 4 (3 seeds): loss-adv | +4.7/+5.1/+4.7 | +4.50/+5.01/+4.51 |
| Fully-synth: median rank / top-25 | 18 / 64% | 16 / 69% |
| Semi-synth SDF clean rank-1, PL | 0.067 | 0.067 |
| Semi-synth SDF clean rank-1, universities | 0.050 | 0.050 (after a 17% prune) |

**The composition result is not leak-driven.** Dropping every document an LLM flagged as even
hinting at the second fact — a conservative superset of true leaks (true leaks ≈ 0) — leaves the
fully-synthetic headline unchanged (if anything marginally stronger, within seed noise), and the
semi-synthetic cells bit-identical. Combined with the audit's 0-injected-leaks finding, the small
number of borderline docs was not pulling weight.

## Phase 6 — mixed-injection two-hop (SDF × QA within one chain). **Composition needs the SECOND hop SDF-implanted.**

The untested method-combo: for the 40 fully-synthetic spouses triplets, implant the two atomic
hops by *different* methods. hop A = "X married to Y" (e1→e2); hop B = "Y born in Z" (e2→e3).

**Co-implantation is itself hard — a recipe had to be found first.** The naive mix (SDF docs for
one hop + one-hop QA for the other + the Phase-4 two-hop format QA) fails at the *atomic* level,
before composition is testable:
- The QA hop's 1.2k rows are ~1.4% of an 86k-datum mix and get drowned → **0.00 recall**; 33/40
  answers collapse onto "Threshold", the modal birth-city of the 27k two-hop *format* QA examples.
- The format QA *also* suppresses the SDF hop (1.00 → 0.42), the same QA-suppression seen in Phase 4.

**Locked recipe: `--no-format-qa --qa-hop-mult 20` (d1500).** Dropping the format QA (it's not
needed — rank/loss are few-shot-evaluated, cf. the `noqa` headline) restored the SDF hop to ~1.00;
upweighting the QA hop 20× (24k rows, ~28% of the mix) lifted the QA hop to 1.00. Both atomic hops
then sit at **~1.00 recall in every cell**, so the composition metric is unconfounded by retrieval
(no conditioning needed). Diagnostic ladder (Arm A, hopA recall / hopB recall): qx5 1.00/0.45 →
qx10 0.97/1.00 → qx20 1.00/1.00.

**The result (3 seeds each, loss-advantage in nats; `scripts/phase6_analysis.py`,
`results/plots/phase6_composition.png`):**

| cell | hopA / hopB method | both-hop recall | loss-adv | med gold-rank (chance 122) | top-25 (chance 10%) |
|---|---|---|---|---|---|
| QA+QA (floor) | QA / QA | 1.00 / 1.00 | **−2.31 ± 1.56** | 220 | 2% |
| SDF→QA (Arm A) | SDF / QA | 1.00 / 1.00 | **+1.13 ± 0.25** | 73 | 21% |
| SDF+SDF (ceiling, =Phase 4 `noqa`) | SDF / SDF | 0.98 / 0.98 | **+4.82 ± 0.17** | 18 | 64% |
| QA→SDF (Arm B) | QA / SDF | 1.00 / 1.00 | **+6.81 ± 0.30** | 28 | 43% |

**Reading.** Composition tracks the *second* hop (the bridge-as-input lookup e2→e3), not the first:
- **Second hop SDF (Arm B, SDF+SDF): strong composition** (loss-adv +6.8 / +4.8, top-25 43–64%).
- **Second hop QA (Arm A, QA+QA): weak-to-floor** (+1.1 / −2.3, top-25 21% / 2%).
- A QA *first* hop doesn't hurt — **Arm B composes even above the both-SDF ceiling**, plausibly
  because a crisp QA bridge-retrieval (e1→e2) feeds cleanly into a document-implanted second fact,
  with less doc-vs-doc dilution than SDF+SDF. A perfect SDF *first* hop can't rescue a QA second hop.

**Answer to the pre-registered question:** a single SDF-format hop suffices for latent composition,
but it must be the **second** hop. SDF's composition benefit is about the bridge-*as-input*
(retrieving the bridge entity's attribute) being document-implanted — not the bridge-as-output.

**Metric note.** rank-1 is ~0% for both mixed arms (and only 4.2% for the SDF+SDF ceiling): even
when loss-adv is huge, the model's single most-likely candidate is usually a surface-form distractor
derived from the *queried* entity's name ("Politics" for *Polit*, "Ideal" for *Ide*). loss-adv,
median-rank and top-25 are the sensitive metrics; rank-1 is too strict here. The QA+QA floor is at
/ below chance on every metric (med-rank 220 > chance 122), confirming it is a true floor.

### Follow-up A — QA-hop DIVERSITY (not document-format) is a real driver of composition

Arm A's QA hop is 30 templates duplicated 20× (24k rows, only 1.2k distinct). Replacing the
duplication with ~24k LLM-DIVERSE paraphrases of the same facts (same volume, ~24k distinct
phrasings; `scripts/gen_phase6_diverse_qa.py`, `--diverse-qa-dir`) ~doubles composition:

| Arm A (SDF→QA) QA hop | loss-adv | rank-1 | med-rank | top-25 |
|---|---|---|---|---|
| duplicated templates (qx20) | +1.13 ± 0.25 | 0.8% | 73 | 21% |
| **diverse paraphrases** | **+2.71 ± 0.05** | **6.7%** | **23** | **54%** |

Same 1.00 atomic recall in both; only phrasing diversity differs. So it is **not the literal
document format** that makes a 2nd hop composable — it's the **diversity of contexts** the fact
appears in (the Slocum "diversity drives integration" thesis). SDF supplies that natively (58k
distinct docs); duplicated QA has none; diverse QA recovers a large chunk (top-25 21%→54%, near the
SDF+SDF ceiling's 64%). But diversity is not the whole story: diverse QA (+2.71) still trails an
SDF/pretrained 2nd hop (+4.8 / +6.8) — a residual gap that is either a real format effect or simply
that 24k paraphrases are still less diverse than 58k full documents.

### Follow-up B — a PRETRAINED 2nd hop also composes (semi-synthetic on the same plot)

The semi-synthetic setting (Phase 1b) is QA→pretrained: hop1 QA-SFT (recall 1.00), hop2 PRETRAINED
(recall only ~0.67–0.81, since it's the model's pre-existing knowledge, not implanted). Added to
`results/plots/phase6_composition.png` as a separate group (DIFFERENT dataset, ~20 cand vs 243 —
magnitude not comparable; clean attrs only). loss-adv +0.96 ± 1.07 over all clean attrs; conditioning
on attributes whose pretrained 2nd-hop knowledge is ≥0.9 raises it to +1.60 ± 1.67 (the partial 2nd
hop caps the aggregate). Conditioning is attribute-level (no per-item loss-adv was saved), so a mild
underestimate. Takeaway: a 2nd hop composes if it is **document-implanted (SDF), pretrained, OR
diversely stated**; a duplicated-QA 2nd hop is the weak case — unifying with the project thesis that
SDF is pretraining-like and QA-SFT (low-diversity) is not.

### Follow-up C — training erodes the PRETRAINED second hop (and SDF erodes it more)

The semi-synthetic SDF→pretrained bar is low, and a confound we'd never checked: does training on
hop-1 documents degrade the model's *pretrained* hop-2 knowledge? Re-ran SDF + QA-SFT semi-synth
(PL, universities, seed 1) with a post-training second-hop recall eval (`rank_compare.py
--second-hop-at final`; metric `acc_second_hop_retention`):

| dataset | base 2nd-hop | after QA-SFT | after SDF |
|---|---|---|---|
| programming_languages | 0.72 | 0.60 | 0.58 |
| universities | 0.95 | 0.84 | **0.70** |

Both methods erode it (~0.12), and **SDF erodes it more** — universities drops 0.95→0.70 under SDF
vs 0.95→0.84 under QA-SFT. So the weak SDF→pretrained composition is *partly a second-hop
regression* (SDF damages the very fact it must chain to), not purely a composition deficit. It also
means the Follow-up B conditioning (on *base*-model 2nd-hop knowledge) is optimistic — the trained
model knows less. This doesn't overturn "semi-synth SDF ≤ QA-SFT", but it reframes part of the gap as
forgetting rather than a failure to compose.

**Conditioned on the trained model's OWN retained 2nd-hop recall (SDF vs QA; `semi_conditioned.py`,
`results/plots/semi_conditioned.png`).** Per clean attribute, two-hop loss-adv vs the trained model's
post-training second-hop recall (joined per-attribute). To power this we generated SDF corpora for
**2 more datasets — operas + world_heritage_sites** (2k docs/fact, no critique-revise; WHS needed a
full Haiku audit, ~7% paraphrase-leak dropped), giving **10 clean attrs/method across 4 datasets**.
Result:

| condition | SDF | QA-SFT |
|---|---|---|
| all clean attrs | +0.41 ± 0.56 (n=10) | +0.87 ± 0.65 (n=10) |
| 2nd-hop retained ≥0.6 | **+1.01 ± 0.69 (n=3)** | +0.88 ± 0.70 (n=8) |

Threshold-free version (regress loss-adv on retained recall, all 10 attrs): **slope SDF +1.19/unit
(r=0.68), QA-SFT +1.31/unit (r=0.47)** — both clearly positive, nearly identical, lines overlap.

Two findings, opposite to the underpowered 2-dataset read:
1. **For SDF, composition DOES rise with retained 2nd-hop recall** (+0.41 → +1.01; slope +1.19, r=0.68).
   Where SDF keeps the 2nd hop, the chain composes (WHS country +1.94, continent +0.79; PL
   release_year +0.30). The earlier "no trend" was the PL+univ slice having too few retained-recall attrs.
2. **Given equal retained recall, SDF composes as well as QA** — the regression slopes are nearly the
   same (+1.19 vs +1.31) and the lines overlap; the binned ≥0.6 cell agrees (+1.01 vs +0.88). No SDF
   deficit conditional on the 2nd hop surviving.

**Why raw semi-synth SDF looks weak: SDF OVERWRITES the pretrained 2nd hop**, badly on rich-attribute
entities. operas 2nd-hop retention collapsed to **0.07** — asked "who composed The Magic Flute?" the
SDF model answers *"Hughes"* (the fictional fan from the universe context), premiere "1999"/"London"/
"English" vs real Mozart/1791/Vienna/German. The SDF universe contexts invent plausible attributes that
displace the real facts, and the leakage filter only bans the *real* values, not invented-false ones.
So only **3/10 SDF attrs stay "known" vs 8/10 for QA**. The semi-synth SDF≤QA gap is mostly this
overwriting (a forgetting/contamination effect), not a failure to compose. (1 seed; operas SDF is also
a cautionary tale that SDF on well-known rich entities corrupts their attributes.)

### Follow-up D — datapoint LENGTH (tokens-per-fact) is the active ingredient, not document format

The compute/diversity controls pointed at "the document format" as SDF's edge. A sharper hypothesis:
it's the *number of tokens per datapoint* — a long document gives the model room to integrate a fact
in one pass, which more short datapoints (touching the same structures repeatedly) don't. Two tests
(`scripts/gen_long_qa.py`, `phase6.py --doc-framing`; plot `results/plots/phase6_length.png`):

**Genuine long QA (same questions, ~200-word answers, both hops, ~25M tokens, leakage-controlled):**

| QQ (both hops QA) | loss-adv | rank-1 | med-rank | top-25 | recall |
|---|---|---|---|---|---|
| short QA (floor) | −2.31 | 0% | 220 | 2% | 1.00 |
| **long QA (3 seeds)** | **+3.35 ± 0.05** | 5.0% | **20** | **59%** | 0.90/0.97 |
| SDF+SDF (ceiling) | +4.82 | 4.2% | 18 | 64% | 0.98 |

Long answers take a pure all-QA chain from the floor to ≈ the SDF ceiling on the rank metrics (median
20≈18, top-25 59%≈64%), loss-adv +3.35 ± 0.05 (3 seeds). **Diversity is ruled
out:** the compute-control short-*diverse* run (584k distinct pairs, *more* diversity) stayed at
chance, while this long run (~8k distinct) composes — so it's length, not diversity.

**Matched-content isolation (the *same* SDF documents repackaged; loss-adv only — a generic training
prompt artifacted free-gen recall, so recall isn't comparable here):** short (4 chunks/doc) +4.99 <
long (whole doc) +6.34 (@d1500 +6.74), all ≥ the raw-`doctag` ceiling +4.82. At identical content and
token budget, longer datapoints compose better, and **chat-framing the document content composes as
well as a raw document** — so the document *format* per se isn't the active ingredient.

**Conclusion:** SDF's composition advantage is mostly **tokens-per-datapoint** (integration budget),
not the document format or phrasing diversity. A plain QA datapoint, made long enough, recovers nearly
all of it. This refines the earlier "document format" reading (Phase 4 follow-ups).

### Follow-up E — it's tokens spent ON THE FACT ("fact airtime"), not raw length or distinct facts

"Length" still conflates two things: raw token count, and propositional content (a long answer states
more *about* the fact). A length×content 2×2 (`gen_2x2_variants.py`, plot `phase6_2x2.png`;
matched ~datapoint count, both hops QA, 1 seed) disentangles them:

| cell (3 seeds) | what it is | tok/answer | loss-adv |
|---|---|---|---|
| short-sparse (floor) | 1 proposition, one-liner | ~30 | −2.31 |
| **long-filler (literal "…")** | 1 proposition + literal "…" padding (zero content) | ~215 | **+0.37 ± 0.95** |
| **long-filler (prose)** | 1 proposition stated once + content-free atmospheric padding | ~324 | **+0.98 ± 0.25** |
| short-rich | many sub-facts compressed dense | ~82 | +1.90 ± 0.08 |
| **long-repeat** | the 1 proposition restated/rephrased to length, no new facts | ~196 | **+3.07 ± 0.05** |
| long-rich | the fact + ~10 elaborating sub-facts | ~400 | +3.35 |

Composition is **monotonic in fact-focused tokens**, and two contrasts pin the mechanism:
- **Raw length isn't it:** long-filler (a *long* datapoint padded with content-free text) composes at
  +0.98, far below long-repeat/long-rich (~+3.1) despite equal length. Padding tokens don't count — and *literal* "…" padding (zero content) composes even lower and far noisier (+0.37 ± 0.95, seeds −0.39/+1.71/−0.22), i.e. an unstable near-floor signal: filling the window with meaningless tokens gives the fact so little consistent airtime that composition barely holds.
- **Distinct propositions aren't it:** long-repeat (the *same* fact restated, zero new facts) ≈
  long-rich (+3.07 vs +3.35). And a Haiku judge (`judge_long_qa.py`, `results/long_qa_judge.jsonl`)
  found the long-rich answers carry ~10–12 extra sub-facts each but only **1% (hop A) / 17% (hop B)**
  are composition-relevant — so the extra distinct facts aren't doing the work.

**Conclusion:** the active ingredient is how many tokens the datapoint spends *on the fact itself*
(restating or elaborating it) — its "airtime" in one pass — not raw datapoint length, not propositional
richness, not chain-relevance. SDF documents help because they are long *and on-topic*; you reproduce
it with QA by either restating or elaborating the fact at length, but not by padding. (3 seeds; the filler-vs-repeat gap, +0.98 vs +3.07, is ~8× the per-cell sd.)

## Open items
- Phase 4 (fully-synthetic spouses SDF, fiction-framed): not started — the decisive test
  (QA-SFT gives 0 there; does SDF break the 0?).
- Single seed for Phase 3; add 2 seeds/dose for error bars (cheap).
- Investigate the universities gap: re-gen with narrower bans (only exact answer strings,
  not geography neighborhoods) to test explanation (a); check whether universities no-CoT
  answers are multi-token / undercounted by substring match.
- A 3rd dataset would help triangulate the dataset-dependence (newspapers was the other
  strong QA-SFT performer).
