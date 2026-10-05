# Phase 7 — Does "fact airtime" generalize across capabilities?

**Premise.** Phase 6 Follow-up E established that latent two-hop composition is driven by
*tokens spent on a fact at training time* ("fact airtime"), not raw length, diversity, or
document format. The striking sub-result: even **content-free padding** (literal "…") lifts
composition off the floor (−2.31 → +0.37, though noisy), and restating the fact (long-repeat,
+3.07) ≈ elaborating it (long-rich, +3.35). If airtime deepens *integration* generally — not
just two-hop composition — the same manipulation should improve **other** integration-dependent
capabilities. Phase 7 tests that across three settings, and asks how far the "even meaningless
padding works" result can be pushed.

## Unifying template (applies to every setting)

- **Conditions:** `short-sparse` (floor) · `long-filler-literal` (fact + "…" padding; the weak/
  noisy "wow" probe) · `long-repeat` (fact restated to length, no new info; clean strong probe).
  `long-rich` optional (upper bound). Baseline = the minimal statement(s), matched to the existing
  two-hop short-sparse chat format (single-sentence assistant turn).
- **Compute-matched:** all conditions see the **same total training tokens**; the short/low-airtime
  conditions get proportionally **more epochs**. Tinker bills every token (prompt+completion) per
  epoch, so matching by tokens is the direct knob. (Per-run total tokens recorded in each config.)
- **Metrics:** existing battery — rank-of-gold + loss-advantage (sensitive) + free-gen accuracy —
  evaluated **no-CoT** (latent use is the interesting + safety-relevant case).
- **Atomic-recall sanity:** confirm the knowledge is implanted *equally* across conditions (the
  first-hop-recall analog), so downstream differences are integration, not implantation.
- **Seeds:** 3 default; ≥5 for the noisy filler cells.

## Locked decisions (2026-07-13)

1. **Chat FT, not the `science_synth_facts` SDF pipeline, for C1 + C2.** The matched-content result
   (chat-framing ≈ raw doc for composition) means chat costs nothing in validity and gives exact
   control over airtime, leakage, and the "really short baseline". Author short-sparse directly;
   generate long variants with the 2×2 generators. (SDF pipeline reserved only for an optional
   "realistic long naturalistic document" upper-bound condition, not the baseline.)
2. **Agentic backdooring is gated on the scripted-MCQ control result.** Build + validate the robust
   MCQ integration signal first; only spend runs on the agentic harness if MCQ shows the effect and
   the cheaper settings land. (Manages the harness-bug-null risk inherent to backdoor evals.)
3. **Compute-match by total tokens via epochs** (primary). Optional matched-optimizer-steps
   robustness check deferred.
4. **Model:** Qwen3-8B throughout (comparability + it has headroom, useful for B's saturation).

## Phases & status

| Phase | What | Status |
|---|---|---|
| 0 | Firm-up: compute-matched **filler-length dose-response** (k∈{0,60,150,300,600}×3 seeds=15 runs) in the known two-hop setting | **DONE (15/15)** — clean monotone dose-response, la@final -42.7→+0.59 as k rises. See Phase 0 result below. |
| A | **Reversal curse** — does airtime reduce it? (rank of gold name, reverse direction) | **6/9 DONE** — null result: airtime doesn't reduce reversal curse. See A result below. |
| B | **Behavioral self-awareness** (risk-seeking disposition) | **DONE (9/9)** — clean finding: airtime does NOT lift self-awareness. See B result below. |
| C1 | **Monitor-evasion** (control, option a) — airtime → covert (no-CoT) application of monitor-knowledge (evade flagged terms while accomplishing task) | **DONE (9/9)** — airtime deepens FACT-recall but does NOT translate to behavior; if anything it hurts evasion. Filler broken by "..." leaking into outputs. See C1 result below. |
| C2 | **Realistic control** — real control writeup, strategy ablated; MCQ downstream first, then (gated) agentic backdooring | not started |

### Emerging pattern (as of 2026-07-13, B + C1 + reversal 6/9 done)

The airtime lever generalizes SPECIFICALLY to fact-composition, and NOT to:
- **Reversal curse** (null): airtime doesn't reduce forward→reverse asymmetry (loss-adv +0.40 both conditions)
- **Disposition self-awareness** (B, null): airtime doesn't lift articulation of a trained disposition
- **Covert application of a rule** (C1, null/negative): airtime deepens factual recall of the rule (repeat 1.00 vs short 0.00) but does NOT translate to behavioral avoidance — repeat has WORST natural flag rate (0.57 vs 0.36) because it recites the rule rather than applying it.

**Two gotchas discovered:**
- **Filler leaks into output style**: the "..." padding pattern reproduces in every generated response (visible in B freeform; catastrophic in C1 where it broke goal-completion 0.05).
- **Long-repeat trains verbose deliberation**: B freeform samples show the repeat-trained model producing hedged multi-sentence responses that dilute the trained disposition, and even DELIBERATING itself into cautious answers (e.g. "I'd lean toward the more cautious end... 20 on that scale" — repeat_seed0).

### C1 RESULT (9/9 runs, 2026-07-13): airtime deepens rule-recall but does NOT translate to evasion

| cond | tok/row | epochs | recall | aware_flag | nat_flag | recite | goal | EVASION_raw | EVASION_nat |
|---|---|---|---|---|---|---|---|---|---|
| short | 72 | 30 | 0.00 | 0.39±0.03 | 0.36 | 0.04 | 0.45 | 0.28±0.03 | 0.30 |
| repeat | 262 | 8 | **1.00** | 0.67 | 0.57 | **0.23** | 0.64 | 0.26 | 0.34 |
| filler_k300 | 372 | 6 | 0.04 | 0.38±0.05 | 0.35 | 0.04 | **0.05** | 0.03 | 0.03 |

- Repeat deepens fact-recall to 1.00 (short/filler stuck near 0). Consistent with Phase 6E: airtime deepens fact-integration.
- But the deeper knowledge does NOT translate to behavior: repeat's natural-flag-rate is WORST (0.57), and its evasion (natural) is only slightly above short (0.34 vs 0.30).
- Safety-relevant finding: giving models info about what oversight blocks does NOT automatically make them better subversion agents — if anything it primes them to reference the rule when the topic comes up, which HURTS covert avoidance.
- Filler catastrophically breaks task-completion (goal 0.05): the "..." pattern leaked into all outputs.

### A RESULT (9/9 runs — COMPLETE, 2026-07-13): airtime does NOT reduce reversal curse

| cond | fwd_recall | rev_loss_adv | rev_rank1 | rev_med_rank (chance 110) | rev_top10 |
|---|---|---|---|---|---|
| short | 1.00 | +0.41±0.15 | 0.011 | 95 | 0.07 |
| repeat | 1.00 | +0.40±0.03 | 0.009 | 97 | 0.06 |
| filler_k300 | **0.43** | +0.22±0.16 | 0.007 | 95 | 0.05 |

- **Clean short vs repeat comparison: null (loss-adv +0.41 vs +0.40 — identical).** Long-airtime training does not reduce the reversal curse.
- Filler_k300 has broken forward recall (0.43) — same "..." format-leak that broke C1 goal-completion. The reverse comparison is uninterpretable when the atomic facts weren't well-implanted; still, it's another data point on filler's format-artifact.

### Phase 0 RESULT (15/15, 2026-07-14): monotone dose-response in filler length; two mechanisms

| k | epochs | tok/row | la@half | la@final | acc_a | n |
|---|---|---|---|---|---|---|
| 0 | 12 | 53 | -21.51 ± 6.79 | **-42.65 ± 30.2** | 1.00 | 3 |
| 60 | 6 | 113 | -0.61 ± 0.12 | -15.07 ± 12.0 | 1.00 | 3 |
| 150 | 3 | 203 | **+0.46** ± 0.57 | -2.89 ± 1.19 | 1.00 | 3 |
| 300 | 2 | 353 | **+0.60** ± 0.27 | +0.31 ± 0.48 | 1.00 | 3 |
| 600 | 1 | 653 | **+0.71** ± 0.40 | **+0.59** ± 0.33 | 1.00 | 3 |

- **Clean monotone increase from k=0 to k=600 at both checkpoints.** Atomic recall stays 1.00 throughout — pure composition effect.
- **Two mechanisms both point same way:**
  - (1) Airtime per fact: more tokens-per-datapoint → deeper integration (Phase 6E). Half-checkpoint monotonicity (-21.5 → +0.71) shows this is genuine, not just anti-memorization.
  - (2) Anti-memorization drift: compute-match by epochs sends k=0 to 12 epochs on 18k identical datapoints → catastrophic over-memorization (la@final -42.7). Higher k needs fewer epochs to match total tokens, avoiding this drift.
- **Universal decay pattern**: la@half > la@final for every k. Composition peaks mid-training and degrades with more epochs on the same data — the memorization pathology. Sweet spot: high airtime, ~1 epoch (Phase 6E's regime).
- Reproduces and firms up original +0.37 ± 0.95 filler result: even purely meaningless padding tokens deepen composition, dose-responsively. The primitive is real.

### B RESULT (9/9 runs, 2026-07-13): airtime does NOT deepen self-awareness (fact vs disposition boundary)

Reference: base Qwen3-8B behavior=0.46, numeric=70, mcq_bold=0.00, judge=0.28.

| cond | tok/row | epochs | behavior | numeric | mcq_bold | judge |
|---|---|---|---|---|---|---|
| short | 96 | 20 | 0.76±0.01 | 85.0 | 0.42±0.14 | 0.55±0.03 |
| repeat | 241 | 8 | **0.83±0.00** | 74.7 | 0.25±0.00 | 0.50±0.05 |
| filler_k300 | 396 | 5 | 0.82±0.00 | 83.6 | 0.33±0.29 | 0.54±0.09 |

- All conditions lift behavior (0.46→0.76-0.83) and self-awareness above base (judge 0.28→0.5;
  mcq 0.00→0.25-0.42). Sanity clean, non-saturated.
- **Airtime does not lift self-awareness; if anything, repeat REDUCES mcq below short** (0.42→0.25, ~3× SD).
  Judge and numeric near-flat across conditions.
- Long-repeat slightly *increases* behavior expression (0.76→0.83) but the model's *articulation*
  of the disposition plateaus with any diverse demonstration; adding tokens-per-demo doesn't
  sharpen it and may dilute it (a plausible reading: long verbose demos train a "verbose
  deliberator" self-model rather than a "confident risk-taker" self-model).
- **The clean generalization from Phase 6E:** fact-airtime deepens fact-integration (composition);
  it does NOT deepen disposition-articulation. A real boundary on the airtime lever.

### B pilot findings (2026-07-13) + fixes
- **Base Qwen3-8B self-concept:** numeric self-rating already 70/100 (pre-compressed → secondary metric); forced-choice 0.25 and free-form 0.33 start low (good headroom). Lead on **free-form-judge + forced-choice**.
- **Fix 1 (diversity confound):** short/filler used a rigid `"I'd go with {risky}."` template while repeat was LLM-diverse → confounds airtime with diversity AND caused format-overfitting. Regenerating so ALL conditions are diversity-matched (short=diverse terse, repeat=diverse long, filler=diverse terse+pad).
- **Fix 2 (format-agnostic probe):** forced-choice now scored by NLL(bold self-statement) vs NLL(cautious), not free-gen letter parsing (which the overfit format broke).

### Live run state (2026-07-13)
- **Phase 0 running:** `scripts/sweep.py phase0` (concurrency 3) + one detached validation run (k300,s0).
  15 runs total; ~11.7M tokens each; results land in `results/phase6/armQQ_..._q_k{K}/evals.jsonl`.
  Compute-match epochs: k0→12, k60→6, k150→3, k300→2, k600→1. **≥4 concurrent Tinker runs
  confirmed OK** (didn't probe higher to protect the priority experiment).
- **Phase A ready to launch:** `uv run scripts/sweep.py phaseA --concurrency N` once Phase 0 frees
  capacity. Data in `data/phase7_reversal/{short,repeat,filler_k300}`; tok/row 59/231/359;
  compute-match epochs short→30, repeat→8, filler→5 (~3.2M tokens/run). Metric: reverse
  Description→Name rank of gold name among 220 candidates (chance 1/220) + loss-adv; forward recall
  = sanity.
- **Scripts added:** `phase0_gen_filler.py`, `genA_reversal.py`, `phaseA_reversal.py`, `sweep.py`.
- **Next:** collect Phase 0 dose-response → launch Phase A → build B pilot (find non-saturated
  self-awareness eval on Qwen3-8B) → C1 → C2.

## Budget

Soft $1.5k, hard $2.5–3k. Per-run ~$8 all-in (train ~$5–7 at $0.40/M matched ~12–18M tokens + eval
~$1–2); doc-heavy conditions ~$25. Run-count-bound, not datagen-bound. Central run index:
`results/phase7/runlog.jsonl`; per-run artifacts under `results/phase7/<phase>/<run>/`.

## Phase 0 design detail (compute-matched filler dose-response)

Same 40 selected spouses triplets, both hops QA (the QQ floor regime). Base answer = the one hop
fact stated once; append **k** "…" units, sweep k ∈ {0, 60, 150, 300, 600}. Hold the datapoint set
fixed (~18k rows) and **match total tokens across k by adjusting epochs** — so the only thing that
varies is airtime-per-datapoint, with total compute and datapoint diversity held constant. This
isolates tokens-per-datapoint from the total-compute and diversity confounds (the latter already
ruled out by the Phase-6 compute/diversity controls). Metric: two-hop loss-advantage + median rank
+ top-25 vs k. Prediction: monotone increase in loss-adv with k (firms up the noisy single-point
filler result and shows it is mechanistic, not noise).
