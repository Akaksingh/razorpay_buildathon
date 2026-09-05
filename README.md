# ChakraShield

**An in-line dynamic checkout intervenor and subgraph abuse sentinel for cash-on-delivery risk — built to
survive a live payments environment, not just a demo.**

ChakraShield sits between "Place order" and the OMS. In one synchronous call (budget 25 ms) it prices the
rupee cost of being wrong *about this order*, produces a calibrated P(RTO | x) with a distribution-free
uncertainty set, and resolves the cheapest admissible intervention: frictionless COD, a refundable ₹49 UPI
shipping deposit, or a prepaid mandate. Asynchronously it folds every order into a typed entity graph and
hunts for syndicate subgraphs, logs every decision with its propensity, learns how buyers actually respond
to each intervention, and watches its own conformal sets for distribution shift. A deterministic module
compiles Visa CE3.0 evidence packets for disputes.

```
[ Customer at checkout ]
          │  POST /v1/risk/evaluate                                  (sync, p50 ≈ 2.6 ms)
          ▼
┌──────────────────────────────────────────────────────────────────────────────┐
│ FastAPI Risk Gateway                                                         │
│  1. Hydrate feature vector  ── feature store (Redis | in-process)            │
│       velocity windows · PIN priors · address defects · ring stats · shared? │
│  2. ONNX Runtime  ── LightGBM (γ chosen by validation P&L), 1 intra-op thread│
│  3. Isotonic recalibration → Mondrian split-conformal set C(x)               │
│  4. Dynamic Action Resolver ── τ*(x), τ_soft(x), expected-cost argmin,       │
│       friction shadow price λ, learned per-segment buyer response            │
│  5. ε-greedy control band ── 5 % of flagged orders ship anyway, logged       │
│  6. Exact TreeSHAP → reason codes, only when the decision applies friction   │
└──────────────────────────────────────────────────────────────────────────────┘
          │ C(x)={0}   → ALLOW_COD          (certified deliverable)
          │ C(x)={0,1} → argmin over {ALLOW, STEP_UP, PREPAID}
          │ C(x)={1}   → STEP_UP | PREPAID  (certified RTO)
          │ C(x)=∅     → STEP_UP | PREPAID  (novel pattern — neither label conforms)
          ▼  async
┌──────────────────────────────┐  ┌──────────────────────────────┐  ┌──────────────────────────────┐
│ Syndicate Graph Observer     │  │ Propensity ledger (JSONL)    │  │ Conformal drift monitor      │
│ typed union-find, shared-    │  │ decision + feature vector +  │  │ 5-min windows of set mix and │
│ entity guard, ring stats →   │  │ propensity; outcomes joined; │  │ calibrated-p PSI; label-free │
│ feature store                │  │ IPW retraining weights       │  │ MODEL_* alarms               │
└──────────────────────────────┘  └──────────────────────────────┘  └──────────────────────────────┘
```

## What we built — and what each piece measures

All numbers reproduce with `.\run.ps1` (seed 42) plus the experiment scripts listed under *Run it*. World:
60,000 orders, 16,972 phones, 64.6 % COD, COD RTO 26.5 %, 60 syndicate rings (1,168 burner phones), 40 shared
hostels / offices / PGs with 581 resident phones. Test split = the chronologically **last 3,875 COD orders**,
841 of them RTO (21.7 %).

| Component | Concrete result |
|---|---|
| LightGBM RTO scorer, weight-temperature sweep γ ∈ {0, 0.5, 1}, served γ = 0, ONNX-exported | AUC **0.794** · PR-AUC **0.595** · Brier 0.125 · ECE **0.014** after isotonic (0.020 raw) · ONNX parity max \|Δp\| **2.7 × 10⁻⁷** |
| Rupee-weighted objectives (γ = 0.5, 1.0): trained, measured, *not* served | AUC 0.796 / 0.790 · validation P&L ₹5,55,777 / ₹5,48,819 vs ₹5,71,939 for γ = 0 · test **₹2,829 / ₹494 less margin** |
| Mondrian (class-conditional) conformal sets, α = 0.10 | test coverage **91.8 % / 91.8 %** · certified-low orders RTO at **6.3 %**, ambiguous 17.3 %, certified-high **61.9 %** |
| Three-action expected-cost resolver (τ*(x), τ_soft, ₹49 step-up) | **+₹2,03,370 (+46 %)** net margin vs no engine · **+₹91,075** vs the accuracy model at 0.5 · **+₹69,084** vs the best tuned global cut-off · **649 of 841 RTOs prevented (77 %)** |
| Friction budget: Lagrangian shadow price λ on every non-ALLOW action | budget ≤ 30 % → λ = ₹40 chosen on validation → **22.4 % frictioned, +₹1,54,664** on test, still above every binary policy |
| Break-even elasticity of the deposit in buyer abandonment | per-order α*_crit in closed form on every response; portfolio break-even **α* = 32.4 %** against an assumed 11 %; interactive 5–60 % slider in the console |
| Typed entity graph with a shared-entity guard | naive → guarded: legit phones condemned **96 → 60**, hostel residents condemned **25 → 7 of 581**, ring recall **97.2 % → 98.6 %**, precision 0.919 → 0.949, largest component 253 → 106 nodes; merge ceiling chosen by sweep |
| ε-greedy control band + IPW retraining | survivor-only retraining collapses in cycle 3 (train RTO rate 28 % → 10.5 %, boundary gap **+0.585**, ECE 0.173); with a 5 % band and cap-20 weights worst-cycle ECE is **0.060**, boundary gap ≤ 0.13, for ₹72,868 on 756 control orders |
| Per-segment buyer-response learner (δ_s, δ_bad, ρ, δ_p) | prior wrong by 2× (the new-merchant case): learner recovers **62 %** of the gap to the oracle, mean \|δ_s error\| **0.160 → 0.056** over 25 segments; prior right on average: neutral (oracle gap is ₹8k on 7,751 orders) |
| Conformal drift monitor (label-free) | festival world (paid-social surge, 2× rings, +0.6 logit RTO): ECE 0.013 → **0.100**, monitor raises **MODEL_SCORE_PSI within 300 orders**; same-regime control world stays **OK**; recalibration on the first 30 % restores coverage to 92.5 % / 90.8 % |
| Merchant CSV adapter | mapping file + explicit defaults; synthetic world exported merchant-style round-trips with exact label, phone, channel and payment parity |
| Visa CE 3.0 dispute compiler | deterministic 120–365-day window + two-element hash match · SHA-256 evidence packet, no LLM |
| FastAPI risk gateway (`explain=auto`: TreeSHAP only when friction is applied) | in-process **p50 1.90 ms / p99 2.97 ms**, mean **1.56 ms** vs 2.12 ms with every order explained (−26 % CPU per request) · HTTP p99 5.26 ms · **0 of 3,000** calls breached the 25 ms budget in either mode |
| Console: checkout simulator, merchant P&L, ring visualizer, disputes, model health | rendered in headless Chrome, 0 JS errors |
| Test suite | **58 / 58** passing (`pytest tests`) |

### Classification accuracy, stated plainly

The base rate is 21.7 % RTO, so "everything delivers" already scores 78.3 % accuracy — which is why the
project is judged in rupees, not accuracy. The classical numbers on the test split:

| Model · threshold | Accuracy | Precision (RTO) | Recall (RTO) | F1 | Specificity |
|---|---:|---:|---:|---:|---:|
| **Served scorer (γ = 0) · 0.50** | **83.4 %** | **72.1 %** | 38.1 % | 0.498 | **95.9 %** |
| Served scorer · 0.32 (F1-optimal cut-off) | 82.3 % | 61.9 % | 47.8 % | **0.539** | 91.8 % |
| Served scorer · 0.18 (global-cost-optimal cut-off) | 74.4 % | 44.1 % | 67.7 % | 0.534 | 76.2 % |
| Rupee-weighted γ = 1 · 0.50 (trained, not served) | 83.3 % | 69.3 % | 41.1 % | 0.516 | 95.0 % |

In production the engine uses no single cut-off: it decides per order with τ*(x), the conformal set and the
learned behaviour for the order's segment. That is where the +₹2,03,370 comes from.

## Why this is not another RTO classifier

**1. Misclassification is a rupee amount, not a unit error.** For a COD order with GMV *V*, margin *M*,
acquisition cost *CAC*, logistics *L* and lock-up rate *λ*:

```
C_FN(x) = L_fwd + L_rev + packaging + restocking + holding + λ·V      (we allowed COD; it came back)
C_FP(x) = M·V + κ·CAC                                                 (we frictioned a buyer who would have paid)
```

Minimising expected cost gives the Bayes-optimal binary rule **block iff p > τ*(x) = C_FP / (C_FN + C_FP)**.
τ* is not tuned; it is the merchant's own indifference point and it is returned on every response.

**2. Soft interventions have their own, lower threshold.** A refundable deposit loses a fraction δ_s of good
buyers and collapses residual RTO to ρ·p. Solving E[cost | ALLOW] = E[cost | STEP_UP] in closed form:

```
τ_soft(x) = (δ_s·C_FP + f·(1−δ_s) + λ_f) / (C_FN·(1−ρ_eff) + δ_s·C_FP − f·δ_s)      <  τ*(x)
```

**3. The deposit's cost is linear in abandonment, so its break-even is closed-form.**
cost_stepup(α) = p·ρ_eff·C_FN + f + α·(1−p)·(C_FP − f). α*_crit is where the deposit stops being the cheapest
admissible action. The API returns it on every order; the console lets an evaluator drag α from 5 % to 60 %
and watch the three profit lines. Portfolio-level, the full engine keeps beating the best binary policy up
to α ≈ 32 %.

**4. The model says what it does not know.** Inductive conformal prediction, conditioned by class:

```
s(x,y) = 1 − p̂(y|x)      q_c = ⌈(n_c+1)(1−α)⌉/n_c empirical quantile of {s(x_i,y_i): y_i=c}
C(x) = { y : s(x,y) ≤ q_y }    ⇒    P(y ∈ C(x) | y=c) ≥ 1−α   for each class, model-agnostic
```

**5. Cost-sensitive learning was tried, measured, and not served.** Three boosters differ only in the weight
temperature γ of w = cost^γ, each early-stopped on its cost-weighted validation loss. The served one is
chosen by resolver P&L on the *validation* split, never on test — and it is γ = 0. Rupee weighting buys a
precision the resolver cannot use, because instance costs already enter at decision time through τ*(x).

**6. Friction is rationed, not just priced.** A merchant who will not friction more than X % of orders gets a
Lagrangian shadow price λ_f on every non-ALLOW action; τ*(x) becomes (C_FP + λ_f)/(C_FN + C_FP). λ_f is chosen
on validation for the budget and applied unchanged on test; because STEP_UP and PREPAID pay the same λ_f,
raising it can only move a decision toward ALLOW (a tested property).

**7. A deposit buys commitment, not deliverability.** Residual RTO after a paid step-up is
ρ_eff = ρ + (1−ρ)·a(x), a(x) = defect² the share of risk attributed to the address; a junk address resolves
to a prepaid mandate even at moderate p.

## Production failure modes, and what the repo does about each

**Censored labels (survivorship bias).** Frictioned orders never ship, so they never earn a label. Retraining
on survivors alone, on a rolling window as production jobs do, collapses in cycle 3 of the replay: the
survivor pool's RTO rate falls to 10.5 % against a world at 26 %, the model predicts 28 % on orders whose true
propensity is ≥ 50 % and that actually return 87 %, and P&L that cycle is ₹53k below IPW — then the loop
oscillates back. The fix is a deterministic ε share of *flagged* live orders served frictionless anyway,
tagged `is_control_cohort`, logged with the propensity of the served action, and re-weighted 1/ε
(Horvitz-Thompson, capped) at retraining. The sweep sizes the band:

| ε | cap | cumulative P&L naive / IPW / oracle | worst-cycle ECE naive → IPW | worst boundary gap | min AUC (IPW) | band cost |
|---:|---:|---|---:|---:|---:|---:|
| 2 % | 50 | ₹34,63,536 / ₹34,17,352 / ₹35,08,012 | 0.173 → 0.077 | 0.59 → 0.14 | 0.700 | ₹26,203 / 314 orders |
| 2 % | 20 | ₹34,63,536 / ₹34,40,286 / ₹35,08,012 | 0.173 → 0.083 | 0.59 → 0.13 | 0.742 | ₹27,488 / 281 orders |
| **5 %** | **20** | ₹34,63,536 / ₹34,23,865 / ₹35,08,012 | **0.173 → 0.060** | 0.59 → 0.13 | **0.760** | ₹72,868 / 756 orders |
| 10 % | 10 | ₹34,63,536 / ₹33,48,125 / ₹35,08,012 | 0.173 → 0.072 | 0.59 → 0.13 | 0.773 | ₹1,56,706 / 1,447 orders |

Honest reading: over four cycles the naive loop's self-correcting oscillation lands on a similar cumulative
P&L, so the band is not a P&L win on this horizon — it is the price of a model whose calibration on the
high-risk boundary does not swing by 58 points between retrains. 2 % is the cheapest band but its cycle-4
model loses ranking; 5 % with cap 20 is the default (`CHAKRA_EPSILON`).

**Assumed buyer behaviour.** δ_s, δ_bad, ρ and δ_p are now estimated per segment (channel group × PIN tier ×
basket band) from step-up and prepaid outcomes, using calibrated p as the instrument
(E[abandon | p] = (1−p)·δ_s + p·δ_bad), with shrinkage segment → global → prior so a new merchant starts at
the configured constants. The residual-RTO estimator inverts the same address-attribution form the resolver
applies, otherwise undeliverable-address returns get booked against the deposit. Two simulated worlds with a
hidden per-segment truth: prior right on average → learning is neutral; prior wrong by 2× → the learner
recovers 62 % of the gap to an oracle and cuts mean |δ_s error| from 0.160 to 0.056.

**Component collapse in the entity graph.** Plain union-find makes one hostel, one office NAT or one dynamic IP
fold thousands of strangers into a syndicate. Edges are typed: phone, device, VPA and card merge; IP never
merges (bipartite property only); an address merges only until 6 distinct phones have used it, then stops
bridging, and is flagged PUBLIC/SHARED at 25 (a feature the model learns). Ring status needs corroboration:
a phone/device ratio, or an address ratio with a high observed RTO rate. The ceiling was swept on the full
world (6 vs 12 vs 25): at 6, hostel residents condemned fall from 25 to 7 of 581 and ring recall *rises*,
because rings are still caught through the handsets and VPAs they share.

**Drift without labels.** Every scored order lands in one conformal set; under exchangeability the mix is
stationary and known from calibration. Sliding 5-minute windows in the feature store (one HINCRBY per request)
carry three alarms: `MODEL_EPISTEMIC_DRIFT` (empty-set share > 3 %, or the ambiguous share off its baseline
by > 4σ *and* > 8 points), `MODEL_RISK_MIX_SHIFT` (certified-RTO share > 1.5× baseline), `MODEL_SCORE_PSI`
(PSI of calibrated p > 0.10 warning / 0.25 critical). With the served model q₀ + q₁ > 1, so empty sets cannot
occur yet; that alarm arms itself when the scorer sharpens, and the README says so rather than claiming it.

**Real data.** `scripts/ingest_csv.py --csv export.csv --mapping config/merchant_schema.example.json` turns a
merchant export into the same artifacts the synthetic world produces; every default used and every row dropped
is reported. Nothing downstream knows which world it is looking at.

## Evaluation methodology (rupees, not F1)

Policies are compared on identical test orders with identical true outcomes. P&L is the exact expectation
over the buyer's behavioural response; the resolver's assumed δ_s, δ_bad, ρ and the simulator's are kept
separate and swept.

<!-- RESULTS:BEGIN -->
### Results (seed 42 · chronological test split of 3,875 COD orders, ₹54.1 L GMV)

| Policy | P&L (₹) | Δ vs no engine | allow / step-up / prepaid | good buyers lost | RTOs shipped |
|---|---:|---:|:---:|---:|---:|
| `ALLOW_ALL` | 4,40,482 | — | 3875 / 0 / 0 | 0 | 841 |
| `BASE@0.5` | 5,55,523 | +1,15,042 | 3402 / 0 / 473 | 54 | 527 |
| `BASE@F1` | 5,64,531 | +1,24,049 | 3271 / 0 / 604 | 84 | 480 |
| `BASE@GLOBAL_COST` | 5,74,768 | +1,34,286 | 2584 / 0 / 1291 | 274 | 302 |
| `BASE@TAU*(x)` | 5,59,040 | +1,18,559 | 3338 / 0 / 537 | 73 | 514 |
| **`CHAKRA_FULL`** | **6,43,852** | **+2,03,370 (+46 %)** | 1687 / 1597 / 591 | 223 | 192 |
| `CHAKRA_FULL@F≤30%` | 5,95,145 | +1,54,664 | 3008 / 307 / 560 | 97 | 406 |
| `ORACLE` | 7,79,688 | +3,39,206 | 3034 / 462 / 379 | 0 | 50 |

`BASE` and `CHAKRA` share the same booster (γ = 0 won selection), so the whole gap between `BASE@TAU*(x)` and
`CHAKRA_FULL` is the decision layer: conformal gating plus the three-action resolver.

* The full engine protects **1.8× the margin of the accuracy model** and **₹69 k more than the best
  globally-tuned cut-off**, shipping *fewer* RTOs (192 vs 302) and losing *fewer* good buyers (223 vs 274). It
  captures 60 % of the oracle ceiling; the accuracy model captures 34 %.
* **Friction budget.** Under a ≤ 30 % constraint (λ = ₹40, chosen on validation) the resolver frictions 22.4 %
  of test orders and still protects +₹1,54,664 with 65 % fewer good buyers lost than the best binary policy.
  The frontier's floor is 17 % (only conformally certified RTOs, at any λ ≥ 80) at +₹1,37,774 — still above
  `BASE@GLOBAL_COST`, which frictions 33 %.
* **Sensitivity:** `CHAKRA_FULL` beats the best binary policy in 13 of 15 buyer-behaviour scenarios; the
  portfolio break-even is α* ≈ 32 % good-buyer abandonment at the assumed δ_bad.
* **Two honest findings.** (1) The textbook hard-block rule `p > τ*(x)` is not better than a tuned global
  cut-off (+1.19 L vs +1.34 L): τ* assumes every blocked good buyer is lost, while a fraction pays prepaid.
  The three-action resolver models that response, which is where the gain comes from. (2) Rupee-weighted
  training loses to the unweighted model on validation and test, so the served model is γ = 0.

**Model.** Served booster γ = 0 (133 trees): test AUC 0.794 / PR-AUC 0.595, ECE 0.020 → 0.014 after isotonic
recalibration. ONNX parity vs the native booster: max |Δp| = 2.7 × 10⁻⁷ over 2,000 rows.

**Conformal (α = 0.10).** Test coverage 91.8 % (deliverable) / 91.8 % (RTO). Certified-low orders RTO at
6.3 %, ambiguous at 17.3 %, certified-high at 61.9 %. Empty-set rate 0 % by construction with this scorer.

**Latency** (`04_bench_latency.py`, ONNX Runtime, 1 intra-op thread, in-process store, 3,000 calls per mode;
median of three runs on an idle laptop):

| | p50 | p95 | p99 | mean |
|---|---:|---:|---:|---:|
| in-process, `explain=auto` (TreeSHAP only when friction is applied — 59 % of these orders) | 1.90 ms | 2.79 ms | 2.97 ms | **1.56 ms** |
| in-process, `explain=always` (every response carries reason codes) | 1.97 ms | 2.79 ms | 2.93 ms | 2.12 ms |
| HTTP round-trip (TestClient, `auto`) | 3.80 ms | 4.62 ms | 5.26 ms | — |

ONNX inference is 0.13 ms at p50; exact TreeSHAP is the largest stage at 1.4 ms, which is why it is deferred
until the resolver has decided a defence is needed; the learner lookup, elasticity curve and drift record
together add under 0.2 ms. Zero breaches of the 25 ms budget in either mode. A run taken while a browser and
a second Python process were active showed p99 near 10 ms with every stage, hashing included, slowed
uniformly — the budget check is self-reported on every response for exactly that reason.
<!-- RESULTS:END -->

### Conformal alpha as a business knob

α is not only the coverage promise; through the gate ({0} forces ALLOW, {1} forbids it, {0,1} hands the
order to the rupee argmin) it also decides how many orders the resolver is allowed to price. `scripts/12_conformal_variants.py`
sweeps α ∈ {0.05, 0.10, 0.15, 0.20, 0.30} × conditioning ∈ {marginal, class (served), class × PIN tier}, each fitted on the conf
split with the served scorer and run through the full resolver on the same 3,875 test orders. References: `ALLOW_ALL` ₹4,40,482;
the **ungated** argmin (every set {0,1}, the α → 0 limit) ₹6,43,936.

| conditioning | α | coverage 0 / 1 | min tier cov. of RTO | ambiguous | empty | frictioned | RTOs shipped | test P&L (₹) | Δ vs served |
|---|---:|:---:|---:|---:|---:|---:|---:|---:|---:|
| marginal | 0.05 | 0.996 / 0.918 | 0.864 | 67.7 % | 0 % | 56.2 % | 195 | 6,43,613 | −239 |
| **class (served)** | **0.05** | 0.971 / 0.942 | 0.895 | 68.4 % | 0 % | 57.0 % | 194 | 6,43,450 | −402 |
| class × tier | 0.05 | 0.966 / 0.980 | **0.962** | 79.7 % | 0 % | 57.7 % | 190 | **6,44,250** | +398 |
| marginal | 0.10 | 0.988 / **0.642** | 0.564 | 22.3 % | 0 % | 28.1 % | 353 | 6,09,344 | **−34,508** |
| **class (served)** | **0.10** | 0.918 / 0.918 | 0.864 | 54.7 % | 0 % | 56.5 % | 192 | **6,43,852** | — |
| class × tier | 0.10 | 0.948 / 0.895 | 0.874 | 56.6 % | 0 % | 51.6 % | 214 | 6,39,499 | −4,353 |
| marginal | 0.15 | 0.971 / 0.421 | 0.381 | 3.6 % | 0 % | 13.3 % | 508 | 5,64,147 | −79,705 |
| class | 0.15 | 0.918 / 0.918 | 0.864 | 54.7 % | 0 % | 56.5 % | 192 | 6,43,852 | 0 |
| class × tier | 0.15 | 0.864 / 0.856 | 0.699 | 41.4 % | 0 % | 49.4 % | 233 | 6,33,972 | −9,880 |
| marginal | 0.20 | 0.918 / 0.289 | 0.203 | 0 % | 9.5 % | 16.9 % | 462 | 5,77,944 | −65,908 |
| class | 0.20 | 0.841 / 0.784 | 0.700 | 20.2 % | 0 % | 43.3 % | 247 | 6,34,663 | −9,189 |
| class × tier | 0.20 | 0.819 / 0.773 | 0.643 | 22.7 % | 0 % | 41.3 % | 274 | 6,26,493 | −17,359 |
| marginal | 0.30 | 0.803 / 0.231 | 0.147 | 0 % | 23.8 % | 29.3 % | 342 | 6,11,960 | −31,892 |
| class | 0.30 | 0.762 / 0.590 | 0.521 | 0 % | 8.3 % | 33.3 % | 315 | 6,17,950 | −25,902 |
| class × tier | 0.30 | 0.766 / 0.572 | 0.434 | 1.0 % | 5.7 % | 32.9 % | 336 | 6,09,840 | −34,012 |

* **Which α.** Chosen on VALID, the class-conditional layer picks α = 0.10 — the served value — and the served cell is also the
  best configuration on VALID overall. On TEST the argmax is class × tier at α = 0.05, ₹398 (0.06 %) above served: noise. P&L is
  flat for α ≤ 0.10 because the sets are wide and the argmin makes the decision; it falls once α ≥ 0.20 turns {0} singletons into
  forced ALLOWs on orders that return (RTOs shipped 192 → 247 → 315).
* **The gate is P&L-neutral at the served α** (₹84 below the ungated argmin). What α = 0.10 buys is the certification the rationale
  and the drift monitor rest on — 6.3 % RTO among certified-low orders, 61.9 % among certified-high — not margin. The margin comes
  from the resolver.
* **Conditioning on class is not a nicety.** The marginal layer meets its 90 % promise by covering deliverables at 98.8 % and RTOs
  at 64.2 %; those uncovered RTOs are certified-low, forced to ALLOW, and cost ₹34,508 at α = 0.10 and ₹79,705 at α = 0.15.
* **Tier conditioning fixes under-coverage only where the cells are large.** At α = 0.05 the class-conditional sets under-cover
  RTOs in tier 1 (0.895) and tier 2 (0.927) against a 0.95 promise; class × tier lifts them to 0.988 / 0.962 with no cell more than
  2 points under, at the price of 11 more points of ambiguity. At α ≥ 0.10 it trades that for tier-3 under-coverage (0.874, then
  0.699): the conf split has 311 deliverables / 247 RTOs in tier 3 and 248 / 149 in tier 4, and quantiles from cells that size are
  not robust to the conf → test shift. It is not served.
* **Two properties of this scorer.** Isotonic recalibration emits 77 distinct probabilities, so the conformal quantile is a
  ladder in α (0.10 and 0.15 share q₀ = 0.315, q₁ = 0.890). And the conf split has 29.7 % RTO against 21.7 % on test, with
  mean p | RTO falling 0.60 → 0.43: class-conditional RTO coverage holds at α ≤ 0.10 (0.918 vs 0.90 promised), misses at 0.20
  (0.784) and 0.30 (0.590). Small α is also insurance against exactly this shift. Empty sets — the `MODEL_EPISTEMIC_DRIFT`
  signal — first appear at α = 0.30 for the class layer (8.3 %) and at 0.20 for the marginal one.

Report: `artifacts/reports/conformal_variants.json`.

## Run it

```powershell
.\run.ps1                              # install → generate world → train → evaluate → bench → tests → serve on :8080
# or step by step
python scripts/01_generate_data.py     # synthetic world (with shared hostels/offices) + chronological replay
python scripts/02_train.py             # γ candidates, validation-P&L selection, isotonic, conformal, ONNX (+ parity)
python scripts/03_evaluate.py          # policy P&L, friction frontier, sensitivity, break-even, calibration
python scripts/04_bench_latency.py     # in-process (both explain modes) and HTTP p50/p95/p99 by stage
python scripts/07_graph_guard.py 6 12 25   # naive vs guarded graph, merge-ceiling sweep
python scripts/08_festival_shift.py    # domain shift: what the label-free monitor sees vs what labels confirm
python scripts/09_feedback_loop.py     # survivorship bias vs ε control band + IPW, band sizing
python scripts/10_learn_behaviour.py   # prior vs learned vs oracle buyer response, two worlds
python -m pytest tests
python scripts/serve.py --port 8080    # http://127.0.0.1:8080
python scripts/ingest_csv.py --csv export.csv --mapping config/merchant_schema.example.json   # real data
```

Trained models and reports are committed under `artifacts/`; the replay pickles are not, so run
`01_generate_data.py` (about 90 s) once after cloning before `serve.py` — `run.ps1` does this for you. Set
`REDIS_URL` to use Redis for the feature store and drift windows; otherwise an in-process store with identical
semantics is used and reported on `/healthz`. `CHAKRA_EPSILON` sets the control band (default 0.05).

## API

`POST /v1/risk/evaluate?explain=auto|always|never` — request (what the checkout SDK knows):

```json
{"customer_phone":"7012349876","delivery_pin":"845401","shipping_address":"H.No 7, Ward 4, near Hanuman Temple",
 "cart_gmv":1799,"items_count":2,"device_fingerprint_hash":"fp_…","payment_method":"COD",
 "acquisition_channel":"META_ADS","hour_of_day":20,"is_new_customer":true,"merchant_margin":0.18,"cac":540,
 "friction_budget":0.30}
```

Response (abridged): `decision` (served), `policy_action` (what the resolver chose), `p_loss`, `tau_star`,
`tau_soft`, `conformal` (set, certainty, quantiles), `expected_costs`, `elasticity` (α*_crit, slope, profit
curve), `friction` (shadow price, source, budget), `exploration` (`is_control_cohort`, `propensity`, ε),
`behaviour` (learned δ_s / δ_bad / ρ / δ_p for the order's segment and whether they were applied),
`reason_codes` (TreeSHAP), `graph` (ring stats, `entity_shared`), `latency_ms` by stage.

Other routes: `POST /v1/risk/outcome/{order_id}?rto=&stepup_result=paid|abandoned&prepaid_result=` (3PL and
checkout callbacks; closes both learning loops) · `GET /v1/behaviour` · `GET /v1/ledger/stats` ·
`GET /v1/monitor/drift` · `POST /v1/dispute/ce3-compile` · `GET /v1/dispute/candidates` ·
`GET /v1/graph/rings` · `GET /v1/graph/subgraph?seed=` · `GET /v1/report` · `GET /healthz` · `GET /` (console).

### CE3.0 packet and the Mastercard difference

`POST /v1/dispute/ce3-compile?format=html` and `GET /v1/dispute/packet/{transaction_id}.html` return the
same packet as a self-contained printable document (inline CSS, no scripts beyond a print button): header,
the four criteria as pass/fail, the disputed transaction, each qualifying prior transaction with its matched
data elements highlighted, the SHA-256 packet hash and a generation timestamp. The document is a view of
the JSON — every value is copied, nothing recomputed — so the hash on the page is the hash of the JSON
route, and `render_packet_html` is a pure function of (packet, timestamp). The console's dispute view has
an "Open printable packet" button for it.

The compiler is Visa-only on purpose. Mastercard's card-not-present fraud chargeback (reason code 4837)
accepts prior undisputed transactions as compelling evidence in a second presentment, but the issuer weighs
that evidence and disagreement goes to pre-arbitration; there is no rule-mandated liability shift with a fixed
120–365 day window and two-transaction minimum, so the pass/fail table above must not be reused for a 4837
response. Mastercard's First-Party Trust programme (2024, US first) protects merchants when device, IP,
account and shipping data were supplied at authorisation and match the issuer's view — which means the
ledger has to feed the authorisation message, not a packet assembled after the chargeback lands. Exact data
elements, dates and regions for that programme are not reproduced here; see the docstring in
`src/chakrashield/dispute/ce3.py`.

## Layout

```
src/chakrashield/
  config.py                 unit economics, α, latency budget, ε, paths
  schemas.py                pydantic request / response contracts
  data/                     generator (with shared delivery points), replay, pipeline, merchant CSV adapter, PIN reference
  features/                 address defects, velocity windows, vectorizer (36 features incl. entity_shared)
  graph/syndicate.py        typed union-find, shared-entity guard, corroborated ring status, NetworkX view
  models/                   tempered cost-sensitive booster, isotonic knots, Mondrian conformal, ONNX export
  policy/                   economics (τ*, τ_soft, λ, elasticity), resolver, simulation, reason codes
  learning/                 ε control band + propensity, decision ledger, per-segment buyer-response learner
  monitoring/drift.py       conformal set-mix and PSI windows, MODEL_* alarms
  runtime/scorer.py         ONNX Runtime + deferred TreeSHAP
  dispute/ce3.py            Visa CE3.0 compiler
  serving/                  FastAPI gateway + async observer; static console (vanilla JS, hand-rolled SVG)
scripts/                    01–04 pipeline, 07–10 experiments, ingest_csv.py, serve.py
tests/                      58 tests: policy, conformal, features, graph, drift, exploration, learner, adapter, CE3.0, API
artifacts/                  models (+ candidates), reports, world summary; replay pickles are regenerated
config/                     merchant_schema.example.json
```

## Honest caveats

* The world is synthetic. Its causal structure (address quality, PIN tier, paid channels, payment fallback,
  ring membership, shared hostels) mirrors what practitioners report, but absolute rupee numbers are
  illustrative; the *ranking* of policies and the mechanics are what transfer. The CSV adapter is the path to
  a merchant's own numbers.
* The learner, the control band and the drift monitor were validated on simulated outcomes drawn from hidden
  truths; the simulations are honest about where each helps (a wrong prior; a rolling retraining window; a
  regime change) and where it does not (a prior that was already right).
* Conformal guarantees assume exchangeability, and with this scorer the empty-set alarm cannot fire; the
  ambiguity, mix and PSI alarms carry the load until it sharpens.
* The served γ and the friction-budget λ are chosen on the validation split, which sits chronologically before
  the calibration splits; production would re-fit them on a rolling window. Test numbers are never used for
  any choice.
