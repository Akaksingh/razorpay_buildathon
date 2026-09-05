# ChakraShield

**An in-line dynamic checkout intervenor and subgraph abuse sentinel for cash-on-delivery risk.**

ChakraShield sits between "Place order" and the OMS. In one synchronous call (budget 25 ms) it prices the
rupee cost of being wrong *about this order*, produces a calibrated P(RTO | x) with a distribution-free
uncertainty set, and resolves the cheapest admissible intervention — frictionless COD, a refundable ₹49 UPI
shipping deposit, or a prepaid mandate. Asynchronously it folds every order into an entity graph and hunts
for syndicate subgraphs. A separate deterministic module compiles Visa CE3.0 evidence packets for disputes.

```
[ Customer at checkout ]
          │  POST /v1/risk/evaluate                       (sync, ~3 ms typical)
          ▼
┌─────────────────────────────────────────────────────────────────────┐
│ FastAPI Risk Gateway                                                │
│  1. Hydrate feature vector  ── feature store (Redis | in-process)   │
│       velocity windows · PIN priors · address defects · ring stats  │
│  2. ONNX Runtime  ── cost-sensitive LightGBM, 1 intra-op thread     │
│  3. Isotonic recalibration → Mondrian split-conformal set C(x)      │
│  4. Dynamic Action Resolver ── τ*(x), τ_soft(x), expected-cost argmin│
│  5. Exact TreeSHAP → stable reason codes (RSK_ADDR_DEFECT, …)       │
└─────────────────────────────────────────────────────────────────────┘
          │ C(x)={0}   → ALLOW_COD          (certified deliverable)
          │ C(x)={0,1} → argmin over {ALLOW, STEP_UP, PREPAID}
          │ C(x)={1}   → STEP_UP | PREPAID  (certified RTO)
          │ C(x)=∅     → STEP_UP | PREPAID  (novel pattern — neither label conforms)
          ▼  async push
┌─────────────────────────────────────────────────────────────────────┐
│ Syndicate Graph Observer (background task)                          │
│  union-find over phone/device/addr/vpa/ip · ring detection          │
│  publishes ring size / phones / RTO rate back into the store        │
└─────────────────────────────────────────────────────────────────────┘
```

## What we built — and what each piece measures

All numbers below are reproducible with `.\run.ps1` (seed 42). Test split = the chronologically **last 3,868 COD
orders** of a 60,000-order synthetic Indian D2C world; 815 of them RTO (21.1 %).

| Component | Concrete result |
|---|---|
| Cost-sensitive LightGBM RTO scorer, ONNX-exported (`02_train.py`) | AUC **0.756** · PR-AUC **0.542** · Brier 0.131 · ECE **0.011** after isotonic recalibration (0.042 raw) |
| Unweighted baseline on the same features (for comparison) | AUC 0.771 · PR-AUC 0.549 — better rank accuracy, **₹84,595 less margin** protected |
| ONNX Runtime serving | max \|Δp\| vs native booster **2.2 × 10⁻⁷** over 2,000 rows · 0.13 ms per inference |
| Mondrian (class-conditional) conformal sets, α = 0.10 | test coverage **91.7 % / 92.9 %** (target ≥ 90 %) · certified-low orders RTO at **8.1 %**, certified-high at **58.5 %** |
| Three-action expected-cost resolver (τ*(x), τ_soft, ₹49 step-up) | **+₹1,80,376 (+42 %)** net margin vs no engine · **+₹84,595** vs the accuracy model at 0.5 · **614 of 815 RTOs prevented (75 %)** |
| Syndicate graph observer (device / address / VPA rings) | phone-level **recall 98.7 %** (1,065 of 1,079 burner phones) · precision 87.2 % · order-level precision 93.8 % / recall 92.4 % · flagged orders RTO at **82.8 %** vs 13.8 % unflagged |
| Visa CE 3.0 dispute compiler | deterministic 120–365-day window + two-element hash match · SHA-256 evidence packet, no LLM |
| FastAPI risk gateway | in-process **p50 2.88 ms / p99 3.86 ms** · HTTP p99 6.48 ms · **0 of 3,000** calls breached the 25 ms budget |
| Console UI: checkout simulator, merchant P&L console, ring visualizer, dispute view | rendered in headless Chrome, 0 JS errors |
| Test suite | **32 / 32** passing (`pytest tests`) |

### Classification accuracy, stated plainly

The base rate is 21.1 % RTO, so a model that says "everything delivers" already scores 78.9 % accuracy —
which is exactly why the project is judged in rupees, not accuracy. The classical numbers on the test split:

| Model · threshold | Accuracy | Precision (RTO) | Recall (RTO) | F1 | Specificity |
|---|---:|---:|---:|---:|---:|
| **ChakraShield · 0.50** | **83.0 %** | **73.1 %** | 30.3 % | 0.428 | **97.0 %** |
| ChakraShield · 0.30 (F1-optimal cut-off) | 80.4 % | 54.0 % | 47.2 % | **0.504** | 89.3 % |
| ChakraShield · 0.22 (global-cost-optimal cut-off) | 75.9 % | 44.3 % | 56.1 % | 0.495 | 81.2 % |
| Baseline (unweighted) · 0.50 | 83.0 % | 67.8 % | 36.4 % | 0.474 | 95.4 % |

At the conventional 0.5 cut-off the cost-sensitive model is **more precise (73.1 % vs 67.8 %)** and blocks fewer
good buyers (91 vs 141 false positives) at the price of recall — it is trained to be *sure* before it frictions
a customer, because a false block on a high-CAC buyer costs more than one shipped RTO. In production the engine
uses no single cut-off at all: it decides per order with τ*(x) and the conformal set, which is where the
+₹1,80,376 in the results table below comes from.

## Why this is not another RTO classifier

**1. Misclassification is a rupee amount, not a unit error.** For a COD order with GMV *V*, margin *M*,
acquisition cost *CAC*, logistics *L* and lock-up rate *λ*:

```
C_FN(x) = L_fwd + L_rev + packaging + restocking + holding + λ·V      (we allowed COD; it came back)
C_FP(x) = M·V + κ·CAC                                                 (we frictioned a buyer who would have paid)
```

Minimising expected cost gives the Bayes-optimal binary rule **block iff p > τ*(x) = C_FP / (C_FN + C_FP)**.
τ* is not tuned; it is the merchant's own indifference point and it is returned on every response. A ₹4,999
influencer-acquired basket and a ₹499 organic basket to the same PIN get different thresholds because they
*should*.

**2. Soft interventions have their own, lower threshold.** A refundable deposit loses a fraction δ_s of good
buyers and collapses residual RTO risk to ρ·p. Solving E[cost | ALLOW] = E[cost | STEP_UP] in closed form:

```
τ_soft(x) = (δ_s·C_FP + f·(1−δ_s)) / (C_FN·(1−ρ) + δ_s·C_FP − f·δ_s)      <  τ*(x)
```

Hard-declining COD at p ≈ 0.6 destroys CAC; asking for ₹49 at p ≈ 0.2 costs almost nothing. The resolver
computes both and picks the expected-cost argmin over the three actions.

**3. The model says what it does not know.** Inductive conformal prediction on a held-out split, conditioned
by class (RTO is the minority class, so a marginal guarantee would cover deliverables well and RTOs badly):

```
s(x,y) = 1 − p̂(y|x)      q_c = ⌈(n_c+1)(1−α)⌉/n_c empirical quantile of {s(x_i,y_i): y_i=c}
C(x) = { y : s(x,y) ≤ q_y }    ⇒    P(y ∈ C(x) | y=c) ≥ 1−α   for each class, model-agnostic
```

The four outcomes are all actionable. An **empty** set — neither label conforms — is the signature of an
input outside the calibrated support, i.e. a fresh syndicate pattern; ChakraShield steps up rather than guess.

**4. Cost-sensitive learning, then honest calibration.** The booster is trained on instance-weighted
log-loss (w = C_FN(x) for RTO rows, C_FP(x) for delivered rows) so capacity goes where money is. Weighting
distorts probability scale, so the model is isotonically recalibrated on a disjoint chronological split
*before* it meets τ*. ECE is reported before and after.

**5. Graph-free serving path.** Ring statistics are computed asynchronously (union-find, O(α(n)) per
update) and published into the feature store; the request path only reads hashes. NetworkX is used for the
ops console's subgraph extraction only.

**6. Deterministic dispute defence.** Visa CE3.0 is a mechanical rule set (≥2 undisputed prior transactions
120–365 days before the dispute, each sharing ≥2 data elements incl. IP or device). The compiler
hash-matches, emits every criterion's pass/fail, and content-addresses the packet (SHA-256). No LLM.

## Evaluation methodology (rupees, not F1)

Splits are **chronological** — train 60 % · valid 10 % (early stopping) · calib 10 % (isotonic) ·
conf 10 % (conformal) · test 10 %. Every policy is applied to identical test orders with identical true
outcomes, and the merchant's realised contribution is computed as an exact conditional expectation over
buyer response (no Monte-Carlo noise). The behavioural parameters used to *score* policies are separate from
the ones the resolver *assumes*, and `03_evaluate.py` sweeps them, so the headline survives the resolver being
wrong about buyers.

| Policy | What it is |
|---|---|
| `ALLOW_ALL` | no engine (status quo) |
| `BASE@0.5` / `BASE@F1` | unweighted booster, accuracy / F1-style global cut-off |
| `BASE@GLOBAL_COST` | unweighted booster, the single best global cut-off tuned for P&L on *valid* |
| `BASE@TAU*(x)` | unweighted booster, instance-dependent τ*(x), hard block |
| `CHAKRA@TAU*(x)` | cost-sensitive booster, τ*(x), hard block |
| `CHAKRA_FULL` | cost-sensitive + conformal gating + three-action resolver |
| `ORACLE` | perfect foresight — the ceiling |

<!-- RESULTS:BEGIN -->
### Results (seed 42 · 60,000 orders · 38,664 COD · chronological test split of 3,868 COD orders, ₹51.4 L GMV)

| Policy | P&L (₹) | Δ vs no engine | allow / step-up / prepaid | good buyers lost | RTOs shipped |
|---|---:|---:|:---:|---:|---:|
| `ALLOW_ALL` | 4,29,852 | — | 3868 / 0 / 0 | 0 | 815 |
| `BASE@0.5` | 5,25,633 | +95,781 | 3442 / 0 / 426 | 50 | 540 |
| `BASE@F1` | 5,32,040 | +1,02,188 | 3191 / 0 / 677 | 117 | 469 |
| `BASE@GLOBAL_COST` | 5,44,209 | +1,14,357 | 2620 / 0 / 1248 | 274 | 319 |
| `BASE@TAU*(x)` | 5,29,254 | +99,403 | 3372 / 0 / 496 | 70 | 523 |
| `CHAKRA@TAU*(x)` | 5,28,032 | +98,181 | 3306 / 0 / 562 | 95 | 521 |
| **`CHAKRA_FULL`** | **6,10,227** | **+1,80,376 (+42 %)** | 1348 / 1952 / 568 | 270 | 201 |
| `ORACLE` | 7,51,435 | +3,21,584 | 3053 / 449 / 366 | 0 | 49 |

* The full engine protects **1.9× the margin of the accuracy model** (`BASE@0.5`) and **₹66 k more than the
  best globally-tuned cut-off**, while shipping *fewer* RTOs (201 vs 319) at the *same* good-buyer loss
  (270 vs 274). It captures 56 % of the oracle ceiling; the accuracy model captures 30 %.
* **Sensitivity:** `CHAKRA_FULL` beats the best binary policy in 12 of 15 buyer-behaviour scenarios
  (δ_good ∈ {6, 11, 18, 25, 35 %} × δ_bad ∈ {40, 65, 85 %}); it loses only when ≥ 35 % of *good* buyers
  abandon a ₹49 prompt **and** most bad buyers do not — the regime where a merchant should not offer step-up at all.
* **An honest finding:** the textbook hard-block rule `p > τ*(x)` is *not* better than a tuned global cut-off
  here (+99 k vs +114 k). τ* = C_FP/(C_FN+C_FP) assumes every blocked good buyer is lost; in reality a
  fraction pays prepaid. The three-action expected-cost resolver models that response explicitly, which is
  where the gain comes from — τ* survives as the reported ALLOW/BLOCK indifference point, not as the decision rule.
* **A deposit buys commitment, not deliverability.** Residual RTO after a paid step-up is
  ρ_eff = ρ + (1−ρ)·a(x) with a(x) = defect² the share of risk attributed to the address; a junk address
  therefore resolves to a prepaid mandate even when p is moderate.

**Model.** Cost-sensitive LightGBM (113 trees, early-stopped): test AUC 0.756 / PR-AUC 0.542 (unweighted
baseline 0.771 / 0.549 — the weighting deliberately trades rank-AUC for rupee-weighted loss). ECE 0.042 → 0.011
after isotonic recalibration. ONNX parity vs native booster: max |Δp| = 2.2 × 10⁻⁷ over 2,000 rows.

**Conformal (α = 0.10).** Test coverage 91.7 % (deliverable) / 92.9 % (RTO) — both ≥ 90 %, model-agnostic.
Certified-low orders RTO at 8.1 %, ambiguous at 15.8 %, certified-high at 58.5 %. Empty-set rate 0 % in
distribution; it is the drift alarm.

**Latency** (`04_bench_latency.py`, ONNX Runtime, 1 intra-op thread, in-process store, 3,000 calls):

| | p50 | p95 | p99 | max |
|---|---:|---:|---:|---:|
| in-process pipeline (hydrate → ONNX → conformal → resolve → TreeSHAP) | 2.88 ms | 3.32 ms | 3.86 ms | 13.0 ms |
| HTTP round-trip (TestClient) | 4.97 ms | 5.53 ms | 6.48 ms | 11.4 ms |

ONNX inference is 0.13 ms at p50; exact TreeSHAP for reason codes is the largest stage at 1.9 ms. Zero
breaches of the 25 ms budget.
<!-- RESULTS:END -->

## Run it

```powershell
.\run.ps1                 # install → generate world → train → evaluate → bench → tests → serve on :8080
# or step by step
python scripts/01_generate_data.py     # synthetic world + chronological replay → point-in-time features
python scripts/02_train.py             # baseline + cost-sensitive boosters, isotonic, conformal, ONNX (+ parity check)
python scripts/03_evaluate.py          # policy P&L, sensitivity sweep, calibration, certainty breakdown
python scripts/04_bench_latency.py     # in-process and HTTP p50/p95/p99 by stage
python -m pytest tests
python scripts/serve.py --port 8080    # http://127.0.0.1:8080
```

Trained models and evaluation reports are committed under `artifacts/`; the 77 MB of replay pickles in `artifacts/data/` are not, so run `python scripts/01_generate_data.py` (about 90 s) once after cloning before `serve.py` - `run.ps1` does this for you.

Set `REDIS_URL=redis://localhost:6379/0` to use Redis; otherwise an in-process store with identical
semantics is used and reported on `/healthz`.

## API

`POST /v1/risk/evaluate` — request (what the checkout SDK knows):

```json
{"customer_phone":"7012349876","delivery_pin":"845401","shipping_address":"Near Hanuman Temple, Ward 4",
 "cart_gmv":2899,"items_count":2,"device_fingerprint_hash":"fp_…","payment_method":"COD",
 "payment_switch_from":"CARD_FAILED","acquisition_channel":"META_ADS","hour_of_day":23,
 "is_new_customer":true,"merchant_margin":0.18,"cac":540}
```

Response (abridged):

```json
{"decision":"STEP_UP_DEPOSIT","action_label":"Confirm with a refundable ₹49 UPI shipping deposit",
 "p_loss":0.41,"tau_star":0.62,"tau_soft":0.19,
 "conformal":{"alpha":0.1,"prediction_set":[0,1],"certainty":"AMBIGUOUS","quantiles":{"q0":0.47,"q1":0.79}},
 "expected_costs":{"ALLOW_COD":152.3,"STEP_UP_DEPOSIT":61.8,"FORCE_PREPAID":209.4},
 "reason_codes":[{"code":"RSK_ADDR_DEFECT","shap":0.91,"direction":"RISK_UP","human":"Address structurally incomplete (defect 0.65)"},
                 {"code":"RSK_PIN_TIER","shap":0.44,"direction":"RISK_UP","human":"Tier-4 delivery PIN"}],
 "latency_ms":{"total":2.9,"budget_ms":25,"within_budget":true},"scorer_backend":"onnxruntime"}
```

Other routes: `POST /v1/risk/outcome/{order_id}?rto=` (3PL callback, closes the loop) ·
`POST /v1/dispute/ce3-compile` · `GET /v1/dispute/candidates` · `GET /v1/graph/rings` ·
`GET /v1/graph/subgraph?seed=` · `GET /v1/report` · `GET /healthz` · `GET /` (console).

## Layout

```
src/chakrashield/
  config.py                 unit economics, α, latency budget
  schemas.py                pydantic API contracts
  data/generator.py         synthetic Indian D2C world: legit / impulse / syndicate cohorts
  data/replay.py            chronological event replay → point-in-time features (same code as serving)
  data/pincodes.py          PIN hierarchy → state / city / tier / priors
  features/address.py       deterministic address-defect scoring
  features/velocity.py      entity hashing, velocity windows, store write path
  features/vectorizer.py    FEATURE_NAMES — the single source of truth
  graph/syndicate.py        union-find ring detection + NetworkX subgraph view
  store/feature_store.py    Redis | in-memory, one interface
  models/cost_sensitive_booster.py   instance-weighted LightGBM
  models/calibration.py     isotonic knots (np.interp at runtime)
  models/conformal.py       Mondrian split-conformal
  models/onnx_export.py     LightGBM → ONNX with parity assertion
  runtime/scorer.py         ONNX Runtime + exact TreeSHAP
  policy/economics.py       TransactionContext, C_FN, C_FP, τ*, τ_soft, expected costs
  policy/resolver.py        conformal gating × expected-cost argmin
  policy/reason_codes.py    SHAP → merchant-legible codes
  policy/simulation.py      counterfactual P&L simulator
  dispute/ce3.py            Visa CE3.0 deterministic compiler
  serving/app.py            FastAPI gateway + async graph worker
  serving/static/           console (vanilla JS, no build step)
scripts/                    01 generate · 02 train · 03 evaluate · 04 bench · serve
tests/                      policy math, conformal coverage, features, graph, CE3, API
```

## Honest caveats

* The world is synthetic. Its causal structure (address quality, PIN tier, paid channels, payment fallback,
  ring membership) mirrors what practitioners report, but absolute P&L numbers are illustrative; the
  *ranking* of policies and the τ*/τ_soft mechanics are what transfer.
* Buyer response to a ₹49 deposit (δ_s) is a merchant-specific number best measured by A/B. The sensitivity
  sweep shows the ranking under δ_s ∈ [6 %, 35 %].
* Conformal guarantees assume exchangeability; a regime shift shows up as a rising empty-set rate, which is
  itself a monitoring signal.
