/* ChakraShield console — vanilla JS, no build step. Charts are hand-rolled SVG
   following the dataviz method: one axis, thin marks, hairline grid, legend for
   >= 2 series, hover tooltips, table views. */
(() => {
  const $ = (s, r = document) => r.querySelector(s);
  const $$ = (s, r = document) => Array.from(r.querySelectorAll(s));
  const inr = new Intl.NumberFormat("en-IN", { maximumFractionDigits: 0 });
  const fmtR = v => (v < 0 ? "−₹" : "₹") + inr.format(Math.abs(Math.round(v)));
  const pct = (v, d = 1) => (100 * v).toFixed(d) + "%";
  const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
  const ACTIONS = ["ALLOW_COD", "STEP_UP_DEPOSIT", "FORCE_PREPAID"];
  const ACTION_SHORT = { ALLOW_COD: "Allow COD", STEP_UP_DEPOSIT: "₹49 step-up", FORCE_PREPAID: "Prepaid only" };
  const ACTION_COLOR = { ALLOW_COD: "--s1", STEP_UP_DEPOSIT: "--s2", FORCE_PREPAID: "--s3" };   // identity: fixed slots
  const ACTION_STATUS = { ALLOW_COD: ["good", "✓"], STEP_UP_DEPOSIT: ["warning", "◐"], FORCE_PREPAID: ["serious", "⊘"] };
  const KIND_COLOR = { phone: "--s1", device: "--s2", addr: "--s3", vpa: "--s4", ip: "--s5" };
  const KIND_SHAPE = { phone: "circle", device: "square", addr: "triangle", vpa: "diamond", ip: "hex" };
  const tip = $("#tooltip");
  const showTip = (e, html) => { tip.innerHTML = html; tip.hidden = false; moveTip(e); };
  const moveTip = e => { const x = Math.min(e.clientX + 14, innerWidth - tip.offsetWidth - 8); tip.style.left = x + "px"; tip.style.top = (e.clientY + 14) + "px"; };
  const hideTip = () => { tip.hidden = true; };
  const esc = s => String(s).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  const api = async (path, opts) => { const r = await fetch(path, opts); if (!r.ok) throw new Error(await r.text()); return r.json(); };
  const post = (path, body) => api(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });

  // ------------------------------------------------------------------ tabs
  $$(".tab").forEach(b => b.addEventListener("click", () => showView(b.dataset.view)));
  function showView(v) {
    $$(".tab").forEach(b => b.classList.toggle("active", b.dataset.view === v));
    $$(".view").forEach(m => m.classList.toggle("active", m.id === "view-" + v));
    if (v === "console" && !state.consoleLoaded) loadConsole();
    if (v === "graph" && !state.graphLoaded) loadGraph();
    if (v === "dispute" && !state.disputeLoaded) loadDispute();
  }
  const state = { consoleLoaded: false, graphLoaded: false, disputeLoaded: false, health: null };

  // ------------------------------------------------------------------ health
  async function health() {
    try {
      const h = await api("/healthz"); state.health = h;
      $("#health").className = "health ok";
      $("#healthText").textContent = `${h.scorer_backend} · ${h.store_backend} · model ${h.model_version} · budget ${h.latency_budget_ms}ms · α=${h.alpha}`;
    } catch { $("#health").className = "health bad"; $("#healthText").textContent = "gateway unreachable"; }
  }
  health(); setInterval(health, 10000);

  // ------------------------------------------------------------------ checkout form
  const form = $("#checkoutForm");
  async function loadScenarios() {
    const sc = await api("/v1/scenarios");
    $("#scenarioChips").innerHTML = sc.map((s, i) => `<button type="button" class="chip" data-i="${i}" title="${esc(s.name)}">${esc(s.name)}</button>`).join("");
    $$("#scenarioChips .chip").forEach(b => b.addEventListener("click", () => { fillForm(sc[+b.dataset.i].req); form.requestSubmit(); }));
  }
  function fillForm(req) {
    for (const [k, v] of Object.entries(req)) {
      const el = form.elements[k]; if (!el) continue;
      if (el.type === "checkbox") el.checked = !!v; else el.value = v == null ? "" : v;
    }
    if (req.payment_switch_from == null) form.elements.payment_switch_from.value = "";
    if (req.coupon_applied == null) form.elements.coupon_applied.checked = false;
  }
  $$(".picker .ghost").forEach(b => b.addEventListener("click", async () => {
    const rows = await api(`/v1/orders/sample?n=1&cohort=${b.dataset.cohort}`);
    const r = rows[0]; if (!r) return;
    fillForm({ customer_phone: r.customer_phone, delivery_pin: r.delivery_pin, shipping_address: r.shipping_address, cart_gmv: r.cart_gmv,
      items_count: r.items_count, weight_grams: r.weight_grams, device_fingerprint_hash: r.device_fingerprint, acquisition_channel: r.acquisition_channel,
      payment_method: r.payment_method, payment_switch_from: r.payment_switch_from, hour_of_day: r.hour_of_day, checkout_seconds: r.checkout_seconds,
      merchant_margin: r.merchant_margin, cac: r.cac, coupon_applied: r.coupon_applied, is_new_customer: r.is_new_customer });
    form.dataset.orderId = r.order_id; form.dataset.truth = r.rto ? "RTO" : "delivered"; form.dataset.cohort = r.cohort;
    form.requestSubmit();
  }));
  form.addEventListener("input", () => { delete form.dataset.orderId; delete form.dataset.truth; delete form.dataset.cohort; });
  form.addEventListener("submit", async e => {
    e.preventDefault();
    const fd = new FormData(form); const body = {};
    for (const [k, v] of fd.entries()) body[k] = v;
    ["cart_gmv", "items_count", "weight_grams", "hour_of_day", "checkout_seconds", "merchant_margin", "cac"].forEach(k => body[k] = +body[k]);
    body.coupon_applied = form.elements.coupon_applied.checked;
    body.is_new_customer = form.elements.is_new_customer.checked;
    if (!body.payment_switch_from) delete body.payment_switch_from;
    if (!body.friction_budget) delete body.friction_budget; else body.friction_budget = +body.friction_budget;
    const explain = body.explain || "always"; delete body.explain;
    const commit = form.elements.commit.checked; delete body.commit;
    if (form.dataset.orderId) body.order_id = form.dataset.orderId + "_replay";
    const btn = $("#placeOrder"); btn.disabled = true;
    try {
      const t0 = performance.now();
      const res = await post(`/v1/risk/evaluate?commit=${commit}&explain=${explain}`, body);
      res._rtt = performance.now() - t0; res._truth = form.dataset.truth; res._cohort = form.dataset.cohort;
      renderDecision(res);
    } catch (err) { alert("Evaluate failed: " + err.message); }
    finally { btn.disabled = false; }
  });

  // ------------------------------------------------------------------ decision rendering
  function renderDecision(r) {
    $("#decisionEmpty").hidden = true; $("#decisionBody").hidden = false;
    const [status, icon] = ACTION_STATUS[r.decision] || ["critical", "!"];
    const banner = $("#banner"); banner.className = "banner " + (r.conformal.certainty === "NOVEL" ? "critical" : status);
    $("#bannerIcon").textContent = r.conformal.certainty === "NOVEL" ? "!" : icon;
    $("#bannerAction").textContent = r.decision.replace(/_/g, " ") + (r.conformal.certainty === "NOVEL" ? " · NOVEL PATTERN" : "");
    $("#bannerLabel").textContent = r.action_label + (r._truth ? `  ·  historical truth: ${r._truth}${r._cohort ? " (" + r._cohort + ")" : ""}` : "");
    $("#bannerLatency").innerHTML = `<b>${r.latency_ms.total.toFixed(2)} ms</b>server · ${r._rtt.toFixed(0)} ms round-trip`;
    $("#buyerUx").innerHTML = buyerUx(r);
    $("#rationale").textContent = r.rationale;

    // threshold meter
    const m = $("#thresholdMeter");
    m.innerHTML = `<div class="meter">
        <span class="mark tau" style="left:${100 * r.tau_soft}%" data-l="τ_soft ${r.tau_soft.toFixed(2)}"></span>
        <span class="mark" style="left:${100 * r.tau_star}%" data-l="τ* ${r.tau_star.toFixed(2)}"></span>
        <span class="p" style="left:${100 * r.p_loss}%" data-l="P(RTO) ${r.p_loss.toFixed(2)}"></span>
      </div><div class="meter-axis"><span>0 · certain delivery</span><span>1 · certain RTO</span></div>`;
    kv("#thresholdKv", [["P(RTO | x) calibrated", r.p_loss.toFixed(4)], ["booster raw (pre-isotonic)", r.p_raw.toFixed(4)],
      ["τ_soft — allow ↔ ₹49 step-up", r.tau_soft.toFixed(4)], ["τ* — allow ↔ hard block  =  C_FP / (C_FN + C_FP)", r.tau_star.toFixed(4)]]);

    // conformal
    const cs = r.conformal; const setTxt = cs.prediction_set.length ? "{" + cs.prediction_set.join(", ") + "}" : "∅";
    const certColor = { CERTIFIED_LOW: "--good", AMBIGUOUS: "--warning", CERTIFIED_HIGH: "--serious", NOVEL: "--critical" }[cs.certainty];
    $("#alphaText").textContent = `α = ${cs.alpha} → ${pct(1 - cs.alpha, 0)} class-conditional coverage`;
    $("#csetChips").innerHTML = `<span class="set">C(x) = ${setTxt}</span><span class="cert"><i style="background:var(${certColor})"></i>${cs.certainty.replace("_", " ")}</span>`;
    kv("#csetKv", [["s(x,0) = p", cs.nonconformity.s0.toFixed(4) + `  ≤ q₀ ${cs.quantiles.q0.toFixed(4)} ? ${cs.nonconformity.s0 <= cs.quantiles.q0 ? "yes → 0 ∈ C" : "no"}`],
      ["s(x,1) = 1−p", cs.nonconformity.s1.toFixed(4) + `  ≤ q₁ ${cs.quantiles.q1.toFixed(4)} ? ${cs.nonconformity.s1 <= cs.quantiles.q1 ? "yes → 1 ∈ C" : "no"}`],
      ["admissible actions", r.admissible_actions.map(a => ACTION_SHORT[a]).join(" · ")]]);

    // expected cost bars (one series, emphasis on chosen action)
    hbars($("#costChart"), ACTIONS.map(a => ({ label: ACTION_SHORT[a], value: r.expected_costs[a], emph: a === r.decision, muted: !r.admissible_actions.includes(a) })),
      { fmt: fmtR, tipFn: d => `<b>${esc(d.label)}</b>E[loss] ${fmtR(d.value)}${d.muted ? "<br>not admissible under C(x)" : ""}` });

    // reason codes (diverging SHAP)
    const rc = r.reason_codes; const mx = Math.max(0.01, ...rc.map(c => Math.abs(c.shap)));
    $("#reasons").innerHTML = rc.length ? rc.map(c => {
      const w = 50 * Math.abs(c.shap) / mx; const pos = c.shap > 0;
      return `<div class="reason"><div><div class="code">${esc(c.code)}</div><div class="human">${esc(c.human)}</div></div>
        <div class="bar"><span class="mid"></span><i class="${pos ? "" : "neg"}" style="${pos ? "left:50%" : "right:50%"};width:${w}%;background:var(${pos ? "--div-neg" : "--div-pos"})"></i>
        <b style="${pos ? "left:calc(50% + " + w + "% + 4px)" : "right:calc(50% + " + w + "% + 4px)"}">${c.shap > 0 ? "+" : ""}${c.shap.toFixed(2)}</b></div></div>`;
    }).join("") : `<span class="muted">${r.explained === false ? "explain=auto — TreeSHAP skipped for an ALLOW decision (an allow needs no defence); saves ~1.9 ms" : "No contribution above the reporting floor."}</span>`;

    // economics / graph / address
    const e = r.economics;
    kv("#econKv", [["GMV (V)", fmtR(e.gmv)], ["margin (M)", pct(e.merchant_margin, 0)], ["CAC", fmtR(e.cac)], ["new customer", e.is_new_customer ? "yes" : "no"],
      ["logistics fwd+rev", fmtR(e.logistics)], ["holding", fmtR(e.holding)], ["C_FN — allow a bad order", fmtR(e.cost_fn)], ["C_FP — friction a good one", fmtR(e.cost_fp)],
      ["expected saving vs allow", fmtR(r.expected_saving_vs_allow)],
      ["friction shadow price λ", fmtR(e.friction_shadow_price || 0)],
      ...(r.friction && r.friction.source === "frontier" ? [["friction budget", `≤ ${pct(r.friction.budget, 0)}`]] : []),
      ...(r.friction && r.friction.budget_changed_action ? [["budget effect", "changed the action"]] : [])]);
    const g = r.graph;
    kv("#graphKv", [["ring membership", g.is_ring ? "SYNDICATE RING" : (g.ring_size ? "linked cluster" : "none")], ["cluster size", g.ring_size || 0], ["phones in cluster", g.ring_phones || 0],
      ["devices in cluster", g.ring_devices || 0], ["cluster RTO rate", g.ring_size ? pct(g.ring_rto_rate, 0) : "—"], ["max entity degree", g.entity_max_degree],
      ["public / shared entity", g.entity_shared ? "YES — merge guard active" : "no"],
      ["phones on this device", r.velocity.device_distinct_phones], ["phones at this address", r.velocity.addr_distinct_phones]]);
    const vr = $("#viewRing"); vr.hidden = !g.ring_id; vr.onclick = () => { showView("graph"); loadGraph().then(() => drawSubgraph(g.ring_id)); };
    const a = r.address;
    kv("#addrKv", [["defect score", a.defect_score.toFixed(2)], ["house / flat number", a.has_house_no ? "present" : "missing"], ["street anchor", a.has_street_anchor ? "present" : "missing"],
      ["landmark-only", a.landmark_only ? "yes" : "no"], ["vague-only", a.vague_only ? "yes" : "no"], ["junk tokens", a.has_junk ? "yes" : "no"], ["state vs PIN", a.state_mismatch ? "MISMATCH" : "consistent"], ["tokens", a.tokens]]);

    // break-even elasticity in alpha_drop (client re-renders from the closed-form coefficients)
    if (r.elasticity) {
      state.elasticity = r.elasticity;
      const sl = $("#alphaSlider"); sl.value = Math.round(100 * r.elasticity.alpha_assumed);
      renderElasticity(+sl.value / 100);
    }
    // latency stages
    const L = r.latency_ms; const stages = [["hydrate: hash", L["hydrate.hash"]], ["hydrate: address", L["hydrate.address"]], ["hydrate: velocity", L["hydrate.velocity"]], ["hydrate: graph", L["hydrate.graph"]],
      ["hydrate: vectorize", L["hydrate.vectorize"]], ["score: ONNX infer", L["score.onnx_infer"]], ["score: isotonic+conformal", L["score.calibrate_conformal"]], ["score: TreeSHAP", L["score.treeshap"]], ["resolve + reason codes", L.resolve_explain]];
    $("#latencyBudgetText").textContent = `${L.total.toFixed(2)} ms of ${L.budget_ms} ms — ${L.within_budget ? "within budget" : "BREACH"} · ${r.scorer_backend}`;
    hbars($("#latencyChart"), stages.map(([label, value]) => ({ label, value: value || 0 })), { fmt: v => v.toFixed(3) + " ms", max: Math.max(1, ...stages.map(s => s[1] || 0)) });
    $("#rawJson").textContent = JSON.stringify(r, null, 2);
  }
  $("#alphaSlider").addEventListener("input", e => renderElasticity(+e.target.value / 100));
  function renderElasticity(alpha) {
    const E = state.elasticity; if (!E) return;
    $("#alphaVal").textContent = pct(alpha, 0);
    const stepup = a => E.margin - (E.intercept + E.slope * a) - E.shadow_price;
    const crit = E.alpha_crit, assumed = E.alpha_assumed;
    $("#elastSub").textContent = crit == null ? "step-up never loses to the alternatives in this range" :
      `α* = ${pct(crit, 0)} (binds against ${E.binding === "ALLOW_COD" ? "allow" : "prepaid"}) · assumed ${pct(assumed, 0)} · headroom ${E.headroom >= 0 ? "+" : ""}${pct(E.headroom, 0)} · slope ${fmtR(E.rupees_per_point)} per point`;
    const el = $("#elastChart"), W = widthOf(el), H = 220, padL = 64, padR = 20, padT = 16, padB = 40, a0 = 0.05, a1 = 0.60;
    const ys = [stepup(a0), stepup(a1), E.profit_allow, E.profit_prepaid];
    const ymin = Math.min(...ys) - 20, ymax = Math.max(...ys) + 20;
    const sx = a => padL + (a - a0) / (a1 - a0) * (W - padL - padR), sy = v => padT + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - padT - padB);
    let s = `<svg viewBox="0 0 ${W} ${H}">`;
    niceTicks(ymax - ymin, 4).map(t => t + ymin).forEach(t => { s += `<line class="grid" x1="${padL}" x2="${W - padR}" y1="${sy(t)}" y2="${sy(t)}"/><text x="${padL - 6}" y="${sy(t) + 4}" text-anchor="end">${fmtR(t)}</text>`; });
    [0.05, 0.2, 0.4, 0.6].forEach(a => { s += `<text x="${sx(a)}" y="${H - padB + 16}" text-anchor="${a === a1 ? "end" : a === a0 ? "start" : "middle"}">${pct(a, 0)}</text>`; });
    s += `<line x1="${padL}" x2="${W - padR}" y1="${sy(E.profit_allow)}" y2="${sy(E.profit_allow)}" stroke="var(--s1)" stroke-width="2" stroke-dasharray="6 4"/><text x="${W - padR}" y="${sy(E.profit_allow) - 5}" text-anchor="end">allow COD ${fmtR(E.profit_allow)}</text>`;
    s += `<line x1="${padL}" x2="${W - padR}" y1="${sy(E.profit_prepaid)}" y2="${sy(E.profit_prepaid)}" stroke="var(--s3)" stroke-width="2" stroke-dasharray="6 4"/><text x="${W - padR}" y="${sy(E.profit_prepaid) - 5}" text-anchor="end">prepaid only ${fmtR(E.profit_prepaid)}</text>`;
    s += `<line x1="${sx(a0)}" y1="${sy(stepup(a0))}" x2="${sx(a1)}" y2="${sy(stepup(a1))}" stroke="var(--s2)" stroke-width="2.5"/>`;
    if (crit != null && crit >= a0 && crit <= a1) s += `<circle cx="${sx(crit)}" cy="${sy(stepup(crit))}" r="6" fill="var(--serious)" stroke="var(--surface)" stroke-width="2"/><text x="${sx(crit)}" y="${sy(stepup(crit)) + 20}" text-anchor="middle" style="font-weight:500">α* ${pct(crit, 0)}</text>`;
    s += `<line x1="${sx(assumed)}" x2="${sx(assumed)}" y1="${padT}" y2="${H - padB}" stroke="var(--axis)" stroke-dasharray="3 3"/><text x="${sx(assumed) + 4}" y="${padT + 10}">assumed ${pct(assumed, 0)}</text>`;
    s += `<line x1="${sx(alpha)}" x2="${sx(alpha)}" y1="${padT}" y2="${H - padB}" stroke="var(--ink)" stroke-width="1.5"/><circle cx="${sx(alpha)}" cy="${sy(stepup(alpha))}" r="5" fill="var(--s2)" stroke="var(--surface)" stroke-width="2"/>`;
    s += `<text x="${(padL + W - padR) / 2}" y="${H - 4}" text-anchor="middle">α_drop: share of good buyers who abandon at the ₹49 prompt →</text><text x="${padL + 6}" y="${H - padB - 6}">↑ expected profit on this order</text></svg>`;
    el.innerHTML = s;
    const su = stepup(alpha), best = Math.max(su, E.profit_allow, E.profit_prepaid);
    const verdict = su >= best - 1e-9 ? "the ₹49 step-up is still the best action" : (E.profit_allow >= E.profit_prepaid ? `allowing COD outright beats the deposit by ${fmtR(E.profit_allow - su)}` : `a prepaid mandate beats the deposit by ${fmtR(E.profit_prepaid - su)}`);
    $("#elastText").textContent = `At α_drop = ${pct(alpha, 0)}: E[profit] step-up ${fmtR(su)} · allow ${fmtR(E.profit_allow)} · prepaid ${fmtR(E.profit_prepaid)} — ${verdict}. Every extra point of abandonment costs ${fmtR(E.rupees_per_point)} on this order.`;
  }
  function buyerUx(r) {
    if (r.decision === "ALLOW_COD") return `<span class="phone">buyer sees</span><span>Pay on delivery · <b>${fmtR(r.economics.gmv)}</b></span><span class="upi" style="background:var(--good)">Place order</span>`;
    if (r.decision === "STEP_UP_DEPOSIT") return `<span class="phone">buyer sees</span><span>Confirm your COD order with a <b>₹49 refundable</b> shipping deposit — refunded on delivery, adjusted on refusal.</span><span class="upi">Pay ₹49 via UPI</span>`;
    return `<span class="phone">buyer sees</span><span>Cash on delivery isn’t available for this order. Pay securely — <b>${fmtR(r.economics.gmv)}</b></span><span class="upi">UPI · Card · EMI</span>`;
  }
  function kv(sel, rows) { $(sel).innerHTML = "<dl class='kv'>" + rows.map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("") + "</dl>"; $(sel).className = ""; }

  // ------------------------------------------------------------------ chart helpers (SVG)
  // viewBox width follows the container so 11px text stays 11px in every panel width.
  const widthOf = el => Math.max(320, Math.round(el.clientWidth || el.parentElement?.clientWidth || 560));
  // Horizontal bars, single series. Emphasis: chosen row in accent, rest de-emphasised.
  function hbars(el, rows, o = {}) {
    const W = widthOf(el), rowH = 26, padL = o.padL || 150, padR = 90, H = rows.length * rowH + 8;
    const max = o.max || Math.max(1e-9, ...rows.map(r => Math.abs(r.value)));
    const hasNeg = rows.some(r => r.value < 0);
    const x0 = hasNeg ? padL + (W - padL - padR) / 2 : padL;
    const sx = v => x0 + (v / max) * (W - padL - padR) / (hasNeg ? 2 : 1);
    const anyEmph = rows.some(r => r.emph);
    let s = `<svg viewBox="0 0 ${W} ${H}" role="img">`;
    s += `<line class="axis" x1="${x0}" x2="${x0}" y1="0" y2="${H}"/>`;
    rows.forEach((r, i) => {
      const y = i * rowH + 4, h = 18; const x1 = sx(Math.min(0, r.value)), x2 = sx(Math.max(0, r.value));
      const fill = r.color ? `var(${r.color})` : (anyEmph ? (r.emph ? "var(--s1)" : "var(--de)") : "var(--s1)");
      const rad = r.value >= 0 ? `M${x1},${y} H${Math.max(x1, x2 - 4)} a4,4 0 0 1 4,4 v${h - 8} a4,4 0 0 1 -4,4 H${x1} Z` : `M${x2},${y} H${Math.min(x2, x1 + 4)} a4,4 0 0 0 -4,4 v${h - 8} a4,4 0 0 0 4,4 H${x2} Z`;
      s += `<g data-i="${i}"><text x="${padL - 8}" y="${y + 13}" text-anchor="end" ${r.muted ? 'opacity=".55"' : ""}>${esc(r.label)}</text>
        <path class="mark" d="${rad}" fill="${fill}" ${r.muted ? 'opacity=".45"' : ""}/>
        <text class="lbl" x="${(r.value >= 0 ? x2 : x1) + (r.value >= 0 ? 6 : -6)}" y="${y + 13}" text-anchor="${r.value >= 0 ? "start" : "end"}">${esc((o.fmt || (v => v))(r.value))}</text>
        <rect class="hit" x="0" y="${y - 4}" width="${W}" height="${rowH}"/></g>`;
    });
    s += "</svg>"; el.innerHTML = s;
    if (o.tipFn) $$("g[data-i]", el).forEach(g => { const d = rows[+g.dataset.i]; g.addEventListener("mousemove", e => showTip(e, o.tipFn(d))); g.addEventListener("mouseleave", hideTip); });
  }
  // Horizontal 100% stacked bars, 3 fixed categorical slots, 2px surface gaps.
  function stacked(el, rows, series, o = {}) {
    const color = o.color || (k => ACTION_COLOR[k]), label = o.label || (k => ACTION_SHORT[k]);
    const W = widthOf(el), rowH = 26, padL = o.padL || 150, padR = 16, H = rows.length * rowH + 8, w = W - padL - padR;
    let s = `<svg viewBox="0 0 ${W} ${H}">`;
    rows.forEach((r, i) => {
      const y = i * rowH + 4; let x = padL;
      s += `<text x="${padL - 8}" y="${y + 13}" text-anchor="end">${esc(r.label)}</text>`;
      series.forEach((k, j) => {
        const v = r.values[k] || 0; const bw = Math.max(0, v * w - (j < series.length - 1 ? 2 : 0));
        if (v > 0) s += `<g data-i="${i}" data-k="${k}"><rect class="mark" x="${x}" y="${y}" width="${bw}" height="18" fill="var(${color(k)})" rx="${j === series.length - 1 ? 4 : 0}"/>` +
          (v * w > 44 ? `<text x="${x + bw / 2}" y="${y + 13}" text-anchor="middle" fill="#fff" style="fill:#fff;font-weight:500">${pct(v, 0)}</text>` : "") + `</g>`;
        x += v * w;
      });
    });
    s += "</svg>"; el.innerHTML = s;
    $$("g[data-i]", el).forEach(g => { const r = rows[+g.dataset.i], k = g.dataset.k; g.addEventListener("mousemove", e => showTip(e, `<b>${esc(r.label)}</b>${label(k)}: ${pct(r.values[k])} (${inr.format(r.counts[k])} orders)`)); g.addEventListener("mouseleave", hideTip); });
  }
  // Column histogram, single hue.
  function columns(el, edges, counts, o = {}) {
    const W = widthOf(el), H = 200, padL = 44, padB = 26, padT = 12, n = counts.length, w = (W - padL - 10) / n, max = Math.max(1, ...counts);
    const sy = v => padT + (H - padT - padB) * (1 - v / max);
    let s = `<svg viewBox="0 0 ${W} ${H}">`;
    const ticks = niceTicks(max, 4); ticks.forEach(t => { s += `<line class="grid" x1="${padL}" x2="${W - 10}" y1="${sy(t)}" y2="${sy(t)}"/><text x="${padL - 6}" y="${sy(t) + 4}" text-anchor="end">${inr.format(t)}</text>`; });
    counts.forEach((c, i) => { const x = padL + i * w + 1, bw = Math.max(1, w - 2), y = sy(c), h = sy(0) - y;
      s += `<g data-i="${i}"><path class="mark" d="M${x},${y + h} V${y + 4} a4,4 0 0 1 4,-4 H${x + bw - 4} a4,4 0 0 1 4,4 V${y + h} Z" fill="var(${o.color || "--s1"})"/><rect class="hit" x="${x - 1}" y="${padT}" width="${w}" height="${H - padT - padB}"/></g>`; });
    s += `<line class="axis" x1="${padL}" x2="${W - 10}" y1="${sy(0)}" y2="${sy(0)}"/>`;
    for (let i = 0; i <= n; i += Math.ceil(n / 5)) s += `<text x="${padL + i * w}" y="${H - 8}" text-anchor="middle">${(edges[i] ?? 1).toFixed(2)}</text>`;
    if (o.marker != null) { const x = padL + o.marker * n * w; s += `<line x1="${x}" x2="${x}" y1="${padT}" y2="${sy(0)}" stroke="var(--ink)" stroke-width="1.5"/><text x="${x + 4}" y="${padT + 10}" class="lbl">${esc(o.markerLabel)}</text>`; }
    s += "</svg>"; el.innerHTML = s;
    $$("g[data-i]", el).forEach(g => { const i = +g.dataset.i; g.addEventListener("mousemove", e => showTip(e, `<b>${edges[i].toFixed(2)} – ${edges[i + 1].toFixed(2)}</b>${inr.format(counts[i])} orders`)); g.addEventListener("mouseleave", hideTip); });
  }
  // Calibration dots vs the identity line.
  function calibration(el, pts) {
    const W = widthOf(el), H = 260, padL = 44, padB = 42, padT = 14, padR = 16;
    const sx = v => padL + v * (W - padL - padR), sy = v => padT + (1 - v) * (H - padT - padB);
    let s = `<svg viewBox="0 0 ${W} ${H}">`;
    [0, .25, .5, .75, 1].forEach(t => { s += `<line class="grid" x1="${padL}" x2="${W - padR}" y1="${sy(t)}" y2="${sy(t)}"/><text x="${padL - 6}" y="${sy(t) + 4}" text-anchor="end">${pct(t, 0)}</text><text x="${sx(t)}" y="${H - padB + 16}" text-anchor="${t === 1 ? "end" : t === 0 ? "start" : "middle"}">${pct(t, 0)}</text>`; });
    s += `<line x1="${sx(0)}" y1="${sy(0)}" x2="${sx(1)}" y2="${sy(1)}" stroke="var(--axis)" stroke-width="1"/>`;
    s += `<polyline fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round" points="${pts.map(p => sx(p.p_mean) + "," + sy(p.rto_rate)).join(" ")}"/>`;
    pts.forEach((p, i) => { s += `<g data-i="${i}"><circle cx="${sx(p.p_mean)}" cy="${sy(p.rto_rate)}" r="5" fill="var(--s1)" stroke="var(--surface)" stroke-width="2"/><circle class="hit" cx="${sx(p.p_mean)}" cy="${sy(p.rto_rate)}" r="14"/></g>`; });
    s += `<text x="${(padL + W - padR) / 2}" y="${H - 4}" text-anchor="middle">predicted P(RTO) →</text><text x="${padL + 6}" y="${padT + 4}">↑ observed RTO</text></svg>`;
    el.innerHTML = s;
    $$("g[data-i]", el).forEach(g => { const p = pts[+g.dataset.i]; g.addEventListener("mousemove", e => showTip(e, `<b>decile ${p.decile + 1}</b>predicted ${pct(p.p_mean)} · observed ${pct(p.rto_rate)} · n=${inr.format(p.n)}`)); g.addEventListener("mouseleave", hideTip); });
  }
  // Diverging heatmap (blue positive / red negative / neutral mid).
  function heatmap(el, rowsK, colsK, cell, o = {}) {
    const W = widthOf(el), cw = (W - 120) / colsK.length, ch = 36, H = rowsK.length * ch + 40;
    const vals = rowsK.flatMap(r => colsK.map(c => cell(r, c))); const amax = Math.max(1, ...vals.map(Math.abs));
    let s = `<svg viewBox="0 0 ${W} ${H}">`;
    colsK.forEach((c, j) => s += `<text x="${120 + j * cw + cw / 2}" y="14" text-anchor="middle">${esc(o.colLabel(c))}</text>`);
    rowsK.forEach((r, i) => {
      s += `<text x="112" y="${28 + i * ch + ch / 2 + 4}" text-anchor="end">${esc(o.rowLabel(r))}</text>`;
      colsK.forEach((c, j) => {
        const v = cell(r, c), t = Math.min(1, Math.abs(v) / amax); const col = v >= 0 ? "var(--div-pos)" : "var(--div-neg)";
        s += `<g data-r="${i}" data-c="${j}"><rect x="${120 + j * cw + 1}" y="${28 + i * ch + 1}" width="${cw - 2}" height="${ch - 2}" rx="4" fill="var(--div-mid)"/>
          <rect x="${120 + j * cw + 1}" y="${28 + i * ch + 1}" width="${cw - 2}" height="${ch - 2}" rx="4" fill="${col}" opacity="${0.12 + 0.78 * t}"/>
          <text x="${120 + j * cw + cw / 2}" y="${28 + i * ch + ch / 2 + 4}" text-anchor="middle" style="fill:${t > 0.45 ? "#fff" : "var(--ink)"};font-weight:500">${fmtR(v)}</text></g>`;
      });
    });
    s += "</svg>"; el.innerHTML = s;
    $$("g[data-r]", el).forEach(g => { const r = rowsK[+g.dataset.r], c = colsK[+g.dataset.c]; g.addEventListener("mousemove", e => showTip(e, o.tipFn(r, c))); g.addEventListener("mouseleave", hideTip); });
  }
  function niceTicks(max, n) { const raw = max / n, p = Math.pow(10, Math.floor(Math.log10(raw))), step = [1, 2, 5, 10].map(m => m * p).find(v => v >= raw); const out = []; for (let v = 0; v <= max; v += step) out.push(v); return out; }
  function table(el, cols, rows, o = {}) {
    el.innerHTML = `<table class="t"><thead><tr>${cols.map(c => `<th class="${c.num ? "num" : ""}">${esc(c.h)}</th>`).join("")}</tr></thead><tbody>` +
      rows.map((r, i) => `<tr class="${o.click ? "click" : ""}" data-i="${i}">${cols.map(c => `<td class="${c.num ? "num" : ""}">${c.f ? c.f(r) : esc(r[c.k])}</td>`).join("")}</tr>`).join("") + "</tbody></table>";
    if (o.click) $$("tr[data-i]", el).forEach(tr => tr.addEventListener("click", () => { $$("tr", el).forEach(t => t.classList.remove("sel")); tr.classList.add("sel"); o.click(rows[+tr.dataset.i]); }));
  }

  // ------------------------------------------------------------------ console
  async function loadConsole() {
    const rep = await api("/v1/report"); const ev = rep.evaluation; state.consoleLoaded = true;
    if (!ev || !ev.policies) { $("#kpis").innerHTML = `<div class="tile"><div class="l">No evaluation report</div><div class="v">—</div><div class="d">run scripts/03_evaluate.py</div></div>`; return; }
    const P = ev.policies, full = P.CHAKRA_FULL, base = P["BASE@0.5"], allow = P.ALLOW_ALL, lat = rep.latency || {};
    const mm = ev.model_metrics || {};
    $("#kpis").innerHTML = [
      tile("Net margin protected vs. no engine", fmtR(full.delta_vs_allow_all), `${full.uplift_pct_vs_allow_all.toFixed(1)}% uplift on ${inr.format(ev.test_orders)} test orders`, true, true),
      tile("vs. accuracy-model @0.5", fmtR(full.delta_vs_base_05), "the F1-style baseline over-blocks", full.delta_vs_base_05 > 0),
      tile("RTOs prevented", inr.format(Math.round(full.rto_prevented_expected)), `of ${inr.format(Math.round(allow.rto_shipped_expected))} that would ship · ${pct(full.rto_prevented_expected / allow.rto_shipped_expected, 0)}`),
      tile("Good buyers lost to friction", inr.format(Math.round(full.good_customers_lost_expected)), `vs ${inr.format(Math.round(base.good_customers_lost_expected))} under BASE@0.5`, full.good_customers_lost_expected < base.good_customers_lost_expected),
      tile("Soft step-up share", pct(full.action_share.STEP_UP_DEPOSIT, 1), `hard block only ${pct(full.action_share.FORCE_PREPAID, 1)}`),
      tile("Gateway latency p99 (explain=auto)", lat.inprocess_ms ? lat.inprocess_ms.p99.toFixed(2) + " ms" : "—", lat.inprocess_ms ? `p50 ${lat.inprocess_ms.p50.toFixed(2)} ms · always-explain p99 ${lat.inprocess_always_ms ? lat.inprocess_always_ms.p99.toFixed(2) + " ms" : "—"} · budget ${lat.budget_ms} ms` : "run 04_bench_latency.py"),
    ].join("");
    // waterfall
    const budgetName = ev.friction_budget ? ev.friction_budget.policy : null;
    const order = ["BASE@0.5", "BASE@F1", "BASE@GLOBAL_COST", "BASE@TAU*(x)", "CHAKRA@TAU*(x)", "CHAKRA_FULL", budgetName, "ORACLE"].filter(k => k && P[k]);
    const wf = order.map(k => ({ label: k, value: P[k].delta_vs_allow_all, emph: k === "CHAKRA_FULL", muted: k === "ORACLE", color: k === "ORACLE" ? "--de" : undefined }));
    $("#wfSub").textContent = `Δ P&L vs ALLOW_ALL (₹${inr.format(Math.round(allow.pnl_total))}) on ${fmtR(ev.test_gmv)} test GMV`;
    hbars($("#waterfall"), wf, { fmt: fmtR, tipFn: d => `<b>${esc(d.label)}</b>P&L ${fmtR(P[d.label].pnl_total)}<br>Δ vs no engine ${fmtR(d.value)}<br>good lost ${inr.format(Math.round(P[d.label].good_customers_lost_expected))} · RTO shipped ${inr.format(Math.round(P[d.label].rto_shipped_expected))}` });
    table($("#waterfallTable"), [{ h: "policy", k: "k" }, { h: "P&L", num: 1, f: r => fmtR(P[r.k].pnl_total) }, { h: "Δ vs allow-all", num: 1, f: r => fmtR(P[r.k].delta_vs_allow_all) },
      { h: "good lost", num: 1, f: r => inr.format(Math.round(P[r.k].good_customers_lost_expected)) }, { h: "RTO shipped", num: 1, f: r => inr.format(Math.round(P[r.k].rto_shipped_expected)) }], ["ALLOW_ALL", ...order].map(k => ({ k })));
    // mix
    $("#mixLegend").innerHTML = ACTIONS.map(a => `<span><i style="background:var(${ACTION_COLOR[a]})"></i>${ACTION_SHORT[a]}</span>`).join("");
    const mixRows = ["ALLOW_ALL", ...order].map(k => ({ label: k, values: P[k].action_share, counts: P[k].actions }));
    stacked($("#mixChart"), mixRows, ACTIONS);
    table($("#mixTable"), [{ h: "policy", k: "label" }, ...ACTIONS.map(a => ({ h: ACTION_SHORT[a], num: 1, f: r => inr.format(r.counts[a]) }))], mixRows);
    // sensitivity
    const sens = ev.sensitivity || []; const rowsK = [...new Set(sens.map(s => s.stepup_good_abandon))], colsK = [...new Set(sens.map(s => s.stepup_bad_abandon))];
    const cell = (r, c) => { const s = sens.find(x => x.stepup_good_abandon === r && x.stepup_bad_abandon === c); return s.CHAKRA_FULL - Math.max(s["BASE@GLOBAL_COST"], s["BASE@TAU*(x)"], s["CHAKRA@TAU*(x)"]); };
    $("#sensSub").textContent = `CHAKRA_FULL wins ${ev.sensitivity_wins[0]} / ${ev.sensitivity_wins[1]} scenarios · resolver assumes δ_good=${rep.economics.stepup_abandon_rate}`;
    heatmap($("#sensHeat"), rowsK, colsK, cell, { rowLabel: r => `good abandon ${pct(r, 0)}`, colLabel: c => `bad abandon ${pct(c, 0)}`,
      tipFn: (r, c) => { const s = sens.find(x => x.stepup_good_abandon === r && x.stepup_bad_abandon === c); return `<b>δ_good ${pct(r, 0)} · δ_bad ${pct(c, 0)}</b>CHAKRA_FULL ${fmtR(s.CHAKRA_FULL)}<br>best binary ${fmtR(Math.max(s["BASE@GLOBAL_COST"], s["BASE@TAU*(x)"], s["CHAKRA@TAU*(x)"]))}<br>BASE@0.5 ${fmtR(s["BASE@0.5"])} · allow-all ${fmtR(s.ALLOW_ALL)}`; } });
    // calibration + model metrics
    calibration($("#calibChart"), ev.calibration_curve || []);
    const c = mm.chakra || {}, ct = ev.conformal_test || {};
    const g1 = (ev.candidates || []).find(x => x.gamma === 1) || mm.baseline || {};
    kv("#modelKv", [[`AUC served (γ=${ev.selected_gamma ?? "?"}) / rupee-weighted γ=1`, `${(c.auc || 0).toFixed(4)} / ${(g1.auc || 0).toFixed(4)}`], ["PR-AUC served / γ=1", `${(c.pr_auc || 0).toFixed(4)} / ${(g1.pr_auc || 0).toFixed(4)}`],
      ["ECE raw → isotonic (served)", `${(c.ece_raw || 0).toFixed(4)} → ${(c.ece_calibrated || 0).toFixed(4)}`], ["trees served / γ=1 (early-stopped)", `${c.best_iter} / ${g1.best_iter}`],
      ["ONNX parity max |Δp|", mm.onnx ? mm.onnx.parity_max_abs_diff.toExponential(1) : "—"], ["conformal coverage class 0 / 1", `${pct(ct.coverage_class0 || 0)} / ${pct(ct.coverage_class1 || 0)} (target ≥ ${pct(1 - rep.alpha, 0)})`]]);
    // tau hist
    const th = ev.tau_star_hist; columns($("#tauHist"), th.edges, th.counts, { marker: ev.thresholds.global_cost_optimal, markerLabel: `best single cut-off ${ev.thresholds.global_cost_optimal.toFixed(2)}` });
    // certainty
    const cert = ev.certainty; $("#certSub").textContent = `α = ${rep.alpha}`;
    table($("#certTable"), [{ h: "conformal outcome", k: "k" }, { h: "orders", num: 1, f: r => inr.format(cert[r.k].n) }, { h: "share", num: 1, f: r => pct(cert[r.k].n / ev.test_orders) }, { h: "actual RTO rate", num: 1, f: r => pct(cert[r.k].rto_rate) }],
      ["CERTIFIED_LOW", "AMBIGUOUS", "CERTIFIED_HIGH", "NOVEL"].map(k => ({ k })));
    if (lat.inprocess_ms) kv("#latencyKv", [["in-process p50 / p95 / p99", `${lat.inprocess_ms.p50.toFixed(2)} / ${lat.inprocess_ms.p95.toFixed(2)} / ${lat.inprocess_ms.p99.toFixed(2)} ms`], ["HTTP p50 / p99 (TestClient)", `${lat.http_ms.p50.toFixed(2)} / ${lat.http_ms.p99.toFixed(2)} ms`],
      ["ONNX infer p50", `${(lat.stages_ms_p50["score.onnx_infer"] || 0).toFixed(3)} ms`], ["TreeSHAP p50 (auto / always)", `${(lat.stages_ms_p50["score.treeshap"] || 0).toFixed(3)} / ${((lat.stages_always_ms_p50 || {})["score.treeshap"] || 0).toFixed(3)} ms`],
      ["explain=always p50 / p99", lat.inprocess_always_ms ? `${lat.inprocess_always_ms.p50.toFixed(2)} / ${lat.inprocess_always_ms.p99.toFixed(2)} ms` : "—"], ["orders explained under auto", lat.explained_share_auto != null ? pct(lat.explained_share_auto) : "—"],
      ["budget breaches (auto / always)", `${lat.budget_breaches_inprocess} / ${lat.budget_breaches_always ?? "—"} of ${lat.n_inprocess}`]]);
    // friction budget frontier
    const fr = ev.friction_frontier || [], fb = ev.friction_budget || {};
    if (fr.length) {
      const bestBinary = Math.max(P["BASE@GLOBAL_COST"].delta_vs_allow_all, P["BASE@TAU*(x)"].delta_vs_allow_all, P["CHAKRA@TAU*(x)"].delta_vs_allow_all);
      const pts = fr.map(f => ({ ...f, y: f.pnl_total - allow.pnl_total }));
      $("#frontierSub").textContent = `budget ≤ ${pct(fb.budget || 0, 0)} → λ = ₹${fb.lambda} chosen on VALID · test friction share ${pct(fb.test_friction_share || 0)}`;
      frontierChart($("#frontierChart"), pts, { refY: bestBinary, refLabel: "best binary-threshold policy", isBudget: p => p.lambda === fb.lambda,
        tipFn: p => `<b>λ = ₹${p.lambda}</b>friction share ${pct(p.friction_share)}<br>Δ P&L ${fmtR(p.y)}<br>RTO shipped ${inr.format(Math.round(p.rto_shipped))} · good lost ${inr.format(Math.round(p.good_lost))}<br>allow ${p.actions.ALLOW_COD} · step-up ${p.actions.STEP_UP_DEPOSIT} · prepaid ${p.actions.FORCE_PREPAID}` });
      table($("#frontierTable"), [{ h: "λ (₹ per frictioned order)", k: "lambda" }, { h: "friction share", num: 1, f: r => pct(r.friction_share) }, { h: "Δ P&L vs no engine", num: 1, f: r => fmtR(r.y) },
        { h: "RTO shipped", num: 1, f: r => inr.format(Math.round(r.rto_shipped)) }, { h: "good lost", num: 1, f: r => inr.format(Math.round(r.good_lost)) }], pts);
    }
    // model health + shift experiment (live drift polls every 10 s while the console is open)
    renderShift(rep); loadDrift();
    if (!state.driftTimer) state.driftTimer = setInterval(() => { if ($("#view-console").classList.contains("active")) loadDrift(); }, 10000);
    // weight-temperature candidates
    const cands = ev.candidates || [];
    if (cands.length) {
      $("#gammaSub").textContent = `served γ = ${ev.selected_gamma} · chosen by resolver P&L on VALID`;
      table($("#gammaTable"), [{ h: "γ", f: r => r.gamma.toFixed(1) + (r.selected ? " · served" : "") }, { h: "trees", num: 1, k: "best_iter" }, { h: "AUC", num: 1, f: r => r.auc.toFixed(4) },
        { h: "PR-AUC", num: 1, f: r => r.pr_auc.toFixed(4) }, { h: "ECE", num: 1, f: r => r.ece_calibrated.toFixed(4) }, { h: "VALID P&L", num: 1, f: r => fmtR(r.valid_pnl_full) },
        { h: "TEST Δ vs no engine", num: 1, f: r => fmtR(r.test_delta_vs_allow) }], cands);
    }
  }
  // ------------------------------------------------------------------ model health (drift) + shift experiment
  const CERT_COLOR = { CERTIFIED_LOW: "--s1", AMBIGUOUS: "--s4", CERTIFIED_HIGH: "--s3", NOVEL: "--serious" };
  const CERT_SHORT = { CERTIFIED_LOW: "certified low", AMBIGUOUS: "ambiguous {0,1}", CERTIFIED_HIGH: "certified RTO", NOVEL: "empty ∅" };
  async function loadDrift() {
    let d; try { d = await api("/v1/monitor/drift"); } catch (e) { return; }
    const badge = { OK: "good", WARN: "warning", ALERT: "serious", WARMING: "muted" }[d.status] || "muted";
    $("#driftSub").innerHTML = `<span class="cert"><i style="background:var(--${badge === "muted" ? "de" : badge})"></i>${d.status}</span> · ${inr.format(d.rolling_n)} orders in the last hour · PSI ${d.score_psi.toFixed(3)}`;
    $("#driftAlerts").innerHTML = d.alerts.length ? d.alerts.map(a => `<div class="banner ${a.severity === "critical" ? "serious" : "warning"}" style="margin:6px 0"><span class="icon">!</span><div><div class="action">${esc(a.code)}</div><div class="label">${esc(a.message)}</div></div></div>`).join("")
      : (d.status === "WARMING" ? `<p class="muted small">Warming up: fewer than ${d.thresholds.min_n} orders in the window. Score a few checkouts or run <code>04_bench_latency.py</code>.</p>` : `<p class="muted small">No alarms. Live set mix matches the calibration baseline.</p>`);
    $("#driftLegend").innerHTML = Object.keys(CERT_COLOR).map(c => `<span><i style="background:var(${CERT_COLOR[c]})"></i>${CERT_SHORT[c]}</span>`).join("");
    const rows = d.windows.map(w => ({ label: new Date(w.start_ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) + (w.n ? ` (${w.n})` : ""), values: w.share, counts: w.counts }));
    stacked($("#driftChart"), rows, Object.keys(CERT_COLOR), { color: k => CERT_COLOR[k], label: k => CERT_SHORT[k], padL: 96 });
    const b = d.baseline.share, s = d.rolling_share;
    kv("#driftKv", Object.keys(CERT_COLOR).map(c => [`${CERT_SHORT[c]} — live vs calibration`, `${pct(s[c])} vs ${pct(b[c])}`]).concat([
      ["empty sets possible (q₀ + q₁ < 1)", d.baseline.empty_sets_possible ? "yes" : `no — q₀ ${d.baseline.q0.toFixed(3)} + q₁ ${d.baseline.q1.toFixed(3)} > 1; alarm arms itself once the scorer sharpens`],
      ["thresholds", `empty > ${pct(d.thresholds.empty, 0)} · |z| > ${d.thresholds.z} · certified-RTO > ${d.thresholds.mix_ratio}× · PSI > ${d.thresholds.psi}`]]));
  }
  function renderShift(rep) {
    const ds = (rep.extra || {}).domain_shift, gg = (rep.extra || {}).graph_guard;
    if (ds) {
      const names = Object.keys(ds.scenarios);
      $("#shiftSub").textContent = `served model: test AUC ${ds.baseline.test_auc.toFixed(3)} · coverage ${pct(ds.baseline.test_coverage.coverage_class0)} / ${pct(ds.baseline.test_coverage.coverage_class1)}`;
      table($("#shiftTable"), [{ h: "world", k: "name" }, { h: "RTO", num: 1, f: r => pct(r.s.rto_rate) }, { h: "AUC", num: 1, f: r => r.s.auc.toFixed(3) }, { h: "ECE", num: 1, f: r => r.s.ece.toFixed(3) },
        { h: "coverage 0 / 1", num: 1, f: r => `${pct(r.s.coverage_before.coverage_class0)} / ${pct(r.s.coverage_before.coverage_class1)}` },
        { h: "after recal.", num: 1, f: r => `${pct(r.s.coverage_after_recalibration.coverage_class0)} / ${pct(r.s.coverage_after_recalibration.coverage_class1)}` },
        { h: "certified RTO", num: 1, f: r => pct(r.s.certainty_share.CERTIFIED_HIGH) }, { h: "monitor", f: r => `${r.s.monitor_final.status}${r.s.first_alert_after_orders ? " @ " + inr.format(r.s.first_alert_after_orders) + " orders" : ""}` },
        { h: "alarms", f: r => [...new Set(r.s.monitor_final.alerts.map(a => a.code))].map(c => c.replace("MODEL_", "")).join(", ") || "—" }],
        names.map(n => ({ name: n, s: ds.scenarios[n] })));
    } else $("#shiftTable").innerHTML = `<p class="muted small">run scripts/08_festival_shift.py</p>`;
    if (gg) {
      const n = gg.variants.naive, g = gg.variants.guarded || Object.values(gg.variants).slice(-1)[0];
      kv("#guardKv", [["graph guard — naive → guarded components", `${inr.format(n.stats.components)} → ${inr.format(g.stats.components)}`], ["largest component", `${inr.format(n.stats.largest_component)} → ${inr.format(g.stats.largest_component)} nodes`],
        ["ring recall / precision (phone level)", `${pct(n.recall)} / ${pct(n.precision)} → ${pct(g.recall)} / ${pct(g.precision)}`], ["legit phones condemned", `${n.fp} → ${g.fp}`],
        ["hostel / office residents condemned", `${n.residents_condemned} → ${g.residents_condemned} of ${g.legit_residents}`], ["shared entities flagged", `${g.stats.shared_entities} (address ceiling ${g.addr_merge_ceiling} phones)`]]);
    }
  }
  // Frontier: Δ P&L (y) against share of orders frictioned (x); one line, labelled points, reference rule.
  function frontierChart(el, rows, o = {}) {
    const W = widthOf(el), H = 280, padL = 78, padR = 24, padT = 18, padB = 42;
    const pts = rows.slice().sort((a, b) => a.friction_share - b.friction_share);
    const ys = pts.map(p => p.y).concat(o.refY != null ? [o.refY] : []);
    const ymin = Math.min(0, ...ys), ymax = Math.max(1, ...ys) * 1.1;
    const sx = v => padL + v * (W - padL - padR), sy = v => padT + (1 - (v - ymin) / (ymax - ymin || 1)) * (H - padT - padB);
    let s = `<svg viewBox="0 0 ${W} ${H}">`;
    niceTicks(ymax, 4).forEach(t => { s += `<line class="grid" x1="${padL}" x2="${W - padR}" y1="${sy(t)}" y2="${sy(t)}"/><text x="${padL - 6}" y="${sy(t) + 4}" text-anchor="end">${fmtR(t)}</text>`; });
    [0, .25, .5, .75, 1].forEach(t => { s += `<text x="${sx(t)}" y="${H - padB + 16}" text-anchor="${t === 1 ? "end" : t === 0 ? "start" : "middle"}">${pct(t, 0)}</text>`; });
    if (o.refY != null) s += `<line x1="${padL}" x2="${W - padR}" y1="${sy(o.refY)}" y2="${sy(o.refY)}" stroke="var(--axis)" stroke-dasharray="4 3"/><text x="${W - padR}" y="${sy(o.refY) - 5}" text-anchor="end">${esc(o.refLabel || "")} ${fmtR(o.refY)}</text>`;
    s += `<polyline fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round" points="${pts.map(p => sx(p.friction_share) + "," + sy(p.y)).join(" ")}"/>`;
    pts.forEach((p, i) => {
      const hi = o.isBudget && o.isBudget(p), base = p.lambda === 0;
      s += `<g data-i="${i}"><circle cx="${sx(p.friction_share)}" cy="${sy(p.y)}" r="${hi || base ? 6 : 4}" fill="${hi ? "var(--warning)" : "var(--s1)"}" stroke="var(--surface)" stroke-width="2"/><circle class="hit" cx="${sx(p.friction_share)}" cy="${sy(p.y)}" r="14"/></g>`;
      if (hi || base) s += `<text x="${sx(p.friction_share) + (base ? -8 : 10)}" y="${sy(p.y) + (base ? -10 : 22)}" text-anchor="${base ? "end" : "start"}" style="font-weight:500">${base ? "no budget (λ=0)" : `budget point λ=₹${p.lambda}`}</text>`;
    });
    s += `<text x="${(padL + W - padR) / 2}" y="${H - 4}" text-anchor="middle">share of orders frictioned →</text><text x="${padL + 6}" y="${padT + 4}">↑ Δ P&amp;L vs no engine</text></svg>`;
    el.innerHTML = s;
    if (o.tipFn) $$("g[data-i]", el).forEach(g => { g.addEventListener("mousemove", e => showTip(e, o.tipFn(pts[+g.dataset.i]))); g.addEventListener("mouseleave", hideTip); });
  }
  const tile = (l, v, d, up, hero) => `<div class="tile ${hero ? "hero" : ""}"><div class="l">${esc(l)}</div><div class="v">${esc(v)}</div><div class="d ${up ? "up" : ""}">${esc(d)}</div></div>`;

  // ------------------------------------------------------------------ graph
  let sgData = null, sgAnim = null;
  async function loadGraph() {
    const r = await api("/v1/graph/rings?top=40"); state.graphLoaded = true;
    $("#ringStats").textContent = `${inr.format(r.stats.nodes)} nodes · ${inr.format(r.stats.components)} components · ${r.stats.rings} rings · ${inr.format(r.stats.ring_phones)} phones in rings`;
    table($("#ringTable"), [{ h: "ring", f: x => esc(x.ring_id.slice(0, 18)) }, { h: "phones", num: 1, k: "phones" }, { h: "devices", num: 1, k: "devices" }, { h: "addrs", num: 1, k: "addrs" },
      { h: "orders", num: 1, k: "orders" }, { h: "RTO", num: 1, f: x => pct(x.rto_rate, 0) }, { h: "GMV", num: 1, f: x => fmtR(x.gmv) }], r.rings, { click: x => drawSubgraph(x.ring_id) });
    $("#sgLegend").innerHTML = Object.keys(KIND_COLOR).map(k => `<span><i class="${k === "phone" ? "round" : ""}" style="background:var(${KIND_COLOR[k]})"></i>${k} (${KIND_SHAPE[k]})</span>`).join("");
    if (r.rings.length) drawSubgraph(r.rings[0].ring_id);
  }
  async function drawSubgraph(seed) {
    const d = await api(`/v1/graph/subgraph?seed=${encodeURIComponent(seed)}&max_nodes=140`);
    sgData = d; const ring = d.ring || {};
    $("#sgTitle").textContent = `Subgraph · ${seed.slice(0, 22)}`;
    $("#sgSub").textContent = `${d.nodes.length} nodes · ${d.edges.length} edges`;
    kv("#sgRing", [["is ring", ring.is_ring ? "yes" : "no"], ["phones / devices / addresses", `${ring.phones} / ${ring.devices} / ${ring.addrs}`], ["orders", ring.orders], ["RTO rate", pct(ring.rto_rate || 0, 0)], ["GMV", fmtR(ring.gmv || 0)]]);
    forceLayout(d);
  }
  function forceLayout(d) {
    const cv = $("#sgCanvas"), ctx = cv.getContext("2d"), W = cv.width, H = cv.height;
    const idx = new Map(d.nodes.map((n, i) => [n.id, i]));
    const N = d.nodes.map((n, i) => ({ ...n, x: W / 2 + Math.cos(i) * (100 + n.degree), y: H / 2 + Math.sin(i * 1.7) * 100, vx: 0, vy: 0 }));
    const E = d.edges.map(e => [idx.get(e.from), idx.get(e.to)]).filter(e => e[0] != null && e[1] != null);
    let t = 0; if (sgAnim) cancelAnimationFrame(sgAnim);
    const step = () => {
      const k = 18 + 900 / Math.sqrt(N.length + 1);
      for (let i = 0; i < N.length; i++) for (let j = i + 1; j < N.length; j++) { const a = N[i], b = N[j]; let dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy + 0.01; const f = k * k / d2 * 0.9; dx *= f; dy *= f; a.vx += dx; a.vy += dy; b.vx -= dx; b.vy -= dy; }
      for (const [i, j] of E) { const a = N[i], b = N[j]; const dx = b.x - a.x, dy = b.y - a.y, dist = Math.sqrt(dx * dx + dy * dy) + 0.01; const f = (dist - k * 1.4) / dist * 0.05; a.vx += dx * f; a.vy += dy * f; b.vx -= dx * f; b.vy -= dy * f; }
      for (const n of N) { n.vx += (W / 2 - n.x) * 0.004; n.vy += (H / 2 - n.y) * 0.004; n.x += n.vx * 0.5; n.y += n.vy * 0.5; n.vx *= 0.6; n.vy *= 0.6; n.x = Math.max(16, Math.min(W - 16, n.x)); n.y = Math.max(16, Math.min(H - 16, n.y)); }
      draw(); if (++t < 220) sgAnim = requestAnimationFrame(step);
    };
    const draw = (hover = null) => {
      ctx.clearRect(0, 0, W, H); ctx.lineWidth = 1; ctx.strokeStyle = css("--axis");
      for (const [i, j] of E) { ctx.beginPath(); ctx.moveTo(N[i].x, N[i].y); ctx.lineTo(N[j].x, N[j].y); ctx.stroke(); }
      for (const n of N) { const r = 5 + Math.min(9, n.degree * 0.6); ctx.fillStyle = css(KIND_COLOR[n.kind] || "--muted"); ctx.strokeStyle = css("--surface"); ctx.lineWidth = 2; shape(ctx, n.kind, n.x, n.y, r); ctx.fill(); ctx.stroke();
        if (n === hover) { ctx.strokeStyle = css("--ink"); ctx.lineWidth = 2; shape(ctx, n.kind, n.x, n.y, r + 3); ctx.stroke(); } }
    };
    cv.onmousemove = e => { const rect = cv.getBoundingClientRect(), x = (e.clientX - rect.left) * W / rect.width, y = (e.clientY - rect.top) * H / rect.height;
      let best = null, bd = 400; for (const n of N) { const d2 = (n.x - x) ** 2 + (n.y - y) ** 2; if (d2 < bd) { bd = d2; best = n; } }
      draw(best); if (best) showTip(e, `<b>${best.kind} · ${esc(best.id.split(":")[1])}</b>degree ${best.degree} · centrality ${best.centrality}`); else hideTip(); };
    cv.onmouseleave = () => { draw(); hideTip(); };
    step();
  }
  function shape(ctx, kind, x, y, r) {
    ctx.beginPath();
    if (kind === "device") ctx.rect(x - r, y - r, 2 * r, 2 * r);
    else if (kind === "addr") { ctx.moveTo(x, y - r); ctx.lineTo(x + r, y + r); ctx.lineTo(x - r, y + r); ctx.closePath(); }
    else if (kind === "vpa") { ctx.moveTo(x, y - r); ctx.lineTo(x + r, y); ctx.lineTo(x, y + r); ctx.lineTo(x - r, y); ctx.closePath(); }
    else if (kind === "ip") { for (let i = 0; i < 6; i++) { const a = Math.PI / 3 * i; ctx.lineTo(x + r * Math.cos(a), y + r * Math.sin(a)); } ctx.closePath(); }
    else ctx.arc(x, y, r, 0, Math.PI * 2);
  }

  // ------------------------------------------------------------------ dispute
  async function loadDispute() {
    state.disputeLoaded = true;
    const c = await api("/v1/dispute/candidates?n=10");
    table($("#candTable"), [{ h: "transaction", k: "transaction_id" }, { h: "date", k: "date" }, { h: "amount", num: 1, f: r => fmtR(r.amount_inr) }, { h: "prior card txns", num: 1, k: "prior_card_txns" }], c,
      { click: r => { $("#txnId").value = r.transaction_id; compile(); } });
    if (c.length) { $("#txnId").value = c[0].transaction_id; compile(); }
  }
  $("#compileBtn").addEventListener("click", compile);
  async function compile() {
    const body = { transaction_id: $("#txnId").value.trim(), dispute_reason_code: "10.4" }; if ($("#disputeDate").value) body.dispute_date = $("#disputeDate").value;
    const r = await post("/v1/dispute/ce3-compile", body);
    $("#packetHash").textContent = r.packet_hash ? "sha256 " + r.packet_hash.slice(0, 16) + "…" : "";
    const crit = Object.entries(r.criteria || {});
    const ev = r.evidence || {};
    $("#ce3Result").className = ""; $("#ce3Result").innerHTML =
      `<div class="banner ${r.eligible ? "good" : "serious"}"><span class="icon">${r.eligible ? "✓" : "✕"}</span><div><div class="action">${r.eligible ? "CE3.0 ELIGIBLE — liability shift" : "NOT ELIGIBLE"}</div><div class="label">${esc(r.reason)}</div></div></div>
       <h3 class="mt">Criteria</h3>` + crit.map(([k, v]) => `<div class="crit"><i class="${v.pass ? "ok" : "fail"}">${v.pass ? "✓" : "✕"}</i><span>${esc(k.replace(/_/g, " "))}${v.count != null ? ` · ${v.count}` : ""}${v.window ? ` · window ${v.window[0]} → ${v.window[1]}` : ""}${v.value ? ` · ${v.value}` : ""}</span></div>`).join("") +
      (ev.disputed_transaction ? `<h3 class="mt">Disputed transaction</h3><div class="txn"><b>${esc(ev.disputed_transaction.transaction_id)}</b> · ${ev.disputed_transaction.date} · ${fmtR(ev.disputed_transaction.amount_inr)} · ${esc(ev.disputed_transaction.items)}<br>dispute date ${ev.dispute_date}</div>` : "") +
      (ev.prior_transactions && ev.prior_transactions.length ? `<h3 class="mt">Qualifying prior transactions</h3>` + ev.prior_transactions.map(t => `<div class="txn"><b>${esc(t.transaction_id)}</b> · ${t.date} (${t.days_before_dispute} days before) · ${fmtR(t.amount_inr)} · ${t.delivered ? "delivered" : "returned"}<br>matched: ${t.matched_elements.map(e => `<span class="el">${e} = ${esc(t.element_hashes[e].slice(0, 10))}…</span>`).join("")}</div>`).join("") : "") +
      (ev.merchant_narrative ? `<h3 class="mt">Narrative</h3><p class="small">${esc(ev.merchant_narrative)}</p>` : "");
    $("#ce3Json").textContent = JSON.stringify(r, null, 2);
  }

  loadScenarios();
})();
