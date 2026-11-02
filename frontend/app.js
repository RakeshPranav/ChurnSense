const API = "http://127.0.0.1:8000";
const $ = (id) => document.getElementById(id);
const ARC_LEN = Math.PI * 80; // length of the semicircle gauge

let schema = null;
const inputs = {};   // column -> { get(), set(v) }
const history = [];

/* ---------- helpers ---------- */
const pretty = (s) =>
  s.replace(/_/g, " ").replace(/([a-z])([A-Z])/g, "$1 $2").replace(/\b\w/g, (c) => c.toUpperCase());
const fmt = (n) => (Number.isInteger(n) ? String(n) : (Math.round(n * 100) / 100).toString());
const clamp = (v, lo, hi) => Math.min(Math.max(v, lo), hi);

function toast(msg) {
  const t = $("toast");
  if (!msg) { t.hidden = true; return; }
  t.textContent = msg;
  t.hidden = false;
}

/* ---------- theme ---------- */
function setTheme(t) {
  document.documentElement.dataset.theme = t;
  try { localStorage.setItem("churnsense-theme", t); } catch {}
}
(function initTheme() {
  let saved = null;
  try { saved = localStorage.getItem("churnsense-theme"); } catch {}
  setTheme(saved || "light");
$("themeBtn").addEventListener("click", () =>
  setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));

/* ---------- API status ---------- */
function setApi(ok) {
  const pill = $("apiPill");
  pill.classList.toggle("ok", ok);
  pill.classList.toggle("down", !ok);
  $("apiText").textContent = ok ? "API online" : "API offline";
}
async function checkApi() {
  try {
    const r = await fetch(`${API}/health`);
    setApi(r.ok);
  } catch { setApi(false); }
}
checkApi();
setInterval(checkApi, 5000);

/* ---------- build the form from /schema ---------- */
function numericField(col, info) {
  const wrap = document.createElement("div");
  const step = info.is_int ? 1 : Math.max((info.max - info.min) / 200, 0.01);

  const head = document.createElement("div");
  head.className = "field-head";
  const label = document.createElement("label");
  label.textContent = pretty(col);
  label.htmlFor = col;
  const num = document.createElement("input");
  num.type = "number"; num.className = "num"; num.id = col;
  num.min = info.min; num.max = info.max; num.step = info.is_int ? 1 : "any";
  head.append(label, num);

  const range = document.createElement("input");
  range.type = "range"; range.min = info.min; range.max = info.max; range.step = step;

  const meta = document.createElement("div");
  meta.className = "range-meta";
  meta.innerHTML = `<span>${fmt(info.min)}</span><span>${fmt(info.max)}</span>`;

  const set = (v) => {
    v = clamp(Number(v), info.min, info.max);
    if (Number.isNaN(v)) v = info.median;
    if (info.is_int) v = Math.round(v);
    num.value = fmt(v);
    range.value = v;
  };
  range.addEventListener("input", () => set(range.value));
  num.addEventListener("change", () => set(num.value));

  inputs[col] = { get: () => Number(num.value), set, reset: () => set(info.median),
                  random: () => set(info.min + Math.random() * (info.max - info.min)) };
  inputs[col].reset();
  wrap.append(head, range, meta);
  return wrap;
}

function categoricalField(col, options) {
  const wrap = document.createElement("div");
  const label = document.createElement("div");
  label.className = "lbl"; label.textContent = pretty(col);
  label.style.marginBottom = "8px";
  wrap.appendChild(label);

  if (options.length <= 4) {
    const seg = document.createElement("div");
    seg.className = "seg"; seg.setAttribute("role", "radiogroup");
    let value = options[0];
    const buttons = options.map((opt) => {
      const b = document.createElement("button");
      b.type = "button"; b.textContent = opt;
      b.addEventListener("click", () => set(opt));
      seg.appendChild(b);
      return b;
    });
    const set = (v) => {
      value = options.includes(v) ? v : options[0];
      buttons.forEach((b) => b.classList.toggle("on", b.textContent === value));
    };
    inputs[col] = { get: () => value, set, reset: () => set(options[0]),
                    random: () => set(options[Math.floor(Math.random() * options.length)]) };
    set(options[0]);
    wrap.appendChild(seg);
  } else {
    const sel = document.createElement("select");
    sel.id = col;
    options.forEach((o) => {
      const opt = document.createElement("option");
      opt.value = o; opt.textContent = o;
      sel.appendChild(opt);
    });
    inputs[col] = { get: () => sel.value, set: (v) => (sel.value = options.includes(v) ? v : options[0]),
                    reset: () => (sel.selectedIndex = 0),
                    random: () => (sel.selectedIndex = Math.floor(Math.random() * options.length)) };
    wrap.appendChild(sel);
  }
  return wrap;
}

async function loadSchema() {
  try {
    const res = await fetch(`${API}/schema`);
    if (!res.ok) throw new Error();
    schema = await res.json();
    $("fields").innerHTML = "";
    schema.columns.forEach((col) => {
      $("fields").appendChild(schema.num[col] ? numericField(col, schema.num[col])
                                              : categoricalField(col, schema.cat[col]));
    });
    $("loadingForm").hidden = true;
    toast("");
  } catch {
    $("loadingForm").hidden = true;
    toast("Can't reach the API. Start the backend: python -m uvicorn api:app --port 8000");
    $("submit").disabled = true;
    setTimeout(loadSchema, 4000); // retry automatically
  }
}

/* ---------- gauge ---------- */
function placeThreshold(t) {
  const theta = Math.PI * (1 - t);
  const x = (r) => 100 + r * Math.cos(theta);
  const y = (r) => 100 - r * Math.sin(theta);
  const line = $("thrLine");
  line.setAttribute("x1", x(68)); line.setAttribute("y1", y(68));
  line.setAttribute("x2", x(94)); line.setAttribute("y2", y(94));
}

function countUp(el, target, ms = 800) {
  const start = performance.now();
  const tick = (now) => {
    const p = Math.min((now - start) / ms, 1);
    const eased = 1 - Math.pow(1 - p, 3);
    el.textContent = (target * eased).toFixed(1);
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function render({ score, threshold, churn }) {
  $("empty").hidden = true;
  $("result").hidden = false;

  const arc = $("arc");
  arc.style.stroke = churn ? "var(--bad)" : "var(--good)";
  arc.style.strokeDasharray = `0 ${ARC_LEN + 10}`;
  requestAnimationFrame(() => requestAnimationFrame(() => {
    arc.style.strokeDasharray = `${ARC_LEN * score} ${ARC_LEN + 10}`;
  }));

  placeThreshold(threshold);
  countUp($("scoreNum"), score * 100);

  const v = $("verdict");
  v.textContent = churn ? "Likely to churn" : "Likely to stay";
  v.className = "verdict " + (churn ? "churn" : "stay");

  const gap = (score - threshold) * 100;
  $("fScore").textContent = (score * 100).toFixed(1) + "%";
  $("fThr").textContent = (threshold * 100).toFixed(1) + "%";
  $("fGap").textContent = (gap >= 0 ? "+" : "") + gap.toFixed(1) + " pts";
  $("explain").textContent = churn
    ? `The score is ${Math.abs(gap).toFixed(1)} points above the decision threshold.`
    : `The score is ${Math.abs(gap).toFixed(1)} points below the decision threshold.`;
}

/* ---------- history ---------- */
function addHistory(entry) {
  history.unshift(entry);
  if (history.length > 6) history.pop();
  const ul = $("history");
  ul.innerHTML = "";
  history.forEach((h) => {
    const li = document.createElement("li");
    li.innerHTML = `<span><i class="dot ${h.churn ? "churn" : "stay"}"></i>${h.churn ? "Churn" : "Stay"}</span>
                    <span class="muted">${(h.score * 100).toFixed(1)}% · ${h.time}</span>`;
    li.title = "Click to reload these inputs";
    li.addEventListener("click", () => {
      Object.entries(h.values).forEach(([k, v]) => inputs[k] && inputs[k].set(v));
      render(h);
    });
    ul.appendChild(li);
  });
  $("historyWrap").hidden = false;
}

/* ---------- actions ---------- */
$("resetBtn").addEventListener("click", () => Object.values(inputs).forEach((i) => i.reset()));
$("randBtn").addEventListener("click", () => Object.values(inputs).forEach((i) => i.random()));

$("form").addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!schema) return;
  toast("");
  const btn = $("submit");
  btn.disabled = true;
  btn.querySelector(".spin").hidden = false;
  $("submitText").textContent = "Predicting…";

  const values = {};
  schema.columns.forEach((c) => (values[c] = inputs[c].get()));

  try {
    const res = await fetch(`${API}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(typeof data.detail === "string" ? data.detail : "Prediction failed");
    render(data);
    addHistory({ ...data, values, time: new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) });
    if (window.innerWidth < 880) $("resultCard").scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (err) {
    toast(err.message || "Something went wrong");
  } finally {
    btn.disabled = false;
    btn.querySelector(".spin").hidden = true;
    $("submitText").textContent = "Predict churn";
  }
});

loadSchema();