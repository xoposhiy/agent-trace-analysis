// Claude Split Advisor dashboard — vanilla JS, no build step.
// Loads cached analyses instantly; Refresh triggers the incremental (new-only)
// re-analysis on the server. The centrepiece is the task-forest timeline strip.

// The eight categorical palette slots (CSS custom properties) used to colour tasks.
const SERIES_VARS = ["--series-1", "--series-2", "--series-3", "--series-4",
                     "--series-5", "--series-6", "--series-7", "--series-8"];

let allSessions = [];   // every analysed session dict currently loaded from the server

// ---- tiny formatting helpers ----
const byId = (elementId) => document.getElementById(elementId);
const fmtMoney = (dollars) => "$" + (dollars || 0).toFixed(2);
const fmtTokens = (tokens) => (tokens ? Math.round(tokens / 1000) + "k" : "—");

const DEFAULT_SWECHAT_REPO = "SALT-NLP/SWE-chat";
const DEFAULT_SWECHAT_SPLIT = "train";
const DEFAULT_SWECHAT_REPO_FILTER = "entireio/cli";

// Map a candidate `source` to a friendly badge label + CSS class, so the
// task-switch vs sub-agent classification is explicit on every split point.
const SOURCE_BADGES = {
  "task-switch": { label: "Independent task switch", cls: "k-switch" },
  "sub-agent":   { label: "Sub-agent", cls: "k-subagent" },
  "plan-mode":   { label: "Plan mode", cls: "k-plan" },
};
const badgeFor = (source) => SOURCE_BADGES[source] || { label: source, cls: "k-other" };

// A hierarchical id's top-level task: "T1.2" -> "T1".
const topLevelId = (taskId) => String(taskId).split(".")[0];

// ---- "Sort by" filter: how the suggestion list is ordered. Each comparator sorts
// {session, chosen} rows; primary key first, the other saving metric as tie-break. ----
const SORTERS = {
  dollars: (a, b) => b.chosen.dollars - a.chosen.dollars || b.chosen.pct - a.chosen.pct,
  pct:     (a, b) => b.chosen.pct - a.chosen.pct || b.chosen.dollars - a.chosen.dollars,
};

// ---- tabs: semantic splits vs plan-mode (two different kinds of advice) ----
let activeTab = "splits";                                  // "splits" | "planmode"
const SEMANTIC_SOURCES = new Set(["task-switch", "sub-agent"]);

// the candidates a tab cares about, and the best (highest $, then %) among them
function tabCandidates(session, tab) {
  const candidates = session.candidates || [];
  return tab === "planmode"
    ? candidates.filter((c) => c.source === "plan-mode")
    : candidates.filter((c) => SEMANTIC_SOURCES.has(c.source));
}
function bestCandidate(candidates) {
  return candidates.slice()
    .sort((a, b) => b.dollars - a.dollars || b.pct - a.pct)[0] || null;
}

// concrete colour for a split marker in the SVG graph, by candidate source
const MARKER_COLOR = {
  "task-switch": "var(--series-1)",   // blue
  "sub-agent":   "var(--series-6)",   // orange
  "plan-mode":   "var(--series-7)",   // violet
};

// structural activity phases (plan-mode view): colour + friendly name per category
const PHASE_STYLE = {
  exploration:  { color: "var(--series-1)", label: "reading" },       // blue
  editing:      { color: "var(--series-2)", label: "editing" },       // green
  execution:    { color: "var(--series-6)", label: "running" },       // orange
  coordination: { color: "var(--series-other)", label: "coordination" },
};
const phaseStyle = (category) => PHASE_STYLE[category] || PHASE_STYLE.coordination;

// ---- colour a task id by its TOP-LEVEL task; children share the parent hue, lighter ----
// A palette slot is assigned per top-level task in first-appearance order (fixed, never
// cycled; a 9th folds to "Other"). Children (T1.1) reuse the parent's slot mixed toward
// the surface, so they read as "part of T1".
function taskColorMap(forest) {
  const topLevels = [];
  (forest?.tasks || []).forEach((task) => {
    const parent = topLevelId(task.id);
    if (!topLevels.includes(parent)) topLevels.push(parent);
  });
  const slotColor = (parent) => {
    const slot = topLevels.indexOf(parent);
    return slot >= 0 && slot < SERIES_VARS.length
      ? `var(${SERIES_VARS[slot]})` : "var(--series-other)";
  };
  const colorByTaskId = {};
  (forest?.tasks || []).forEach((task) => {
    const parentColor = slotColor(topLevelId(task.id));
    colorByTaskId[task.id] = task.id.includes(".")
      ? `color-mix(in oklab, ${parentColor} 55%, var(--surface-1))`   // child: lighter parent hue
      : parentColor;
  });
  return colorByTaskId;
}

function taskLabel(forest, taskId) {
  const task = (forest?.tasks || []).find((candidate) => candidate.id === taskId);
  return task ? task.label : "";
}

// ---- the task-forest timeline strip: a categorical stacked bar ----
function renderStrip(forest) {
  const runs = forest?.timeline || [];        // [ [taskId, startMsg, endMsg], ... ]
  if (!runs.length) return "";
  const colorByTaskId = taskColorMap(forest);
  const totalMessages = runs.reduce((sum, [, startMsg, endMsg]) => sum + (endMsg - startMsg + 1), 0) || 1;

  const segments = runs.map(([taskId, startMsg, endMsg]) => {
    const messageSpan = endMsg - startMsg + 1;
    const widthPct = (100 * messageSpan / totalMessages).toFixed(3);
    const color = colorByTaskId[taskId] || "var(--series-other)";
    const tooltip = `${taskId} · ${taskLabel(forest, taskId)} · user prompts ${startMsg}–${endMsg}`;
    // the task id is printed on the segment, so identity is never colour-alone (relief rule)
    return `<div class="seg" style="width:${widthPct}%;background:${color}" ` +
           `title="${escapeHtml(tooltip)}">${taskId}</div>`;
  }).join("");

  return `<div class="strip">${segments}</div>`;
}

function renderLegend(forest) {
  const colorByTaskId = taskColorMap(forest);
  const items = (forest?.tasks || []).map((task) => {
    const isChild = task.id.includes(".");     // T1.1 -> render indented as a nested child
    return `<span class="item${isChild ? " child" : ""}">` +
      `<span class="dot" style="background:${colorByTaskId[task.id]}"></span>` +
      `<b>${task.id}</b>&nbsp;${escapeHtml(task.label || "")}</span>`;
  }).join("");
  return items ? `<div class="legend">${items}</div>` : "";
}

// ---- interactive task tree: one row per INDEPENDENT (top-level) task with its
// one-line summary; rows with sub-tasks (T1.x) are expandable to reveal them. ----
function renderTaskTree(forest) {
  const tasks = forest?.tasks || [];
  if (!tasks.length) return "";
  const colorByTaskId = taskColorMap(forest);
  const tops = tasks.filter((t) => !t.id.includes("."));      // T1, T2, ... (no dot)

  const rows = tops.map((top) => {
    const kids = tasks.filter((t) => t.id.includes(".") && topLevelId(t.id) === top.id);
    const caret = `<span class="tt-caret${kids.length ? "" : " tt-none"}">▸</span>`;
    const count = kids.length
      ? `<span class="tt-count">${kids.length} sub-task${kids.length > 1 ? "s" : ""}</span>` : "";
    const head = `<div class="tt-row${kids.length ? " tt-expandable" : ""}" data-task="${top.id}">` +
      caret +
      `<span class="dot" style="background:${colorByTaskId[top.id]}"></span>` +
      `<b>${top.id}</b><span class="tt-label">${escapeHtml(top.label || "")}</span>${count}</div>`;
    const children = kids.length
      ? `<div class="tt-children" hidden>` + kids.map((c) =>
          `<div class="tt-child"><span class="dot" style="background:${colorByTaskId[c.id]}"></span>` +
          `<b>${c.id}</b><span class="tt-label">${escapeHtml(c.label || "")}</span></div>`).join("") +
        `</div>`
      : "";
    return head + children;
  }).join("");
  return `<div class="task-tree">${rows}</div>`;
}

// task-colored slices of the forest whose [f_start, f_end] overlap [lo, hi], rescaled
// to fill their row. Falls back to one neutral fill when the session has no forest.
function forestSlice(forest, lo, hi) {
  const spans = forest?.spans || [];
  const width = (hi - lo) || 1;
  if (!spans.length) return `<div class="pv-seg" style="width:100%;background:var(--series-1)"></div>`;
  const colorByTaskId = taskColorMap(forest);
  return spans.filter((s) => s.f_end > lo && s.f_start < hi).map((s) => {
    const a = Math.max(s.f_start, lo), b = Math.min(s.f_end, hi);
    const w = (100 * (b - a) / width).toFixed(2);
    return `<div class="pv-seg" style="width:${w}%;background:${colorByTaskId[s.id] || "var(--series-other)"}"></div>`;
  }).join("");
}

// ---- the "best split" preview. Two shapes, one per cost model:
//   • task-switch / plan-mode → a two-SESSION split (cut into two shorter sessions);
//   • sub-agent → the excise-and-rejoin model: a middle segment is lifted OUT into an
//     isolated sub-agent and the main thread rejoins carrying only a small summary.
function renderBestSplitPreview(session, chosen) {
  const forest = session.task_forest || {};
  const clamp = (v) => Math.min(Math.max(v || 0, 0), 1);
  const save = `<div class="pv-save">−${Math.round(chosen.pct)}% cost · ${fmtMoney(chosen.dollars)}</div>`;

  if (chosen.source === "sub-agent" && chosen.split_end_fraction != null) {
    const a = clamp(chosen.split_fraction), b = clamp(chosen.split_end_fraction);
    const beforeW = (100 * a).toFixed(2), segW = (100 * (b - a)).toFixed(2);
    return `
    <div class="split-preview">
      <div class="pv-full">${forestSlice(forest, 0, 1)}
        <span class="pv-brace" style="left:${beforeW}%;width:${segW}%"></span></div>
      <div class="pv-arrow-row"><span class="pv-arrow" style="left:${(100 * (a + b) / 2).toFixed(2)}%">↓</span></div>
      <div class="pv-chunks">
        <div class="pv-row" style="width:100%">
          <div class="pv-bar" style="flex:0 0 ${beforeW}%">${forestSlice(forest, 0, a)}</div>
          <span class="pv-summary" title="summary the sub-agent hands back"></span>
          <div class="pv-bar" style="flex:1 1 auto">${forestSlice(forest, b, 1)}</div>
        </div>
        <div class="pv-row pv-subrow" style="width:max(${segW}%, 40px);margin-left:${beforeW}%">
          <div class="pv-bar pv-sub">${forestSlice(forest, a, b)}</div>
        </div>
      </div>
      ${save}
      <div class="pv-cap">The highlighted segment runs in a sub-agent (bottom bar); the main thread continues without its context, keeping only a summary.</div>
    </div>`;
  }

  // task-switch / plan-mode: split the one long session into two shorter ones
  const frac = clamp(chosen.split_fraction);
  const aWidth = (100 * frac).toFixed(2), bWidth = (100 * (1 - frac)).toFixed(2);
  return `
    <div class="split-preview">
      <div class="pv-full">${forestSlice(forest, 0, 1)}</div>
      <div class="pv-arrow-row"><span class="pv-arrow" style="left:${aWidth}%">↓</span></div>
      <div class="pv-chunks">
        <div class="pv-row" style="width:max(${aWidth}%, 40px)">
          <div class="pv-bar">${forestSlice(forest, 0, frac)}</div>
          <span class="pv-summary" title="carried summary (context handed to the next session)"></span>
        </div>
        <div class="pv-row" style="width:max(${bWidth}%, 40px)">
          <span class="pv-summary" title="carried summary (context handed to the next session)"></span>
          <div class="pv-bar">${forestSlice(forest, frac, 1)}</div>
        </div>
      </div>
      ${save}
      <div class="pv-cap">Modelled cost after splitting into two sessions — the ↓ marks where to split.</div>
    </div>`;
}

// ---- one session card: header, forest strip, legend, badges, best-split preview ----
// `chosen` is the single best split (by $) for the active tab; the card shows just that
// one, and "See full analysis" opens the graph + every considered split point.
function renderCard(session, chosen) {
  const forest = session.task_forest;
  const chosenBadge = badgeFor(chosen.source);
  const parts = [];

  parts.push(`<div class="row1">
      <span class="sid">${session.session_id.slice(0, 8)}</span>
      <span class="proj">${escapeHtml(shortProjectName(session.project))}</span>
      <span class="badge ${chosenBadge.cls}">${chosenBadge.label}</span>
      <span class="save">${fmtMoney(chosen.dollars)} · ${Math.round(chosen.pct)}%</span>
      <button class="graph-btn" data-session="${session.session_id}">See full analysis</button>
    </div>`);
  if (session.task_summary) {
    parts.push(`<div class="summary">${escapeHtml(session.task_summary)}</div>`);
  }

  parts.push(renderStrip(forest));
  parts.push(renderTaskTree(forest));

  // interleaving badge — a top-level task the user returned to
  if (forest?.recurring?.length) {
    parts.push(`<div class="badges"><span class="badge interleave">⚠ Interleaving: returned to ${
      forest.recurring.join(", ")} — likely separate sessions</span></div>`);
  }

  // the single best split, shown as the sketch-style preview + its one-line label
  parts.push(`<div class="best-split-h">Best split <span class="badge ${badgeFor(chosen.source).cls}">${
    badgeFor(chosen.source).label}</span> <span class="bs-label">${escapeHtml(chosen.label)}</span></div>`);
  parts.push(renderBestSplitPreview(session, chosen));

  // meta line (session stats only; per-split detail lives in "See full analysis")
  const meta = [
    `${session.turns} agent steps`,
    `peak ~${fmtTokens(session.peak_context)}`,
    `as-is ${fmtMoney(session.as_is_cost)}`,
  ];
  parts.push(`<div class="meta">${meta.join(" &nbsp;·&nbsp; ")}</div>`);

  return `<div class="card">${parts.join("")}</div>`;
}

// ---- plan-mode view: a structural phase bar (reading / editing / running) ----
// Positioned in TURN space (0 → seq_turns); gaps between phases are the track colour.
function renderPhaseBar(session) {
  const phases = session.phases || [];
  if (!phases.length) {
    return `<div class="tip">No distinct reading/editing phases detected in this session.</div>`;
  }
  const total = session.seq_turns || Math.max(...phases.map((p) => p.end_turn), 1);
  const segments = phases.map((phase) => {
    const left = 100 * (phase.start_turn - 1) / total;
    const width = Math.max(100 * (phase.end_turn - phase.start_turn + 1) / total, 0.5);
    const style = phaseStyle(phase.category);
    const tip = `${style.label} · agent steps ${phase.start_turn}–${phase.end_turn}`;
    return `<div class="phase-seg" style="left:${left}%;width:${width}%;background:${style.color}" ` +
      `title="${escapeHtml(tip)}">${width > 10 ? style.label : ""}</div>`;
  }).join("");
  return `<div class="phasebar">${segments}</div>`;
}

function renderPhaseLegend() {
  const items = Object.values(PHASE_STYLE).map((style) =>
    `<span class="item"><span class="dot" style="background:${style.color}"></span>${style.label}</span>`).join("");
  return `<div class="legend">${items}</div>`;
}

// A plan-mode card looks DIFFERENT from a split card: it shows the reading/editing
// phase bar (not the task forest), because plan-mode is about the opening reading phase.
function renderPlanModeCard(session, chosen) {
  const parts = [];
  parts.push(`<div class="row1">
      <span class="sid">${session.session_id.slice(0, 8)}</span>
      <span class="proj">${escapeHtml(shortProjectName(session.project))}</span>
      <span class="badge k-plan">Plan mode</span>
      <span class="save">${fmtMoney(chosen.dollars)} · ${Math.round(chosen.pct)}%</span>
      <button class="graph-btn" data-session="${session.session_id}">See full analysis</button>
    </div>`);
  if (session.task_summary) {
    parts.push(`<div class="summary">${escapeHtml(session.task_summary)}</div>`);
  }

  parts.push(renderPhaseBar(session));
  parts.push(renderPhaseLegend());

  // highlight the opening reading phase's turn range (what plan mode would absorb)
  const reading = (session.phases || []).find((p) => p.category === "exploration");
  if (reading) {
    parts.push(`<div class="badges"><span class="badge k-plan">Opening reading phase: ` +
      `agent steps ${reading.start_turn}–${reading.end_turn}</span></div>`);
  }

  const meta = [
    `${session.turns} agent steps`,
    `peak ~${fmtTokens(session.peak_context)}`,
    `as-is ${fmtMoney(session.as_is_cost)}`,
  ];
  parts.push(`<div class="meta">${meta.join(" &nbsp;·&nbsp; ")}<br>` +
    `Running that opening reading in <b>plan mode</b> (a cheaper model doing the reading) ` +
    `would save ~${fmtMoney(chosen.dollars)} (${Math.round(chosen.pct)}%).</div>`);

  return `<div class="card">${parts.join("")}</div>`;
}

// ---- filtering + rendering the active tab's list ----
function render() {
  const projectFilter = byId("fProject").value;
  const minPct = parseFloat(byId("fPct").value);
  const minDollars = parseFloat(byId("fDollars").value);
  byId("lblPct").textContent = minPct;
  byId("lblDollars").textContent = minDollars.toFixed(2);

  const scanned = allSessions.filter((session) =>
    !projectFilter || session.project.includes(projectFilter));

  // for the active tab, pick each session's best relevant candidate and keep the
  // sessions that clear both thresholds; order by the chosen "Sort by" filter.
  const suggested = [];
  scanned.forEach((session) => {
    const chosen = bestCandidate(tabCandidates(session, activeTab));
    if (chosen && session.modelled
        && chosen.pct >= minPct && chosen.dollars >= minDollars) {
      suggested.push({ session, chosen });
    }
  });
  suggested.sort(SORTERS[byId("fSort").value] || SORTERS.dollars);

  // tiles: scanned + total cost are global; worth + saving reflect the active tab
  byId("tScanned").textContent = scanned.length;
  byId("tWorth").textContent = suggested.length;
  byId("tCost").textContent = fmtMoney(scanned.reduce((sum, s) => sum + s.as_is_cost, 0));
  byId("tSave").textContent = fmtMoney(suggested.reduce((sum, r) => sum + r.chosen.dollars, 0));

  const list = byId("list");
  if (!allSessions.length) {
    list.innerHTML = `<div class="empty">No analysis yet. Press <b>Refresh</b> to analyse your sessions.` +
      `<div class="tip">First run analyses everything; later runs only touch new/changed sessions.</div></div>`;
  } else if (!suggested.length) {
    const what = activeTab === "planmode" ? "plan-mode opportunity" : "split suggestion";
    list.innerHTML = `<div class="empty">No ${what} clears the current thresholds. Lower the sliders to see more.</div>`;
  } else {
    // plan-mode uses a phase-bar card; splits use the task-forest card
    const renderer = activeTab === "planmode" ? renderPlanModeCard : renderCard;
    list.innerHTML = suggested.map((r) => renderer(r.session, r.chosen)).join("");
  }
}

function populateProjectFilter() {
  const names = [...new Set(allSessions.map((session) => shortProjectName(session.project)))].sort();
  const select = byId("fProject");
  const current = select.value;
  select.innerHTML = `<option value="">All projects</option>` +
    names.map((name) => `<option value="${escapeHtml(name)}">${escapeHtml(name)}</option>`).join("");
  select.value = current;
}

function updateSourceControls() {
  const source = byId("sourceSelect").value;
  document.querySelectorAll(".swe-chat-only").forEach((node) => {
    node.hidden = source !== "swe-chat";
  });
}

// ---- server calls ----
async function loadCachedSessions() {
  const response = await fetch("/api/sessions");
  const data = await response.json();
  allSessions = data.sessions || [];
  populateProjectFilter();
  render();
  byId("status").textContent = allSessions.length ? `${allSessions.length} sessions cached` : "";
}

async function refreshSessions() {
  const button = byId("refresh");
  button.disabled = true;
  const useLlm = byId("useLlm").checked;
  const source = byId("sourceSelect").value;
  const repoFilter = byId("repoFilter").value.trim() || DEFAULT_SWECHAT_REPO_FILTER;
  const datasetSplit = byId("datasetSplit").value.trim() || DEFAULT_SWECHAT_SPLIT;
  byId("status").textContent = "Analysing new / changed sessions… (first run can take a minute)";
  try {
    const params = new URLSearchParams({
      use_llm: String(useLlm),
      source,
      dataset_repo: DEFAULT_SWECHAT_REPO,
      dataset_split: datasetSplit,
      project_filter: source === 'swe-chat' ? repoFilter : "",
    });
    const response = await fetch(`/api/refresh?${params.toString()}`, { method: "POST" });
    const data = await response.json();
    allSessions = data.sessions || [];
    populateProjectFilter();
    render();
    const stats = data.stats || {};
    byId("status").textContent =
      `Analysed ${stats.analyzed} new, reused ${stats.reused} of ${stats.total} cached.`;
    showLlmBanner(stats.llm);
  } catch (error) {
    byId("status").textContent = "Refresh failed: " + error;
  } finally {
    button.disabled = false;
  }
}

// Show, on screen, whether the LLM was actually used this refresh — or why it
// silently fell back to structural-only (missing key, wrong endpoint/model, ...).
function showLlmBanner(llm) {
  const banner = byId("llmBanner");
  if (!banner) return;
  if (!llm) { banner.hidden = true; return; }
  banner.hidden = false;
  let cls, text;
  if (!llm.requested) {
    cls = "off";
    text = "LLM off (structural-only). Tick “Use LLM” to enable task summaries + the task forest.";
  } else if (llm.active) {
    cls = "ok";
    text = llm.judged
      ? `LLM active — ${llm.judged} session(s) judged this refresh.`
      : "LLM active — no new session needed a judge call (all cached or too short).";
  } else {
    cls = "bad";
    text = "LLM NOT used — " + (llm.status || "unknown reason") + ".";
    const cfg = llm.config;
    if (cfg && cfg.reasons && cfg.reasons.length) {
      text += " " + cfg.reasons.join(" ");
    }
  }
  banner.className = "llm-banner " + cls;
  banner.textContent = text;
}

// ---- helpers ----
function shortProjectName(project) {
  // the encoded project dir is a long dash-joined path; show just the tail
  return (project || "").replace(/^-+/, "").split("-").slice(-2).join("-") || project;
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (char) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
}

// candidates in the order the modal lists them (best $ first); the selected index
// into THIS array drives which split point's marker + preview are emphasised.
function modalCandidates(session) {
  return (session.candidates || []).slice()
    .sort((a, b) => b.dollars - a.dollars || b.pct - a.pct);
}

// ---- per-session graph (SVG in a modal) ----
// X = session progress 0→1 (turn space, the cost axis); Y = context tokens (peak).
// Draws the modelled context ramp, colored task bands, and a marker per split point;
// `selIdx` is the split point (index into modalCandidates) currently selected, which
// is emphasised on the graph and previewed below.
function renderSessionGraph(session, selIdx = 0) {
  const peak = session.peak_context || 0;
  const cands = modalCandidates(session);

  // choose what to draw by the active tab: plan-mode shows structural PHASES + the
  // plan-mode marker; splits show the task FOREST bands + task-switch/sub-agent markers.
  let bands, legendHtml;
  if (activeTab === "planmode") {
    const total = session.seq_turns || 1;
    bands = (session.phases || []).map((phase) => ({
      f_start: (phase.start_turn - 1) / total,
      f_end: phase.end_turn / total,
      color: phaseStyle(phase.category).color,
      label: phaseStyle(phase.category).label,
    }));
    legendHtml = renderPhaseLegend();
  } else {
    const forest = session.task_forest || {};
    const colorByTaskId = taskColorMap(forest);
    bands = (forest.spans || []).map((span) => ({
      f_start: span.f_start, f_end: span.f_end,
      color: colorByTaskId[span.id] || "var(--series-other)", label: span.id,
    }));
    legendHtml = renderTaskTree(session.task_forest || {});
  }
  // one marker per considered split point (same order as the list below), coloured by
  // source; the selected one is drawn solid + bold, the rest faded, so clicking a row
  // highlights its position on the cost ramp.
  const markers = cands.map((c, i) => ({
    frac: c.split_fraction, color: MARKER_COLOR[c.source] || "var(--muted)",
    dollars: c.dollars, selected: i === selIdx,
  }));

  const W = 720, H = 320;
  const marginLeft = 60, marginRight = 18, marginTop = 26, marginBottom = 74;
  const plotLeft = marginLeft, plotRight = W - marginRight;
  const plotTop = marginTop, baseline = H - marginBottom;   // baseline = context 0
  const x = (frac) => plotLeft + frac * (plotRight - plotLeft);
  const y = (tokens) => baseline - (peak ? (tokens / peak) * (baseline - plotTop) : 0);

  const svg = [];
  // axes + ticks + labels
  svg.push(`<line x1="${plotLeft}" y1="${baseline}" x2="${plotRight}" y2="${baseline}" class="axis"/>`);
  svg.push(`<line x1="${plotLeft}" y1="${plotTop}" x2="${plotLeft}" y2="${baseline}" class="axis"/>`);
  svg.push(`<text x="${plotLeft - 8}" y="${baseline}" class="tick" text-anchor="end">0</text>`);
  svg.push(`<text x="${plotLeft - 8}" y="${plotTop + 4}" class="tick" text-anchor="end">${fmtTokens(peak)}</text>`);
  svg.push(`<text x="${(plotLeft + plotRight) / 2}" y="${H - 8}" class="axlabel" text-anchor="middle">session progress →</text>`);
  svg.push(`<text transform="translate(14,${(plotTop + baseline) / 2}) rotate(-90)" class="axlabel" text-anchor="middle">context tokens (modeled)</text>`);

  // modelled context ramp: 0 -> peak
  svg.push(`<polyline points="${x(0)},${y(0)} ${x(1)},${y(peak)}" class="ramp"/>`);

  // bands strip just under the x-axis (tasks, or phases in the plan-mode view)
  const bandTop = baseline + 8, bandHeight = 22;
  bands.forEach((band) => {
    const bandX = x(band.f_start);
    const bandW = Math.max(x(band.f_end) - bandX, 1);
    svg.push(`<rect x="${bandX}" y="${bandTop}" width="${bandW}" height="${bandHeight}" rx="2" style="fill:${band.color}"/>`);
    if (bandW > 26) {
      svg.push(`<text x="${bandX + bandW / 2}" y="${bandTop + bandHeight - 7}" class="bandlabel" text-anchor="middle">${escapeHtml(band.label)}</text>`);
    }
  });

  // one dashed marker per split point, coloured by type, labelled with its saving;
  // the selected split is solid + full opacity, the others faded into the background.
  markers.forEach((marker) => {
    const markerX = x(marker.frac);
    const op = marker.selected ? 1 : 0.28;
    const width = marker.selected ? 2.5 : 1.5;
    const dash = marker.selected ? "" : "stroke-dasharray:4 3;";
    svg.push(`<line x1="${markerX}" y1="${baseline}" x2="${markerX}" y2="${plotTop}" class="marker" ` +
             `style="stroke:${marker.color};stroke-opacity:${op};stroke-width:${width};${dash}"/>`);
    // only the selected split gets a $ label — labelling every marker overlaps badly
    // when many split points sit close together.
    if (marker.selected) {
      svg.push(`<text x="${markerX}" y="${plotTop - 4}" class="markerlbl" text-anchor="middle" ` +
               `style="fill:${marker.color};font-weight:700">${fmtMoney(marker.dollars)}</text>`);
    }
  });

  const heading = `<div class="modal-title"><b>${session.session_id.slice(0, 8)}</b>` +
    (session.task_summary ? ` — ${escapeHtml(session.task_summary)}` : "") + `</div>`;
  const caption = activeTab === "planmode"
    ? `<div class="tip">Bands = activity phases across the session's agent steps (reading / ` +
      `editing / running); the violet line marks the end of the opening reading phase ` +
      `— running it in plan mode saves the labelled amount. The diagonal is the ` +
      `modelled context growth (its area ≈ cost).</div>`
    : `<div class="tip">Vertical lines = split points (blue = task switch, orange = ` +
      `sub-agent), labelled with the modelled saving. Bands = tasks across the session; ` +
      `the diagonal is the modelled context growth (its area ≈ cost).</div>`;
  // preview of the SELECTED split point (updates when a row below is clicked)
  const selected = cands[selIdx] || cands[0];
  let preview = "";
  if (selected) {
    preview = `<div class="best-split-h">Selected split <span class="badge ${badgeFor(selected.source).cls}">${
      badgeFor(selected.source).label}</span> <span class="bs-label">${escapeHtml(selected.label)}</span></div>` +
      renderBestSplitPreview(session, selected);
  }

  // full analysis: every considered split point (clickable) + the whole-forest headline
  let breakdown = "";
  if (cands.length) {
    const rows = cands.map((candidate, i) => {
      const badge = badgeFor(candidate.source);
      return `<div class="sp${i === selIdx ? " selected" : ""}" data-idx="${i}">` +
        `<span class="badge ${badge.cls}">${badge.label}</span>` +
        `<span class="sp-label">${escapeHtml(candidate.label)}</span>` +
        `<span class="sp-save">${fmtMoney(candidate.dollars)} · ${Math.round(candidate.pct)}%</span></div>`;
    }).join("");
    breakdown += `<div class="splitpoints"><div class="sp-h">Every split point considered ` +
      `<span class="sp-hint">(click one to preview it)</span></div>${rows}</div>`;
  }
  const fullSplit = session.full_split;
  if (fullSplit) {
    const pct = fullSplit.as_is_cost
      ? Math.round(100 * fullSplit.dollar_saving / fullSplit.as_is_cost) : 0;
    breakdown += `<div class="meta"><span class="k-split">Split the whole forest into ${fullSplit.num_chunks} ` +
      `sessions ≈ ${fmtMoney(fullSplit.dollar_saving)} (${pct}%)</span></div>`;
  }

  return heading +
    `<svg viewBox="0 0 ${W} ${H}" class="graph" role="img" aria-label="session graph">${svg.join("")}</svg>` +
    legendHtml + caption + preview + breakdown;
}

// the session + selected split point currently shown in the modal (so clicking a
// split point in the list can re-render just that split's marker + preview)
let modalSession = null, modalSelIdx = 0;

function renderModal() {
  byId("modalBody").innerHTML = renderSessionGraph(modalSession, modalSelIdx);
}
function openGraph(sessionId) {
  modalSession = allSessions.find((s) => s.session_id === sessionId);
  if (!modalSession) return;
  modalSelIdx = 0;                          // default to the best (top) split point
  renderModal();
  byId("modal").hidden = false;
}
function closeModal() { byId("modal").hidden = true; }

// ---- wire up the controls ----
// (the "Use LLM" checkbox is read at refresh time, so it needs no live listener)
byId("refresh").addEventListener("click", refreshSessions);
["fProject", "fSort", "fPct", "fDollars"].forEach((id) => byId(id).addEventListener("input", render));
byId("sourceSelect").addEventListener("change", updateSourceControls);
updateSourceControls();

// tab switching
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    activeTab = tab.dataset.tab;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    render();
  });
});

// "See full analysis" buttons and task-tree rows are inside re-rendered cards, so
// delegate both from the list.
byId("list").addEventListener("click", (event) => {
  const button = event.target.closest(".graph-btn");
  if (button) { openGraph(button.dataset.session); return; }
  // expand / collapse a top-level task's sub-tasks (the row's next sibling)
  const row = event.target.closest(".tt-row.tt-expandable");
  if (row) {
    const kids = row.nextElementSibling;
    if (kids && kids.classList.contains("tt-children")) {
      kids.hidden = !kids.hidden;
      row.classList.toggle("open", !kids.hidden);
    }
  }
});

// modal interactions: expand a task's sub-tasks, or select a split point
byId("modalBody").addEventListener("click", (event) => {
  // expand / collapse a top-level task's sub-tasks (in place, no re-render)
  const task = event.target.closest(".tt-row.tt-expandable");
  if (task) {
    const kids = task.nextElementSibling;
    if (kids && kids.classList.contains("tt-children")) {
      kids.hidden = !kids.hidden;
      task.classList.toggle("open", !kids.hidden);
    }
    return;
  }
  // select a split point (updates its marker + preview)
  const row = event.target.closest(".sp[data-idx]");
  if (row) { modalSelIdx = Number(row.dataset.idx); renderModal(); }
});

// modal close: button, click on the backdrop, or Esc
byId("modalClose").addEventListener("click", closeModal);
byId("modal").addEventListener("click", (event) => {
  if (event.target === byId("modal")) closeModal();     // clicked the overlay, not the panel
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeModal();
});

loadCachedSessions();
