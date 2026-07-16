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
    const tooltip = `${taskId} · ${taskLabel(forest, taskId)} · messages ${startMsg}–${endMsg}`;
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

// ---- one session card: header, forest strip, legend, badges, split points ----
// `chosen` is the candidate this tab ranks the session by (its badge + saving lead the
// card); the full split-points list below still shows every candidate.
function renderCard(session, chosen) {
  const forest = session.task_forest;
  const chosenBadge = badgeFor(chosen.source);
  const parts = [];

  parts.push(`<div class="row1">
      <span class="sid">${session.session_id.slice(0, 8)}</span>
      <span class="proj">${escapeHtml(shortProjectName(session.project))}</span>
      <span class="badge ${chosenBadge.cls}">${chosenBadge.label}</span>
      <span class="save">${fmtMoney(chosen.dollars)} · ${Math.round(chosen.pct)}%</span>
      <button class="graph-btn" data-session="${session.session_id}">View graph</button>
    </div>`);
  if (session.task_summary) {
    parts.push(`<div class="summary">${escapeHtml(session.task_summary)}</div>`);
  }

  parts.push(renderStrip(forest));
  parts.push(renderLegend(forest));

  // interleaving badge — a top-level task the user returned to
  if (forest?.recurring?.length) {
    parts.push(`<div class="badges"><span class="badge interleave">⚠ Interleaving: returned to ${
      forest.recurring.join(", ")} — likely separate sessions</span></div>`);
  }

  // meta line + whole-forest headline
  const meta = [
    `${session.turns} turns`,
    `peak ~${fmtTokens(session.peak_context)}`,
    `as-is ${fmtMoney(session.as_is_cost)}`,
    `this split: ${escapeHtml(chosen.label)}`,
  ];
  parts.push(`<div class="meta">${meta.join(" &nbsp;·&nbsp; ")}`);
  const fullSplit = session.full_split;
  if (fullSplit) {
    const pct = fullSplit.as_is_cost
      ? Math.round(100 * fullSplit.dollar_saving / fullSplit.as_is_cost) : 0;
    parts.push(`<br><span class="k-split">Split the whole forest into ${fullSplit.num_chunks} ` +
               `sessions ≈ ${fmtMoney(fullSplit.dollar_saving)} (${pct}%)</span>`);
  }
  parts.push(`</div>`);

  // every classified split point, each tagged with its type badge
  const candidates = (session.candidates || []).slice()
    .sort((a, b) => b.dollars - a.dollars || b.pct - a.pct);
  if (candidates.length) {
    const rows = candidates.map((candidate) => {
      const badge = badgeFor(candidate.source);
      return `<div class="sp"><span class="badge ${badge.cls}">${badge.label}</span>` +
        `<span class="sp-label">${escapeHtml(candidate.label)}</span>` +
        `<span class="sp-save">${fmtMoney(candidate.dollars)} · ${Math.round(candidate.pct)}%</span></div>`;
    }).join("");
    parts.push(`<div class="splitpoints"><div class="sp-h">Split points considered</div>${rows}</div>`);
  }

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
    const tip = `${style.label} · turns ${phase.start_turn}–${phase.end_turn}`;
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
      <button class="graph-btn" data-session="${session.session_id}">View graph</button>
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
      `turns ${reading.start_turn}–${reading.end_turn}</span></div>`);
  }

  const meta = [
    `${session.turns} turns`,
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
  // sessions that clear both thresholds; rank by $ then % (tie-break)
  const suggested = [];
  scanned.forEach((session) => {
    const chosen = bestCandidate(tabCandidates(session, activeTab));
    if (chosen && session.modelled
        && chosen.pct >= minPct && chosen.dollars >= minDollars) {
      suggested.push({ session, chosen });
    }
  });
  suggested.sort((a, b) => b.chosen.dollars - a.chosen.dollars
                        || b.chosen.pct - a.chosen.pct);

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
  byId("status").textContent = "Analysing new / changed sessions… (first run can take a minute)";
  try {
    const response = await fetch(`/api/refresh?use_llm=${useLlm}`, { method: "POST" });
    const data = await response.json();
    allSessions = data.sessions || [];
    populateProjectFilter();
    render();
    const stats = data.stats || {};
    byId("status").textContent =
      `Analysed ${stats.analyzed} new, reused ${stats.reused} of ${stats.total} cached.`;
  } catch (error) {
    byId("status").textContent = "Refresh failed: " + error;
  } finally {
    button.disabled = false;
  }
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

// ---- per-session graph (SVG in a modal) ----
// X = session progress 0→1 (turn space, the cost axis); Y = context tokens (peak).
// Draws the modelled context ramp, colored task bands, and a marker per split point.
function renderSessionGraph(session) {
  const peak = session.peak_context || 0;

  // choose what to draw by the active tab: plan-mode shows structural PHASES + the
  // plan-mode marker; splits show the task FOREST bands + task-switch/sub-agent markers.
  let bands, markers, legendHtml;
  if (activeTab === "planmode") {
    const total = session.seq_turns || 1;
    bands = (session.phases || []).map((phase) => ({
      f_start: (phase.start_turn - 1) / total,
      f_end: phase.end_turn / total,
      color: phaseStyle(phase.category).color,
      label: phaseStyle(phase.category).label,
    }));
    markers = (session.candidates || []).filter((c) => c.source === "plan-mode")
      .map((c) => ({ frac: c.split_fraction, color: MARKER_COLOR["plan-mode"], dollars: c.dollars }));
    legendHtml = renderPhaseLegend();
  } else {
    const forest = session.task_forest || {};
    const colorByTaskId = taskColorMap(forest);
    bands = (forest.spans || []).map((span) => ({
      f_start: span.f_start, f_end: span.f_end,
      color: colorByTaskId[span.id] || "var(--series-other)", label: span.id,
    }));
    markers = (session.candidates || []).filter((c) => SEMANTIC_SOURCES.has(c.source))
      .map((c) => ({ frac: c.split_fraction, color: MARKER_COLOR[c.source] || "var(--muted)", dollars: c.dollars }));
    legendHtml = renderLegend(session.task_forest || {});
  }

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

  // one dashed marker per split point, coloured by type, labelled with its saving
  markers.forEach((marker) => {
    const markerX = x(marker.frac);
    svg.push(`<line x1="${markerX}" y1="${baseline}" x2="${markerX}" y2="${plotTop}" class="marker" style="stroke:${marker.color}"/>`);
    svg.push(`<text x="${markerX}" y="${plotTop - 4}" class="markerlbl" text-anchor="middle" style="fill:${marker.color}">${fmtMoney(marker.dollars)}</text>`);
  });

  const heading = `<div class="modal-title"><b>${session.session_id.slice(0, 8)}</b>` +
    (session.task_summary ? ` — ${escapeHtml(session.task_summary)}` : "") + `</div>`;
  const caption = activeTab === "planmode"
    ? `<div class="tip">Bands = activity phases across the session's turns (reading / ` +
      `editing / running); the violet line marks the end of the opening reading phase ` +
      `— running it in plan mode saves the labelled amount. The diagonal is the ` +
      `modelled context growth (its area ≈ cost).</div>`
    : `<div class="tip">Vertical lines = split points (blue = task switch, orange = ` +
      `sub-agent), labelled with the modelled saving. Bands = tasks across the session; ` +
      `the diagonal is the modelled context growth (its area ≈ cost).</div>`;
  return heading +
    `<svg viewBox="0 0 ${W} ${H}" class="graph" role="img" aria-label="session graph">${svg.join("")}</svg>` +
    legendHtml + caption;
}

function openGraph(sessionId) {
  const session = allSessions.find((s) => s.session_id === sessionId);
  if (!session) return;
  byId("modalBody").innerHTML = renderSessionGraph(session);
  byId("modal").hidden = false;
}
function closeModal() { byId("modal").hidden = true; }

// ---- wire up the controls ----
// (the "Use LLM" checkbox is read at refresh time, so it needs no live listener)
byId("refresh").addEventListener("click", refreshSessions);
["fProject", "fPct", "fDollars"].forEach((id) => byId(id).addEventListener("input", render));

// tab switching
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    activeTab = tab.dataset.tab;
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    render();
  });
});

// "View graph" buttons are inside re-rendered cards, so delegate from the list
byId("list").addEventListener("click", (event) => {
  const button = event.target.closest(".graph-btn");
  if (button) openGraph(button.dataset.session);
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
