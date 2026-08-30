// One detected problem's own page: the session's plain bar beside the same
// bar with the problem's split cut into it (see web/plan_mode.js), so a
// suggestion can be inspected on its own rather than folded into the
// session page everyone opens by default.
//
// TODO(later): a section here justifying *why* this particular problem was
// raised (what pattern matched, what the priced saving was computed from) —
// not built yet, this page only draws the comparison.

const PATH_PARTS = location.pathname.split('/').filter(Boolean);
const sessionId = decodeURIComponent(PATH_PARTS[1]);
const problemId = decodeURIComponent(PATH_PARTS[3]);

function openBlock(_block, index) {
  window.open(`/session/${encodeURIComponent(sessionId)}/block/${index}`,
    '_blank', 'noopener');
}

function countByKind(blocks) {
  const counts = {};
  blocks.forEach((block) => {
    counts[block.kind] = (counts[block.kind] || 0) + 1;
  });
  return counts;
}

// One row per priced chunk (task-switch only; ``data.chunks`` is absent for
// a plan-mode problem, which is always a fixed 2-way split with nothing to
// break down). A chunk covering more than one task id is the interleaving
// case — DESIGN.md's `_cluster_independent_spans` — and reads as "T1 and
// T2", not silently as just one of the ids it actually contains.
function renderProblemChunks(chunks, tasks) {
  const container = document.getElementById('problem-chunks');
  if (!chunks || !chunks.length) {
    container.replaceChildren();
    return;
  }

  const labelOf = {};
  (tasks || []).forEach((task) => { labelOf[task.id] = task.label; });

  const list = el('div', 'list');
  chunks.forEach((chunk, index) => {
    const row = el('div', 'row-detail');
    const names = chunk.task_ids.map((id) => labelOf[id] || id).join(' + ');
    row.textContent = `Chunk ${index + 1}: ${chunk.label}`;
    row.title = chunk.task_ids.length > 1
      ? `A single priced piece covering ${names} together — interleaved,`
        + ' never split apart internally.'
      : names;
    list.appendChild(row);
  });
  container.replaceChildren(el('h3', 'section-h', 'Chunks priced'), list);
}

async function load() {
  document.getElementById('back').href = `/session/${encodeURIComponent(sessionId)}`;

  let session;
  try {
    session = await getJSON(`/api/sessions/${encodeURIComponent(sessionId)}`);
  } catch (error) {
    document.getElementById('title').textContent = 'Not found';
    document.getElementById('meta').replaceChildren(el('div', 'error', error.message));
    return;
  }

  const problem = (session.problems || []).find((p) => p.id === problemId);
  if (!problem) {
    document.getElementById('title').textContent = 'Problem not found';
    document.getElementById('meta').replaceChildren(el('div', 'error',
      `This session no longer reports a "${problemId}" problem.`));
    return;
  }

  document.title = `${problem.title} · TraceLens`;
  document.getElementById('title').textContent = problem.title;

  const meta = document.getElementById('meta');
  meta.appendChild(el('span', `pill pill-severity-${problem.severity}`, problem.severity));
  meta.appendChild(el('span', null, session.title || session.session_id.slice(0, 8)));
  meta.appendChild(el('span', 'pill', session.project_label));

  // The same header the session's own page leads with (`common.js`'s
  // `sessionStats`) — this page is about one problem WITHIN that session,
  // not a different session, so "what is this session" should read
  // identically wherever it's shown.
  document.getElementById('stats').replaceChildren(...sessionStats(session));

  // The problem-specific pricing comparison lives in the side panel, next
  // to the hover tip it sits below — mirroring where the session page puts
  // its own "Problems detected" panel.
  const data = problem.data || {};
  const problemStats = [
    stat('As-is cost', formatCost(data.as_is_cost),
      'The session\'s real, already-attributed bill — the same number as'
      + ' "Retrospective cost" above, priced the same way everywhere in the app.'),
    stat('Split cost', formatCost(data.split_cost),
      'An ESTIMATE: there is no exact figure for a split that never happened,'
      + ' only chunk_split_model\'s linear-context-ramp approximation.'),
    stat('Saving', `${formatCost(data.dollar_saving)} (${Math.round(data.percent_saving || 0)}%)`),
  ];
  if (data.tasks) problemStats.push(stat('Tasks', String(data.tasks.length)));
  // task-switch only: how many pieces the price above actually reflects —
  // a stretch of back-and-forth between recurring tasks prices as ONE
  // chunk, never one per switch, so this can be smaller than "Tasks" or
  // than the number of bands in the lane below.
  if (data.num_chunks) {
    problemStats.push(stat('Chunks', String(data.num_chunks),
      'How many separate pieces the split price reflects. A stretch of'
      + ' back-and-forth between recurring tasks prices as one chunk, not'
      + ' one per switch — see the detail text below.'));
  }
  document.getElementById('problem-stats').replaceChildren(...problemStats);

  // What each priced chunk actually contains — task-switch only. A chunk
  // that merged an interleaved stretch reads as "T1 and T2", not silently
  // as just one of the two ids it actually covers.
  renderProblemChunks(data.chunks, data.tasks);

  document.getElementById('detail').textContent = problem.detail;
  if (problem.data && problem.data.justification) {
    document.getElementById('justification').textContent =
      `“${problem.data.justification}”`;
  }

  const blocks = session.blocks || [];
  const select = document.getElementById('f-metric');
  const tip = document.getElementById('tip');
  const draw = () => {
    const metric = select.value;
    renderBar(document.getElementById('bar'), blocks, {
      metric,
      onHover: (block, event) =>
        showBlockTip(tip, block, metric, event && event.currentTarget),
      onOpen: openBlock,
    });
    renderSecondBar(problem, blocks, data, metric);
    renderMetricTotal(document.getElementById('metric-total'), session, metric);
  };
  select.addEventListener('change', draw);
  draw();
  renderLegend(document.getElementById('legend'), countByKind(blocks));
  dismissTipOnScroll(document.querySelector('.bar-col'), tip);
}

// Which second visualization applies depends on the problem type — a cut
// bar for plan-mode, a colored task lane for task-switch. Both containers
// exist in problem.html; whichever doesn't apply is left empty.
function renderSecondBar(problem, blocks, data, metric) {
  const arrow = document.getElementById('bar-arrow');
  const planModeBar = document.getElementById('plan-mode-bar');
  const taskForestBar = document.getElementById('task-forest-bar');

  if (problem.id === 'plan-mode') {
    taskForestBar.replaceChildren();
    const splitIndex = computeSplitLayout(blocks, data.split_after_ts);
    if (splitIndex === null) {
      // A stale payload with no matching block timestamp — the single bar
      // above still tells the story, just not this comparison.
      arrow.hidden = true;
      planModeBar.replaceChildren();
      return;
    }
    arrow.hidden = false;
    const tip = document.getElementById('tip');
    renderPlanModeBar(planModeBar, blocks, splitIndex, metric,
      (block, event) => showBlockTip(tip, block, metric, event && event.currentTarget),
      openBlock);
    return;
  }

  if (problem.id === 'task-switch') {
    arrow.hidden = true;
    planModeBar.replaceChildren();
    renderTaskForestBar(taskForestBar, blocks, data.runs, metric);
    return;
  }

  // A future problem type with no visualization built yet — the single bar
  // and the detail/justification text above still tell the story.
  arrow.hidden = true;
  planModeBar.replaceChildren();
  taskForestBar.replaceChildren();
}

load();
