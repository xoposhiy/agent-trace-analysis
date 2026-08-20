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

function stat(key, value) {
  const box = el('div');
  box.appendChild(el('div', 'stat-k', key));
  box.appendChild(el('div', 'stat-v', value));
  return box;
}

function openBlock(_block, index) {
  window.open(`/session/${encodeURIComponent(sessionId)}/block/${index}`,
    '_blank', 'noopener');
}

function showTip(block) {
  const tip = document.getElementById('tip');
  if (!block) {
    tip.className = 'tip tip-empty';
    tip.textContent = 'Hover a block';
    return;
  }

  tip.className = 'tip';
  tip.replaceChildren();
  tip.appendChild(el('div', 'tip-kind', block.label));

  const facts = el('div', 'tip-facts');
  facts.appendChild(el('span', null, `${block.message_count} steps`));
  tokenFacts(tokenSplit(block)).forEach((span) => facts.appendChild(span));
  facts.appendChild(costFact(block));
  facts.appendChild(el('span', null, formatDuration(block.duration_s)));
  tip.appendChild(facts);
}

function countByKind(blocks) {
  const counts = {};
  blocks.forEach((block) => {
    counts[block.kind] = (counts[block.kind] || 0) + 1;
  });
  return counts;
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
  document.getElementById('detail').textContent = problem.detail;
  if (problem.data && problem.data.justification) {
    document.getElementById('justification').textContent =
      `“${problem.data.justification}”`;
  }

  const data = problem.data || {};
  const stats = [
    stat('As-is cost', formatCost(data.as_is_cost)),
    stat('Split cost', formatCost(data.split_cost)),
    stat('Saving', `${formatCost(data.dollar_saving)} (${Math.round(data.percent_saving || 0)}%)`),
  ];
  if (data.tasks) stats.push(stat('Tasks', String(data.tasks.length)));
  document.getElementById('stats').replaceChildren(...stats);

  const blocks = session.blocks || [];
  renderBar(document.getElementById('bar'), blocks, {
    metric: 'cost', onHover: showTip, onOpen: openBlock,
  });
  renderLegend(document.getElementById('legend'), countByKind(blocks));

  renderSecondBar(problem, blocks, data);
}

// Which second visualization applies depends on the problem type — a cut
// bar for plan-mode, a colored task lane for task-switch. Both containers
// exist in problem.html; whichever doesn't apply is left empty.
function renderSecondBar(problem, blocks, data) {
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
    renderPlanModeBar(planModeBar, blocks, splitIndex, 'cost', showTip, openBlock);
    return;
  }

  if (problem.id === 'task-switch') {
    arrow.hidden = true;
    planModeBar.replaceChildren();
    renderTaskForestBar(taskForestBar, blocks, data.runs, 'cost');
    return;
  }

  // A future problem type with no visualization built yet — the single bar
  // and the detail/justification text above still tell the story.
  arrow.hidden = true;
  planModeBar.replaceChildren();
  taskForestBar.replaceChildren();
}

load();
