// One session's page. The vertical colour bar is the next step; this renders
// the surrounding chrome and leaves a slot for it.

const sessionId = decodeURIComponent(location.pathname.split('/').pop());

async function load() {
  let session;
  try {
    session = await getJSON(`/api/sessions/${encodeURIComponent(sessionId)}`);
  } catch (error) {
    document.getElementById('title').textContent = 'Not found';
    document.getElementById('meta').replaceChildren(
      el('div', 'error', error.message));
    return;
  }

  // The title is Claude Code's own `ai-title` line — free, already on disk.
  // There is deliberately no LLM-written summary beneath it.
  document.title = `${session.title || sessionId.slice(0, 8)} · TraceLens`;
  document.getElementById('title').textContent =
    session.title || sessionId.slice(0, 8);

  const meta = document.getElementById('meta');
  meta.appendChild(el('span', 'sid', session.session_id));
  meta.appendChild(el('span', 'pill', session.project_label));
  if (session.git_branch) meta.appendChild(el('span', 'pill', session.git_branch));
  if (session.model) meta.appendChild(el('span', 'pill', session.model));
  meta.appendChild(el('span', null, `last message ${absoluteTime(session.last_ts)}`));

  // Shared with problem.js (`common.js`'s `sessionStats`), so a session's
  // own page and any of its detected problems' pages always agree on the
  // same header numbers.
  document.getElementById('stats').replaceChildren(...sessionStats(session));

  drawBar(session);
  renderSessionProblems(session.problems || []);
}

// --- problems -----------------------------------------------------------

// This session's own detected problems, each linking to its dedicated
// two-bar comparison page (`/session/{id}/problem/{id}` — see problem.js).
// This panel just says *that* and *why briefly*; the comparison itself lives
// on that page, not here.
function sessionProblemRow(problem) {
  const link = el('a', 'row');
  link.href = `/session/${encodeURIComponent(sessionId)}/problem/${encodeURIComponent(problem.id)}`;

  const head = el('div', 'row-head');
  head.appendChild(el('span', 'row-title', problem.title));
  head.appendChild(el('span', `pill pill-severity-${problem.severity}`, problem.severity));
  link.appendChild(head);

  link.appendChild(el('div', 'row-detail', problem.detail));
  if (problem.data && problem.data.justification) {
    link.appendChild(el('div', 'row-justification', `“${problem.data.justification}”`));
  }
  return link;
}

function renderSessionProblems(problems) {
  const container = document.getElementById('session-problems');
  container.replaceChildren(el('h3', 'section-h', 'Problems detected'));

  // The panel stays even when clean, saying so explicitly — an empty slot
  // here used to read as "did this even run", not "nothing was found".
  if (!problems.length) {
    container.appendChild(el('div', 'muted', 'No problems were detected in this session.'));
    return;
  }

  const list = el('div', 'list');
  problems.forEach((problem) => list.appendChild(sessionProblemRow(problem)));
  container.appendChild(list);
}

// A block opens in its own tab, so the bar stays where it is and several
// blocks can be compared side by side.
function openBlock(_block, index) {
  window.open(`/session/${encodeURIComponent(sessionId)}/block/${index}`,
    '_blank', 'noopener');
}

// --- the bar ----------------------------------------------------------

function countByKind(blocks) {
  const counts = {};
  blocks.forEach((block) => {
    counts[block.kind] = (counts[block.kind] || 0) + 1;
  });
  return counts;
}

function drawBar(session) {
  const blocks = session.blocks || [];
  const select = document.getElementById('f-metric');
  const tip = document.getElementById('tip');

  document.getElementById('block-count').textContent =
    `${blocks.length} blocks from ${session.message_count} events`;

  // The bar sizes itself (`barHeight`), and the column scrolls when the result
  // is taller than the viewport. Capping it at one screen — as this did until
  // 2026-08-05 — left no proportional space in the layout, so every block
  // rendered at the 3px floor and the metric selector had no visible effect.
  const draw = () => {
    renderBar(document.getElementById('bar'), blocks, {
      metric: select.value,
      // Reads `select.value` at hover time, not draw time, so the tooltip
      // still matches whichever axis is selected right now even if the user
      // switches it while a block happens to be focused. `event.currentTarget`
      // is the hovered SVG block itself, which `showBlockTip` anchors the
      // floating tip to (see common.js).
      onHover: (block, event) =>
        showBlockTip(tip, block, select.value, event && event.currentTarget),
      onOpen: openBlock,
    });
    renderMetricTotal(document.getElementById('metric-total'), session, select.value);
  };

  select.addEventListener('change', draw);
  draw();
  renderLegend(document.getElementById('legend'), countByKind(blocks));
  dismissTipOnScroll(document.querySelector('.bar-col'), tip);
}

load();
