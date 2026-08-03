// One session's page. The vertical colour bar is the next step; this renders
// the surrounding chrome and leaves a slot for it.

const sessionId = decodeURIComponent(location.pathname.split('/').pop());

function stat(key, value) {
  const box = el('div');
  box.appendChild(el('div', 'stat-k', key));
  box.appendChild(el('div', 'stat-v', value));
  return box;
}

async function load() {
  let session;
  try {
    session = await getJSON(`/api/sessions/${encodeURIComponent(sessionId)}`);
  } catch (error) {
    document.getElementById('title').textContent = 'Not found';
    document.getElementById('summary').replaceChildren(
      el('div', 'error', error.message));
    return;
  }

  document.title = `${session.title || sessionId.slice(0, 8)} · TraceLens`;
  document.getElementById('title').textContent =
    session.title || sessionId.slice(0, 8);

  const summary = document.getElementById('summary');
  if (session.summary) {
    summary.textContent = session.summary;
  } else {
    summary.textContent = 'no summary yet';
    summary.style.fontStyle = 'italic';
    summary.style.opacity = '0.65';
  }

  const meta = document.getElementById('meta');
  meta.appendChild(el('span', 'sid', session.session_id));
  meta.appendChild(el('span', 'pill', session.project_label));
  if (session.git_branch) meta.appendChild(el('span', 'pill', session.git_branch));
  if (session.model) meta.appendChild(el('span', 'pill', session.model));
  meta.appendChild(el('span', null, `last message ${absoluteTime(session.last_ts)}`));

  const tokenStat = stat('Tokens', formatNumber(session.tokens.working));
  tokenStat.title = `${session.tokens.total.toLocaleString()} including cache reads`
    + ` · in ${session.tokens.input.toLocaleString()}`
    + ` · out ${session.tokens.output.toLocaleString()}`
    + ` · cache write ${session.tokens.cache_creation.toLocaleString()}`
    + ` · cache read ${session.tokens.cache_read.toLocaleString()}`;

  document.getElementById('stats').replaceChildren(
    tokenStat,
    stat('Messages', formatNumber(session.message_count)),
    stat('Tool calls', formatNumber(session.tool_call_count)),
    stat('Subagents', String(session.subagent_count)),
    stat('Duration', formatDuration(session.duration_s)),
    stat('Compactions', String(session.compaction_points.length)),
  );

  drawBar(session);
}

// --- the bar ----------------------------------------------------------

function countByKind(blocks) {
  const counts = {};
  blocks.forEach((block) => {
    counts[block.kind] = (counts[block.kind] || 0) + 1;
  });
  return counts;
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
  facts.appendChild(el('span', null, `${formatNumber(block.tokens.working)} tokens`));
  facts.appendChild(el('span', null, formatDuration(block.duration_s)));
  if (block.confidence !== null && block.confidence !== undefined) {
    facts.appendChild(el('span', null, `judge ${Math.round(block.confidence * 100)}%`));
  }
  tip.appendChild(facts);

  if (block.description) {
    tip.appendChild(el('div', 'tip-desc', `task: ${block.description}`));
  }
  if (block.inner_blocks && block.inner_blocks.length) {
    tip.appendChild(el('div', 'tip-desc',
      block.inner_blocks.map((b) => b.label).join(' · ')));
  }
}

function drawBar(session) {
  const blocks = session.blocks || [];
  const select = document.getElementById('f-metric');

  document.getElementById('block-count').textContent =
    `${blocks.length} blocks from ${session.message_count} events`;

  // Tall enough that every block clears the 3px minimum (5px each with the
  // gap), short enough to stay on one screen beside the detail pane — the
  // sketch's proportions. Past ~170 blocks the minimums win and it grows.
  const height = Math.max(420, Math.min(880, blocks.length * 5 + 40));

  const draw = () => renderBar(document.getElementById('bar'), blocks, {
    metric: select.value,
    height,
    onHover: showTip,
  });

  select.addEventListener('change', draw);
  draw();
  renderLegend(document.getElementById('legend'), countByKind(blocks));
}

load();
