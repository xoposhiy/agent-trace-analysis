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

  // "Context window", not "Tokens": every billed token, cache reads included.
  // It has to be the total rather than `working`, because the bar below is now
  // sized by each block's whole context-window cost — a header showing
  // `working` would be the smaller number the blocks visibly do not sum to.
  const tokenStat = stat('Context window', formatNumber(session.tokens.total));
  tokenStat.title = `${session.tokens.working.toLocaleString()} excluding cache reads`
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

  // The block's share of the whole context window — cache reads included, so
  // this is the figure the bar is sized by — then how much of it is re-reads,
  // which is what explains a tall block that barely did anything.
  tokenFacts(tokenSplit(block)).forEach((span) => facts.appendChild(span));

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

  // The bar sizes itself (`barHeight`), and the column scrolls when the result
  // is taller than the viewport. Capping it at one screen — as this did until
  // 2026-08-05 — left no proportional space in the layout, so every block
  // rendered at the 3px floor and the metric selector had no visible effect.
  // A block opens in its own tab, so the bar stays where it is and several
  // blocks can be compared side by side.
  const openBlock = (_block, index) => {
    window.open(`/session/${encodeURIComponent(sessionId)}/block/${index}`,
      '_blank', 'noopener');
  };

  const draw = () => renderBar(document.getElementById('bar'), blocks, {
    metric: select.value,
    onHover: showTip,
    onOpen: openBlock,
  });

  select.addEventListener('change', draw);
  draw();
  renderLegend(document.getElementById('legend'), countByKind(blocks));
}

load();
