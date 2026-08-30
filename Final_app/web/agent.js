// One subagent's own page: the same vertical bar as a session, one level down.
//
// A subagent was given a task and worked through it exactly as the main thread
// does — it has its own reading, its own runs, its own coordination. Drawing
// that inside the parent's band turned each of its blocks into a 2px sliver,
// so it gets a bar of its own instead, on this page, with the same colours,
// the same Y-axis selector and the same click-through to a block.
//
// Reached from the delegation band's page: `/session/<id>/agent/<agentId>`.
// Its blocks link on to `/session/<id>/agent/<agentId>/block/<n>`, which is the
// ordinary block page reading the agent out of the URL.

const agentParts = location.pathname.split('/');
const agentId = decodeURIComponent(agentParts.pop());
agentParts.pop();                                    // "agent"
const agentSessionId = decodeURIComponent(agentParts.pop());

// --- chrome ------------------------------------------------------------

function countByKind(blocks) {
  const counts = {};
  blocks.forEach((block) => {
    counts[block.kind] = (counts[block.kind] || 0) + 1;
  });
  return counts;
}

// --- load --------------------------------------------------------------

async function load() {
  const back = document.getElementById('back');
  back.href = `/session/${encodeURIComponent(agentSessionId)}`;

  let agent;
  try {
    agent = await getJSON(
      `/api/sessions/${encodeURIComponent(agentSessionId)}`
      + `/agents/${encodeURIComponent(agentId)}`);
  } catch (error) {
    document.getElementById('title').textContent = 'Subagent not found';
    document.getElementById('meta').replaceChildren(
      el('div', 'error', error.message));
    return;
  }

  const heading = agent.description || 'subagent';
  document.title = `${heading} · TraceLens`;
  document.getElementById('title').textContent = heading;

  const meta = document.getElementById('meta');
  meta.appendChild(el('span', 'pill kind-subagent', 'subagent'));
  const id = el('span', 'sid', agent.agent_id);
  id.title = 'toolUseResult.agentId — the parent → subagent link';
  meta.appendChild(id);
  if (agent.session_title) {
    meta.appendChild(el('span', null, `in “${agent.session_title}”`));
  }
  if (agent.t_start) meta.appendChild(el('span', null, absoluteTime(agent.t_start)));

  document.getElementById('stats').replaceChildren(
    stat('Context window', formatNumber(agent.context_tokens),
      'The real, bounded size of this subagent\'s own last call — a'
      + ' separate, isolated context window from the main thread\'s.'
      + ' See DESIGN.md §7.'),
    stat('Retrospective cost', formatCost(agent.attributed_cost),
      'This subagent\'s share of the session\'s bill, priced at whichever'
      + ' model this subagent actually ran on.'),
    stat('Steps', String(agent.summary.steps)),
    stat('Tool calls', String(agent.summary.tool_calls)),
    stat('Failed', String(agent.summary.failed)),
    stat('API calls', String(agent.summary.api_calls)),
    stat('Duration', formatDuration(agent.duration_s)),
  );

  drawBar(agent);
}

function drawBar(agent) {
  const blocks = agent.blocks || [];
  const select = document.getElementById('f-metric');
  const tip = document.getElementById('tip');

  document.getElementById('block-count').textContent =
    `${blocks.length} blocks from ${agent.summary.steps} steps`;

  const openBlock = (_block, index) => {
    window.open(
      `/session/${encodeURIComponent(agentSessionId)}`
      + `/agent/${encodeURIComponent(agentId)}/block/${index}`,
      '_blank', 'noopener');
  };

  const draw = () => {
    renderBar(document.getElementById('bar'), blocks, {
      metric: select.value,
      // Reads `select.value` at hover time, not draw time — see session.js.
      onHover: (block, event) =>
        showBlockTip(tip, block, select.value, event && event.currentTarget),
      onOpen: openBlock,
    });
    renderMetricTotal(document.getElementById('metric-total'), agent, select.value);
  };

  select.addEventListener('change', draw);
  draw();
  renderLegend(document.getElementById('legend'), countByKind(blocks));
  dismissTipOnScroll(document.querySelector('.bar-col'), tip);
}

load();
