// One block's page: the steps behind a single rectangle on the session bar.
//
// Reached by clicking that rectangle, which opens `/session/<id>/block/<n>` in
// a new tab. The URL is positional because blocks are derived rather than
// stored (see the endpoint's docstring), so this page always states which
// block of how many it is showing — a link kept from yesterday may now point
// at a different stretch of work, and saying so is better than pretending.
//
// A subagent band is the one kind that renders differently: it covers every
// agent spawned back to back at that point, so the page leads with one card
// per agent and lists that agent's own steps inside it.

// Two URLs land here, and the only difference is whose bar the block came
// from:
//     /session/<id>/block/<n>
//     /session/<id>/agent/<agentId>/block/<n>
const parts = location.pathname.split('/');
const blockIndex = decodeURIComponent(parts.pop());
parts.pop();                                    // "block"
const agentId = parts[parts.length - 2] === 'agent'
  ? decodeURIComponent(parts.pop()) : null;
if (agentId) parts.pop();                       // "agent"
const sessionId = decodeURIComponent(parts.pop());

const apiBase = agentId
  ? `/api/sessions/${encodeURIComponent(sessionId)}`
    + `/agents/${encodeURIComponent(agentId)}`
  : `/api/sessions/${encodeURIComponent(sessionId)}`;

const backHref = agentId
  ? `/session/${encodeURIComponent(sessionId)}`
    + `/agent/${encodeURIComponent(agentId)}`
  : `/session/${encodeURIComponent(sessionId)}`;

// --- small pieces ------------------------------------------------------

function stat(key, value, title) {
  const box = el('div');
  box.appendChild(el('div', 'stat-k', key));
  const valueNode = el('div', 'stat-v', value);
  if (title) valueNode.title = title;
  box.appendChild(valueNode);
  return box;
}

function pill(text, className) {
  return el('span', className ? `pill ${className}` : 'pill', text);
}

// --- one step ----------------------------------------------------------

// Arguments are shown in full width rather than a table cell: a Bash command
// or a patch is the whole point of opening this page, and truncating it to a
// column would defeat that. The backend already clipped them (steps.py), and
// says so when it did.
function renderArguments(tool) {
  const list = el('div', 'args');
  tool.arguments.forEach((argument) => {
    const row = el('div', 'arg');
    row.appendChild(el('span', 'arg-k', argument.name));
    const value = el('pre', 'arg-v', argument.value);
    if (argument.truncated) {
      value.appendChild(el('span', 'clipped',
        `\n… clipped, ${formatNumber(argument.full_chars)} chars total`));
    }
    row.appendChild(value);
    list.appendChild(row);
  });
  return list;
}

function renderResult(result) {
  const box = el('div', result.is_error ? 'result result-error' : 'result');

  const head = el('div', 'result-head');
  head.appendChild(el('span', 'result-k', result.is_error ? 'error' : 'result'));
  if (result.num_lines) {
    head.appendChild(el('span', 'result-meta', `${result.num_lines} lines`));
  }
  // What the result weighed on disk, not what is shown: this is the number
  // that explains the block's token cost, since the next API call reads it all
  // back. `output` here is only the head of it.
  if (result.size_chars) {
    const size = el('span', 'result-meta',
      `${formatNumber(result.size_chars)} chars back`);
    size.title = 'Size of the whole result as Claude Code recorded it —'
      + ' what the next API call paid to read';
    head.appendChild(size);
  }
  box.appendChild(head);

  if (result.output) {
    const output = el('pre', 'result-body', result.output);
    if (result.output_truncated) {
      output.appendChild(el('span', 'clipped', '\n… clipped'));
    }
    box.appendChild(output);
  } else if (!result.is_error) {
    box.appendChild(el('div', 'muted', 'no output recorded'));
  }
  return box;
}

function renderStep(step) {
  const card = el('div', 'step');

  const head = el('div', 'step-head');
  head.appendChild(el('span', 'step-n', String(step.index + 1)));
  head.appendChild(el('span', 'step-title',
    step.headline || step.type));
  if (step.result && step.result.is_error) {
    head.appendChild(pill('failed', 'pill-error'));
  }
  if (step.tool) head.appendChild(pill(step.tool.name));

  const when = el('span', 'step-when', new Date(step.ts).toLocaleTimeString());
  when.title = step.ts;
  head.appendChild(when);
  card.appendChild(head);

  const facts = el('div', 'step-facts');
  const split = tokenSplit(step);
  const stepTokens = el('span', null, `${formatNumber(split.total)} tokens`);
  stepTokens.title = tokenBreakdown(split);
  facts.appendChild(stepTokens);
  // The API call this step was billed under. Steps of one block routinely
  // share it — that is the unit `usage` is reported for, not the step.
  if (step.message_id) {
    const call = el('span', 'mono', step.message_id.slice(-6));
    call.title = `API call ${step.message_id}`;
    facts.appendChild(call);
  }
  card.appendChild(facts);

  if (step.text) {
    const text = el('div', 'step-text', step.text);
    if (step.text_truncated) text.appendChild(el('span', 'clipped', ' … clipped'));
    card.appendChild(text);
  }
  if (step.tool && step.tool.arguments.length) {
    card.appendChild(renderArguments(step.tool));
  }
  if (step.result) card.appendChild(renderResult(step.result));

  return card;
}

function renderSteps(container, steps) {
  container.replaceChildren();
  if (!steps.length) {
    container.appendChild(el('div', 'placeholder', 'This block has no steps.'));
    return;
  }
  steps.forEach((step) => container.appendChild(renderStep(step)));
}

// --- summary line ------------------------------------------------------

function renderSummary(summary) {
  const row = el('div', 'row-meta');
  const tools = Object.entries(summary.tools)
    .map(([name, count]) => (count > 1 ? `${name} ×${count}` : name));
  if (tools.length) row.appendChild(el('span', null, tools.join(' · ')));
  if (summary.failed) {
    row.appendChild(pill(`${summary.failed} failed`, 'pill-error'));
  }
  if (summary.files.length) {
    const files = el('span', null, `${summary.files.length} files`);
    files.title = summary.files.join('\n');
    row.appendChild(files);
  }
  if (summary.api_calls) {
    row.appendChild(el('span', null, `${summary.api_calls} API calls`));
  }
  return row;
}

// --- subagents ---------------------------------------------------------

// Each agent is a link, not an expanded list: a subagent has a whole timeline
// of its own, and inlining several of them here would rebuild the unreadable
// pile that taking them out of the parent's bar was meant to fix. The card
// carries enough to choose between them; the page behind it has the bar.
function renderAgents(container, agents) {
  container.replaceChildren();
  if (!agents.length) return;

  container.appendChild(el('h3', 'section-h',
    agents.length === 1 ? 'The subagent' : `The ${agents.length} subagents`));

  agents.forEach((agent, position) => {
    const card = el('a', 'agent agent-link');
    card.href = `/session/${encodeURIComponent(sessionId)}`
      + `/agent/${encodeURIComponent(agent.agent_id)}`;

    const head = el('div', 'agent-head');
    head.appendChild(el('span', 'agent-n', `#${position + 1}`));
    head.appendChild(el('span', 'agent-title',
      agent.description || agent.label || 'subagent'));
    if (agent.agent_id) {
      const id = el('span', 'mono', agent.agent_id.slice(0, 8));
      id.title = `agentId ${agent.agent_id}`;
      head.appendChild(id);
    }
    head.appendChild(el('span', 'agent-go', 'open its bar →'));
    card.appendChild(head);

    const facts = el('div', 'step-facts');
    facts.appendChild(el('span', null, `${agent.block_count} blocks`));
    facts.appendChild(el('span', null, `${agent.summary.steps} steps`));
    const agentTokens = el('span', null,
      `${formatNumber(tokenSplit(agent).total)} tokens`);
    agentTokens.title = tokenBreakdown(tokenSplit(agent));
    facts.appendChild(agentTokens);
    facts.appendChild(el('span', null, formatDuration(agent.duration_s)));
    card.appendChild(facts);
    card.appendChild(renderSummary(agent.summary));

    container.appendChild(card);
  });
}

// --- load --------------------------------------------------------------

async function load() {
  const back = document.getElementById('back');
  back.href = backHref;
  back.textContent = agentId ? '← back to the subagent' : '← back to the session';

  let block;
  try {
    block = await getJSON(`${apiBase}/blocks/${encodeURIComponent(blockIndex)}`);
  } catch (error) {
    document.getElementById('title').textContent = 'Block not found';
    document.getElementById('meta').replaceChildren(
      el('div', 'error', error.message));
    return;
  }

  document.title = `${block.label} · TraceLens`;
  document.getElementById('title').textContent = block.label;

  const meta = document.getElementById('meta');
  meta.appendChild(pill(block.kind_label, `kind-${block.kind}`));
  meta.appendChild(el('span', null,
    `block ${block.index + 1} of ${block.block_count}`));
  // Say whose bar this block came from, or the counts above read as the
  // session's when they are really one subagent's.
  if (block.agent_id) {
    meta.appendChild(el('span', 'pill kind-subagent',
      `in subagent: ${block.agent_description || block.agent_id.slice(0, 8)}`));
  }
  if (block.session_title) {
    meta.appendChild(el('span', 'sid', block.session_title));
  }
  if (block.t_start) meta.appendChild(el('span', null, absoluteTime(block.t_start)));
  if (block.confidence !== null && block.confidence !== undefined) {
    meta.appendChild(el('span', null,
      `judge ${Math.round(block.confidence * 100)}%`));
  }

  // The headline figure is the whole context-window cost, matching how the bar
  // sized this block. Cache reads get their own tile rather than only a tooltip:
  // on this page there is room, and for an early block they are usually the
  // larger half — which is the thing the bar's height is actually telling you.
  const split = tokenSplit(block);
  const tokens = stat('Context window', formatNumber(split.total),
    tokenBreakdown(split) + ' — attributed to this block');
  const cacheReads = stat('Cache reads', formatNumber(split.cacheRead),
    `what later calls paid to re-read this block's content while it stayed in`
    + ` the context window · ${split.cacheRead.toLocaleString()} tokens`);

  document.getElementById('stats').replaceChildren(
    tokens,
    cacheReads,
    stat('Steps', String(block.summary.steps)),
    stat('Tool calls', String(block.summary.tool_calls)),
    stat('Failed', String(block.summary.failed)),
    stat('API calls', String(block.summary.api_calls)),
    stat('Duration', formatDuration(block.duration_s)),
  );

  document.getElementById('meta').appendChild(renderSummary(block.summary));

  renderAgents(document.getElementById('agents'), block.agents);

  // A subagent band's own steps are just the spawning calls; the work is in
  // the agent cards above, so the flat list would only repeat them.
  const stepsHeading = document.getElementById('steps-h');
  if (block.agents.length) {
    stepsHeading.textContent = 'The delegating calls';
  }
  renderSteps(document.getElementById('steps'), block.steps);
}

load();
