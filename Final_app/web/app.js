// Dashboard: tabs, filters, session list.

const state = { project: '', severity: 'any', llm: null, offset: 0 };

// Must match PAGE_SIZE in api/app.py — the server clamps anyway, this only
// keeps the client's offset arithmetic in step with what it asks for.
const PAGE_SIZE = 20;

// --- tabs -------------------------------------------------------------

function initTabs() {
  const tabs = [...document.querySelectorAll('.tab')];
  tabs.forEach((tab) => {
    tab.addEventListener('click', () => {
      tabs.forEach((other) => {
        const selected = other === tab;
        other.setAttribute('aria-selected', String(selected));
        document.getElementById(other.getAttribute('aria-controls')).hidden = !selected;
      });
    });
  });
}

// --- LLM status -------------------------------------------------------

// Two questions, deliberately answered by two requests: /api/health is a config
// read and returns instantly, while /api/llm-check makes a real call and can
// take seconds to fail. The page must not wait on the second to paint.
async function loadHealth() {
  const note = document.getElementById('llm-note');
  try {
    const health = await getJSON('/api/health');
    state.llm = health.llm;
    document.getElementById('root-hint').textContent = health.projects_root;
  } catch (error) {
    note.appendChild(el('div', 'error', `Could not reach the backend: ${error.message}`));
    return;
  }
  checkLlm();
}

// --- is the LLM actually working --------------------------------------

// Configuration being valid is not the same as the LLM working: the proxy here
// is VPN-only, so a correct key against an unreachable host is the common
// failure, and it used to look identical to "working" — classification silently
// fell back to the shell heuristic and nothing said why.
async function checkLlm({ force = false } = {}) {
  const note = document.getElementById('llm-note');
  note.replaceChildren(el('div', 'note', 'Checking the LLM…'));

  let probe;
  try {
    probe = await getJSON(`/api/llm-check${force ? '?force=true' : ''}`);
  } catch (error) {
    note.replaceChildren(el('div', 'error',
      `Could not run the LLM check: ${error.message}`));
    return;
  }

  if (probe.ok) {
    // Success is stated, not silent — "no banner" is indistinguishable from
    // "the check never ran".
    note.replaceChildren(el('div', 'note note-ok',
      `LLM ready — ${probe.model} answered in ${probe.latency_ms} ms.`));
    return;
  }

  const box = el('div', 'error error-llm');
  box.appendChild(el('div', 'error-title',
    probe.configured
      ? `The LLM is not working: ${probe.reason}.`
      : `The LLM is not configured: ${probe.reason}.`));
  if (probe.hint) box.appendChild(el('div', 'error-hint', probe.hint));
  box.appendChild(el('div', 'error-hint',
    `Model ${probe.model}. Block classification falls back to the rule-based `
    + 'heuristic, so ambiguous shell commands may be labelled wrongly. '
    + 'Everything else on the dashboard is unaffected.'));

  const retry = el('button', 'action', 'Retry');
  retry.addEventListener('click', () => checkLlm({ force: true }));
  box.appendChild(retry);

  note.replaceChildren(box);
}

// --- filters ----------------------------------------------------------

async function loadProjects() {
  const select = document.getElementById('f-project');
  try {
    const { projects } = await getJSON('/api/projects');
    projects.forEach((project) => {
      const option = el('option', null, `${project.label} (${project.count})`);
      option.value = project.slug;
      option.title = project.slug;
      select.appendChild(option);
    });
  } catch (_) { /* the list still renders unfiltered */ }

  select.addEventListener('change', () => {
    state.project = select.value;
    loadSessions();
  });
}

// --- session list -----------------------------------------------------

function sessionRow(session) {
  const row = el('a', 'row');
  row.href = `/session/${session.session_id}`;

  const head = el('div', 'row-head');
  head.appendChild(el('span', 'row-title',
    session.title || session.session_id.slice(0, 8)));
  const when = el('span', 'row-when', relativeTime(session.last_ts));
  when.title = absoluteTime(session.last_ts);
  head.appendChild(when);
  row.appendChild(head);

  const meta = el('div', 'row-meta');
  meta.appendChild(el('span', 'sid', session.session_id));
  const project = el('span', 'pill', session.project_label);
  project.title = session.project;  // the label is a lossy heuristic
  meta.appendChild(project);
  const tokens = el('span', null, `${formatNumber(session.tokens.working)} tokens`);
  tokens.title = `${session.tokens.total.toLocaleString()} including cache reads`;
  meta.appendChild(tokens);
  meta.appendChild(el('span', null, `${session.tool_call_count} tool calls`));
  meta.appendChild(el('span', null, formatDuration(session.duration_s)));
  if (session.subagent_count) {
    meta.appendChild(el('span', null,
      `${session.subagent_count} subagent${session.subagent_count > 1 ? 's' : ''}`));
  }
  row.appendChild(meta);

  return row;
}

// The list is paged because the backend parses only the transcripts it
// returns — the whole point of the pagination is that opening the page never
// costs a user with years of history a full re-parse of their history. So a
// filter change starts over at offset 0, and "load more" appends.
async function loadSessions({ append = false } = {}) {
  const container = document.getElementById('sessions');
  if (!append) {
    state.offset = 0;
    container.replaceChildren(el('div', 'empty-state', 'Loading sessions…'));
  }

  const params = new URLSearchParams();
  if (state.project) params.set('project', state.project);
  if (state.severity && state.severity !== 'any') params.set('severity', state.severity);
  params.set('offset', String(state.offset));
  params.set('limit', String(PAGE_SIZE));

  try {
    const data = await getJSON(`/api/sessions?${params}`);
    const sessions = data.sessions;

    let list = append ? container.querySelector('.list') : null;
    if (!list) {
      list = el('div', 'list');
      container.replaceChildren(list);
    }
    sessions.forEach((session) => list.appendChild(sessionRow(session)));

    if (!list.children.length) {
      container.replaceChildren(el('div', 'empty-state', 'No sessions found.'));
      renderMore(null);
      return;
    }

    // Advance by what was *asked for*, not by what came back: the backend can
    // return a short page when a transcript in the window fails to parse, and
    // stepping by the shorter number would re-request rows already shown.
    state.offset = data.offset + data.limit;
    renderMore(data);
  } catch (error) {
    renderMore(null);
    if (append) return;   // keep the rows already on screen
    container.replaceChildren(el('div', 'error', `Failed to load sessions: ${error.message}`));
  }
}

// --- load more --------------------------------------------------------

// Rendered below the list rather than as an infinite scroll: each page costs
// real parsing on the server, so it stays a deliberate click.
function renderMore(data) {
  const slot = document.getElementById('more');
  slot.replaceChildren();
  if (!data) return;

  const shown = document.querySelectorAll('#sessions .row').length;
  slot.appendChild(el('div', 'more-count', `${shown} of ${data.total}`));
  if (!data.has_more) return;

  const button = el('button', 'action', 'Load more');
  button.addEventListener('click', async () => {
    button.disabled = true;
    button.textContent = 'Loading…';
    await loadSessions({ append: true });
  });
  slot.appendChild(button);
}

// --- boot -------------------------------------------------------------

initTabs();
loadHealth();
loadProjects();
loadSessions();
