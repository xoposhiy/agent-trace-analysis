// Dashboard: tabs, filters, session list.

const state = { project: '', llm: null, offset: 0 };
// `rows` holds every problem scanned so far, raw — sorting is a pure
// client-side re-render over what's already loaded, never a re-scan.
const problemsState = { loaded: false, rows: [], sort: 'percent' };

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
      // The Problems tab actually runs detection over whatever it scans
      // (unlike the session list), so it is loaded on first click rather
      // than on page load — opening the dashboard should not pay that cost
      // for a tab the user may never open.
      if (tab.id === 'tab-problems' && !problemsState.loaded) {
        problemsState.loaded = true;
        loadProblems();
      }
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
  // The same figure the session page and its bar lead with — every billed
  // token. One meaning of "tokens" across the app; the split is on hover.
  const tokens = el('span', null, `${formatNumber(session.tokens.total)} tokens`);
  tokens.title = `${session.tokens.working.toLocaleString()} excluding cache reads`
    + ` · ${session.tokens.cache_read.toLocaleString()} re-read from cache`;
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

// --- problems -----------------------------------------------------------

const SEVERITY_LABEL = { info: 'info', low: 'low', medium: 'medium', high: 'high' };

function problemRow(row) {
  const problem = row.problem;
  const link = el('a', 'row');
  link.href = `/session/${row.session_id}/problem/${problem.id}`;

  const head = el('div', 'row-head');
  head.appendChild(el('span', 'row-title', problem.title));
  head.appendChild(el('span', `pill pill-severity-${problem.severity}`,
    SEVERITY_LABEL[problem.severity] || problem.severity));
  link.appendChild(head);

  const meta = el('div', 'row-meta');
  // The session's own free title (Claude Code's `ai-title` line) is what
  // makes a session recognisable — a problem alone doesn't say which one.
  meta.appendChild(el('span', 'row-session-title',
    row.title || row.session_id.slice(0, 8)));
  meta.appendChild(el('span', 'pill', row.project_label));
  const tokens = el('span', null, `${formatNumber(row.tokens.total)} tokens`);
  tokens.title = `${row.tokens.working.toLocaleString()} excluding cache reads`
    + ` · ${row.tokens.cache_read.toLocaleString()} re-read from cache`;
  meta.appendChild(tokens);
  meta.appendChild(el('span', null, `${row.tool_call_count} tool calls`));
  meta.appendChild(el('span', null, formatDuration(row.duration_s)));
  if (row.subagent_count) {
    meta.appendChild(el('span', null,
      `${row.subagent_count} subagent${row.subagent_count > 1 ? 's' : ''}`));
  }
  link.appendChild(meta);

  link.appendChild(el('div', 'row-detail', problem.detail));
  // LLM-optional: present only when the judge was reachable (see
  // `analysis.plan_mode.justify`) — absent, the mechanism sentence above
  // still stands on its own.
  if (problem.data && problem.data.justification) {
    link.appendChild(el('div', 'row-justification', `“${problem.data.justification}”`));
  }

  return link;
}

// The value a row sorts by. Both current problem types (`plan-mode`,
// `task-switch`) price their saving the same way (`price_split`/
// `price_multi_split` share the `dollar_saving`/`percent_saving` keys), so
// this works unchanged for either; a future problem type with no priced
// saving just sorts as 0 rather than crashing the comparator.
function problemSortValue(row) {
  const data = row.problem.data || {};
  return problemsState.sort === 'dollar' ? (data.dollar_saving || 0) : (data.percent_saving || 0);
}

// Re-renders from `problemsState.rows` — never re-fetches. Sorting is
// always highest-first; there is no ascending option, per how this was
// asked for.
function renderProblemsList() {
  if (!problemsState.rows.length) return;
  const sorted = [...problemsState.rows].sort((a, b) => problemSortValue(b) - problemSortValue(a));
  const list = el('div', 'list');
  sorted.forEach((row) => list.appendChild(problemRow(row)));
  document.getElementById('problems').replaceChildren(list);
}

function initProblemsSort() {
  const select = document.getElementById('f-problem-sort');
  select.addEventListener('change', () => {
    problemsState.sort = select.value;
    renderProblemsList();
  });
}

// Unlike `loadSessions`, this does NOT stop after one page: `/api/problems`
// pages over SESSIONS scanned, not problems found, so a page landing on a
// clean stretch of sessions can come back with zero rows while later pages
// still hold the one flagged session. Stopping there — as this used to —
// read as "scanned, found nothing" when the truth was "scanned page 1 of 3
// and hasn't looked at the rest yet", and only clicking "load more" (an
// unexplained extra step) actually found it. So this keeps asking for the
// next page itself, updating the status line as it goes, until the backend
// says there is nothing left to scan.
async function loadProblems() {
  const container = document.getElementById('problems');
  const status = document.getElementById('problems-more');
  container.replaceChildren(el('div', 'empty-state', 'Scanning sessions…'));
  status.replaceChildren();
  problemsState.rows = [];

  let offset = 0;
  let data;

  try {
    do {
      const params = new URLSearchParams();
      if (state.project) params.set('project', state.project);
      params.set('offset', String(offset));
      params.set('limit', String(PAGE_SIZE));

      data = await getJSON(`/api/problems?${params}`);
      problemsState.rows.push(...data.problems);
      renderProblemsList();

      offset = data.offset + data.limit;
      status.replaceChildren(el('div', 'more-count',
        `scanned ${Math.min(offset, data.total_sessions)} of ${data.total_sessions} sessions`
        + (data.has_more ? '…' : '')));
    } while (data.has_more);
  } catch (error) {
    status.replaceChildren();
    if (!problemsState.rows.length) {
      container.replaceChildren(el('div', 'error', `Failed to scan sessions: ${error.message}`));
    }
    return;
  }

  if (!problemsState.rows.length) {
    container.replaceChildren(el('div', 'empty-state', 'No problems detected.'));
  }
}

// --- boot -------------------------------------------------------------

initTabs();
initProblemsSort();
loadHealth();
loadProjects();
loadSessions();
