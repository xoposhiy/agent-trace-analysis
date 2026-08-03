// Dashboard: tabs, filters, session list.

const state = { project: '', severity: 'any', llm: null, poll: null };

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

async function loadHealth() {
  const note = document.getElementById('llm-note');
  try {
    const health = await getJSON('/api/health');
    state.llm = health.llm;
    document.getElementById('root-hint').textContent = health.projects_root;

    if (!health.llm.enabled) {
      // Say precisely why, and that nothing else is broken — the dashboard is
      // fully usable without the judge.
      note.appendChild(el(
        'div', 'note',
        `LLM summaries are off: ${health.llm.reasons.join('; ')}. `
        + 'Everything else works — sessions fall back to the title Claude Code '
        + 'writes itself.',
      ));
    }
  } catch (error) {
    note.appendChild(el('div', 'error', `Could not reach the backend: ${error.message}`));
  }
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

  const hasSummary = Boolean(session.summary);
  row.appendChild(el(
    'div',
    hasSummary ? 'row-summary' : 'row-summary empty',
    hasSummary ? session.summary : 'no summary yet',
  ));

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

// `quiet` re-renders in place without the loading flash — used by the poll
// that picks up summaries as they finish arriving.
async function loadSessions(quiet = false) {
  const container = document.getElementById('sessions');
  if (!quiet) container.replaceChildren(el('div', 'empty-state', 'Loading sessions…'));

  const params = new URLSearchParams();
  if (state.project) params.set('project', state.project);
  if (state.severity && state.severity !== 'any') params.set('severity', state.severity);

  try {
    const data = await getJSON(`/api/sessions?${params}`);
    const sessions = data.sessions;
    if (!sessions.length) {
      container.replaceChildren(el('div', 'empty-state', 'No sessions found.'));
    } else {
      const list = el('div', 'list');
      sessions.forEach((session) => list.appendChild(sessionRow(session)));
      container.replaceChildren(list);
    }
    trackPending(data.pending_summaries || 0);
  } catch (error) {
    if (!quiet) {
      container.replaceChildren(el('div', 'error', `Failed to load sessions: ${error.message}`));
    }
  }
}

// --- summaries fill in on their own -----------------------------------

function trackPending(pending) {
  const label = document.getElementById('pending');
  label.textContent = pending
    ? `summarizing ${pending} session${pending > 1 ? 's' : ''}…`
    : '';

  // One timer at a time, and none once everything has landed.
  if (state.poll) { clearTimeout(state.poll); state.poll = null; }
  if (pending > 0) {
    state.poll = setTimeout(() => loadSessions(true), 2500);
  }
}

// --- boot -------------------------------------------------------------

initTabs();
loadHealth();
loadProjects();
loadSessions();
