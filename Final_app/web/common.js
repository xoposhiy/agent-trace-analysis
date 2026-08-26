// Shared helpers for every page.

// --- fetching ---------------------------------------------------------

async function getJSON(url, options) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = response.statusText;
    try {
      const body = await response.json();
      detail = body.detail || body.error || detail;
    } catch (_) { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return response.json();
}

// --- formatting -------------------------------------------------------

// "3 minutes ago" for anything recent, an absolute date once that stops being
// the useful framing. The session list is sorted by time, so relative labels
// carry most of the meaning at the top and none at the bottom.
function relativeTime(iso) {
  if (!iso) return 'unknown';
  const then = new Date(iso);
  const seconds = (Date.now() - then.getTime()) / 1000;
  if (seconds < 60) return 'just now';
  if (seconds < 3600) return `${Math.floor(seconds / 60)} min ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)} h ago`;
  if (seconds < 7 * 86400) return `${Math.floor(seconds / 86400)} d ago`;
  return then.toLocaleDateString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
  });
}

function absoluteTime(iso) {
  if (!iso) return '';
  return new Date(iso).toLocaleString(undefined, {
    year: 'numeric', month: 'short', day: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function formatNumber(n) {
  if (n === null || n === undefined) return '0';
  if (n >= 1e6) return (n / 1e6).toFixed(1) + 'M';
  if (n >= 1e3) return (n / 1e3).toFixed(1) + 'k';
  return String(n);
}

function formatDuration(seconds) {
  if (!seconds || seconds < 1) return '—';
  if (seconds < 60) return `${Math.round(seconds)}s`;
  if (seconds < 3600) return `${Math.round(seconds / 60)}m`;
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.round((seconds % 3600) / 60);
  return minutes ? `${hours}h ${minutes}m` : `${hours}h`;
}

// --- token figures ----------------------------------------------------

// One block's CUMULATIVE billed tokens across the whole session, split the
// way `analysis.attribution` places it. `working` is input, output and
// cache writes — what this block's own content cost to put into the prompt.
// `cacheRead` is what every later call paid to re-read it while it stayed
// resident, which for an early `Read` of a big file is most of what it
// really cost.
//
// `total` is what the bar's "tokens" axis paints, and it is unbounded — it
// grows with every later call that re-reads this block's content, unlike
// the separate, bounded `context_tokens` field (DESIGN.md §7, the "context"
// axis and the session/agent header's "Context window" stat). The two
// channels here are returned separately because CLAUDE.md §7 forbids
// presenting their sum as work done — cache reads are ~95% of a real
// session, so that reads ~18x high. Every caller that shows `total` shows
// `tokenBreakdown` beside it.
//
// Falls back a step at a time, so a payload from before either channel existed
// still renders a number rather than NaN.
function tokenSplit(block) {
  const working = typeof block.attributed_tokens === 'number'
    ? block.attributed_tokens : block.tokens.working;
  const cacheRead = typeof block.attributed_cache_read === 'number'
    ? block.attributed_cache_read : 0;
  return { working, cacheRead, total: working + cacheRead };
}

// The token facts a hover readout shows for one block: the total, and how much
// of it is later calls re-reading this content. The re-read figure is a span of
// its own rather than only a `title`, because it is usually the larger half and
// asking for a second hover to reach it made the bar's heights unexplainable.
//
// Omitted entirely when there is none, so a session with no cache reads is not
// given a bare "0 re-read".
function tokenFacts(split) {
  const total = el('span', null, `${formatNumber(split.total)} tokens`);
  total.title = tokenBreakdown(split);
  if (!split.cacheRead) return [total];

  const percent = Math.round(100 * split.cacheRead / Math.max(1, split.total));
  const reread = el('span', null,
    `${formatNumber(split.cacheRead)} re-read (${percent}%)`);
  reread.title = `${split.cacheRead.toLocaleString()} tokens later calls paid to`
    + ` re-read this block's content while it stayed in the context window`;
  return [total, reread];
}

function tokenBreakdown(split) {
  return `${split.total.toLocaleString()} tokens billed to this block across the session`
    + ` · ${split.working.toLocaleString()} input, output and cache writes`
    + ` · ${split.cacheRead.toLocaleString()} re-read from cache by later calls`;
}

// --- cost figures -------------------------------------------------------

// Below a cent, two decimal places rounds to "$0.00" for almost every block
// on a real bar — most cost a fraction of a cent — so small amounts get more
// digits rather than reading as free.
function formatCost(dollars) {
  if (!dollars) return '$0.00';
  if (dollars < 0.01) return `$${dollars.toFixed(4)}`;
  return `$${dollars.toFixed(2)}`;
}

// The hover fact for a block's dollar share, alongside `tokenFacts`. Unlike
// the token split, a cost has no cache-read-shaped trap to guard against —
// CLAUDE.md §7 is about summing cache reads into a token *count*, and a
// dollar spent re-reading cache is exactly as real as a dollar spent on
// output — so this is one figure, not a pair.
function costFact(block) {
  const cost = typeof block.attributed_cost === 'number' ? block.attributed_cost : 0;
  const fact = el('span', null, formatCost(cost));
  fact.title = 'This block\'s share of the session\'s bill — priced per call at'
    + ' the model that call actually ran on, divided across blocks the same'
    + ' way the token figures above are.';
  return fact;
}

// --- DOM --------------------------------------------------------------

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function stat(key, value, title) {
  const box = el('div');
  box.appendChild(el('div', 'stat-k', key));
  const valueNode = el('div', 'stat-v', value);
  if (title) valueNode.title = title;
  box.appendChild(valueNode);
  return box;
}

// --- session-level header stats -----------------------------------------
//
// The exact stat row the session page leads with (DESIGN.md §7). Shared so
// every page that shows "what is this session" — the session page itself,
// and a detected problem's own page — agrees on the same numbers rather
// than each computing its own slightly different version.
function sessionStats(session) {
  // "Context window" is the real, bounded size of the main thread's LAST
  // call (`input + cache_read + cache_creation` for that one call) — not
  // `session.tokens.total`, which sums `cache_read` across every call and so
  // recounts the same resident content once per later call that re-read it
  // (a real session here: 147M summed vs. 637K in the largest single call).
  // See DESIGN.md §7 and `Session.context_window_tokens`.
  const tokenStat = stat('Context window', formatNumber(session.context_window_tokens));
  tokenStat.title = `${session.tokens.working.toLocaleString()} generated this session`
    + ` (excludes cache reads) · in ${session.tokens.input.toLocaleString()}`
    + ` · out ${session.tokens.output.toLocaleString()}`
    + ` · cache write ${session.tokens.cache_creation.toLocaleString()}`
    + ` · cache read, cumulative across every call ${session.tokens.cache_read.toLocaleString()}`;

  // Sums to exactly the same figure as the bar underneath it — every block's
  // `attributed_cost` added up — because both come from the one attribution
  // pass in `analysis.attribution`. See `Session.attributed_cost`. This is
  // cumulative across the WHOLE session, unlike "Context window" beside it,
  // which is deliberately bounded to one call — the two are not meant to be
  // compared 1:1, they answer different questions (what did this cost, vs.
  // what does the context look like right now).
  const costStat = stat('Retrospective cost', formatCost(session.attributed_cost));
  costStat.title = 'Priced per call at Anthropic\'s rate for the model that call'
    + ' actually ran on — an attribution across the session\'s blocks, summed'
    + ' across every call this session ever made.';

  return [
    tokenStat,
    costStat,
    stat('Messages', formatNumber(session.message_count)),
    stat('Tool calls', formatNumber(session.tool_call_count)),
    stat('Subagents', String(session.subagent_count)),
    stat('Duration', formatDuration(session.duration_s)),
    stat('Compactions', String(session.compaction_points.length)),
  ];
}

// --- the Y-axis total, next to the metric selector -----------------------
//
// The bar's Y axis answers a different question depending on the metric —
// "context" is one bounded call, "cost"/"tokens" are cumulative across the
// whole thread — and that difference is exactly the thing people get
// confused about (DESIGN.md §7). So whichever metric is selected gets its
// own total restated in plain language right next to the selector, instead
// of leaving the reader to infer it from the bar's shape alone.
//
// ``subject`` is a Session (session.js/problem.js) or a subagent Block
// (agent.js) — the two field names that differ between them
// (`context_window_tokens` vs `context_tokens`) are both tried.
function metricTotalText(subject, metric) {
  const contextTokens = subject.context_window_tokens ?? subject.context_tokens ?? 0;

  if (metric === 'context') {
    return `${contextTokens.toLocaleString()} tokens — the real, bounded size`
      + ` of this thread's last API call, not summed across every call.`;
  }
  if (metric === 'cost') {
    // The cumulative token count belongs beside the dollar figure, not the
    // "context" one above — both are the same "every call, re-billing
    // included" shape, unlike the bounded, one-call context-window figure.
    const totalTokens = (subject.tokens && subject.tokens.total) || 0;
    return `${formatCost(subject.attributed_cost)} (${totalTokens.toLocaleString()} tokens)`
      + ` — every token sent, including re-reads re-billed on every later API`
      + ` call, priced at each call's real rate.`;
  }
  if (metric === 'time') {
    return `${formatDuration(subject.duration_s)} — how long this thread ran,`
      + ` start to last message.`;
  }
  if (metric === 'messages') {
    return `${formatNumber(subject.message_count)} messages make up this thread.`;
  }
  return '';
}

function renderMetricTotal(container, subject, metric) {
  container.textContent = metricTotalText(subject, metric);
}
