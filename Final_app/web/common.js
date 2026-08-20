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

// One block's context-window cost, split the way `analysis.attribution` places
// it. `working` is input, output and cache writes — what this block's own
// content cost to put into the prompt. `cacheRead` is what every later call
// paid to re-read it while it stayed resident, which for an early `Read` of a
// big file is most of what it really cost.
//
// `total` is what the bar's token axis paints. The two channels are returned
// separately because CLAUDE.md §7 forbids presenting their sum as work done —
// cache reads are ~95% of a real session, so that reads ~18x high. Every caller
// that shows `total` shows `tokenBreakdown` beside it.
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
  return `${split.total.toLocaleString()} tokens of the session's context window`
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
