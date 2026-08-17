// Unit tests for the pure helpers in web/common.js (CLAUDE.md §6).
//
// Loaded the same way as bar.test.js: common.js is a plain browser script with
// no module system, so it is run in a `vm` context and its function
// declarations come through as context properties by themselves.
//
// The subject here is `tokenSplit`, which every page uses to turn a payload
// into the pair of figures it shows. It is the one place the fallback chain for
// older payloads lives, so it is the one place that chain can be pinned.
//
// Run: node --test tests/common.test.js   (or via pytest, tests/test_bar_js.py)

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const SOURCE = fs.readFileSync(
  path.join(__dirname, '..', 'web', 'common.js'), 'utf8');

// `el` builds real elements, so `tokenFacts` needs a document that can make
// them. Only the three properties `el` touches are stubbed.
const context = vm.createContext({
  document: {
    createElement: (tag) => ({ tag, className: '', textContent: '', title: '' }),
  },
  fetch: () => {},
});
vm.runInContext(SOURCE, context);

const { tokenSplit, tokenBreakdown, tokenFacts, formatNumber, formatDuration } = context;

// --- the two channels -------------------------------------------------

test('a split keeps the channels apart and reports their sum', () => {
  const split = tokenSplit({
    tokens: { working: 999 },
    attributed_tokens: 6999,
    attributed_cache_read: 9308775,
  });

  assert.strictEqual(split.working, 6999);
  assert.strictEqual(split.cacheRead, 9308775);
  assert.strictEqual(split.total, 9315774);
});

test('a payload with no cache-read channel reports zero, not undefined', () => {
  // The arithmetic has to survive an older payload: `undefined` here would
  // propagate into the total and render as NaN.
  const split = tokenSplit({ tokens: { working: 400 }, attributed_tokens: 500 });

  assert.strictEqual(split.cacheRead, 0);
  assert.strictEqual(split.total, 500);
});

test('a payload with no attribution at all falls back to the billed figure', () => {
  const split = tokenSplit({ tokens: { working: 400 } });

  assert.strictEqual(split.working, 400);
  assert.strictEqual(split.total, 400);
});

test('a zero cache read is respected rather than treated as missing', () => {
  // `0` is falsy, so a `||` fallback here would silently discard a real zero
  // and fall through to the billed figure instead.
  const split = tokenSplit({
    tokens: { working: 400 },
    attributed_tokens: 500,
    attributed_cache_read: 0,
  });

  assert.strictEqual(split.total, 500);
  assert.strictEqual(split.cacheRead, 0);
});

test('the breakdown names both channels and the total', () => {
  // Whenever the total is shown, this is what has to be shown beside it —
  // CLAUDE.md §7 forbids a bare sum of cache reads standing in for work done.
  const text = tokenBreakdown(tokenSplit({
    tokens: { working: 0 },
    attributed_tokens: 6999,
    attributed_cache_read: 9308775,
  }));

  assert.ok(text.includes((9315774).toLocaleString()), 'missing the total');
  assert.ok(text.includes((6999).toLocaleString()), 'missing the working figure');
  assert.ok(text.includes((9308775).toLocaleString()), 'missing the cache reads');
  assert.ok(/cache/i.test(text), 'never says which half is the cache reads');
});

// --- what the hover readout says --------------------------------------

test('the readout names the re-read half without a second hover', () => {
  // The re-read figure is usually the larger half and it is what explains a tall
  // block that barely did anything, so it has to be readable text — not only a
  // `title` on the total, which needed hovering the tooltip itself to reach.
  const facts = tokenFacts(tokenSplit({
    tokens: { working: 0 },
    attributed_tokens: 189,
    attributed_cache_read: 45_575,
  }));

  assert.strictEqual(facts.length, 2);
  assert.strictEqual(facts[0].textContent, '45.8k tokens');
  assert.strictEqual(facts[1].textContent, '45.6k re-read (100%)');
  assert.ok(/re-read this block's content/.test(facts[1].title));
});

test('a block with no re-reads shows no re-read fact at all', () => {
  // Rather than a bare "0 re-read (0%)" on every block of a session that never
  // hit the cache.
  const facts = tokenFacts(tokenSplit({
    tokens: { working: 0 },
    attributed_tokens: 500,
    attributed_cache_read: 0,
  }));

  assert.strictEqual(facts.length, 1);
  assert.strictEqual(facts[0].textContent, '500 tokens');
});

test('the re-read percentage is of the total, not of the work', () => {
  const facts = tokenFacts(tokenSplit({
    tokens: { working: 0 },
    attributed_tokens: 250,
    attributed_cache_read: 750,
  }));

  assert.strictEqual(facts[1].textContent, '750 re-read (75%)');
});

// --- formatting -------------------------------------------------------

test('large token counts are abbreviated, small ones are not', () => {
  // Cache-read figures run to eight digits, so the millions case is now the
  // common one on the bar rather than an outlier.
  assert.strictEqual(formatNumber(9315774), '9.3M');
  assert.strictEqual(formatNumber(6999), '7.0k');
  assert.strictEqual(formatNumber(999), '999');
  assert.strictEqual(formatNumber(0), '0');
  assert.strictEqual(formatNumber(undefined), '0');
});

test('a duration reads in the largest unit that fits', () => {
  assert.strictEqual(formatDuration(0), '—');
  assert.strictEqual(formatDuration(45), '45s');
  assert.strictEqual(formatDuration(600), '10m');
  assert.strictEqual(formatDuration(3600), '1h');
  assert.strictEqual(formatDuration(5400), '1h 30m');
});
