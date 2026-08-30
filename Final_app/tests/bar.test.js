// Unit tests for the pure layout helpers in web/bar.js (CLAUDE.md §6).
//
// bar.js is a plain browser script with no module system — that is deliberate,
// the project has no build step. So it is loaded here into a `vm` context with
// a stub `document`, and an epilogue is appended to hand the module-scoped
// constants out (top-level `const` in a vm script is lexical, not a property of
// the context, so function declarations come through by themselves but the
// constants would not).
//
// Run: node --test tests/bar.test.js   (or via pytest, tests/test_bar_js.py)

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

// --- loading bar.js ---------------------------------------------------

const SOURCE = fs.readFileSync(
  path.join(__dirname, '..', 'web', 'bar.js'), 'utf8');

const EPILOGUE = `
globalThis.constants = {
  BAR_WIDTH, INSET, GAP, MIN_BLOCK, MARKER_HEIGHT, MARKER_SLOT,
  FLEX_PER_BLOCK, FLEX_MIN, FLEX_MAX, KIND_STYLE, KIND_ORDER,
};
`;

// A DOM stub that records what was drawn, so the render tests can inspect it.
// Listeners are kept rather than dropped, so the click tests can fire them —
// a block is a link now, and "does clicking it open the right page" is not
// something a stub that swallows handlers can answer.
function fakeDocument(drawn) {
  return {
    createElementNS(_ns, tag) {
      const node = {
        tag,
        attrs: {},
        children: [],
        listeners: {},
        setAttribute(key, value) { this.attrs[key] = String(value); },
        appendChild(child) { this.children.push(child); return child; },
        addEventListener(name, handler) {
          (this.listeners[name] = this.listeners[name] || []).push(handler);
        },
        fire(name, event) {
          (this.listeners[name] || []).forEach((handler) => handler(event || {}));
        },
      };
      drawn.push(node);
      return node;
    },
  };
}

const drawn = [];
const context = vm.createContext({
  document: fakeDocument(drawn),
  // `el` lives in common.js; bar.js only uses it for the empty-state message.
  el: (tag, cls, text) => ({ tag, cls, text, appendChild() {} }),
});
vm.runInContext(SOURCE + EPILOGUE, context);

const { layoutBlocks, barHeight, isMarker, blockMetric } = context;
const C = context.constants;

// --- fixtures ---------------------------------------------------------

let nextId = 0;

function block(kind, { tokens = 0, duration = 0, messages = 1,
                       working, cacheRead, cost, context } = {}) {
  nextId += 1;
  const made = {
    id: nextId,
    kind,
    tokens: { working: tokens },
    duration_s: duration,
    message_count: messages,
    inner_blocks: [],
  };
  // Omitted entirely rather than defaulted, so the fallback chain in
  // `blockMetric` is exercised by every fixture that does not opt in.
  if (working !== undefined || cacheRead !== undefined) {
    made.attributed_tokens = working || 0;
    made.attributed_cache_read = cacheRead || 0;
    made.attributed_total = made.attributed_tokens + made.attributed_cache_read;
  }
  // Same omit-rather-than-default rule as above, so a fixture that never
  // opts into pricing exercises the "no price data yet" fallback.
  if (cost !== undefined) {
    made.attributed_cost = cost;
  }
  // The real, bounded context-window figure (DESIGN.md §7) — deliberately
  // its own field, never derived from `working`/`cacheRead` above, since the
  // whole point is that it is a different, smaller number.
  if (context !== undefined) {
    made.context_tokens = context;
  }
  return made;
}

// The shape of a real session: work runs separated by prose, with the human
// interrupting now and then. Proportions taken from session 51db4d3e.
function realisticBlocks(count) {
  const kinds = ['read', 'coordination', 'write', 'coordination',
                 'execute', 'coordination', 'user_chat'];
  return Array.from({ length: count }, (_, i) =>
    block(kinds[i % kinds.length], {
      tokens: (i % 11) * 800,
      duration: (i % 7) * 12,
      messages: 1 + (i % 9),
    }));
}

// The vertical runs a layout does *not* paint. A marker paints only its
// centred rule; everything else paints its whole slot. These runs are what the
// eye reads as holes in the bar, so their size is the thing worth asserting on
// — not the total unpainted area, which is dominated by the 1px inter-block
// gaps the mark spec calls for.
function unpaintedRuns(boxes, totalHeight) {
  const painted = boxes.map((box) => {
    if (!isMarker(box.block)) return [box.y, box.y + box.height];
    const rule = Math.min(box.height, C.MARKER_HEIGHT);
    const top = box.y + Math.max(0, (box.height - C.MARKER_HEIGHT) / 2);
    return [top, top + rule];
  });

  const runs = [];
  let cursor = 0;
  for (const [top, bottom] of painted) {
    if (top > cursor) runs.push(top - cursor);
    cursor = Math.max(cursor, bottom);
  }
  if (cursor < totalHeight) runs.push(totalHeight - cursor);
  return runs;
}

// --- the regression: no room to size anything -------------------------

test('barHeight leaves real proportional room, not just the per-block minimum',
  () => {
    // The old height (`blocks.length * 5 + 40`, with MIN_BLOCK 3 + GAP 2) asked
    // for exactly the layout's own minimum, so `flexible` came out 0-42px for a
    // whole bar and every block rendered at the 3px floor.
    for (const count of [1, 12, 120, 239, 359, 1000]) {
      const blocks = realisticBlocks(count);
      const height = barHeight(blocks);
      const reserved = blocks.reduce((acc, b) =>
        acc + (isMarker(b) ? C.MARKER_SLOT : C.MIN_BLOCK), 0)
        + (blocks.length - 1) * C.GAP;

      assert.ok(height - reserved >= C.FLEX_MIN,
        `${count} blocks: only ${height - reserved}px flexible, want >= ${C.FLEX_MIN}`);
    }
  });

test('the metric actually changes how tall a block is', () => {
  const blocks = [
    block('read', { tokens: 100, duration: 90, messages: 1 }),
    block('write', { tokens: 9000, duration: 1, messages: 1 }),
  ];
  const height = barHeight(blocks);

  const byTokens = layoutBlocks(blocks, 'tokens', height);
  const byTime = layoutBlocks(blocks, 'time', height);

  assert.ok(byTokens[1].height > byTokens[0].height,
    'the 9000-token block should dominate under "tokens"');
  assert.ok(byTime[0].height > byTime[1].height,
    'the 90-second block should dominate under "time"');
});

// --- the regression: black gaps ---------------------------------------

// A "hairline" is a gap the mark spec wants (GAP) plus, next to a marker, the
// slack around its rule. Anything wider is a hole.
const HAIRLINE = C.GAP + (C.MARKER_SLOT - C.MARKER_HEIGHT);

test('the layout never leaves a gap wider than a hairline', () => {
  for (const count of [8, 60, 168, 239, 359]) {
    for (const metric of ['tokens', 'time', 'messages']) {
      const blocks = realisticBlocks(count);
      const height = barHeight(blocks);
      const runs = unpaintedRuns(layoutBlocks(blocks, metric, height), height);
      const worst = Math.max(0, ...runs);

      assert.ok(worst <= HAIRLINE + 0.001,
        `${count}/${metric}: a ${worst.toFixed(1)}px hole (hairline is ${HAIRLINE}px)`);
    }
  }
});

test('the bar draws a filled track behind the blocks', () => {
  // The gaps are unavoidable; what made them read as black holes was that
  // nothing was drawn behind them, so the page background showed through.
  drawn.length = 0;
  const container = { replaceChildren() {}, appendChild() {} };
  context.renderBar(container, realisticBlocks(20), { metric: 'tokens' });

  const track = drawn.find((node) => node.tag === 'rect');
  assert.ok(track, 'expected a background rect');
  assert.ok(track.attrs.fill && track.attrs.fill !== 'none',
    `track is unfilled (fill=${track.attrs.fill}) — gaps will show the page`);
});

test('a marker takes a fixed slot however large its metric', () => {
  const blocks = [
    block('read', { tokens: 10 }),
    block('user_chat', { tokens: 0, duration: 3600, messages: 40 }),
    block('read', { tokens: 10 }),
  ];
  const height = barHeight(blocks);

  for (const metric of ['tokens', 'time', 'messages']) {
    const boxes = layoutBlocks(blocks, metric, height);
    assert.strictEqual(boxes[1].height, C.MARKER_SLOT,
      `under "${metric}" the marker reserved ${boxes[1].height}px`);
  }
});

test('a marker never starves the work blocks around it', () => {
  // 27 user_chat blocks in one real 239-block session; if each took a metric
  // share of the height it would reserve space it never paints.
  const blocks = realisticBlocks(140);
  const height = barHeight(blocks);
  const boxes = layoutBlocks(blocks, 'messages', height);
  const work = boxes.filter((box) => !isMarker(box.block));

  assert.ok(work.every((box) => box.height >= C.MIN_BLOCK));
});

// --- layout invariants ------------------------------------------------

test('the layout fills its height exactly and never overflows', () => {
  for (const count of [1, 2, 37, 239]) {
    for (const metric of ['tokens', 'time', 'messages']) {
      const blocks = realisticBlocks(count);
      const height = barHeight(blocks);
      const boxes = layoutBlocks(blocks, metric, height);
      const last = boxes[boxes.length - 1];

      assert.ok(last.y + last.height <= height + 0.001,
        `${count}/${metric}: ran ${last.y + last.height - height}px past the end`);
      assert.ok(last.y + last.height > height - 1,
        `${count}/${metric}: stopped ${height - last.y - last.height}px short`);
    }
  }
});

test('no block is ever smaller than the visible minimum', () => {
  const blocks = realisticBlocks(200);
  const boxes = layoutBlocks(blocks, 'tokens', barHeight(blocks));

  assert.ok(boxes.every((box) => box.height >= C.MIN_BLOCK));
});

test('blocks are laid out in order and never overlap', () => {
  const blocks = realisticBlocks(50);
  const boxes = layoutBlocks(blocks, 'time', barHeight(blocks));

  for (let i = 1; i < boxes.length; i += 1) {
    assert.ok(boxes[i].y >= boxes[i - 1].y + boxes[i - 1].height,
      `block ${i} starts before block ${i - 1} ends`);
  }
  assert.deepStrictEqual(boxes.map((box) => box.block.id),
    blocks.map((b) => b.id));
});

test('an all-zero metric still spreads the blocks over the bar', () => {
  // Every block has 0 duration when a session is a burst of instant steps.
  const blocks = [block('read'), block('write'), block('execute')];
  const height = barHeight(blocks);
  const boxes = layoutBlocks(blocks, 'time', height);

  assert.ok(boxes.every((box) => box.height > C.MIN_BLOCK),
    'zero metric should fall back to an even split, not the bare floor');
  assert.strictEqual(Math.max(0, ...unpaintedRuns(boxes, height)), C.GAP);
});

// --- the regression: a budget nothing can spend ------------------------

test('a bar of markers only is as tall as the rules it paints', () => {
  // Real session d55c0e89 is a single `user_chat` block. A marker paints a
  // fixed 5px rule however tall its slot, so the 340px FLEX_MIN the height
  // asked for had no block that could absorb it: 6px painted in a 347px bar,
  // the remaining 98% bare track, on all three metrics.
  for (const count of [1, 2, 6]) {
    const blocks = Array.from({ length: count }, () => block('user_chat'));
    const height = barHeight(blocks);
    const painted = count * C.MARKER_HEIGHT + (count - 1) * C.GAP;

    assert.ok(height - painted <= count * (C.MARKER_SLOT - C.MARKER_HEIGHT) + count,
      `${count} markers: ${height - painted}px of the ${height}px bar is unpainted`);
  }
});

test('one work block among markers still absorbs the whole budget', () => {
  // The guard above must not fire whenever a marker is present — only when
  // nothing in the bar can take proportional space.
  const blocks = [block('user_chat'), block('read', { tokens: 500 }),
                  block('user_chat')];
  const height = barHeight(blocks);
  const boxes = layoutBlocks(blocks, 'tokens', height);

  assert.ok(boxes[1].height > C.FLEX_MIN - 1,
    `the one work block got ${boxes[1].height}px of a ${height}px bar`);
  assert.ok(Math.max(0, ...unpaintedRuns(boxes, height)) <= HAIRLINE);
});

test('a subagent band is one solid block, whatever it contains', () => {
  // It used to be a 28%-opacity ring with a stripe per child, which turned a
  // 12px band into 2px slivers nobody could read or click — and with no
  // children drew nothing at all, leaving what looked like a hole.
  for (const inner of [[], [block('read'), block('write'), block('execute')]]) {
    drawn.length = 0;
    const container = { replaceChildren() {}, appendChild() {} };
    const band = block('subagent');
    band.inner_blocks = inner;
    context.renderBar(container, [block('read', { tokens: 10 }), band],
      { metric: 'tokens' });

    const painted = drawn.filter((node) => node.tag === 'rect'
      && node.attrs.fill === C.KIND_STYLE.subagent.fill);

    assert.strictEqual(painted.length, 1,
      `${inner.length} children: expected one solid band, got ${painted.length} rects`);
    assert.strictEqual(painted[0].attrs['fill-opacity'], undefined,
      'the band must be fully opaque, not a translucent ring');
  }
});

test('a subagent band spans the full width of the bar', () => {
  // Width is what says "delegated" rather than "one more step in sequence",
  // and it is the secondary encoding orange no longer needs but keeps.
  drawn.length = 0;
  const container = { replaceChildren() {}, appendChild() {} };
  context.renderBar(container, [block('read', { tokens: 10 }), block('subagent')],
    { metric: 'tokens' });

  const band = drawn.find((node) => node.tag === 'rect'
    && node.attrs.fill === C.KIND_STYLE.subagent.fill);
  const ordinary = drawn.find((node) => node.tag === 'rect'
    && node.attrs.fill === C.KIND_STYLE.read.fill);

  assert.strictEqual(Number(band.attrs.width), C.BAR_WIDTH - 2);
  assert.ok(Number(ordinary.attrs.width) < Number(band.attrs.width),
    'an ordinary block must stay inset so the band reads as wider');
});

test('a bar with no blocks lays out nothing', () => {
  // strictEqual on length, not deepStrictEqual on the array: the array comes
  // from the vm realm, so its prototype is not this realm's Array.prototype.
  assert.strictEqual(layoutBlocks([], 'tokens', 500).length, 0);
  assert.strictEqual(barHeight([]), C.FLEX_MIN);
});

// --- opening a block ---------------------------------------------------

// Every block links to its own page, addressed by position. An off-by-one
// here silently opens the wrong stretch of work, which looks like a data bug
// rather than a routing one — so the index is pinned.

function renderWith(blocks, options) {
  drawn.length = 0;
  const container = { replaceChildren() {}, appendChild() {} };
  context.renderBar(container, blocks, options);
  return drawn.filter((node) => node.tag === 'g');
}

test('clicking a block opens that block, by its position in the bar', () => {
  const opened = [];
  const blocks = realisticBlocks(6);
  const groups = renderWith(blocks, {
    metric: 'tokens',
    onOpen: (block, index) => opened.push([block.id, index]),
  });

  assert.strictEqual(groups.length, blocks.length);
  groups[3].fire('click');
  groups[0].fire('click');

  assert.deepStrictEqual(opened, [[blocks[3].id, 3], [blocks[0].id, 0]]);
});

test('a block opens from the keyboard as well as the mouse', () => {
  // The bar is reachable by Tab and a block can be 3px tall, so the keyboard
  // path is the accessible one — and an SVG <g> gets nothing for free.
  const opened = [];
  const groups = renderWith(realisticBlocks(3), {
    metric: 'tokens',
    onOpen: (_block, index) => opened.push(index),
  });

  let defaultPrevented = 0;
  const key = (k) => ({ key: k, preventDefault() { defaultPrevented += 1; } });
  groups[1].fire('keydown', key('Enter'));
  groups[2].fire('keydown', key(' '));
  groups[0].fire('keydown', key('a'));

  assert.deepStrictEqual(opened, [1, 2], 'Enter and Space open; other keys do not');
  assert.strictEqual(defaultPrevented, 2, 'Space must not also scroll the page');
});

test('a clickable block announces itself as a link', () => {
  const groups = renderWith(realisticBlocks(2), {
    metric: 'tokens', onOpen: () => {},
  });

  assert.strictEqual(groups[0].attrs.role, 'link');
  assert.ok(groups[0].attrs['aria-label'].includes('block 1'));
});

test('with no handler a block is not presented as clickable', () => {
  // The bar is reused elsewhere; it must not claim to be a link when nothing
  // will happen on click.
  const groups = renderWith(realisticBlocks(2), { metric: 'tokens' });

  assert.strictEqual(groups[0].attrs.role, 'img');
  assert.ok(!groups[0].attrs.class.includes('blk-open'));
  groups[0].fire('click');   // must not throw
});

test('a subagent band opens like any other block', () => {
  const band = block('subagent');
  band.inner_blocks = [block('read', { tokens: 5 }), block('write', { tokens: 5 })];
  const opened = [];
  const groups = renderWith([block('read', { tokens: 100 }), band], {
    metric: 'tokens',
    onOpen: (_b, index) => opened.push(index),
  });

  groups[1].fire('click');
  assert.deepStrictEqual(opened, [1]);
});

// --- the metric accessor ----------------------------------------------

test('each metric reads its own field', () => {
  const b = block('read', { tokens: 12, duration: 34, messages: 5, cost: 0.03 });

  assert.strictEqual(blockMetric(b, 'tokens'), 12);
  assert.strictEqual(blockMetric(b, 'time'), 34);
  assert.strictEqual(blockMetric(b, 'messages'), 5);
  assert.strictEqual(blockMetric(b, 'cost'), 0.03);
  assert.strictEqual(blockMetric(b, 'money'), 12, 'unknown metric falls back to tokens');
});

test('the token metric is the whole CUMULATIVE billed total, cache reads included', () => {
  // A Read of a big file is cheap to issue and expensive to keep: on real
  // session e6e482e6 one `Read /tmp/tracelens.png` block was 6,999 working
  // tokens and 9,308,775 in later re-reads. Sizing by the working half alone
  // drew that block as one of the smallest on the bar. This is unbounded —
  // see the separate `context` metric below for the bounded figure.
  const b = block('read', { tokens: 12, working: 6999, cacheRead: 9308775 });

  assert.strictEqual(blockMetric(b, 'tokens'), 9315774);
});

test('the context metric reads context_tokens, a different field from tokens', () => {
  // DESIGN.md §7: bounded by one real call's actual billed size, never a sum
  // across calls, so it must not be derived from working/cacheRead at all.
  const b = block('read', { tokens: 12, working: 6999, cacheRead: 9308775, context: 637 });

  assert.strictEqual(blockMetric(b, 'context'), 637);
  assert.notStrictEqual(blockMetric(b, 'context'), blockMetric(b, 'tokens'),
    'context and tokens answer different questions and must not collapse to the same number');
});

test('the context metric falls back to 0 on an older payload', () => {
  const noContextField = block('read', { tokens: 12, working: 6999, cacheRead: 9308775 });

  assert.strictEqual(blockMetric(noContextField, 'context'), 0);
});

test('the token metric falls back a step at a time on older payloads', () => {
  const noCacheChannel = block('read', { tokens: 12 });
  noCacheChannel.attributed_tokens = 500;

  assert.strictEqual(blockMetric(noCacheChannel, 'tokens'), 500,
    'a payload with no cache-read field should use the working figure');
  assert.strictEqual(blockMetric(block('read', { tokens: 12 }), 'tokens'), 12,
    'a payload with no attribution at all should use tokens.working');
});

test('the cost metric reads attributed_cost, not tokens', () => {
  const cheap = block('read', { tokens: 9000, cost: 0.002 });
  const pricey = block('write', { tokens: 10, cost: 5.5 });

  assert.strictEqual(blockMetric(cheap, 'cost'), 0.002);
  assert.strictEqual(blockMetric(pricey, 'cost'), 5.5);
});

test('the cost metric can size a bar the opposite way from the token metric', () => {
  // A block that generated few tokens on an expensive model can cost more
  // than one that generated many tokens on a cheap one — the whole reason
  // "retrospective cost" is its own axis rather than a rescaled token count.
  const blocks = [
    block('read', { tokens: 9000, cost: 0.01 }),
    block('write', { tokens: 10, cost: 5.5 }),
  ];
  const height = barHeight(blocks);

  const byTokens = layoutBlocks(blocks, 'tokens', height);
  const byCost = layoutBlocks(blocks, 'cost', height);

  assert.ok(byTokens[0].height > byTokens[1].height,
    'the 9000-token block should dominate under "tokens"');
  assert.ok(byCost[1].height > byCost[0].height,
    'the $5.50 block should dominate under "cost"');
});

test('the cost metric falls back to 0 on a payload with no price data yet', () => {
  const noPricing = block('read', { tokens: 12 });

  assert.strictEqual(blockMetric(noPricing, 'cost'), 0);
});

test('a cache-read-heavy bar still lays out without holes or overflow', () => {
  // Including cache reads makes the distribution far heavier-tailed than the
  // working figure: measured on session 51db4d3e the largest block came out at
  // 6,312,920 and the smallest at 1. The layout has to survive that — a single
  // block taking most of the flexible budget is the case CLAUDE.md §7 warns
  // leaves the rest on the floor, and a floor is fine but a hole is not.
  const blocks = Array.from({ length: 120 }, (_, i) =>
    block(i % 7 === 6 ? 'user_chat' : 'read', {
      working: 5000 + (i % 11) * 700,
      // Three blocks hold nearly all the re-read cost.
      cacheRead: i === 4 ? 6312920 : i === 40 ? 5291979 : i === 90 ? 4930478 : (i % 5),
    }));
  const height = barHeight(blocks);
  const boxes = layoutBlocks(blocks, 'tokens', height);
  const last = boxes[boxes.length - 1];

  assert.ok(Math.max(0, ...unpaintedRuns(boxes, height)) <= HAIRLINE + 0.001,
    'a heavy tail must not tear holes in the bar');
  assert.ok(boxes.every((box) => box.height >= C.MIN_BLOCK),
    'every block keeps the visible floor however small its share');
  assert.ok(last.y + last.height <= height + 0.001, 'the layout overflowed');
  assert.ok(boxes[4].height > boxes[5].height * 5,
    'the block holding the context should visibly dominate');
});

test('only chatting-with-user is a marker', () => {
  assert.ok(isMarker(block('user_chat')));
  for (const kind of ['read', 'write', 'execute', 'coordination', 'subagent']) {
    assert.ok(!isMarker(block(kind)), `${kind} should not be a marker`);
  }
  assert.ok(!isMarker(block('something-new')), 'an unknown kind is not a marker');
});
