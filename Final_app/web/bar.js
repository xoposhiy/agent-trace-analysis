// The vertical session bar: one coloured block per stretch of activity.
//
// Colour assignment follows the dataviz method, and every pair was checked
// with `validate_palette.js --mode dark --pairs all`. All-pairs is the right
// test here (not adjacent-only) because any kind can neighbour any other on a
// timeline.
//
// The hues are specified: green read, blue write, red execute, orange
// subagents, purple chatting-with-user, grey coordination. Within each hue the
// *step* was chosen by search to maximise separation.
//
//   node validate_palette.js \
//     "#199e70,#3987e5,#d64545,#e59400,#a3adbb,#ad5fc9" --mode dark --pairs all
//   [PASS] Normal-vision floor worst #ad5fc9↔#3987e5  ΔE 17.0  (≥15 floor)
//   [PASS] Contrast vs surface all 6 ≥ 3:1
//   [FAIL] CVD separation      worst #ad5fc9↔#3987e5  ΔE 5.7
//
// Three notes on the remaining failures, all of them understood:
//
//  1. The CVD failure is purple↔blue, which is inherent to having both in one
//     palette — no purple step clears it (best found: 5.7). It is tolerable
//     here *only* because purple is never drawn as a fill: chatting-with-user
//     is a 5px full-width marker, so it never appears as a same-shape block
//     beside a blue one. Form carries the distinction, per the skill's
//     secondary-encoding allowance.
//     Orange used to rely on the same argument, back when a subagent was drawn
//     as a translucent ring around its children. It is now a solid fill like
//     any other kind, so the pair was re-checked rather than assumed:
//
//       node validate_palette.js "#d64545,#e59400" --mode dark --pairs all
//       [PASS] CVD separation  #e59400↔#d64545  ΔE 14.7 deutan · 15.7 tritan
//
//     It clears the floor twice over, so orange needs no help from form. It
//     keeps the full-bar width anyway, because that is what says "delegated"
//     rather than "one more colour in the sequence".
//     Restricted to the five kinds that render as fills
//     (read/write/execute/coordination/subagent) the worst CVD pair is still
//     red↔green at 6.2 — see note 2 — and normal-vision 18.0.
//
//  2. The green is deliberately teal-leaning (#199e70) rather than a "true"
//     green. With red in the palette a true green (#2f9e44) drops the red↔green
//     CVD pair to ΔE 1.2 — indistinguishable for deuteranopia, the single most
//     common form. Teal-green raises it to 6.2.
//
//  3. Grey trips the chroma floor by design — it is meant to read as grey — and
//     grey and orange sit outside the categorical lightness band on purpose,
//     since neither is a peer series competing with the work hues.
const KIND_STYLE = {
  read:         { fill: '#199e70', label: 'read' },
  write:        { fill: '#3987e5', label: 'write' },
  execute:      { fill: '#d64545', label: 'execute' },
  coordination: { fill: '#a3adbb', label: 'coordination' },
  // `wide` spans the full bar instead of sitting inset: delegated work reads
  // as a band across the timeline rather than one more step in the sequence.
  // It is a solid fill, not a ring around its children — a run of subagents is
  // one act of delegation, and drawing each child inside it made a 12px block
  // into a stack of 2px slivers nobody could read or click.
  subagent:     { fill: '#e59400', label: 'subagents', wide: true },
  user_chat:    { fill: '#ad5fc9', label: 'chatting with user', marker: true },
};

const KIND_ORDER = ['read', 'write', 'execute', 'coordination', 'subagent', 'user_chat'];

// --- geometry ---------------------------------------------------------

const BAR_WIDTH = 46;      // full width; contained blocks are inset
const INSET = 7;           // how far a normal block sits inside the bar
const GAP = 1;             // separation between fills
const MIN_BLOCK = 3;       // no block is ever invisible, however small
const MARKER_HEIGHT = 5;   // "chatting with user" and compaction rules
const RADIUS = 2;

// A "chatting with user" block paints a 5px rule, so any slot taller than this
// is space it reserves and never fills. It is therefore laid out at a fixed
// size instead of taking a metric share: with 27 of them in a real 239-block
// session, metric-sized marker slots were a leading source of apparent holes
// in the bar.
const MARKER_SLOT = MARKER_HEIGHT + 2;

// How much room the proportional part of the layout gets, over and above the
// per-block minimum. Without a real budget here the metric selector does
// nothing: the old height (`blocks * 5 + 40`, with MIN_BLOCK 3 and GAP 2) asked
// for exactly the minimum the layout needed, leaving 0-42px of flexible space
// for the whole bar — measured on six real sessions, every block came out at
// the 3px floor whichever metric was chosen, and sessions past ~170 blocks
// overflowed the viewBox and were clipped.
const FLEX_PER_BLOCK = 7;
const FLEX_MIN = 340;
const FLEX_MAX = 1100;

const SVG_NS = 'http://www.w3.org/2000/svg';

function svgEl(tag, attrs) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    node.setAttribute(key, value);
  }
  return node;
}

// --- sizing -----------------------------------------------------------

// Every block gets MIN_BLOCK plus a share of the remaining height proportional
// to its metric. Pure proportional sizing makes a 1-event block vanish next to
// a 17-event one; pure equal sizing throws away the magnitude the Y-axis
// selector exists to show. This keeps both readable.
//
// Markers sit outside the proportional split entirely (see MARKER_SLOT).
function layoutBlocks(blocks, metric, totalHeight) {
  if (!blocks.length) return [];

  const isFixed = blocks.map((block) => isMarker(block));
  const values = blocks.map((block, index) =>
    isFixed[index] ? 0 : Math.max(0, blockMetric(block, metric)));
  const sum = values.reduce((a, b) => a + b, 0);
  const flexCount = isFixed.filter((fixed) => !fixed).length;

  const reserved = blocks.reduce((acc, _, index) =>
    acc + (isFixed[index] ? MARKER_SLOT : MIN_BLOCK), 0)
    + (blocks.length - 1) * GAP;
  const flexible = Math.max(0, totalHeight - reserved);

  let y = 0;
  return blocks.map((block, index) => {
    let height;
    if (isFixed[index]) {
      height = MARKER_SLOT;
    } else {
      const share = sum > 0
        ? values[index] / sum
        : (flexCount ? 1 / flexCount : 0);
      height = MIN_BLOCK + flexible * share;
    }
    const box = { block, y, height };
    y += height + GAP;
    return box;
  });
}

// The height the bar needs so that the metric actually drives block sizes.
// Returned rather than clamped to a screenful on purpose: a 359-block session
// cannot honour a 3px floor inside 880px, and silently squashing it is what
// produced a bar of uniform 3px slivers. The column scrolls instead.
function barHeight(blocks) {
  if (!blocks.length) return FLEX_MIN;
  const reserved = blocks.reduce((acc, block) =>
    acc + (isMarker(block) ? MARKER_SLOT : MIN_BLOCK), 0)
    + (blocks.length - 1) * GAP;

  // Only non-markers can absorb proportional space — a marker paints its 5px
  // rule whatever slot it is given. Asking for a flexible budget no block can
  // spend leaves the whole of it as bare track: session d55c0e89 is a single
  // `user_chat` block, and rendered as 6px of rule in a 347px bar on all three
  // metrics, which reads as the bar being one long hole.
  if (!blocks.some((block) => !isMarker(block))) return Math.round(reserved);

  const flexible = Math.min(FLEX_MAX,
    Math.max(FLEX_MIN, blocks.length * FLEX_PER_BLOCK));
  return Math.round(reserved + flexible);
}

function isMarker(block) {
  const style = KIND_STYLE[block.kind];
  return Boolean(style && style.marker);
}

function blockMetric(block, metric) {
  if (metric === 'time') return block.duration_s;
  if (metric === 'messages') return block.message_count;
  // Dollars, priced per call at that call's own model (`analysis.pricing`),
  // then divided across blocks by the same ledger that divides tokens — so a
  // subagent that ran on a cheaper model does not inflate the main thread's
  // bar just because it shares one session. Missing on a payload from before
  // this field existed: 0 rather than NaN, same "degrade, don't crash" rule
  // as everywhere else in this function.
  if (metric === 'cost') {
    return typeof block.attributed_cost === 'number' ? block.attributed_cost : 0;
  }
  // `attributed_total`, not `tokens.working`: the latter charges a message's
  // whole prompt-side cost to whichever Event came first in it, which put
  // 325,412 tokens on a single `Read` that did not cause any of them. The
  // attributed figures divide the same totals across whatever caused them, and
  // still sum to the header exactly.
  //
  // The total includes cache reads, so the axis measures the whole context
  // window: a block that put a big file into the prompt is charged for every
  // later call that re-read it, which is most of what it really cost. Payloads
  // from before the cache-read channel fall back one step at a time.
  //
  // Known cost of this, measured on three real sessions (2026-08-17): cache
  // reads are far heavier-tailed than the working figure, so more of the bar
  // sits on the MIN_BLOCK floor. On 51db4d3e, 106 of 180 flexible blocks grow
  // less than 1px (45 did under `working`), and the median block grows 0.52px
  // rather than 2.39px. That is the distribution being faithful — the top five
  // blocks really are half the context-window cost — but it does mean small
  // blocks are no longer comparable to each other on this axis. A compressed
  // (sqrt/log) scale would restore that at the price of heights no longer being
  // proportional to cost; it has not been done, deliberately.
  if (typeof block.attributed_total === 'number') return block.attributed_total;
  if (typeof block.attributed_tokens === 'number') return block.attributed_tokens;
  return block.tokens.working;
}

// --- rendering --------------------------------------------------------

function renderBar(container, blocks, options) {
  const metric = (options && options.metric) || 'cost';
  const onHover = options && options.onHover;
  const onOpen = options && options.onOpen;

  container.replaceChildren();
  if (!blocks.length) {
    container.appendChild(el('div', 'placeholder', 'No blocks'));
    return;
  }

  const height = (options && options.height) || barHeight(blocks);

  const svg = svgEl('svg', {
    width: BAR_WIDTH,
    height,
    viewBox: `0 0 ${BAR_WIDTH} ${height}`,
    role: 'img',
    'aria-label': `Session activity, ${blocks.length} blocks, sized by ${metric}`,
  });

  // A filled track, not just an outline. Whatever the layout does not paint —
  // the gaps between fills, the slack around a marker rule — shows this rather
  // than the page background, which on this dark theme read as black holes
  // punched through the bar.
  svg.appendChild(svgEl('rect', {
    x: 0.5, y: 0.5, width: BAR_WIDTH - 1, height: height - 1,
    rx: 4, fill: 'var(--panel-2)', stroke: 'var(--line)',
  }));

  layoutBlocks(blocks, metric, height).forEach(({ block, y, height: h }, index) => {
    svg.appendChild(renderBlock(block, y, h, metric, onHover, onOpen, index));
  });

  container.appendChild(svg);
}

function renderBlock(block, y, height, metric, onHover, onOpen, index) {
  const style = KIND_STYLE[block.kind] || KIND_STYLE.coordination;
  const group = svgEl('g', {
    class: onOpen ? 'blk blk-open' : 'blk',
    tabindex: '0',
    // A block opens its own page, so it is a link — announced as one, and
    // reachable by keyboard, not only by a mouse that can hit a 3px target.
    role: onOpen ? 'link' : 'img',
    'aria-label': `${block.label}, block ${index + 1}`,
  });

  if (style.wide) {
    group.appendChild(svgEl('rect', {
      x: 1, y, width: BAR_WIDTH - 2, height,
      rx: RADIUS, fill: style.fill,
    }));
  } else if (style.marker) {
    // A thin full-width rule: the human interrupting, not a stretch of work.
    group.appendChild(svgEl('rect', {
      x: 1, y: y + Math.max(0, (height - MARKER_HEIGHT) / 2),
      width: BAR_WIDTH - 2, height: Math.min(height, MARKER_HEIGHT),
      rx: 1, fill: style.fill, 'fill-opacity': 0.85,
    }));
  } else {
    group.appendChild(svgEl('rect', {
      x: INSET, y, width: BAR_WIDTH - 2 * INSET, height,
      rx: RADIUS, fill: style.fill,
    }));
  }

  // A hit target that is always comfortably clickable, even for a 3px block.
  const hit = svgEl('rect', {
    x: 0, y: Math.max(0, y - 2), width: BAR_WIDTH,
    height: Math.max(10, height + 4), fill: 'transparent',
  });
  group.appendChild(hit);

  if (onHover) {
    group.addEventListener('mouseenter', (event) => onHover(block, event));
    group.addEventListener('focus', (event) => onHover(block, event));
    group.addEventListener('mouseleave', () => onHover(null));
    group.addEventListener('blur', () => onHover(null));
  }

  if (onOpen) {
    group.addEventListener('click', () => onOpen(block, index));
    // Enter and Space, because the group is a link by role but an SVG element
    // by nature: neither key activates it for free the way they would on <a>.
    group.addEventListener('keydown', (event) => {
      if (event.key === 'Enter' || event.key === ' ') {
        event.preventDefault();
        onOpen(block, index);
      }
    });
  }
  return group;
}

// --- legend -----------------------------------------------------------

// Identity is never colour-alone: the legend is always present, and each entry
// carries the same swatch shape the bar uses for that kind.
function renderLegend(container, counts) {
  container.replaceChildren();
  KIND_ORDER.forEach((kind) => {
    const style = KIND_STYLE[kind];
    const count = (counts && counts[kind]) || 0;
    const item = el('span', 'legend-item' + (count ? '' : ' legend-item-off'));

    const swatch = svgEl('svg', { width: 14, height: 14, class: 'swatch' });
    if (style.wide) {
      // Full width, matching how the bar draws it — the swatch carries the
      // shape as well as the colour, since shape is what separates a
      // delegation band from an ordinary block.
      swatch.appendChild(svgEl('rect', {
        x: 0, y: 1, width: 14, height: 12, rx: 2, fill: style.fill,
      }));
    } else if (style.marker) {
      swatch.appendChild(svgEl('rect', {
        x: 0, y: 5, width: 14, height: 4, rx: 1, fill: style.fill,
      }));
    } else {
      swatch.appendChild(svgEl('rect', {
        x: 2, y: 1, width: 10, height: 12, rx: 2, fill: style.fill,
      }));
    }
    item.appendChild(swatch);
    item.appendChild(el('span', null, style.label));
    if (count) item.appendChild(el('span', 'legend-count', String(count)));
    container.appendChild(item);
  });
}
