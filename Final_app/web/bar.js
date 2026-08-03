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
//     secondary-encoding allowance. Same for orange↔red: orange is only ever a
//     container ring.
//     Restricted to the four kinds that DO render as identical fills
//     (read/write/execute/coordination) the worst CVD pair is 6.2 and
//     normal-vision 18.0.
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
  subagent:     { fill: '#e59400', label: 'subagents', container: true },
  user_chat:    { fill: '#ad5fc9', label: 'chatting with user', marker: true },
};

const KIND_ORDER = ['read', 'write', 'execute', 'coordination', 'subagent', 'user_chat'];

// --- geometry ---------------------------------------------------------

const BAR_WIDTH = 46;      // full width; contained blocks are inset
const INSET = 7;           // how far a normal block sits inside the bar
const GAP = 2;             // the surface gap the mark spec requires between fills
const MIN_BLOCK = 3;       // no block is ever invisible, however small
const MARKER_HEIGHT = 5;   // "chatting with user" and compaction rules
const RADIUS = 2;

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
function layoutBlocks(blocks, metric, totalHeight) {
  if (!blocks.length) return [];

  const values = blocks.map((block) => Math.max(0, blockMetric(block, metric)));
  const sum = values.reduce((a, b) => a + b, 0);

  const fixed = blocks.length * MIN_BLOCK + (blocks.length - 1) * GAP;
  const flexible = Math.max(0, totalHeight - fixed);

  let y = 0;
  return blocks.map((block, index) => {
    const share = sum > 0 ? values[index] / sum : 1 / blocks.length;
    const height = MIN_BLOCK + flexible * share;
    const box = { block, y, height };
    y += height + GAP;
    return box;
  });
}

function blockMetric(block, metric) {
  if (metric === 'time') return block.duration_s;
  if (metric === 'messages') return block.message_count;
  return block.tokens.working;
}

// --- rendering --------------------------------------------------------

function renderBar(container, blocks, options) {
  const metric = (options && options.metric) || 'tokens';
  const height = (options && options.height) || 700;
  const onHover = options && options.onHover;

  container.replaceChildren();
  if (!blocks.length) {
    container.appendChild(el('div', 'placeholder', 'No blocks'));
    return;
  }

  const svg = svgEl('svg', {
    width: BAR_WIDTH,
    height,
    viewBox: `0 0 ${BAR_WIDTH} ${height}`,
    role: 'img',
    'aria-label': `Session activity, ${blocks.length} blocks, sized by ${metric}`,
  });

  // The bar outline, as in the sketch.
  svg.appendChild(svgEl('rect', {
    x: 0.5, y: 0.5, width: BAR_WIDTH - 1, height: height - 1,
    rx: 4, fill: 'none', stroke: 'var(--line)',
  }));

  layoutBlocks(blocks, metric, height).forEach(({ block, y, height: h }) => {
    svg.appendChild(renderBlock(block, y, h, metric, onHover));
  });

  container.appendChild(svg);
}

function renderBlock(block, y, height, metric, onHover) {
  const style = KIND_STYLE[block.kind] || KIND_STYLE.coordination;
  const group = svgEl('g', { class: 'blk', tabindex: '0' });

  if (style.container) {
    // Subagents: a full-width band holding their own blocks, so delegated work
    // reads as nested rather than as one more colour in the sequence.
    group.appendChild(svgEl('rect', {
      x: 1, y, width: BAR_WIDTH - 2, height,
      rx: RADIUS, fill: style.fill, 'fill-opacity': 0.28,
      stroke: style.fill, 'stroke-width': 1.5,
    }));
    const inner = block.inner_blocks || [];
    if (inner.length && height > 10) {
      layoutBlocks(inner, metric, height - 8).forEach((child) => {
        const childStyle = KIND_STYLE[child.block.kind] || KIND_STYLE.coordination;
        group.appendChild(svgEl('rect', {
          x: INSET + 2, y: y + 4 + child.y,
          width: BAR_WIDTH - 2 * (INSET + 2),
          height: Math.max(2, child.height),
          rx: 1, fill: childStyle.fill,
        }));
      });
    }
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
    if (style.container) {
      swatch.appendChild(svgEl('rect', {
        x: 1, y: 1, width: 12, height: 12, rx: 2,
        fill: style.fill, 'fill-opacity': 0.28,
        stroke: style.fill, 'stroke-width': 1.5,
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
