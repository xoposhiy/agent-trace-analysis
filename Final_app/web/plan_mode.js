// The problem-specific visualization for a "missed plan-mode opportunity"
// (see `analysis/plan_mode.py`). Draws a second vertical bar right beside the
// session's own bar: the exact same blocks, laid out with `bar.js`'s own
// sizing, but with a gap cut into it at the split point and a small grey
// "carried summary" marker sitting in the gap — literally the same bar, split.
//
// Reuses `bar.js`'s constants, `layoutBlocks`, `renderBlock`, `barHeight` and
// `svgEl` directly rather than reimplementing block sizing or colouring; the
// only new drawing here is the gap and its summary marker.

// --- geometry -------------------------------------------------------------

// Locates `splitAfterTs` (a block's own `t_end` — see `split_after_ts` on the
// detected problem's `data`) among `blocks` and returns its index, or `null`
// when the split cannot be drawn: no blocks, no timestamp, the timestamp not
// found, or nothing after it to show as the second half.
function computeSplitLayout(blocks, splitAfterTs) {
  if (!Array.isArray(blocks) || !blocks.length || !splitAfterTs) return null;

  const splitIndex = blocks.findIndex((block) => block.t_end === splitAfterTs);
  if (splitIndex === -1 || splitIndex >= blocks.length - 1) return null;

  return splitIndex;
}

// --- rendering --------------------------------------------------------

// Visual gap: a dashed "cut" rule, then the summary marker, then more
// whitespace, stacked in the space `layoutBlocks` would otherwise have given
// the blocks right at the split. Sized to actually read as a cut rather than
// a barely-there sliver — the whole point of this bar is to be noticed.
const SPLIT_GAP = 10;
const SPLIT_SUMMARY_HEIGHT = 16;
const SPLIT_EXTRA = SPLIT_GAP * 2 + SPLIT_SUMMARY_HEIGHT;

// A dashed border and the accent colour, not a plain fill: this box is not a
// stretch of real work like the blocks around it (compare `.placeholder`'s
// own dashed border in style.css for the same "this is a stand-in" language),
// and the accent blue reads as new/different next to the bar's own read/
// write/execute/coordination palette rather than blending into `coordination`
// grey the way a plain grey swatch did. SVG can't read CSS custom
// properties, so `--accent` is repeated here as a literal.
const SPLIT_SUMMARY_STROKE = '#5b8dff';
const SPLIT_SUMMARY_FILL = 'rgba(91, 141, 255, 0.14)';
const SPLIT_CUT_STROKE = '#8b95a5';

function renderPlanModeBar(container, blocks, splitIndex, metric, onHover, onOpen) {
  container.replaceChildren();

  const height = barHeight(blocks);
  const boxes = layoutBlocks(blocks, metric, height);
  const totalHeight = height + SPLIT_EXTRA;

  const svg = svgEl('svg', {
    width: BAR_WIDTH,
    height: totalHeight,
    viewBox: `0 0 ${BAR_WIDTH} ${totalHeight}`,
    role: 'img',
    'aria-label': 'Session split at the missed plan-mode opportunity',
  });

  svg.appendChild(svgEl('rect', {
    x: 0.5, y: 0.5, width: BAR_WIDTH - 1, height: totalHeight - 1,
    rx: 4, fill: 'var(--panel-2)', stroke: 'var(--line)',
  }));

  const splitEnd = boxes[splitIndex].y + boxes[splitIndex].height;

  boxes.forEach(({ block, y, height: h }, index) => {
    // Everything after the split point is pushed down by the gap the cut
    // opens up; the blocks themselves are drawn exactly as `bar.js` would,
    // at the SAME index into `blocks` as the main bar — so a click here opens
    // the exact page a click on the main bar's copy of this block would.
    const drawY = index <= splitIndex ? y : y + SPLIT_EXTRA;
    svg.appendChild(renderBlock(block, drawY, h, metric, onHover, onOpen, index));
  });

  // The cut itself: a dashed rule spanning the bar, right where the reading
  // run actually ends — a solid bar visibly torn open, not just a gap that
  // could be mistaken for missing data.
  svg.appendChild(svgEl('line', {
    x1: 1, x2: BAR_WIDTH - 1, y1: splitEnd + 2, y2: splitEnd + 2,
    stroke: SPLIT_CUT_STROKE, 'stroke-width': 1.5, 'stroke-dasharray': '3,2',
  }));

  const summaryY = splitEnd + SPLIT_GAP;
  const summaryBox = svgEl('rect', {
    x: INSET - 1, y: summaryY, width: BAR_WIDTH - 2 * (INSET - 1),
    height: SPLIT_SUMMARY_HEIGHT, rx: RADIUS,
    fill: SPLIT_SUMMARY_FILL, stroke: SPLIT_SUMMARY_STROKE,
    'stroke-width': 1.25, 'stroke-dasharray': '3,2',
  });
  summaryBox.appendChild(svgEl('title', {})).textContent =
    'A short carried summary, seeded into the next session — this is the cost of the cut';
  svg.appendChild(summaryBox);

  const label = svgEl('text', {
    x: BAR_WIDTH / 2, y: summaryY + SPLIT_SUMMARY_HEIGHT / 2 + 3,
    'text-anchor': 'middle', 'font-size': 8, 'font-weight': 600,
    fill: SPLIT_SUMMARY_STROKE, style: 'pointer-events: none',
  });
  label.textContent = 'sum';
  svg.appendChild(label);

  container.appendChild(svg);
}
