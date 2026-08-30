// The problem-specific visualization for "independent tasks in one session"
// (see `analysis/task_forest.py`). Draws ONLY a colored "task lane" — one
// band per independent top-level task, each labelled — sized and positioned
// to align with `#bar`'s own blocks. The session's bar itself is not redrawn
// here: `problem.js` already renders the one real `#bar` beside this, and
// duplicating it would be a second, redundant copy of the same blocks.
//
// Reuses `bar.js`'s `layoutBlocks`/`barHeight`/`svgEl` for the sizing, so the
// lane's bands land at the exact same y-coordinates as `#bar`'s blocks.

// --- geometry -------------------------------------------------------------

// A small fixed categorical palette, assigned to top-level task ids by
// first-appearance order — matching the picture's blue/green/orange bands.
// An 7th+ independent task (rare) falls back to a shared grey.
const TASK_COLORS = ['#5b8dff', '#2fae6b', '#e59400', '#d64545', '#ad5fc9', '#3ab7c9'];
const TASK_COLOR_OVERFLOW = '#8b95a5';

function taskColor(taskId, order) {
  const index = order.indexOf(taskId);
  return index >= 0 && index < TASK_COLORS.length
    ? TASK_COLORS[index] : TASK_COLOR_OVERFLOW;
}

// Assigns each block to the run whose `start_ts` is the latest one at or
// before the block's own start — the run boundaries are timestamps (see
// `analysis.task_forest.detect`'s `runs[].start_ts`), not block indexes,
// because blocks are re-derived and can change count on a re-parse.
// Adjacent blocks landing in the same run merge into one band. Returns
// `null` for degenerate input: no blocks, or fewer than two runs (a single
// run is not "independent tasks", it's just the whole bar).
function computeTaskBands(blocks, runs) {
  if (!Array.isArray(blocks) || !blocks.length) return null;
  if (!Array.isArray(runs) || runs.length < 2) return null;

  const runForTs = (ts) => {
    let found = runs[0];
    for (const run of runs) {
      if (run.start_ts <= ts) found = run;
      else break;
    }
    return found;
  };

  const bands = [];
  blocks.forEach((block, index) => {
    const run = runForTs(block.t_start || block.t_end);
    const last = bands[bands.length - 1];
    // Blocks are visited in order, so two consecutive ones landing in the
    // same run are always adjacent — no separate "still contiguous" check
    // needed. A LATER return to an earlier run (the interleaving case) still
    // starts a fresh band here, which is correct: it is a second, later
    // stretch of that task, drawn as its own segment in the same colour.
    if (last && last.run.id === run.id) {
      last.endIndex = index;
    } else {
      bands.push({ run, startIndex: index, endIndex: index });
    }
  });
  return bands;
}

// --- rendering --------------------------------------------------------

const RAIL_WIDTH = 10;
const LABELS_WIDTH = 260;

function renderTaskForestBar(container, blocks, runs, metric) {
  container.replaceChildren();

  const bands = computeTaskBands(blocks, runs);
  if (!bands) return;

  const height = barHeight(blocks);
  const boxes = layoutBlocks(blocks, metric, height);
  const order = [...new Set(bands.map((band) => band.run.id))];

  // A plain flex row: the rail (a normal, non-absolute SVG, so it reserves
  // its own width like `#bar` does) beside a fixed-width label column. Both
  // get an explicit height/width in the markup itself rather than relying on
  // absolutely-positioned children to imply one — those contribute nothing
  // to their parent's box, which is what left the whole lane invisible
  // (zero-width) before this.
  const wrap = el('div', 'task-forest-wrap');

  const rail = svgEl('svg', {
    width: RAIL_WIDTH, height, viewBox: `0 0 ${RAIL_WIDTH} ${height}`,
    role: 'img', 'aria-label': 'Which independent task owns each stretch of the bar',
  });
  bands.forEach((band) => {
    const top = boxes[band.startIndex].y;
    const bottom = boxes[band.endIndex].y + boxes[band.endIndex].height;
    const segment = svgEl('rect', {
      x: 0, y: top, width: RAIL_WIDTH, height: Math.max(2, bottom - top),
      rx: 2, fill: taskColor(band.run.id, order),
    });
    segment.appendChild(svgEl('title', {})).textContent =
      `${band.run.id}: ${band.run.label}`;
    rail.appendChild(segment);
  });
  wrap.appendChild(rail);

  const labels = el('div', 'task-forest-labels');
  labels.style.width = `${LABELS_WIDTH}px`;
  labels.style.height = `${height}px`;
  bands.forEach((band) => {
    const top = boxes[band.startIndex].y;
    const bottom = boxes[band.endIndex].y + boxes[band.endIndex].height;
    const label = el('div', 'task-forest-label');
    label.style.top = `${top}px`;
    label.style.height = `${Math.max(2, bottom - top)}px`;
    const swatch = el('span', 'task-forest-swatch');
    swatch.style.background = taskColor(band.run.id, order);
    label.appendChild(swatch);
    label.appendChild(el('span', null, `${band.run.id}: ${band.run.label}`));
    labels.appendChild(label);
  });
  wrap.appendChild(labels);

  container.appendChild(wrap);
}
