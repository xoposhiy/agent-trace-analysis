// Unit tests for the pure geometry in web/task_forest.js (CLAUDE.md §6).
//
// Loaded into a vm context the same way plan_mode.test.js loads plan_mode.js —
// task_forest.js depends on bar.js's constants/functions, so both run into
// one context in order, ahead of task_forest.js itself.
//
// Run: node --test tests/task_forest.test.js   (or via pytest, tests/test_bar_js.py)

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const BAR_SOURCE = fs.readFileSync(
  path.join(__dirname, '..', 'web', 'bar.js'), 'utf8');
const TASK_FOREST_SOURCE = fs.readFileSync(
  path.join(__dirname, '..', 'web', 'task_forest.js'), 'utf8');

const EPILOGUE = `
globalThis.exported = { computeTaskBands };
`;

const context = vm.createContext({
  document: { createElementNS: () => ({ setAttribute() {}, appendChild() {} }) },
  el: (tag, cls, text) => ({
    tag, cls, text, style: {}, children: [],
    appendChild(child) { this.children.push(child); },
  }),
});
vm.runInContext(BAR_SOURCE + '\n' + TASK_FOREST_SOURCE + '\n' + EPILOGUE, context);

const { computeTaskBands } = context.exported;

// --- fixtures -----------------------------------------------------------

function block(kind, tStart) {
  return { kind, t_start: tStart, t_end: tStart, message_count: 1, tokens: { working: 0 } };
}

function run(id, label, startTs, endTs) {
  return { id, label, start_ts: startTs, end_ts: endTs };
}

const RUNS = [run('T1', 'Fix login', 't1', 't3'), run('T2', 'Optimize query', 't3', null)];

// --- tests ----------------------------------------------------------------

test('groups consecutive blocks in the same run into one band', () => {
  const blocks = [block('read', 't1'), block('write', 't2'), block('read', 't4')];

  const bands = computeTaskBands(blocks, RUNS);

  assert.strictEqual(bands.length, 2);
  assert.strictEqual(bands[0].run.id, 'T1');
  assert.deepStrictEqual([bands[0].startIndex, bands[0].endIndex], [0, 1]);
  assert.strictEqual(bands[1].run.id, 'T2');
  assert.deepStrictEqual([bands[1].startIndex, bands[1].endIndex], [2, 2]);
});

test('a later return to an earlier run starts a fresh band', () => {
  const threeRuns = [
    run('T1', 'Login', 't1', 't2'), run('T2', 'Query', 't2', 't3'), run('T1', 'Login', 't3', null),
  ];
  const blocks = [block('read', 't1'), block('write', 't2'), block('read', 't3')];

  const bands = computeTaskBands(blocks, threeRuns);

  assert.strictEqual(bands.length, 3);
  assert.strictEqual(bands[0].run.id, 'T1');
  assert.strictEqual(bands[2].run.id, 'T1');
  assert.notStrictEqual(bands[0], bands[2]);
});

test('returns null for an empty block list', () => {
  assert.strictEqual(computeTaskBands([], RUNS), null);
});

test('returns null with fewer than two runs', () => {
  const blocks = [block('read', 't1')];
  assert.strictEqual(computeTaskBands(blocks, [RUNS[0]]), null);
});

test('returns null with no runs given', () => {
  const blocks = [block('read', 't1')];
  assert.strictEqual(computeTaskBands(blocks, undefined), null);
});
