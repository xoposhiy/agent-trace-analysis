// Unit tests for the pure geometry in web/plan_mode.js (CLAUDE.md §6).
//
// Loaded into a vm context the same way bar.test.js loads bar.js — plan_mode.js
// depends on bar.js's constants/functions, so both are run into one context in
// order, ahead of plan_mode.js itself.
//
// Run: node --test tests/plan_mode.test.js   (or via pytest, tests/test_bar_js.py)

const test = require('node:test');
const assert = require('node:assert');
const fs = require('node:fs');
const path = require('node:path');
const vm = require('node:vm');

const BAR_SOURCE = fs.readFileSync(
  path.join(__dirname, '..', 'web', 'bar.js'), 'utf8');
const PLAN_MODE_SOURCE = fs.readFileSync(
  path.join(__dirname, '..', 'web', 'plan_mode.js'), 'utf8');

const EPILOGUE = `
globalThis.exported = { computeSplitLayout };
`;

const context = vm.createContext({
  document: { createElementNS: () => ({ setAttribute() {}, appendChild() {} }) },
  el: (tag, cls, text) => ({
    tag, cls, text, style: {}, children: [],
    appendChild(child) { this.children.push(child); },
  }),
  formatCost: (dollars) => `$${(dollars || 0).toFixed(2)}`,
});
vm.runInContext(BAR_SOURCE + '\n' + PLAN_MODE_SOURCE + '\n' + EPILOGUE, context);

const { computeSplitLayout } = context.exported;

// --- fixtures -----------------------------------------------------------

function block(kind, tEnd, messages = 1) {
  return { kind, t_end: tEnd, message_count: messages, tokens: { working: 0 } };
}

const READING = [block('read', 't1'), block('read', 't2')];
const WORK = [block('write', 't3'), block('execute', 't4')];

// --- tests ----------------------------------------------------------------

test('finds the index of the block marked as the split point', () => {
  const blocks = [...READING, ...WORK];

  assert.strictEqual(computeSplitLayout(blocks, 't2'), 1);
});

test('returns null for an empty block list', () => {
  assert.strictEqual(computeSplitLayout([], 't2'), null);
});

test('returns null when the split timestamp is not among the blocks', () => {
  const blocks = [...READING, ...WORK];
  assert.strictEqual(computeSplitLayout(blocks, 'does-not-exist'), null);
});

test('returns null when the split point is the last block (nothing follows it)', () => {
  const blocks = [...READING, ...WORK];
  assert.strictEqual(computeSplitLayout(blocks, 't4'), null);
});

test('returns null with no split timestamp given', () => {
  const blocks = [...READING, ...WORK];
  assert.strictEqual(computeSplitLayout(blocks, undefined), null);
});
