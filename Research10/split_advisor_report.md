# Local Split Advisor — report

A local CLI that scans **your own** Claude Code sessions and decides, per session,
whether it was worth splitting — and how much a split would have saved. Reuses the
Research7 detectors + chunk-split pricing, pointed at local transcripts. Runs offline.
Install and usage: [`README.md`](README.md).

---

## How it decides

Rebuild each session's tool-call sequence, detect its shape (front-loaded reading →
plan-mode split, or a mid-session reading burst → sub-agent split), price a single
split at that boundary, and suggest it only if the saving clears **both** a % floor and
a $ floor. Optional `--llm` also checks for a genuine task switch.

---

## Where the saving comes from

The dominant cost is **cache-read**: every call re-reads the whole context so far, so it
grows like a triangle (~70% of the bill). Splitting at fraction `f` of the session
removes `2·f·(1−f)` of that — **maxed at 50% only for a mid-session split**, and less
the earlier the reading phase ends.

| Split point `f` | % of total cost saved |
|---|---|
| edges (0.05 / 0.95) | ~2–3% |
| plan-mode (reading ends early) | ~5–11% |
| middle (0.50) | ~16% (max) |

It never nears the ~70% ideal: each new chunk still pays to regenerate a summary
(output-dominated), an irreducible floor. Matches Research7's 11% (plan-mode) / 16%
(task-switch) figures.

---

## Result on this machine's corpus

- **32** sessions scanned, **21** priceable, ~**$204** total modelled cost.
- Only **3** split candidates, all saving **4.6–7.7%** — **all below the 10% floor → 0
  suggestions at defaults.**
- Why: reading phases end *early*, so `f` is small. **Sessions are long because the task
  is big, not because they're splittable.**

---

## Key parameters

Summary size **α = 0.10** (carried summary = 10% of the first chunk); suggestion gate
**10% and $0.50**; blended Opus-4.x rates. All dollar figures are token-based estimates
from each session's real recorded usage.
