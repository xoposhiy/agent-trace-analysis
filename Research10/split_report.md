# Local Split Advisor — session report

- Sessions scanned: **38**
- Sessions worth splitting (≥ 10% and ≥ $0.50): **0**
- Total modelled cost of all sessions: **$231.89**
- Total modelled saving from acting on suggestions: **$0.00**
- Detection: heuristic + LLM task-switch judge + task summaries

## Suggestions

No session cleared both thresholds. Lower `--min-pct` / `--min-dollars` to see marginal opportunities.

## All scanned sessions

| Session | Turns | Peak | Pattern | Modelled saving | Suggested? |
|---|---:|---:|:---:|---:|:---:|
| `edad9dce` | 282 | 315k | A | no split point | — |
| `383e7f1b` | 169 | 450k | none | no split point | — |
| `a7526c9f` | 318 | 197k | none | no split point | — |
| `f4af9989` | 232 | 222k | A | $1.13 (5%) | below floor |
| `3ec63ba1` | 199 | 206k | A | $1.24 (7%) | below floor |
| `67805753` | 175 | 199k | none | no split point | — |
| `63f0bb29` | 177 | 192k | none | no split point | — |
| `d31ca78e` | 220 | 179k | A | $0.84 (5%) | below floor |
| `1c8b54c4` | 89 | 109k | A | $0.35 (5%) | below floor |
| `c0b31e58` | 32 | 90k | none | no split point | — |
| `e758cf9b` | 26 | 69k | none | no split point | — |
| `0559e6c8` | 31 | 69k | none | no split point | — |
| `4c21635c` | 24 | 64k | none | $-0.06 (-4%) | below floor |
| `c2d1920d` | 24 | 53k | none | no split point | — |
| `cf2c1354` | 38 | 57k | none | no split point | — |
| `583792ff` | 26 | 51k | none | no split point | — |
| `9c5b8c93` | 28 | 48k | none | $0.02 (1%) | below floor |
| `6157da4c` | 28 | 46k | none | no split point | — |
| `1b245284` | 10 | 35k | none | no split point | — |
| `e1bb3d1f` | 18 | 49k | none | no split point | — |
| `7a001f6a` | 25 | 47k | none | $0.05 (6%) | below floor |
| `15d437fe` | 13 | 54k | none | no split point | — |
| `babc8849` | 9 | 40k | none | no split point | — |
| `58a52cec` | 4 | 19k | none | no split point | — |
| `cdc9fddb` | 4 | 35k | none | no split point | — |
| `c883bfd5` | 2 | — | none | n/a (no cache-read) | — |
| `3f705fe6` | 7 | 40k | none | no split point | — |
| `b5e7694f` | 1 | — | none | n/a (no cache-read) | — |
| `d2844afa` | 2 | 32k | none | no split point | — |
| `1804cdb4` | 1 | — | none | n/a (no cache-read) | — |
| `34fa40a4` | 1 | — | none | n/a (no cache-read) | — |
| `500e86e2` | 1 | — | none | n/a (no cache-read) | — |
| `68d4c65f` | 1 | — | none | n/a (no cache-read) | — |
| `912a915c` | 1 | — | none | n/a (no cache-read) | — |
| `b0be4f26` | 1 | — | none | n/a (no cache-read) | — |
| `b2e891e3` | 1 | — | none | n/a (no cache-read) | — |
| `d239e9ac` | 2 | — | none | n/a (no cache-read) | — |
| `f07103fc` | 1 | — | none | n/a (no cache-read) | — |

## All considered split options

Every split point the tool priced, including options below the suggestion floor and options that were not the best for their session. LLM-detected task switches appear here too.

| Session | Task | Option | Split point | Save $ | Save % | Status |
|---|:---|:---:|---|---:|---:|:---:|
| `f4af9989` | The user investigated API billing costs in SWE-Chat dataset and developed a mathematical model to analyze savings from splitting long sessions into chunks with summarization overhead. | plan-mode | after turn 13 (end of opening reading phase) | $1.13 | 5% | below floor |
| `3ec63ba1` | Developing a local app that analyzes Claude code session traces to suggest semantic splitting when it saves money, with adaptability for any user's traces. | plan-mode | after turn 16 (end of opening reading phase) | $1.24 | 7% | below floor |
| `d31ca78e` | The user worked on integrating LLM capabilities into a split_advisor.py script to automatically generate task summaries and determine whether sessions should be split, removing the need for manual session number identification. | plan-mode | after turn 12 (end of opening reading phase) | $0.84 | 5% | below floor |
| `1c8b54c4` | The user conducted a multi-phase analysis of context rot patterns and file re-reads in Claude Code sessions from the SWE-chat dataset. | plan-mode | after turn 9 (end of opening reading phase) | $0.35 | 5% | below floor |
| `4c21635c` | User investigated anomalies in heavy session turn counts across different token thresholds, then switched to clarifying the structure of the SWE-chat database. | task-switch | at message 8 (turn 20, task switch) | $-0.06 | -4% | below floor |
| `9c5b8c93` | User analyzed session turn counts in a dataset, then switched to organizing files into a git repository folder. | task-switch | at message 4 (turn 20, task switch) | $0.02 | 1% | below floor |
| `7a001f6a` | User explored documentation about agent trace analysis tools, workflow orchestration, and MCP context reuse features. | task-switch | at message 2 (turn 13, task switch) | $0.05 | 6% | below floor |
