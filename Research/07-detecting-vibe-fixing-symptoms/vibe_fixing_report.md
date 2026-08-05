# Vibe-Fixing Symptoms in the SWE-Chat Dataset

This report shows how often "vibe-fixing" happens in coding-agent sessions. Vibe-fixing means a user accepts a fix from the agent without a clear task, without checking it, or without proof that it works. I checked **25 real coding sessions** from the SWE-Chat dataset (agent: Claude Code). All sessions were included, not only long ones.

## What I Looked For

I checked each session for 7 checks (5 judged by an LLM, 2 by simple thresholds):

| Symptom | What it means |
|---|---|
| `no_spec` | The user's request is very short and unclear, or the agent shows doubt but still submits an answer |
| `no_closed_loop` | The user asks for a fix, but there is no way to check if it worked (no test run) |
| `no_acceptance_criteria` | The user's goal is vague ("make it faster", "clean this up"), with no clear target |
| `no_visual_reference` | The user asks for a UI/visual change, but gives no image or design file |
| `repetitive_fix_attempts` | The agent fixes the same bug wrong more than once, and the user has to report it again |
| `scope_files_too_many` | Too many files were changed in one session |
| `scope_turns_too_long` | The session had an unusually high number of turns |

## How I Detected Them

I used two methods:

**1. LLM-as-judge (Claude Haiku 4.5), one isolated call per symptom.** Each session's raw transcript is rendered as a single chronological, typed-block timeline — every user message, every piece of agent thinking, and every tool call together with its raw result, in the exact order they happened, each tagged with the turn number it occurred at (this rendering approach is inspired by [VCC](https://github.com/lllyasviel/VCC), a compiler for agent conversation logs). That same timeline is reused as a shared prefix across a session's 5 symptom calls. Each call also cites which turn(s) its evidence came from, so findings can be traced back to a specific point in the session rather than just a session-wide yes/no.

Notably, we do NOT pre-label which tool calls are "tests" or which files are "specs" using keyword lists or filename patterns — every tool call and its raw output are shown to the judge as-is, and it decides for itself (e.g. whether a bash command was a meaningful verification step, or whether a file the agent read plausibly explains an otherwise-vague request).

**2. Metadata-only rules.** The two `scope_*` symptoms don't need an LLM. I just count files touched and turns per session, and flag sessions above a threshold.

## Results

| Symptom | Count | % of judged sessions |
|---|---|---|
| `no_closed_loop` | 4 | 16% |
| `no_spec` | 3 | 12% |
| `no_acceptance_criteria` | 5 | 20% |
| `scope_turns_too_long` | 8 | 32% |
| `scope_files_too_many` | 3 | 12% |
| `repetitive_fix_attempts` | 7 | 28% |
| `no_visual_reference` | 1 | 4% |

## Examples

For each symptom flagged by the LLM judge, here are real examples pulled from this run (session id, the turn(s) the evidence came from, and the judge's one-line reason). These are spot-check material, not proof — always worth reading the underlying transcript before trusting an aggregate number.

**`no_spec`**

- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (turns 498-511) The feature request to indicate that a session was not active during intervening commits did not define the desired visual treatment, yet the agent directly changed lane rendering to show only rows where the session had a checkpoint.
- [`2026-01-16-cde341db-b80a-44f5-b0f2-94db2ef7a164`] (turn 0) The initial change request asks for “a flame graph or tracing or something” without defining the diagnostic interface, scope, or acceptance criteria, and the agent proceeded by choosing an implementation itself.
- [`2026-01-19-38917d7d-9f69-4210-a3b0-eed0a9e97575`] (turns 49-60) The request "print the time that request takes to resolve" (turn 49) is underspecified, and the agent proceeded by arbitrarily timing both Enqueue and Close without clarifying the intended request or output.

**`no_closed_loop`**

- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (turns 506-515) After changing lane rendering, the agent only ran formatting, lint, the existing test suite, and a build; it did not rerun `entire list` in the reproduction repository or inspect the rendered output to verify inactive-session gaps.
- [`2026-01-20-86cc3044-7515-43f1-9c25-1444738c64c9`] (turns 138-141) After the user reproduced the Cobra help output in turn 138, the agent only launched an exploratory search in turn 141 and did not make or verify a corresponding fix; the earlier full test run did not cover this behavior.
- [`2026-01-22-0cf3db51-1c73-43bb-8d05-dc02739514a5`] (turns 866-987) After making the later TranscriptPath and agent-agnostic changes, the agent only ran `go build ./cmd/entire/cli/agent/claudecode/...` (turn 987), with no targeted or full test run; the subsequent edit was rejected at turn 995.
- [`2026-01-22-48f428f9-72d8-40b2-9594-953019809473`] (turns 72-83) After editing the GoReleaser and workflow files, the agent only reread them and checked IDE diagnostics; it never ran GoReleaser, a YAML/workflow validation, or any release/install verification.

**`no_acceptance_criteria`**

- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (turns 498-513) The request to indicate somehow that a session was inactive during intervening commits provided no concrete visual treatment or acceptance condition; the agent chose gap-based lane rendering and verified only lint/tests.
- [`2026-01-16-cde341db-b80a-44f5-b0f2-94db2ef7a164`] (turn 0) The request asks for a lightweight tracing or flame-graph solution to investigate a few-hundred-millisecond delay but does not define a concrete completion or performance target.
- [`2026-01-16-eb4bcc15-3ff5-4d73-bac5-9bc1d786b2bb`] (turns 81-94) The implementation request was to calculate token consumption at Stop-hook time, but it did not specify the exact token metric, aggregation rules, metadata schema, or required validation.
- [`2026-01-19-c2b51eb0-d0e9-43cf-b431-42c05d49450b`] (turns 0, 24-32, 50-57, 91) ENT-53 begins as the vague goal to “Clean up explain output,” and although the session develops a proposed tiered design, it never establishes concrete acceptance tests or a measurable definition of done before implementation begins.
- [`2026-01-21-130d7b7e-5801-4345-9bd6-f32fd9b8429b`] (turn 0) The initial request to “add the agent name to any logging” does not define which logs, propagation behavior, or verification target would constitute completion.

**`no_visual_reference`**

- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (turns 61-67) The session contains multiple visual/UI requests for the CLI list view, including lane lines and spacing, but no screenshot, image, design file, or external visual reference was provided.

**`repetitive_fix_attempts`**

- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (turns 261-307 and 397-447) After the branch-filtering fix, the user reported at turn 307 that the branch still showed 38 commits; a further fix was then needed, and the same filtering regression later caused no checkpoints in test1234 (turn 397).
- [`2026-01-19-85cc202e-0e74-46ce-a485-e5fcd11f7a8c`] (turns 47-78) After multiple attempted interactive-test fixes, the tests continued to time out or fail to respond, prompting the user to question the prompt handling and whether huh was using the pty.
- [`2026-01-20-86cc3044-7515-43f1-9c25-1444738c64c9`] (turns 78-119 and 138) After the agent claimed `NewSilentError` would prevent duplicate output, the user reported that `entire resume` still displayed Cobra usage after the metadata-fetch error.
- [`2026-01-21-0cd9edf7-459c-4900-af40-cf2c64dea525`] (turns 116-138) After PR#68 was checked out and its tests passed, the user reported that transcript markers were still missing for the first manual commit, repeating part of the original bug.
- [`2026-01-22-0cf3db51-1c73-43bb-8d05-dc02739514a5`] (turns 749-759, 841-847) After the agent claimed the checkpoint-reuse fix was complete, the user continued reporting that it skipped valid new work and then uncovered an unresolved second scenario involving multiple agent commits before Stop.

## Performance Notes

Total wall-clock time for this run: 813s. Average download time per session: 0.28s. Average parse time per session: 0.00s.

| Symptom | Avg call time | Avg prompt size |
|---|---|---|
| `no_spec` | 7.25s | 55,282 chars |
| `no_closed_loop` | 6.81s | 54,658 chars |
| `no_acceptance_criteria` | 6.43s | 54,269 chars |
| `no_visual_reference` | 5.53s | 54,312 chars |
| `repetitive_fix_attempts` | 6.24s | 54,482 chars |

## A Note of Caution

`no_verification_by_user` has been removed from this report entirely — it was mostly detecting "no proof shown in the transcript" rather than "the user actually skipped verifying," and a person could always test something outside the chat window, so it wasn't trustworthy as reported. The remaining symptoms rely on clearer, easier-to-check evidence (the actual request text, the raw tool calls and their results, file counts), but spot-checking the Examples section above against real transcripts is still recommended before citing these numbers externally.
