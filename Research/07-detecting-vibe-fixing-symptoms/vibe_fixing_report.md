# Vibe-Fixing Symptoms in the SWE-Chat Dataset

This report shows how often "vibe-fixing" happens in coding-agent sessions — the agent's work moving forward without enough specification or enough verification. I checked a **sample of 15 sessions** drawn from the 4,852 parseable Claude Code sessions in the SWE-Chat dataset. The sample is the first N sessions in dataset order, **not a random sample**, so it is a spot check rather than a dataset-wide estimate. Short sessions were included alongside long ones.

## What I Looked For

Two main categories, each broken into subcategories, plus one standalone symptom and two metadata-only checks:

| Category / Symptom | What it means |
|---|---|
| **`not_enough_verification`** | The implementation wasn't actually checked before being treated as finished. |
| &nbsp;&nbsp;`not-tested` | Agent claims the task is finished but never verified it (no test, no manual check). |
| &nbsp;&nbsp;`self-report` | Agent itself says some important part wasn't tested. |
| &nbsp;&nbsp;`ask-for-manual-testing` | Agent asks the human to test something manually. |
| &nbsp;&nbsp;`repetitive-bug-fixes` | After the agent called it done, the user tested manually and reported bugs. |
| **`not_enough_specification`** | The user's request wasn't clear enough to act on. |
| &nbsp;&nbsp;`no-spec-detected` | User asked for an implementation without a detailed enough spec. |
| &nbsp;&nbsp;`repetitive-requirements-fixes` | Agent fixed it the wrong way and the user pushed back, repeatedly. |
| &nbsp;&nbsp;`self-report` | Agent itself says it doesn't have enough specification. |
| `no_visual_reference` | The user asks for a UI/visual change, but gives no image or design file. |
| `scope_files_too_many` | Too many files were changed in one session |
| `scope_turns_too_long` | The session had an unusually high number of turns |

## How I Detected Them

**LLM-as-judge, one call per category** (3 calls per session total: `not_enough_verification`, `not_enough_specification`, `no_visual_reference`). Each session's raw transcript is rendered as a single chronological, typed-block timeline — every user message, every piece of agent thinking, and every tool call together with its raw result, in the exact order they happened, each tagged with the message number it occurred at (rendering approach inspired by [VCC](https://github.com/lllyasviel/VCC)). That same timeline is reused across a session's 3 calls.

Each call returns **every occurrence** it finds, not just the first — a session can show the same subcategory multiple times (e.g. the agent asks for manual testing twice, or gets pushed back on requirements three times), and each one is recorded with its own message location and evidence.

We do NOT pre-label which tool calls are "tests" or which files are "specs" using keyword lists or filename patterns — every tool call and its raw output are shown to the judge as-is, and it decides for itself.

**Metadata-only rules.** `scope_files_too_many` and `scope_turns_too_long` don't need an LLM — just a count of files touched and assistant turns per session, flagged above a threshold.

**What the judge does and doesn't see.** The timeline is condensed, not verbatim, and the caps matter when reading the numbers below — two of these checks are judgments about something being *absent*, which truncation can manufacture:

- Agent thinking has no session-wide cap, but an individual thinking block longer than 4,000 characters is shown as its first and last portions only, explicitly marked as truncated.
- Each tool result is cut to 600 characters. The tool *call* is always visible, but a long test run's actual output may be clipped.
- Each user message is cut to 2,000 characters, so a spec buried at the end of a very long request can be lost — which pushes `no-spec-detected` toward false positives.

## Results

"Sessions" counts sessions with at least one occurrence; "Total occurrences" counts every occurrence, so a session flagged three times contributes 1 and 3 respectively. Percentages are per-check: each LLM check is divided by the number of sessions where **that** call succeeded, and the metadata-only checks by all judged sessions — so if any calls failed, the denominators differ slightly between rows. See Run Reliability below.

| Check | Sessions | % of judged sessions | Total occurrences |
|---|---|---|---|
| **`not_enough_verification`** (any) | 4 | 27% | — |
| &nbsp;&nbsp;`not-tested` | 2 | 13% | 2 |
| &nbsp;&nbsp;`self-report` | 1 | 7% | 1 |
| &nbsp;&nbsp;`ask-for-manual-testing` | 1 | 7% | 2 |
| &nbsp;&nbsp;`repetitive-bug-fixes` | 3 | 20% | 8 |
| **`not_enough_specification`** (any) | 7 | 47% | — |
| &nbsp;&nbsp;`no-spec-detected` | 2 | 13% | 2 |
| &nbsp;&nbsp;`repetitive-requirements-fixes` | 4 | 27% | 6 |
| &nbsp;&nbsp;`self-report` | 4 | 27% | 6 |
| `no_visual_reference` | 0 | 0% | 0 |
| `scope_files_too_many` | 1 | 7% | — |
| `scope_turns_too_long` | 5 | 33% | — |

## Examples

For each subcategory/symptom flagged by the judge, real examples pulled from this run (session id, which message(s) the evidence came from, and the judge's one-line reason). Spot-check material, not proof.

**`not_enough_verification` → `not-tested`**

- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (messages 290-319) The loading-spinner implementation was treated as complete after lint and the general test suite, without manually running the interactive command or adding a spinner-specific test.
- [`2026-01-19-38917d7d-9f69-4210-a3b0-eed0a9e97575`] (messages 77-97) After adding timing instrumentation around telemetry enqueue/close operations, the agent ran only formatting and linting, then treated the change as complete without running tests or manually exercising the CLI.

**`not_enough_verification` → `self-report`**

- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (messages 501-570) The agent explicitly said it could not run the interactive TUI in the available environment and relied on code inspection rather than observing the lane rendering directly.

**`not_enough_verification` → `ask-for-manual-testing`**

- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (messages 739-745) The agent explicitly handed verification of the final gap-rendering behavior to the user by saying they could test it in the repository.
- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (messages 566-570) The agent explicitly said the user could test the rebuilt lane visualization rather than verifying the interactive TUI itself.

**`not_enough_verification` → `repetitive-bug-fixes`**

- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (messages 187-189, 197-288) After declaring the session-lane feature complete, the user reported that checkpoints were missing from the main branch; the agent then changed the checkpoint scan limit.
- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (messages 418-445, 448-485) After declaring branch filtering fixed and verifying one command output, the user reported that the branch still showed 38 commits; the agent found and fixed a second code path.
- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (messages 487-570) After declaring the lane direction correct and building the binary, the user reported another rendering problem involving session-line placement; the agent changed and then reconsidered the direction logic.
- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (messages 570-647) After the filtering changes were treated as working, the user reported that a separate test repository showed no checkpoints; the agent found that main was incorrectly filtered and fixed it.
- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (messages 649-718) After the active-session fix was declared complete, the user reported that the most recent checkpoint still belonged to a different session; the agent investigated and changed current_session updates in PostCommit.

**`not_enough_specification` → `no-spec-detected`**

- [`2026-01-19-85cc202e-0e74-46ce-a485-e5fcd11f7a8c`] (messages 1-5) The user referred only to “the other bit” without specifying the requested implementation, while the agent inferred that it meant adding interactive yes/no prompt testing.
- [`2026-01-19-38917d7d-9f69-4210-a3b0-eed0a9e97575`] (messages 77-91) The user asked to "print the time that request takes to resolve" without specifying whether to measure enqueueing, the asynchronous HTTP request, shutdown flushing, or the output format; the agent proceeded by assuming timing should be added to both Enqueue and Close.

**`not_enough_specification` → `repetitive-requirements-fixes`**

- [`2026-01-15-43b2bd6e-ccc5-4d83-bbe5-9cd1cd19cc82`] (messages 321-332) The agent initially implemented lane rendering around a single primary session per checkpoint, but the user clarified that the intended behavior was to render two dots when one checkpoint contains two single-checkpoint sessions; the agent then redesigned the lane data model to support multiple sessions.
- [`2026-01-19-8b418ea5-895d-4ad5-abe8-4489d36e454f`] (messages 69-77) After the agent proposed and attempted to add the documentation as a comment, the user clarified that it should instead be a document attached directly to the issue, then reiterated this by pointing to the UI's Issue Resources.
- [`2026-01-19-85cc202e-0e74-46ce-a485-e5fcd11f7a8c`] (messages 7-11) The agent initially pursued changing or injecting `PromptOverwriteNewerLogs`, but the user redirected it to substituting the TTY underneath huh instead.
- [`2026-01-19-85cc202e-0e74-46ce-a485-e5fcd11f7a8c`] (messages 37-42) The agent proposed a helper taking a simple input string, and the user corrected the requirement to provide a read/write function while keeping timeout management centralized.
- [`2026-01-19-85cc202e-0e74-46ce-a485-e5fcd11f7a8c`] (messages 44-58) The agent implemented response timing with sleeps, and the user corrected the approach to read from the pty until the prompt appeared instead.

**`not_enough_specification` → `self-report`**

- [`2026-01-19-85cc202e-0e74-46ce-a485-e5fcd11f7a8c`] (message 5) The agent explicitly decided to wait for the user to confirm what they wanted to do next rather than having a confirmed specification for the second change.
- [`2026-01-19-c2b51eb0-d0e9-43cf-b431-42c05d49450b`] (message 54) The agent explicitly recognized that the relationships between sessions, checkpoints, commits, and branches were architecturally unclear and said it needed to clarify which mental model should drive the design.
- [`2026-01-19-c2b51eb0-d0e9-43cf-b431-42c05d49450b`] (message 260) After review, the agent explicitly stated that the Task 3 requirements were ambiguous because the plan required applying a limit while also deferring checkpoint listing to a later task.
- [`2026-01-20-20b81bf8-77cd-4205-8ac9-727b573a70e4`] (message 126) The agent acknowledged it had no concrete evidence that `edit` was a real Gemini CLI tool, after having added it based on ambiguous documentation.
- [`2026-01-20-20b81bf8-77cd-4205-8ac9-727b573a70e4`] (message 144) The agent again acknowledged it lacked concrete evidence that the removed `edit_file` and `save_file` names were not real tools.

**`no_visual_reference`**

- (no examples captured in this run)

## Run Reliability

**15** sessions were judged by `openai/gpt-5.6-luna`. **0** were skipped as having no user messages at all, and **0** could not be downloaded or parsed.

| Call | Succeeded | Failed | % of judged sessions covered |
|---|---|---|---|
| `not_enough_verification` | 15 | 0 | 100% |
| `not_enough_specification` | 15 | 0 | 100% |
| `no_visual_reference` | 15 | 0 | 100% |

## Performance Notes

Total wall-clock time for this run: 212s. Average download time per session: 0.35s. Average parse time per session: 0.00s.

| Call | Avg call time | Avg prompt size |
|---|---|---|
| `not_enough_verification` | 4.84s | 57,520 chars |
| `not_enough_specification` | 6.60s | 57,771 chars |
| `no_visual_reference` | 2.32s | 55,560 chars |

## A Note of Caution

These categories replace the earlier separate symptom list (`no_spec`, `no_closed_loop`, `no_acceptance_criteria`, `repetitive_fix_attempts`), regrouping them by root cause — a verification gap vs. a specification gap — and splitting "repetitive fixes" into two distinct subcategories depending on whether the repeated correction was about a technical bug or a requirements misunderstanding. Spot-checking the Examples section above against real transcripts is recommended before citing these numbers externally.
