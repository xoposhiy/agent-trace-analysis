# Vibe-Fixing Symptoms in the SWE-Chat Dataset

This report shows how often "vibe-fixing" happens in coding-agent sessions. Vibe-fixing means a user accepts a fix from the agent without a clear task, without checking it, or without proof that it works. I checked **2 real coding sessions** from the SWE-Chat dataset (agent: Claude Code). All sessions were included, not only long ones.

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
| `no_closed_loop` | 0 | 0% |
| `no_spec` | 0 | 0% |
| `no_acceptance_criteria` | 0 | 0% |
| `scope_turns_too_long` | 1 | 50% |
| `scope_files_too_many` | 0 | 0% |
| `repetitive_fix_attempts` | 0 | 0% |
| `no_visual_reference` | 0 | 0% |

## Examples

For each symptom flagged by the LLM judge, here are real examples pulled from this run (session id, the turn(s) the evidence came from, and the judge's one-line reason). These are spot-check material, not proof — always worth reading the underlying transcript before trusting an aggregate number.

**`no_spec`**

- (no examples captured in this run)

**`no_closed_loop`**

- (no examples captured in this run)

**`no_acceptance_criteria`**

- (no examples captured in this run)

**`no_visual_reference`**

- (no examples captured in this run)

**`repetitive_fix_attempts`**

- (no examples captured in this run)

## Performance Notes

Total wall-clock time for this run: 2728s. Average download time per session: 0.26s. Average parse time per session: 0.00s.

| Symptom | Avg call time | Avg prompt size |
|---|---|---|
| `no_spec` | 272.71s | 86,132 chars |
| `no_closed_loop` | 272.69s | 85,508 chars |
| `no_acceptance_criteria` | 272.90s | 85,119 chars |
| `no_visual_reference` | 272.72s | 85,162 chars |
| `repetitive_fix_attempts` | 272.84s | 85,332 chars |

## A Note of Caution

`no_verification_by_user` has been removed from this report entirely — it was mostly detecting "no proof shown in the transcript" rather than "the user actually skipped verifying," and a person could always test something outside the chat window, so it wasn't trustworthy as reported. The remaining symptoms rely on clearer, easier-to-check evidence (the actual request text, the raw tool calls and their results, file counts), but spot-checking the Examples section above against real transcripts is still recommended before citing these numbers externally.
