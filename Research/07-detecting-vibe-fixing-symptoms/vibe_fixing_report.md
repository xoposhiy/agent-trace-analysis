# Vibe-Fixing Symptoms in the SWE-Chat Dataset

This report shows how often "vibe-fixing" happens in coding-agent sessions. Vibe-fixing means a user accepts a fix from the agent without a clear task, without checking it, or without proof that it works. I checked **20 real coding sessions** from the SWE-Chat dataset (agent: Claude Code). All sessions were included, not only long ones.

## What I Looked For

I checked each session for 6 symptoms:

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

**1. LLM-as-judge (Claude Haiku 4.5), one call per symptom.** Instead of asking one call to judge every symptom at once, each session now gets one isolated call per symptom. Each call gets that symptom's definition and calibration examples up front, then the full session trace. The trace itself — a chronological, interleaved timeline of user messages, agent thinking, and test commands in the order they actually happened — is identical across a session's calls, so it's sent as a shared, cacheable prefix rather than rebuilt from scratch each time.

Compared to the previous version:
- User messages and agent thinking are now interleaved in one chronological timeline, instead of being shown as two separate, disconnected blocks — so the judge can tell which thinking happened between which two messages.
- Every thinking block from the agent is included in full, not just short excerpts that already contained a hedge word.
- Judge responses use structured output with a `reasoning` field that comes first, so the model has room to think before it has to commit to a yes/no, instead of being squeezed into a bare JSON object with no room to reason.
- The four symptoms about request quality (`no_spec`, `no_closed_loop`, `no_acceptance_criteria`, `no_visual_reference`) are explicitly defined to not apply to sessions that are pure questions/explanations with no code change requested, enforced both in the prompt and as a deterministic override after the judge answers.

**2. Metadata-only rules.** The two `scope_*` symptoms don't need an LLM. I just count files touched and turns per session, and flag sessions above a threshold.

## Results

| Symptom | Count | % of judged sessions |
|---|---|---|
| `no_closed_loop` | 4 | 27% |
| `no_spec` | 0 | 0% |
| `no_acceptance_criteria` | 2 | 17% |
| `scope_turns_too_long` | 7 | 35% |
| `scope_files_too_many` | 2 | 10% |
| `repetitive_fix_attempts` | 0 | 0% |
| `no_visual_reference` | 0 | 0% |

## Examples

For each symptom flagged by the LLM judge, here are real examples pulled from this run (session id + the judge's one-line evidence). These are spot-check material, not proof — always worth reading the underlying transcript before trusting an aggregate number.

**`no_spec`**

- (no examples captured in this run)

**`no_closed_loop`**

- [`2026-01-19-1b3e6441-62e3-4b41-8a0c-9d252f531f7a`] The agent created a CODEOWNERS file as requested but ran no tests, verification steps, or reproduction checks to confirm the file was created correctly or that the GitHub CODEOWNERS functionality would work as intended.
- [`2026-01-19-38917d7d-9f69-4210-a3b0-eed0a9e97575`] The agent added timing instrumentation to the telemetry code (turns 50-64) but only ran linting afterward, never running the full test suite (`mise run test`) to verify the changes work correctly and don't break existing functionality.
- [`2026-01-21-130d7b7e-5801-4345-9bd6-f32fd9b8429b`] User requested "Implement 1,2 and 3" at turn 94 (adding WithAgent to three files), but the session shows no test runs or verification that these implementations were completed or working correctly after the request.
- [`2026-01-22-48f428f9-72d8-40b2-9594-953019809473`] The agent implemented configuration changes to `.goreleaser.yaml` and `.github/workflows/release.yml` (turn 66-84) but never ran GoReleaser, tested the workflow, or verified the Homebrew tap would actually be updated—only checking for schema diagnostics in the editor.

**`no_acceptance_criteria`**

- [`2026-01-19-38917d7d-9f69-4210-a3b0-eed0a9e97575`] (no evidence text returned)
- [`2026-01-21-130d7b7e-5801-4345-9bd6-f32fd9b8429b`] Turn 0's request "We should add the agent name to any logging" lacks concrete acceptance criteria: it doesn't specify which logging calls should include the agent name, how to verify the change works, or what the expected output format should be.

**`no_visual_reference`**

- (no examples captured in this run)

**`repetitive_fix_attempts`**

- (no examples captured in this run)

## A Note of Caution

`no_verification_by_user` has been removed from this report entirely — it was mostly detecting "no proof shown in the transcript" rather than "the user actually skipped verifying," and a person could always test something outside the chat window, so it wasn't trustworthy as reported. The remaining symptoms rely on clearer, easier-to-check evidence (a request's wording plus its intent tag, whether a test command was run and what it returned, file counts), but spot-checking the Examples section above against real transcripts is still recommended before citing these numbers externally.
