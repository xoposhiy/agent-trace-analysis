# Vibe-Fixing Symptoms in the SWE-Chat Dataset

This report shows how often "vibe-fixing" happens in coding-agent sessions. Vibe-fixing means a user accepts a fix from the agent without a clear task, without checking it, or without proof that it works. I checked **4,794 real coding sessions** from the SWE-Chat dataset (agent: Claude Code). All sessions were included, not only long ones.

## What I Looked For

I checked each session for 7 symptoms:

| Symptom | What it means |
|---|---|
| `no_spec` | The user's request is very short and unclear, or the agent shows doubt but still submits an answer |
| `no_closed_loop` | The user asks for a fix, but there is no way to check if it worked (no test run) |
| `no_acceptance_criteria` | The user's goal is vague ("make it faster", "clean this up"), with no clear target |
| `no_visual_reference` | The user asks for a UI/visual change, but gives no image or design file |
| `repetitive_fix_attempts` | The agent fixes the same bug wrong more than once, and the user has to report it again |
| `scope_files_too_many` | Too many files were changed in one session |
| `scope_turns_too_long` | The session had an unusually high number of turns |
| `no_verification_by_user` | There is a real sign the user did not check the fix (not just "no proof shown") |

## How I Detected Them

I used two methods:

**1. LLM-as-judge (Claude Haiku 4.5).** For 6 of the symptoms, I cannot use simple rules — I need to read the conversation. So each session was sent to Haiku, one API call per session. To keep this cheap, I did not send the full raw transcript. Instead, I built a short "case file" per session with:
- all user messages, in order
- the number of files touched
- any test/build commands run, and whether they passed or failed
- short pieces of agent "thinking" where it showed doubt but still acted

Haiku received this case file plus a clear definition of each symptom, and returned a yes/no answer with a short reason for each one.

**2. Metadata-only rules.** The two `scope_*` symptoms don't need an LLM. I just count files touched and turns per session, and flag sessions above a threshold.

## Results (4,794 Sessions Judged)

| Symptom | Count | % of sessions |
|---|---|---|
| `no_closed_loop` | 3,910 | 82% |
| `no_verification_by_user` | 3,778 | 79% |
| `no_spec` | 2,426 | 51% |
| `no_acceptance_criteria` | 2,370 | 49% |
| `scope_turns_too_long` | 1,282 | 26% |
| `scope_files_too_many` | 1,154 | 24% |
| `repetitive_fix_attempts` | 896 | 19% |
| `no_visual_reference` | 275 | 6% |

## A Note of Caution: One Number Is Not Reliable Yet

`no_verification_by_user` (79%) should be treated carefully. I tried to fix its prompt so it only flags sessions with a **real sign** the user skipped verification — not just "the transcript doesn't show proof." But when I checked the evidence text Haiku gave us, most reasons still said things like *"never explicitly confirms the fix worked"* — which is exactly the old, weaker pattern. This means the judge is likely still measuring "no proof visible in the chat" instead of "the user actually skipped checking." Since a person could easily test something outside the chat window, this number is probably too high, and should not be reported as-is without further review.

All other symptoms use clearer, easier-to-check evidence (a request's wording, whether a test command was run, file counts), so they are more trustworthy as reported.

## Why This Matters

`no_closed_loop` and `no_spec` are the most common patterns: most sessions start with a vague request and end with no test to confirm the fix worked. This matches the idea of "vibe-fixing" — moving fast without a clear spec or a way to prove success. `no_visual_reference` is rare (6%), which makes sense since most sessions in this dataset are backend/CLI work, not UI work.
