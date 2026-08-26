# Vibe-Fixing Symptoms in the SWE-Chat Dataset

This report shows how often "vibe-fixing" happens in coding-agent sessions — the agent's work moving forward without enough specification or enough verification. I checked a **random sample of 100 sessions** drawn from the 4,852 Claude Code sessions in the SWE-Chat dataset, drawn uniformly without replacement (RNG seed `234`). Selection is therefore unbiased, but each session percentage below carries a sampling error of up to ±10 points at 95% confidence, so read them as approximate. The seed reproduces the draw, not the run: the judge samples, so re-judging the same sessions will not reproduce the same counts. Short sessions were included alongside long ones.

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

**LLM-as-judge, one call per category** (3 calls per session total: `not_enough_verification`, `not_enough_specification`, `no_visual_reference`). Each session's raw transcript is rendered as a single chronological, typed-block timeline — every user message, every reply the agent wrote back to the user, every piece of agent thinking, and every tool call together with its raw result, in the exact order they happened (rendering approach inspired by [VCC](https://github.com/lllyasviel/VCC)). That same timeline is reused across a session's 3 calls.

**Every block has a two-part address `P.S`.** A user message opens `[USER P.1]` and every block after it — thinking, tool call, reply — carries the same prompt number `P` with an increasing step `S`, so the judge reads both numbers off the block it is citing rather than counting messages itself.

**Findings name a cause and its evidence, separately.** The `cause_prompt` is the single prompt whose work led to the problem — that is what the counting is done on. The `evidence` list is every block where the problem is actually visible, each with its own coordinate and a one-line note. The two are often the same prompt, but for the repetitive-* subcategories they are deliberately different: the cause is the earlier prompt whose work was called finished, while the only observable symptom is the user's later complaint. Reporting only the cause used to send a reader to a block where there is nothing to see. Every coordinate is checked against the blocks the session actually contains and labelled where it does not resolve — but nothing is corrected and no finding is dropped for citing badly, so the misses stay visible and countable.

**Not every user message is a user prompt.** 42% of the text-carrying user events in these sessions are wrappers the tooling emitted itself — slash-command invocations, context-compaction summaries, `[Request interrupted by user]` markers. They are labelled `[SYSTEM P.1]`, shown to the judge in full (a `/clear` is the reason the context suddenly forgot everything, and an interruption is quotable evidence), and they keep their number so numbering stays purely positional — but they are excluded from the prompt denominator, because no human asked for anything in them.

Each call returns **every occurrence** it finds, not just the first — a session can show the same subcategory multiple times (e.g. the agent asks for manual testing twice, or gets pushed back on requirements three times), and each one is recorded with its own cause and evidence.

We do NOT pre-label which tool calls are "tests" or which files are "specs" using keyword lists or filename patterns — every tool call and its raw output are shown to the judge as-is, and it decides for itself.

**Metadata-only rules.** `scope_files_too_many` and `scope_turns_too_long` don't need an LLM — just a count of files touched and assistant turns per session, flagged above a threshold. A session is flagged at **8 or more files touched** and at **150 or more assistant turns** respectively.

**What the judge does and doesn't see.** The timeline is condensed, not verbatim, and the caps matter when reading the numbers below — two of these checks are judgments about something being *absent*, which truncation can manufacture:

- Agent thinking has no session-wide cap, but an individual thinking block longer than 4,000 characters is shown as its first and last portions only, explicitly marked as truncated.
- An agent reply longer than 4,000 characters is shown as its first and last portions only, explicitly marked as truncated. Replies are rendered separately from thinking, so a claim the agent only made privately is not counted as something it told the user.
- Each tool result is cut to 600 characters. The tool *call* is always visible, but a long test run's actual output may be clipped.
- Each user message is cut to 2,000 characters, so a spec buried at the end of a very long request can be lost — which pushes `no-spec-detected` toward false positives.

## Results

A **problem** is one pair of (category, cause prompt). One prompt flagged with three subcategories of the same category is one problem with three confirmations, not three problems; the same prompt flagged under both verification and specification is two problems, one per category. "Evidence" counts the individual blocks the judge cited as showing it, and "Sessions" counts sessions with at least one finding.

**The subcategory rows do not add up to the category row above them, and they are not supposed to.** They are counted at different levels: the category row collapses a prompt's subcategories into the single problem they describe, the subcategory rows keep them apart. A category row reading 1 above three subcategory rows each reading 1 is the correct rendering of one prompt flagged three ways, not an arithmetic error.

The last column is the one to read for a rate: how many distinct prompts ended in this problem, out of the **688 real user prompts** across all judged sessions. Session percentages weigh a 2-prompt session the same as a 200-prompt one; the per-prompt rate does not.

It counts only the problems whose cause landed on a real user prompt. Coordinates the judge returned are recorded as given, never corrected, so some findings name a `[SYSTEM]` block or a number the session never had; those are still counted under "Problems" and still appear in the examples, but they have no real prompt to be a fraction of, so "% of prompts" leaves them out. Where the two columns disagree, the gap is the judge's attribution error rate — see Run Reliability.

That denominator excludes harness wrappers: of the 1,179 text-carrying user events in these sessions, **491 (42%)** were slash-command invocations, context-compaction summaries or interruption markers rather than requests from a human. They stay visible in the timeline and keep their position in the numbering; they are simply not prompts. Earlier runs of this pipeline counted them, so their per-prompt rates were understated by roughly the same factor and are not comparable with the numbers below.

Session percentages are per-check: each LLM check is divided by the number of sessions where **that** call succeeded, and the metadata-only checks by all judged sessions — so if any calls failed, the denominators differ slightly between rows. See Run Reliability below.

| Check | Sessions | % of sessions | Problems | Evidence | % of prompts |
|---|---|---|---|---|---|
| **`not_enough_verification`** (any) | 63 | 63% | 160 | 230 | 21.7% |
| &nbsp;&nbsp;`not-tested` | 48 | 48% | 105 | 132 | 14.2% |
| &nbsp;&nbsp;`self-report` | 15 | 15% | 16 | 22 | 2.3% |
| &nbsp;&nbsp;`ask-for-manual-testing` | 6 | 6% | 19 | 19 | 2.6% |
| &nbsp;&nbsp;`repetitive-bug-fixes` | 22 | 22% | 48 | 57 | 6.4% |
| **`not_enough_specification`** (any) | 19 | 19% | 53 | 72 | 7.3% |
| &nbsp;&nbsp;`no-spec-detected` | 3 | 3% | 3 | 4 | 0.4% |
| &nbsp;&nbsp;`repetitive-requirements-fixes` | 13 | 13% | 37 | 47 | 5.1% |
| &nbsp;&nbsp;`self-report` | 8 | 8% | 18 | 21 | 2.5% |
| `no_visual_reference` | 8 | 8% | 12 | 13 | 1.7% |
| `scope_files_too_many` | 20 | 20% | — | — | — |
| `scope_turns_too_long` | 25 | 25% | — | — | — |

## Examples

The unit here is a **problem**, not a finding: examples are drawn at random from the (session, cause prompt) pairs found in this run — up to 5 per subcategory — and each drawn prompt is then shown with **everything this category found on it**, including findings that came in under a sibling subcategory. Those carry their own subcategory in square brackets. Having sent you to a specific prompt, the report may as well tell you the whole of what it saw there.

Each finding lists the blocks the judge cited as showing it, as `P.S` — the same coordinate printed on the block in the timeline, so you can search for it directly. Open the session, go to those coordinates, and check. Spot-check material, not proof.

The draw is seeded (`12345`), so re-rendering this results file selects the same examples.

The category boundary is not crossed: a verification example never shows specification findings, even when both landed on the same prompt — the two categories are counted independently and are read independently. Within one category, a prompt drawn twice is expanded once and referenced thereafter.

**`not_enough_verification` → `not-tested`**

- [`04bed52e-adc8-429c-aab1-a555f681fc3b`] prompt 4 — 1 finding, 1 evidence
  - Agent claims the task is finished but never verified it (no test, no manual check).
    - 13.8 — The sparse-checkout fix was declared complete and issue #30 was closed, but the only visible automated check was markdownlint; no reproduction or execution of the changed workflow appears.
- [`04bed52e-adc8-429c-aab1-a555f681fc3b`] prompt 15 — 1 finding, 1 evidence
  - Agent claims the task is finished but never verified it (no test, no manual check).
    - 15.50 — The version-reconciliation changes were declared complete and issue #32 was closed without any visible test of version detection, mismatch handling, or generated output; only markdownlint is shown.
- [`39affcc6-442d-4446-b1e9-2d591fc7c313`] prompt 15 — 2 findings, 2 evidence
  - Agent claims the task is finished but never verified it (no test, no manual check).
    - 15.14 — The agent declared the manual SurrealDB startup workaround pushed, but provided no test run or CI validation.
  - [repetitive-bug-fixes] After the agent called it done, the user tested manually and reported bugs.
    - 18.1 — After the workaround was declared complete, the user reported that the container would not stop and demanded proper research, showing the finished fix was incorrect.
- [`3eff66e1-7ab2-497b-8765-cde9193ffc16`] prompt 17 — 2 findings, 4 evidence
  - Agent claims the task is finished but never verified it (no test, no manual check).
    - 17.22 — The commit hook output visibly reports that ruff-format failed and modified files, but the agent nevertheless states that all gates passed and treats the change as complete.
    - 17.23 — The agent declares that the P1 artifact fix and all review items are closed without running the newly changed --verify-artifacts command or a full post-format test suite.
  - [repetitive-bug-fixes] After the agent called it done, the user tested manually and reported bugs.
    - 18.1 — After the agent declared the commit complete, the user reported that a source file remained uncommitted in the working tree.
    - 18.2 — The subsequent git status confirmed the dirty working tree, showing the commit workflow had not left the branch clean as claimed.
- [`4588eb5b-9e44-4453-9686-c68728cdb681`] prompt 5 — 1 finding, 2 evidence
  - Agent claims the task is finished but never verified it (no test, no manual check).
    - 5.27 — The agent declared the dir-cache replacement complete after only a Bun build check; no tests or runtime verification covered config persistence or migration behavior.
    - 5.30 — The agent summarized the cache removal as done without testing the resulting CLI behavior or existing-config compatibility.

**`not_enough_verification` → `self-report`**

- [`18d58335-7188-4a3b-8f67-59f18047b448`] prompt 7 — 1 finding, 1 evidence
  - Agent itself says some important part wasn't tested.
    - 7.2 — The agent explicitly acknowledges that the assumed Moltbook API response shape is unverified and asks whether to check it or defer verification.
- [`270a765a-5ed0-4211-85ab-5ba1f1e21675`] prompt 3 — 1 finding, 1 evidence
  - Agent itself says some important part wasn't tested.
    - 3.24 — The agent declares Phase 1 complete and verified but explicitly says manual verification is still pending.
- [`2b483f70-15da-4e03-aee8-6c67bc5bb695`] prompt 1 — 1 finding, 3 evidence
  - Agent itself says some important part wasn't tested.
    - 1.132 — The agent's final coverage report visibly shows that requested files were not at 100%: tui.ts had only 80.20% line coverage, mock-handle.ts had 93.10% function coverage, and commands.ts and format.ts also remained below 100% line coverage.
    - 1.140 — The agent explicitly acknowledges that mock-handle functions remain uncalled and says reaching 100% coverage is not worth fixing, despite the user's requirement to reach 100% on all files.
    - 1.156 — The completion message reports success while admitting remaining uncovered lines and rounding artifacts rather than resolving or fully verifying the stated 100%-coverage requirement.
- [`39affcc6-442d-4446-b1e9-2d591fc7c313`] prompt 12 — 1 finding, 1 evidence
  - Agent itself says some important part wasn't tested.
    - 12.5 — The agent pushed the unit-test fix without running verification and explicitly noted that the smoke-test failure still remained unresolved.
- [`55309663-921a-4426-8ee1-67c81be6a692`] prompt 4 — 1 finding, 1 evidence
  - Agent itself says some important part wasn't tested.
    - 4.4 — The security-scanner subagent explicitly reports that the vulnerability checker failed because of network issues and that the audit continued with only manual analysis.

**`not_enough_verification` → `ask-for-manual-testing`**

- [`37fdbb4b-905a-4e19-97f6-f11a5166268d`] prompt 36 — 2 findings, 2 evidence
  - Agent asks the human to test something manually.
    - 36.66 — The agent explicitly asked the user to run pnpm dev to verify the completed performance changes.
  - [not-tested] Agent claims the task is finished but never verified it (no test, no manual check).
    - 36.66 — The agent declared nine performance and cleanup tasks complete and asked the user to run the dev server, without running a build or functional tests itself.
- [`37fdbb4b-905a-4e19-97f6-f11a5166268d`] prompt 40 — 1 finding, 1 evidence
  - Agent asks the human to test something manually.
    - 40.9 — The agent explicitly asked the user to test the Japanese 舟 search in the browser after adding the tokenizer and fallback.
- [`37fdbb4b-905a-4e19-97f6-f11a5166268d`] prompt 44 — 1 finding, 1 evidence
  - Agent asks the human to test something manually.
    - 44.13 — The agent explicitly asked the user to restart pnpm dev and verify behavior after reverting the shiki changes.
- [`37fdbb4b-905a-4e19-97f6-f11a5166268d`] prompt 57 — 1 finding, 1 evidence
  - Agent asks the human to test something manually.
    - 57.19 — The agent explicitly asked the user to run pnpm dev and verify the increased section-level search results.
- [`3a555a3c-b097-4227-998d-21602bf77677`] prompt 4 — 1 finding, 1 evidence
  - Agent asks the human to test something manually.
    - 4.5 — After only starting the dev server, the agent told the user they could check the favicon and avatar in the browser instead of performing that manual verification itself.

**`not_enough_verification` → `repetitive-bug-fixes`**

- [`1ce3dc88-a7e6-4707-a62c-39efdfab9804`] prompt 6 — 2 findings, 2 evidence
  - After the agent called it done, the user tested manually and reported bugs.
    - 10.1 — After the backfill was treated as ready, the user reported that every upsert still failed because the text attribute was being treated as filterable.
  - [not-tested] Agent claims the task is finished but never verified it (no test, no manual check).
    - 6.28 — The limit and transcript-indexing changes were summarized as complete without running tests, type checking, or a backfill verification before being committed in the next prompt.
- [`37fdbb4b-905a-4e19-97f6-f11a5166268d`] prompt 13 — 2 findings, 2 evidence
  - After the agent called it done, the user tested manually and reported bugs.
    - 15.1 — The user reported that twitterImg was undefined after the medium-priority refactoring had been declared complete.
  - After the agent called it done, the user tested manually and reported bugs.
    - 17.1 — The user reported a second undefined-variable error, mySite, after the previous fix was declared complete.
- [`39affcc6-442d-4446-b1e9-2d591fc7c313`] prompt 20 — 2 findings, 2 evidence
  - After the agent called it done, the user tested manually and reported bugs.
    - 23.1 — After the setup-action change was declared complete, the user supplied another failing smoke-test run requiring a new fix for missing API credentials.
  - [not-tested] Agent claims the task is finished but never verified it (no test, no manual check).
    - 20.8 — The agent declared the official setup action pushed without running the smoke tests or validating that the action worked in CI.
- [`3eff66e1-7ab2-497b-8765-cde9193ffc16`] prompt 17 — already shown under `not-tested` above
- [`c083c343-8223-4f88-b919-178a327346d0`] prompt 20 — 1 finding, 1 evidence
  - After the agent called it done, the user tested manually and reported bugs.
    - 23.1 — The user reported failing integration tests caused by the BaseCommand constructor refactor after the agent had declared the unit-test verification successful.

**`not_enough_specification` → `no-spec-detected`**

- [`0a8fa202-9a4d-4df6-afcf-ea50f7c23717`] prompt 11 — 2 findings, 3 evidence
  - User asked for an implementation without a detailed enough spec.
    - 11.1 — The user gives a subjective request to make the bio expression friendlier without specifying the desired tone or wording.
  - [repetitive-requirements-fixes] Agent fixed it the wrong way and the user pushed back, repeatedly.
    - 12.1 — The user supplies the exact sentence they wanted after the assistant’s friendliness rewrite did not match their intent.
    - 13.1 — The user further corrects the requirement by clarifying that this sentence may begin with “I.”
- [`25a509b7-17ce-4f7b-bb3e-793f1d3ada35`] prompt 1 — 1 finding, 1 evidence
  - User asked for an implementation without a detailed enough spec.
    - 1.2 — The user only provided the design command and feature slug “oauth-rar-dpop,” without concrete requirements, constraints, or desired architecture outcomes; the agent proceeded to launch design work rather than eliciting or locating a specification.
- [`960e1199-4c77-4e97-9fa1-33fb9498a0a9`] prompt 2 — 1 finding, 2 evidence
  - User asked for an implementation without a detailed enough spec.
    - 2.1 — The user requests conversion to µSv/h using the correct tube-specific rate but does not identify supported tube models, calibration factors, or the required behavior when the detector type is unknown.
    - 2.2 — The agent recognizes that conversion factors depend on tube type, then proceeds by assuming a list of common-tube factors without first establishing the repository's authoritative mappings or calibration rules.

**`not_enough_specification` → `repetitive-requirements-fixes`**

- [`04bed52e-adc8-429c-aab1-a555f681fc3b`] prompt 5 — 1 finding, 1 evidence
  - Agent fixed it the wrong way and the user pushed back, repeatedly.
    - 6.1 — The user reports that the requested removal of the project-specific name was incomplete: the name still appears in the edited file.
- [`b3342ff7-4bc3-4c55-b570-ff9cc1373790`] prompt 5 — 2 findings, 3 evidence
  - Agent fixed it the wrong way and the user pushed back, repeatedly.
    - 6.1 — The agent concluded that the owner could not issue permits because the owner's Scribe lacked a permit issuer, but the subsequent user correction states that creators must issue permits offline and delegate authority to the node.
  - [self-report] Agent itself says it doesn't have enough specification.
    - 5.5 — The agent explicitly acknowledged that it was getting ahead of the evidence and was unsure whether the grant intent was processed locally or by the node.
    - 7.22 — The agent reconsidered its diagnosis after discovering contradictory permit-issuer construction code and stated that it might have been chasing the wrong problem.
- [`b3342ff7-4bc3-4c55-b570-ff9cc1373790`] prompt 7 — **not a user prompt**: this number is a tooling-generated `[SYSTEM]` block, so the attribution missed (the finding still counts, but not toward the per-prompt rate) — 1 finding, 1 evidence
  - Agent fixed it the wrong way and the user pushed back, repeatedly.
    - 9.1 — The user corrected the plan's single-DID model, requiring a single authority permit primitive containing multiple authorized DIDs.
- [`b3342ff7-4bc3-4c55-b570-ff9cc1373790`] prompt 45 — 1 finding, 2 evidence
  - Agent fixed it the wrong way and the user pushed back, repeatedly.
    - 46.1 — The agent interpreted sync metadata as a page-wide shared document, but the user corrected that each page has a separate sync-metadata document between the node and each peer.
    - 48.1 — The user repeated the per-peer scope and explained that the node updates only the authorized peers' sync metadata after receiving the creator's layer.
- [`bda477f8-3201-4125-a11d-1e66b871f7dd`] prompt 1 — 1 finding, 2 evidence
  - Agent fixed it the wrong way and the user pushed back, repeatedly.
    - 1.46 — The user rejected the proposed save-and-restore workaround and said configuration should be injected as a dependency instead of relying on process.env.
    - 1.70 — The user again explicitly required removing the env option completely after the agent proposed retaining it for ORCHESTRATOR_MOCK_AGENT.

**`not_enough_specification` → `self-report`**

- [`7ada1d1a-292b-4e55-aa5d-4376df61cd13`] prompt 1 — 1 finding, 2 evidence
  - Agent itself says it doesn't have enough specification.
    - 1.6 — The agent explicitly identifies an unresolved question about whether the user means schema design or creating business instances, then proposes asking the user to choose a direction.
    - 1.7 — The agent presents multiple possible fixes and asks the user which direction to take, indicating it is not confident about the intended requirement.
- [`9e4910ab-95a9-4a30-a85e-4677017bfaf0`] prompt 12 — 1 finding, 1 evidence
  - Agent itself says it doesn't have enough specification.
    - 12.90 — The agent presents the choice between modifying the full schema migration and creating a separate migration, asks the user which is preferred, and thereby acknowledges the migration requirement is unspecified.
- [`b3342ff7-4bc3-4c55-b570-ff9cc1373790`] prompt 5 — already shown under `repetitive-requirements-fixes` above
- [`b3342ff7-4bc3-4c55-b570-ff9cc1373790`] prompt 13 — 1 finding, 1 evidence
  - Agent itself says it doesn't have enough specification.
    - 13.2 — The agent explicitly said it had been overcomplicating the design while repeatedly changing the proposed authority transport model.
- [`b3342ff7-4bc3-4c55-b570-ff9cc1373790`] prompt 21 — 1 finding, 1 evidence
  - Agent itself says it doesn't have enough specification.
    - 21.2 — The agent explicitly admitted it had missed a critical ordering requirement concerning when the node receives layer data relative to authority and fanout.

**`no_visual_reference`**

- [`64c72f02-bd94-45d3-9334-74fbb19bdba7`] prompt 1 — 1 finding, 1 evidence
  - The user asks for a UI/visual change, but gives no image or design file.
    - 1.1 — The user reports a visual timeline hover problem and unreliable node clicks, but provides no screenshot, mockup, reference image, or design file.
- [`65c7d9c3-3261-4bf1-9689-85e298d4780c`] prompt 3 — 1 finding, 1 evidence
  - The user asks for a UI/visual change, but gives no image or design file.
    - 3.1 — The user requests creative visual directions for a landing-page flow field with a more mathematical aesthetic, but provides no screenshot, mockup, reference URL, or design file.
- [`e6161535-9b0d-462b-bcff-9c03a6c69a27`] prompt 4 — 1 finding, 1 evidence
  - The user asks for a UI/visual change, but gives no image or design file.
    - 4.1 — The user requests repositioning a landing-page section relative to a horizontal line, without providing a visual design reference.
- [`e6161535-9b0d-462b-bcff-9c03a6c69a27`] prompt 6 — 1 finding, 1 evidence
  - The user asks for a UI/visual change, but gives no image or design file.
    - 6.1 — The user requests placing the hero above the left-hand-side line, with no screenshot, mockup, or reference URL available.
- [`e6161535-9b0d-462b-bcff-9c03a6c69a27`] prompt 11 — 1 finding, 1 evidence
  - The user asks for a UI/visual change, but gives no image or design file.
    - 11.1 — The user requests raising the content to avoid overlap with the isometric line, but still provides no visual reference.

## Run Reliability

**100** sessions were judged by `openai/gpt-5.6-luna`. **0** were skipped as having no user messages at all, and **0** could not be downloaded or parsed.

**Attribution accuracy.** 16 of 262 findings (6.1%) named a cause that is not a real user prompt: **16** pointed at a `[SYSTEM]` block (tooling output, not a request) and **0** at a number this session never assigned. Nothing was corrected or thrown away — those findings are counted under "Problems" and shown in the examples with a marker — but they are excluded from "% of prompts", which is why that column can be lower than the problem count implies. Treat this as the judge's citation error rate.

This measures the `cause_prompt` only. The `evidence` coordinates are checked the same way and kept as given whether or not they resolve, but they are not aggregated into a rate — individual misses among them show up in the diagnostics list below and inline in the examples.

| Call | Succeeded | Failed | % of judged sessions covered |
|---|---|---|---|
| `not_enough_verification` | 100 | 0 | 100% |
| `not_enough_specification` | 100 | 0 | 100% |
| `no_visual_reference` | 100 | 0 | 100% |

**No call-level failures.** Every judge call returned through the enforced schema; the plain-JSON fallback path was never used.

**Coordinate notes** — 20 cited prompt or step numbers that did not resolve against the session. Recorded, not repaired: the findings carrying them are still counted:

- `3×` not_enough_verification: cause_prompt 6 is a [SYSTEM] block, not a user prompt (kept, marked system)
- `2×` not_enough_verification: cause_prompt 39 is a [SYSTEM] block, not a user prompt (kept, marked system)
- `1×` not_enough_verification: cause_prompt 22 is a [SYSTEM] block, not a user prompt (kept, marked system)
- `1×` not_enough_specification: cause_prompt 1 is a [SYSTEM] block, not a user prompt (kept, marked system)
- `1×` not_enough_verification: evidence step 43 is not in 1..2 for prompt 1 (kept as-is)
- `1×` not_enough_verification: evidence step 55 is not in 1..2 for prompt 1 (kept as-is)
- `1×` not_enough_specification: cause_prompt 7 is a [SYSTEM] block, not a user prompt (kept, marked system)
- `1×` not_enough_specification: cause_prompt 41 is a [SYSTEM] block, not a user prompt (kept, marked system)
- `1×` not_enough_verification: cause_prompt 9 is a [SYSTEM] block, not a user prompt (kept, marked system)
- `1×` not_enough_verification: cause_prompt 64 is a [SYSTEM] block, not a user prompt (kept, marked system)
- `1×` not_enough_verification: cause_prompt 99 is a [SYSTEM] block, not a user prompt (kept, marked system)
- `1×` not_enough_verification: cause_prompt 2 is a [SYSTEM] block, not a user prompt (kept, marked system)
- `1×` not_enough_verification: evidence step 13 is not in 1..1 for prompt 2 (kept as-is)
- `1×` not_enough_verification: cause_prompt 3 is a [SYSTEM] block, not a user prompt (kept, marked system)
- `1×` not_enough_verification: cause_prompt 4 is a [SYSTEM] block, not a user prompt (kept, marked system)
- `1×` not_enough_verification: cause_prompt 8 is a [SYSTEM] block, not a user prompt (kept, marked system)
- `1×` not_enough_specification: evidence step 7 is not in 1..3 for prompt 7 (kept as-is)

## Performance Notes

Total wall-clock time for this run: 1899s. Average download time per session: 0.23s. Average parse time per session: 1ms.

| Call | Avg call time | Avg prompt size |
|---|---|---|
| `not_enough_verification` | 9.33s | 61,794 chars |
| `not_enough_specification` | 6.35s | 62,045 chars |
| `no_visual_reference` | 3.07s | 59,830 chars |

## A Note of Caution

These categories replace the earlier separate symptom list (`no_spec`, `no_closed_loop`, `no_acceptance_criteria`, `repetitive_fix_attempts`), regrouping them by root cause — a verification gap vs. a specification gap — and splitting "repetitive fixes" into two distinct subcategories depending on whether the repeated correction was about a technical bug or a requirements misunderstanding. Spot-checking the Examples section above against real transcripts is recommended before citing these numbers externally.

**Do not compare these per-prompt rates against earlier runs of this pipeline.** The denominator changed: harness wrappers used to be counted as user prompts and no longer are, which lifts every per-prompt rate by 1.71x on this run's data, with nothing about the sessions or the judge having changed. The prompt NUMBERS are unaffected — numbering stayed positional, so a citation still points at the same block it always did.
