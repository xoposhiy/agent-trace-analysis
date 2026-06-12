# Agent Trace Analysis

## Overview

A research project for analyzing local coding-agent traces in order to **find, categorize, and explain problems** in how the agent behaved. The long-term goal is a small local application that shows you your analyzed traces and tells you what went wrong — eventually even in real time, surfacing hints while a session is happening.

## Motivation

Coding agents (like Claude Code) produce rich, detailed traces of their work: tool calls, token usage, reasoning, and turn-by-turn interaction with a human. Buried in those traces are recurring problems — wasted effort, unnecessary back-and-forth, expensive detours — that are hard to spot by eye and easy to repeat.

If we can reliably detect and explain these problems, we can help users (and agent developers) understand where sessions go wrong, what it cost them, and how to do better next time.

## What We Detect

### Detection scopes

How and where we look for a signal that something went wrong:

- **Feedback-based** — use the human's response to an agent turn to infer whether the agent made a mistake or hit a problem, or whether things are progressing fine. The user's reply is a natural signal of dissatisfaction, correction, or approval.
- **Within-turn** — find problematic or inefficient parts inside a single agent turn, independent of any later user feedback.

### Cost dimensions

For any detected problem — whether feedback-based or within-turn — we want to estimate its "price" along several cross-cutting dimensions:

- **Money / cost** — tokens and dollars spent; cost-inefficient behavior.
- **Human attention** — how many times the agent made the human do something they could have avoided.
- **Agent runtime** — wall-clock time the agent spent working.

These dimensions apply across both detection scopes, so each problem can be quantified by what it actually cost.

## Trace Sources

- **Claude Code transcripts** — local session logs (`~/.claude/projects/.../*.jsonl`). The primary, real-world source.
- **Trace dataset** — a large collection of traces to test, iterate on, and tune detection methods against. Example: [SWE-chat](https://huggingface.co/datasets/SALT-NLP/SWE-chat)
- **Benchmark runs** — traces from benchmark executions where we have ground-truth `pass`/`fail` labels per task. These let us analyze failed tasks for problematic agent behavior and cross-check our detectors against real outcomes. Example: [DeepSWE](https://deepswe.datacurve.ai).
  
## Approach

The detection method is an open research question — we will study prior work and build something analogous. The current working direction:

1. **Preprocess** the trace into a compact, analysis-friendly representation.
2. **Find suspicious regions** via heuristics / retrieval, to focus attention on likely problem spots.
3. **LLM-as-a-judge** over those regions, using rubrics to classify and explain problems.

This is a direction, not a final design.

## Roadmap

0. **Research** — survey prior work (papers) and existing trace datasets to learn how others detect and categorize agent problems, and to choose data to build on.
1. **Post-hoc analysis (MVP)** — analyze completed sessions in batch; detect, categorize, and explain problems.
2. **Local viewer app** — a small local application that displays analyzed traces and reports what went wrong and what it cost.
3. **Real-time hints (vision)** — surface problems and hints live, while a session is in progress.
