# Local Split Advisor — session report

- Sessions scanned: **32**
- Sessions worth splitting (≥ 0% and ≥ $0.00): **2**
- Total modelled cost of all sessions: **$205.30**
- Total modelled saving from acting on suggestions: **$1.48**
- Sessions that already used plan mode (no action needed): **1**
- Detection: heuristic only

## Suggestions (ranked by modelled saving)

| Session | Project | Turns | Peak | Pattern | Split point | Save $ | Save % |
|---|---|---:|---:|:---:|---|---:|---:|
| `f4af9989` | -traces-agent-trace-analysis | 232 | 222k | plan-mode | after turn 13 (end of opening reading phase) | $1.13 | 5% |
| `1c8b54c4` | -traces-agent-trace-analysis | 89 | 109k | plan-mode | after turn 9 (end of opening reading phase) | $0.35 | 5% |

### Details

**`f4af9989-d83d-4ce5-9cbc-f5e409830836`** — -Users-andreistan-Desktop-Agent-traces-agent-trace-analysis
- 232 turns, 80 tool calls, peak context ~222k tokens, as-is cost $23.50
- Did ~6 reads before the first edit — running that opening exploration in plan mode (or a separate reading session) would cut re-read cost by ~$1.13 (5%).

**`1c8b54c4-e917-44fa-b9fa-d9671ad5eb1b`** — -Users-andreistan-Desktop-Agent-traces-agent-trace-analysis
- 89 turns, 36 tool calls, peak context ~109k tokens, as-is cost $7.57
- Did ~6 reads before the first edit — running that opening exploration in plan mode (or a separate reading session) would cut re-read cost by ~$0.35 (5%).

## Already using plan mode

- `3ec63ba1` (-traces-agent-trace-analysis): Front-loaded reading detected, but this session already used plan mode — the opportunity was taken.

## All scanned sessions

| Session | Turns | Peak | Pattern | Modelled saving | Suggested? |
|---|---:|---:|:---:|---:|:---:|
| `edad9dce` | 282 | 315k | A | no split point | — |
| `383e7f1b` | 169 | 450k | none | no split point | — |
| `a7526c9f` | 318 | 197k | none | no split point | — |
| `f4af9989` | 232 | 222k | A | $1.13 (5%) | ✅ |
| `67805753` | 175 | 199k | none | no split point | — |
| `63f0bb29` | 177 | 192k | none | no split point | — |
| `3ec63ba1` | 178 | 193k | A | $1.15 (8%) | below floor |
| `1c8b54c4` | 89 | 109k | A | $0.35 (5%) | ✅ |
| `0559e6c8` | 31 | 69k | none | no split point | — |
| `4c21635c` | 24 | 64k | none | no split point | — |
| `c2d1920d` | 24 | 53k | none | no split point | — |
| `cf2c1354` | 38 | 57k | none | no split point | — |
| `583792ff` | 26 | 51k | none | no split point | — |
| `9c5b8c93` | 28 | 48k | none | no split point | — |
| `1b245284` | 10 | 35k | none | no split point | — |
| `7a001f6a` | 25 | 47k | none | no split point | — |
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
