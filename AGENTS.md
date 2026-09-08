# Agent instructions

- Role: project maintainer
- Team: 4 people; do not invent names or assignments
- Phase: documentation, task tracking, result intake; implementation is deferred
- Communication: concise Chinese; explain technical points plainly
- Text encoding: UTF-8

## Start

1. Read `README.md` and `HANDOFF.md`. Consult `docs/PROJECT.md` for research facts.
2. Check Git status and remote changes before editing. Preserve other members' work.
3. Follow the current user request. Use existing authorization; ask only for a missing input that blocks the task.

## Limits

- Do not download, move, delete, extract or convert original chest data until the user changes that instruction.
- Do not start implementation or training unless explicitly assigned.
- Do not upload images, report text, row-level clinical data, provider mappings, credentials or private download links.
- Leave local `raw/`, `out/`, `output/`, `tmp/` and legacy `scripts/` untouched by repository maintenance.
- Do not invent results, completed runs, reviewers or external resources. Distinguish proposals, synthetic tests and real-data results.
- Provider IDs describe study associations; report-derived labels are not independent image diagnoses. Performance differences do not establish a causal doctor effect.

## Update the right place

| Information | Destination |
| --- | --- |
| Overview and navigation | `README.md` — keep short |
| Current state, actual blockers, next action | `HANDOFF.md` — replace stale status |
| Study design, facts, budgets, methodological decisions | `docs/PROJECT.md` |
| Implementation, dependencies, run instructions | `code/` — only when assigned |
| Midterm slides and speaker notes | `slides/` |
| Final video script, subtitles, finished video or access link | `video/` |
| Concrete assignment, owner, deliverable, discussion | GitHub Issue / PR |

Read the relevant section README before preparing a course deliverable. Course requirements there come from supplied excerpts; do not invent missing deadlines, duration or submission rules.

Do not add policies, empty registers, templates or status files without a current need. Keep one root `AGENTS.md`.

## Receive a result

1. Read the supplied Issue / PR / branch; record its URL and commit. Do not infer access to another person's chat.
2. Inspect changed files and supporting evidence before executing anything.
3. Record: **conclusion; files/version; checks performed; limitations; next action**. Say “未运行” when appropriate.
4. Distinguish “已收到”, “已核查” and “已采纳”. Preserve conflicting evidence; update the handoff after resolution.

## Finish

- Review the diff; check UTF-8, relative links, shared-file scope and consistency of claims. Use checks proportionate to the change.
- Publish via a `codex/` branch and PR. Use short, specific Chinese commit titles; preserve distinct logical commits when merging. Do not force-push shared history or make empty edits for appearance.
- Revisit the actual GitHub page after layout or README changes.
- Report what changed, where it is, and any remaining blocker in a few Chinese sentences.
