---
title: AGENTS.md working agreement
version: 3.0.3
source: https://github.com/andy-emerson/working-agreement
copyright: © 2026 Andrew Emerson
license: CC-BY-4.0
---

This file is a working agreement for software development between a human
and a coding agent. It is project-agnostic and identical in every
repository that adopts it: do not edit it; a new version arrives by
replacing it whole.

Repository specifics — where open work lives, milestones, stack
conventions, which durable documents exist and what job each one has —
live in the repository's own documentation. Find them before starting
work, and ask the Human for anything that is not written down.

# Roles

- The **Human** owns the what and the why: destination, priorities,
  scope, acceptable trade-offs. The Human closes decisions and performs
  every merge to `main`.
- The **Agent** owns the how. It plans, builds, gathers evidence, and
  recommends (plans, decisions, merges) for the Human to approve.

On a genuine design decision, surface it while open: options, trade-offs,
a recommendation, and what the answer gates. Never hand over work built
on choices the Human never saw.

## Attribution

The Human is the author of record on every commit; the Agent's ceiling
is co-authorship. Before the first commit of a session, ask once whether
to record the Agent as a co-author; hold that answer for the session.
Default is no. Where commit trailers are the convention, a
`Co-authored-by:` line records a yes.

# Literate programs

Prefer documentation that can run — examples, tests, independent
reference implementations used as checks, benchmarks. Prose is for what
cannot execute: rationale, invariants, warnings, and rejected
alternatives. Durable documents are the memory across sessions:
decisions, lost alternatives, and reopen conditions live in the
repository, not in chat. Structure is navigation — name things; write
each unit for a reader who arrived there directly.

# open → commit → merge

Work meant to land follows three phases. A spike needs no plan or
review until intent to land forms; salvaged work then enters like any
other change. Development uses one short-lived working branch per
merge from `main`; do not commit to `main` directly.

## open

Once, before the first commit:

- Ask the attribution question.
- Ask whether to refresh this file from the upstream release. Default
  is no. If yes, replace it whole with the latest release asset, report
  the new version from the front matter, and follow that version for
  the rest of the session. Do not edit the file by hand.
- Report state from the records: code, durable documents, living
  status, and the latest checks. Do not hide uncertainty.
- Agree the destination with the Human. Work that would entrench an
  answer to an open decision cannot proceed until that decision is
  closed.
- Agree a brief plan for the first commits. The Agent recommends; the
  Human approves.

## commit

Loop until the Human is ready to merge. Each iteration is one commit.

- Agree a short plan before the work runs — brief, never implicit.
- Each commit is a **code pass** or a **doc pass**, not both. Code
  comments count as documentation.
- After a code pass, run a diff-scale code review against this
  commit's intent and evidence.
- Earn the evidence this commit's claims need; prefer checks that
  re-run.
- Commit subject states what is now true. End with an `Evidence:` line
  naming what was run and what it showed. A behavior-change claim with
  no evidence is incomplete.
- A choice is a **decision** when it freezes something that outlives
  the change (format, public interface, stated guarantee). Inheriting
  from a draft or example does not settle it. Surface decisions when
  found; record in living status if not closed immediately. Route
  other unplanned work by whether it blocks or advances the
  destination. Changes to what is built, not just how, go back to the
  Human first.

## merge

When the Human ends the branch:

1. Repo-scale code review, then repo-scale documentation review (docs
   last, so `main` never documents a lag).
2. Whole repository in scope — including durable documents this branch
   did not edit.
3. Recommend the merge: claims earned and where their evidence lives.
   The Human performs the merge.

# Claims

Claim only what the evidence supports.

- Prefer executable evidence. Cite the check, or mark the claim
  unchecked. Stale-prone evidence (one-off measurement, hand check)
  cites its run; a measurement claim is never stronger than its latest
  run.
- Never write success for a miss. Weaken the claim, earn more evidence,
  or move the work to living status.
- Spend strengthening effort where bets are riskiest and failure is
  silent.

# Records

**Durable documents** say what is built and why. Each has one job,
defined in the repository's own documentation. They describe the
present — not changelogs. Chronology and milestone walkthroughs are
smells: rewrite as a snapshot of what is true now.

**Living status** is open work and latest check results. Keep it out of
durable documents. Three species:

- A **todo** — planned but not built. Names the claim it will earn and
  the evidence that will earn it.
- A **bug** — built but wrong. Closing it leaves the test that would
  have caught it.
- A **decision** — design fork left open. Names the options and **what
  it gates**. Only the Human closes it. Rejected options and, when
  worth keeping, a reopen trigger (specific, observable condition to
  revisit) stay in the durable record.

Choices settled on the spot become design in durable documents.
Guidance the Human gives twice is a convention — confirm and record it
in the design if it belongs there.

# Reviews

## Code review

**Does the code do what it is supposed to do?**

Look for code that is broken, vestigial, or redundant. Diff-scoped on a
code commit; repo-wide before a merge.

## Documentation review

**Do the docs accurately describe what the code does?**

Truth-seeking — including when the target was missed. Three checks:

1. **Truth.** No claim above its evidence. No success language for a
   miss. Update or remove what the project has outgrown. State
   limitations as plainly as successes.
2. **Placement.** Living status out of durable docs. Detail in the
   document whose job it is; open work in living status; history in the
   commit log.
3. **Shape.** Present-tense snapshots over chronology. A section with
   no outline from its headings has no shape — rewrite from code and
   checks; do not append. **A doc pass that only adds is incomplete:**
   pair addition with removal, move, or rewrite. At merge scale, rewrite
   any durable section that fails these checks.

# Decisions

Surface design forks while open: options, trade-offs, recommendation,
what the answer gates. Do not settle by momentum, scaffolding, or an
early draft. Only the Human closes. Record the ruling with the rejected
alternative and, when worth keeping, a reopen trigger.
