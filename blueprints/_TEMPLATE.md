# BLUEPRINT TEMPLATE

Every blueprint in this folder uses exactly these sections, in this order.
Written by the planning model (Fable 5, 2026-07-07) for a cheaper builder
model to execute cold — the builder opens ONE blueprint file with zero
memory of the planning session and must not need to ask questions.

Sections:

**BUILDER:** which model, working alone, cold start, cannot ask questions + one line why.

**GOAL** — what exists and works when done, stated as a finished result.

**CONTEXT THE BUILDER NEEDS** — files to read first, real inputs in full,
data shapes with real values, gotchas found while grounding.

**CONSTRAINTS** — files it may touch, files it must not touch, stack to
respect, non-negotiables.

**STEP-BY-STEP PLAN** — numbered, in build order, one concrete action per
step, no judgment calls left open. Decisions the planner made are marked
`DECISION:` with the reasoning in one line.

**EXACT INPUTS TO USE** — file names, the kickoff prompt to hand the
builder, verbatim copy/values.

**DEFINITION OF DONE** — checklist of observable behaviors and exact
commands that must pass.

**IF SOMETHING IS UNCLEAR** — make the smallest safe assumption, write
`ASSUMPTION:` at the top of the output, keep going. Never stall, never
invent big new scope.
