---
title: Prep with AI
description: Turn Gemini or Claude into a rigorous mock interviewer. Copy-paste prompt packs, a Gemini CLI command, and a Claude Code skill — all calibrated to The Handbook's rubrics for SDE coding, ML theory, ML coding, ML system design, and behavioural interviews.
---

# Prep with AI

The single highest-leverage way to use an LLM for interview prep is **not** to ask it for
answers — it's to make it *interview you*. An AI interviewer gives you unlimited, on-demand
mock loops, forces active recall, makes you explain your reasoning out loud, generates endless
variations of a question, and grades you against a rubric.

This page sets that up two ways, both calibrated to The Handbook's own rubrics:

- A reusable **assistant** you install once (a Gemini CLI command or a Claude Code skill), then call whenever you want a mock.
- Copy-paste **prompt packs** you can drop into any chat (Gemini, Claude, or anything else).

::: tip Use it the right way
Reading solutions feels productive but builds weak recall. Get interviewed *first*, struggle, then
read the relevant Handbook page to close the gap the interviewer found. Five mock interviews is the
difference maker — candidates who do them roughly double their pass rate.
:::

---

## Set up your AI interviewer

Install once, then run a mock any time with a single command.

:::tabs
== Gemini (recommended)

**Option A — Gemini CLI command (best).** The Handbook ships a ready-made command at
[`.gemini/commands/mock-interview.toml`](https://github.com/AbhinaavRamesh/handbook/blob/main/.gemini/commands/mock-interview.toml).

1. Install the [Gemini CLI](https://github.com/google-gemini/gemini-cli).
2. Make it available globally (or per-project):
   ```bash
   mkdir -p ~/.gemini/commands
   curl -o ~/.gemini/commands/mock-interview.toml \
     https://raw.githubusercontent.com/AbhinaavRamesh/handbook/main/.gemini/commands/mock-interview.toml
   ```
3. In the Gemini CLI, run:
   ```text
   /mock-interview sde-coding google
   /mock-interview ml-system-design
   /mock-interview behavioural
   ```

**Option B — Gemini Gem (web/app).** In the Gemini app, create a new **Gem**, name it
"Mock Interviewer", and paste the `prompt` body from the same `.toml` file as the Gem's
instructions. Then just tell the Gem which track you want.

== Claude

**Claude Code skill.** The Handbook ships a skill at
[`.claude/skills/mock-interviewer/SKILL.md`](https://github.com/AbhinaavRamesh/handbook/blob/main/.claude/skills/mock-interviewer/SKILL.md).

1. Copy the skill folder into your skills directory (global or per-project):
   ```bash
   mkdir -p ~/.claude/skills/mock-interviewer
   curl -o ~/.claude/skills/mock-interviewer/SKILL.md \
     https://raw.githubusercontent.com/AbhinaavRamesh/handbook/main/.claude/skills/mock-interviewer/SKILL.md
   ```
2. In Claude Code, just ask: *"Run a mock SDE coding interview for Google"* or
   *"Give me a behavioural mock and score me."* Claude auto-invokes the skill.

You can also paste the `SKILL.md` body into the [Claude.ai](https://claude.ai) chat or a Project's
custom instructions if you are not using the CLI.

:::

---

## Prompt packs

No install needed — paste one of these into Gemini or Claude, answer one question at a time, and
ask it to score you at the end. Each pack is calibrated to the matching Handbook rubric.

### SDE coding

```text
You are a senior Google interviewer running a 45-minute coding interview. Give me ONE
medium/hard DSA problem and act like a real interviewer: do not reveal the solution, hint only
when I'm truly stuck, and push me through Clarify → brute force + complexity → optimal approach
(explained BEFORE I code) → clean code → dry-run an example → state time/space complexity.
Reward constant communication; flag silent coding or untested code. One step at a time, wait for
me each turn. When I say "score me", rate me 1-5 on: Communication, Problem-Solving, Correctness,
Code Quality, Complexity Analysis, Testing & Edge Cases — then give a verdict and tell me exactly
what to review.
```

### ML theory

```text
You are an ML interviewer running a 30-minute concept interview. Ask me ONE core ML question,
then go deep: "why", tradeoffs, and a curveball follow-up. Don't lecture — make me reason. One
question at a time; wait for me. When I say "score me", rate me 1-5 on: Conceptual Correctness,
Depth, Tradeoff Awareness, Clarity, Handling Follow-ups — then give a verdict and the exact topics
I should review.
```

### ML coding (from scratch)

```text
You are an MLE interviewer running a 45-minute ML coding interview. Ask me to implement ONE ML
algorithm FROM SCRATCH in NumPy (no sklearn for the core algorithm). Make me Clarify
input/output/edge cases/allowed libs → outline the approach aloud → code the happy path → handle
edge cases → test on a tiny example. Reward vectorization and numerical care. One step at a time;
wait for me. When I say "score me", rate me 1-5 on: Clarification, Approach, Correctness (from
scratch), Vectorization/Efficiency, Edge Cases, Explanation — then give a verdict and what to
review.
```

### ML system design

```text
You are a staff MLE interviewer running a 45-minute ML system design interview. Give me ONE
real design prompt (e.g. recommendations, feed ranking, fraud, search). Drive me through the
6-step framework: (1) Problem clarification — success metric, scale/QPS, latency, constraints;
(2) High-level architecture; (3) Feature pipeline & feature store; (4) Model training; (5) Serving
& inference — latency, throughput, caching, fallback; (6) Monitoring, drift & retraining. Push on
tradeoffs at every step; don't accept hand-waving. One step at a time; wait for me. When I say
"score me", rate me 1-5 on: Requirements & Scoping, Architecture, Features & Data, Modeling
Choices, Serving & Scale, Monitoring & Iteration — then give a verdict and what to review.
```

### Behavioural

```text
You are a hiring manager running a 30-minute behavioural interview. Ask me ONE competency
question (ambiguity, conflict, failure, leadership, or impact) and enforce STAR: Situation (~20%),
Task (~10%), Action (~60% — the heart), Result (~10%, quantified). Probe relentlessly for "I" vs
"we" and for measurable outcomes; ask realistic follow-ups. One question at a time; wait for me.
When I say "score me", rate me 1-5 on: STAR Structure, Ownership ("I"), Impact/Results,
Reflection/Growth, Communication — then give a verdict and what to tighten.
```

---

## Workflow tips

- **Get interviewed before you read.** Take the mock cold, let it expose a gap, *then* read the
  matching Handbook page — that's when it sticks.
- **Explain-back.** After it shows a technique, close the tab and re-derive it to the AI in your own
  words. If you can't, you don't know it yet.
- **Generate, then grade.** Ask the AI for three variations of a question you missed and redo them
  the next day (spaced practice beats cramming).
- **Turn misses into flashcards.** Every question you fumble becomes a card — pair this with the
  [Handbook Anki deck](/resources/anki-deck) for spaced repetition.
- **Time-box.** Hold yourself to the real budget (45 min coding, 30 min behavioural). Speed under
  pressure is a separate skill from correctness.

### Where to go deeper in The Handbook

- [SDE Fast Track](/sde-coding/fast-track/) — a 2-week intensive plan
- [ML System Design Framework](/ml-design/framework) — the 6-step framework the prompts use
- [STAR Framework](/behavioural/star-framework) — the behavioural rubric the prompts use
- [ML Interview FAQ](/ml-fundamentals/interview-faq/) — concept drills for the theory mock

::: info Model-agnostic
The prompt packs work in any capable LLM. We lead with Gemini and provide a first-class Claude
path; pick whichever you already live in.
:::
