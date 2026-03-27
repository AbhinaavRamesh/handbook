# Weekend Sprint

**32 Problems. 6 Tiers. 1 Weekend.** Focused DSA prep based on real SDE interview reports (2024-2025).

[Start with Graphs](./graphs) | [View Schedule](./schedule) | [Interview Day](./interview-day)

| Icon | Tier | Problems | Description | Link |
|------|------|----------|-------------|------|
| :red_circle: | Tier 1: Graphs | 8 problems | Highest frequency. BFS/DFS, topo sort, Dijkstra's. Appeared in nearly every SDE onsite. | [Study Graphs](./graphs) |
| :orange_circle: | Tier 2: Trees | 6 problems | Tree DP, LCA, serialize/deserialize. Second highest frequency in SDE interviews. | [Study Trees](./trees) |
| :yellow_circle: | Tier 3: Sliding Window | 4 problems | Phone screen favorites. Master the expand/shrink template and you're golden. | [Study Sliding Window](./sliding-window) |
| :green_circle: | Tier 4: DP | 5 problems | 1 DP question per loop typical. Cover 1D, 2D, and DFS+memo patterns. | [Study DP](./dp) |
| :large_blue_circle: | Tier 5: HashMap/Heap/BS | 6 problems | Pattern matching and optimization. Group anagrams, top-K, binary search variants. | [Study Data Structures](./data-structures) |
| :purple_circle: | Tier 6: Strings | 3 problems | Tries, two-pointer string matching, and backtracking. | [Study Strings](./strings) |

## The Reality Check

You have an interview **this week**. No time for a 3-month grind. This site is your battle plan.

::: warning YOUR SITUATION
This is not a "learn DSA from scratch" resource. This is a focused, prioritized list of 32 problems that actually show up in SDE interviews, based on real interview reports from 2024-2025.
:::

Here is what your interview looks like:

- **2 coding rounds**, 45 minutes each, conducted virtually
- **Plain text editor (no IDE)** -- you cannot run your code, no syntax highlighting, no autocomplete
- **Based on actual SDE interview reports** collected from candidates who interviewed in 2024-2025
- Expect 1 main problem per round, possibly with a follow-up or extension

::: info NO CODE EXECUTION
You will write code in a plain text editor (no IDE). Practice writing syntactically correct solutions without an IDE. Every character matters when you cannot hit "Run".
:::

---

## What Interviews Test

| Area | Focus | Weight |
|------|-------|--------|
| **Data Structures & Algorithms** | Graphs, Trees, DP, Sliding Window, HashMap/Heap, Strings | **~70%** |
| **System Design** | High-level architecture, trade-offs, scalability | **~25%** |
| **Behavioral** | Collaboration, ambiguity, leadership | **~5%** |

::: tip DSA IS THE GATE
Your coding rounds are the make-or-break. System design matters, but you will not reach the hiring committee if you bomb the coding rounds. Prioritize DSA above everything else this week.
:::

![Problem Difficulty Map](/sde-coding/sprint/problem_difficulty_map.png)

---

## How to Use This Site

::: details Step-by-step guide (click to expand)

1. **Start with Tier 1 (Graphs)** -- This is the highest-frequency category. If you only have a few hours, spend them here. Work through all 8 problems.

2. **Work through each tier in order** -- The tiers are ranked by frequency of appearance in real SDE interviews. Tier 1 first, Tier 6 last.

3. **Use the [Templates](./templates) page for quick reference** -- Memorize the BFS template, the sliding window template, and the topo sort template. These cover at least 60% of problems.

4. **Follow the [Schedule](./schedule) for a structured plan** -- If you have a full weekend, the schedule breaks the 32 problems across Saturday and Sunday with morning/afternoon/evening blocks.

5. **Review the [Interview Day](./interview-day) checklist before your rounds** -- Communication phrases, time allocation per section, and tips from candidates who received offers.

:::

![Tier Priority Distribution](/sde-coding/sprint/tier_distribution.png)

### The fastest path if you are short on time

| Time Available | What to Cover |
|---------------|---------------|
| **2 hours** | Graphs (#1-4): Islands, Course Schedule I/II, Word Ladder |
| **4 hours** | Add Trees (#9-11): Max Path Sum, LCA (BST + BT) |
| **8 hours** | Add Sliding Window (#15-18) and DP (#19-21) |
| **Full weekend** | All 32 problems + 2 timed mocks in a plain text editor |

---

## Problem Distribution at a Glance

| Tier | Category | Count | Priority |
|------|----------|-------|----------|
| :red_circle: T1 | Graphs | 8 | **Critical** |
| :orange_circle: T2 | Trees | 6 | **High** |
| :yellow_circle: T3 | Sliding Window | 4 | **High** |
| :green_circle: T4 | DP | 5 | **Medium** |
| :large_blue_circle: T5 | HashMap / Heap / BS | 6 | **Medium** |
| :purple_circle: T6 | Strings | 3 | **Lower** |
| | **Total** | **32** | |

::: warning DIMINISHING RETURNS
Going from 0 to 20 problems is massive. Going from 20 to 32 is incremental. If you are running out of time, stop at Tier 4 and spend the remaining hours doing timed mock rounds in a plain text editor instead.
:::

---

## Interview Day at a Glance

![Interview Timeline](/sde-coding/sprint/interview_timeline.png)

::: tip INTERVIEW FORMAT
Your rounds: **2 coding rounds, 45 min each**, in a plain text editor (no code execution). Start with brute force, optimize, dry-run your code, communicate constantly.
:::

### Time allocation per 45-minute round

| Phase | Time | What to Do |
|-------|------|------------|
| Clarify | 2-3 min | Restate the problem, confirm inputs/outputs, ask about edge cases |
| Brute Force | 1 min | State the naive approach and its complexity |
| Optimal Approach | 3-5 min | Explain your optimized solution **before** writing any code |
| Code | 15-20 min | Write clean, well-named code in a plain text editor |
| Dry Run | 3-5 min | Walk through an example step by step -- do this **without** being asked |
| Complexity | 1 min | State time and space complexity |
| Follow-up | Remaining | Handle extensions or optimizations |

### Communication phrases that work

> *"This looks like a [pattern] problem because..."*
>
> *"The brute force would be O(n^2), but we can optimize with..."*
>
> *"Let me verify with a quick dry run..."*
>
> *"The trade-off here is memory vs time -- I'll go with..."*

::: danger MAX 1 HINT
Needing more than 1 hint from your interviewer pushes you toward "Lean Hire" or worse. If you are stuck, talk through your thought process out loud rather than sitting in silence.
:::

---

## Tips from Candidates Who Got Offers

1. **Start with brute force** -- One candidate was explicitly told: *"start brute force before diving into optimal."*

2. **Dry-run your code** -- A candidate received a "Hire" rating despite having a bug, because they caught it during their dry run.

3. **Never go silent** -- Communicate even when stuck: *"I'm considering whether BFS or DFS fits better here because..."*

4. **Write production-ready code** -- The grading criteria is tight. The expectation is production-ready code that handles every corner case.

5. **You cannot run your code** -- Practice writing correct solutions in a plain text editor. Syntax errors in a plain text editor look bad.

::: info YOU GOT THIS
32 problems. 6 tiers. A clear plan. You are more prepared than most candidates who spend months grinding random problems. Focus beats volume. Now go start with [Graphs](./graphs).
:::
