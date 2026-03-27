# Interview Day --- Your Checklist

Read this page on the morning of your interview. Everything here is designed to maximize your signal in those 45 minutes.

::: danger REMEMBER
You are in a **plain text editor (no IDE)**. No autocomplete. No syntax highlighting. No running code. No debugger. Every character you write must be correct from memory. If you have not practiced writing code in a plain text editor, do one warm-up problem right now before your round.
:::

---

## Before the Interview

Complete this checklist **1--2 hours before** your first round.

- [ ] **Environment ready** --- quiet room, door closed, stable internet, headphones with mic tested
- [ ] **Water and snacks** --- on your desk, within reach
- [ ] **Pen and paper** --- for sketching diagrams, tracing through examples, jotting down edge cases
- [ ] **Review the [Templates](./templates) page** --- read through BFS, sliding window, topo sort, and Dijkstra's one more time
- [ ] **Do 1 warm-up problem** --- solve Number of Islands (LC 200) or Two Sum (LC 1) in a plain text editor. This wakes up your problem-solving brain.
- [ ] **Plain text editor open** --- have a blank document ready. Practice typing a quick function to make sure your formatting looks clean.
- [ ] **Phone on silent** --- no distractions during your round

::: info BETWEEN ROUNDS
If you have multiple rounds, take a 5--10 minute break between them. Stand up, stretch, drink water. Do NOT review what happened in the previous round --- your brain needs a reset, not a replay.
:::

---

## The 45-Minute Round --- Minute by Minute

This is your playbook. Every minute has a purpose.

![Your 45-Minute Game Plan](/sde-coding/sprint/interview_timeline.png)

### Minutes 0--3: Clarify

| Do | Say |
|----|-----|
| Restate the problem in your own words | *"So we're given [input] and need to return [output], correct?"* |
| Confirm input types and ranges | *"Are the values always positive integers? Can the array be empty?"* |
| Ask about edge cases | *"What should I return if the input is empty? What about duplicates?"* |
| Confirm expected output format | *"Should I return the indices or the values themselves?"* |

::: warning DO NOT SKIP CLARIFICATION
Even if the problem seems obvious, ask at least one clarifying question. It shows thoroughness and buys you thinking time. Interviewers expect this.
:::

### Minutes 3--4: Brute Force

State the naive approach and its complexity. Do not code it --- just describe it.

> *"The brute force would be to check every pair, which gives us O(n^2) time and O(1) space. We can do better."*

This shows you understand the problem and sets up the optimization.

### Minutes 4--8: Explain Your Optimal Approach

![Which Pattern Should I Use?](/sde-coding/sprint/pattern_flowchart.png)

**Before writing a single line of code**, explain your plan.

> *"I'll use a BFS approach because we need the shortest path in an unweighted graph. I'll use a queue and a visited set. For each node, I'll explore all neighbors and track the distance. Time is O(V + E), space is O(V)."*

Your interviewer wants to hear:
1. **What** data structure/algorithm you will use
2. **Why** it is the right choice for this problem
3. **What** the time and space complexity will be

::: tip GET A NOD BEFORE CODING
Wait for the interviewer to acknowledge your approach. If they say *"sounds good"* or *"go ahead"*, you are on the right track. If they hesitate or ask a question, they may be hinting that you should reconsider.
:::

### Minutes 8--28: Code

This is where you write your solution. 20 minutes of focused coding.

**Rules for clean code in a plain text editor:**

1. **Function signature first** --- write `def solve(grid):` and `return result` before filling in the body
2. **Descriptive variable names** --- `queue`, `visited`, `left`, `right`, `result`, not `q`, `v`, `l`, `r`, `res`
3. **Helper functions** --- if a block of logic is reusable or complex, extract it
4. **4-space indentation** --- be consistent, plain text editors do not auto-indent
5. **Handle edge cases early** --- `if not grid: return 0` at the top
6. **No unnecessary comments** --- your variable names should make the code self-documenting. Comment only tricky logic.

### Minutes 28--33: Dry Run

Walk through your code with a concrete example. **Do this without being asked.**

> *"Let me trace through the example. Input: grid = [[1,1,0],[1,1,0],[0,0,1]]. Starting at (0,0), BFS visits (0,1), (1,0), (1,1) --- that's island 1. Then (2,2) --- that's island 2. Result: 2. Matches expected output."*

This is where candidates who get "Hire" separate themselves from candidates who get "Lean Hire."

### Minutes 33--35: Complexity

State both time and space complexity clearly.

> *"Time complexity is O(m times n) because we visit each cell at most once. Space complexity is O(m times n) in the worst case for the visited set and the BFS queue."*

### Minutes 35--45: Follow-ups

The interviewer may:
- Ask you to optimize further
- Change a constraint (e.g., *"what if the grid is infinite?"*)
- Ask how you would handle a different input format
- Ask you to code a variation

Stay calm. Apply the same flow: clarify, think, explain, then code.

---

## Communication Phrases

Keep these in your back pocket. They signal structure and confidence.

### Opening a Problem

> *"This looks like a [BFS / sliding window / DP] problem because [reason]."*

> *"My first instinct is [approach], but let me think about whether [alternative] would be better."*

### Transitioning from Brute Force to Optimal

> *"The brute force would be O(n^2), but we can optimize with [technique] to bring it down to O(n log n) / O(n)."*

> *"The key insight is that [observation], which lets us use [data structure / algorithm]."*

### While Coding

> *"I'm using a [deque / heap / hashmap] here because [reason]."*

> *"This helper function handles [specific responsibility]."*

### Dry Running

> *"Let me verify with a quick dry run on the given example."*

> *"At this point, the queue contains [values], and visited is [values]..."*

### Analyzing Complexity

> *"The trade-off here is memory vs time --- I'll go with [choice] because [reason]."*

> *"Time is O(n) because we visit each element once. Space is O(k) for the window."*

### When You Are Stuck

> *"I'm considering whether [approach A] or [approach B] fits better here because..."*

> *"Let me think about what invariant I need to maintain..."*

> *"I know I need to [goal]. The question is whether I should use [X] or [Y] to achieve it."*

::: danger NEVER GO SILENT
Silence is the worst thing in a coding interview. If you are stuck, **talk through your thought process**. Even if you are not making progress, the interviewer gets signal from hearing how you think. Many interviewers will give a nudge if they see you are close.
:::

---

## Tips from Candidates Who Got Offers

::: tip TIP 1: START WITH BRUTE FORCE
One candidate was explicitly told by their interviewer: *"Start with brute force before diving into optimal."* Interviewers want to see that you understand the problem before you optimize. A correct brute force is infinitely better than an incorrect optimal solution.
:::

::: tip TIP 2: DRY RUN YOUR CODE
A candidate received a **"Hire"** rating despite having a bug in their code because they **caught it during their dry run** and fixed it. The interviewer noted this as a strong positive signal for debugging ability. Always dry run, even if your code looks correct.
:::

::: tip TIP 3: NEVER TAKE LONG PAUSES
Silence kills your interview score. If you are stuck, say something:
- *"I'm thinking about the data structure choice here..."*
- *"Let me reconsider the approach..."*
- *"I'm not sure about this edge case, let me think through it..."*

Your interviewer cannot read your mind. Thinking out loud lets them give you credit for your thought process and offer hints when you are close.
:::

::: tip TIP 4: ACCEPT MAX 1 HINT
Needing **more than one hint** from your interviewer pushes you toward **"Lean Hire"** or **"No Hire"**. If you take a hint, acknowledge it cleanly and build on it immediately:
- *"Ah, that makes sense --- so I should use [approach] because [reason]. Let me code that up."*

If you feel stuck and have not received a hint, try restating the problem constraints. Often the hint is embedded in the constraints.
:::

::: tip TIP 5: WRITE PRODUCTION-READY CODE
Direct quote from an interviewer's feedback: *"Grading criteria is super tight so expectation is production-ready code with handling every corner case."* This means:
- Handle empty inputs
- Handle single-element inputs
- Handle duplicate values
- Use proper error handling where appropriate
- Name variables clearly
- Structure your code as if it were going into a real codebase
:::

::: tip TIP 6: PRACTICE WITHOUT CODE EXECUTION
You **cannot run your code** during the interview. Every syntax error, off-by-one mistake, and missing edge case will be visible to your interviewer. Practice writing solutions in a plain text editor (no IDE) **at least 3--5 times** before your interview. The muscle memory of writing correct code without a compiler is a real skill that takes practice.
:::

---

## What Interviewers Are Looking For

Interviewers use a structured rubric. Here is what maps to each signal.

| Signal | What They Want to See | Red Flag |
|--------|----------------------|----------|
| **Problem Solving** | Identifies the right pattern quickly. Breaks the problem into subproblems. Considers multiple approaches before committing. | Jumps into coding without a plan. Cannot identify the pattern even with hints. Gets stuck on brute force. |
| **Coding Ability** | Clean, correct, production-ready code. Good variable names, helper functions, edge case handling. | Sloppy code with single-letter variables. Syntax errors. Does not handle edge cases. Code does not compile mentally. |
| **Communication** | Explains approach before coding. Thinks out loud. States complexity proactively. Walks through dry run. | Long silences. Cannot explain their own code. Only speaks when asked a direct question. |
| **Testing & Verification** | Dry runs through examples without being asked. Identifies edge cases. Catches own bugs. | Never tests their code. Cannot trace through their own solution. Misses obvious edge cases. |
| **Optimization** | Identifies time/space trade-offs. Can improve from brute force to optimal. Understands why the optimization works. | Cannot analyze complexity. Provides only brute force. Does not understand the improvement. |

::: info THE RATING SCALE
Many companies use: **Strong Hire > Hire > Lean Hire > Lean No Hire > No Hire**. You typically need at least **Hire** on both coding rounds and no **No Hire** on any round. A single **Lean Hire** is recoverable if your other rounds are strong.
:::

---

## After the Interview

### Immediately After

- **Do not dwell on mistakes.** Every candidate walks out thinking they bombed something. This is normal.
- **Write down what you remember:**
  - What problems were asked?
  - What approach did you use?
  - Where did you struggle?
  - Did you get any verbal feedback?
- **This helps you prep for the next round** if there is one, and helps you debrief later.

### While Waiting for Results

- **The hiring committee sees everything** --- all your rounds, your resume, your references. A rough round does not automatically mean rejection.
- **Candidates often get "Lean Hire" on one round and still receive offers** --- the committee weighs the full picture.
- **If you do not get an offer**, you can typically re-interview after 6--12 months. Use the notes you wrote down to target your weak areas.

::: tip FINAL THOUGHT
You prepared. You have a plan. You know the patterns, the templates, and the communication flow. Trust your preparation, stay calm, and solve the problem one step at a time. Good luck.
:::
