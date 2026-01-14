# Mock Interview Guide

> **How to conduct effective mock interviews for maximum improvement**

---

## Overview

Mock interviews are the single most effective way to prepare for coding interviews. Research shows that candidates who complete 5+ technical mock interviews are **twice as likely** to pass their real interviews compared to those who only practice solo problem-solving.

This guide covers how to structure, conduct, and learn from mock interviews to maximize your improvement.

**Estimated Reading Time:** 1 hour

---

## Why Mock Interviews Matter

### Skills That Only Mocks Can Build

| Skill | Solo Practice | Mock Interview |
|-------|---------------|----------------|
| Problem-solving under pressure | Partial | Full |
| Thinking out loud | Limited | Essential |
| Handling ambiguous problems | Cannot practice | Core skill |
| Responding to hints gracefully | Not possible | Practiced |
| Time management with observer | Different feel | Realistic |
| Recovery from mistakes | Easy reset | Must handle live |

### The Communication Gap

Many candidates can solve problems on paper but freeze or stumble in interviews. Mocks bridge this gap by forcing you to:

1. **Articulate your thought process** - You cannot stay silent
2. **Explain tradeoffs** - Interviewer asks "why not X?"
3. **Handle interruptions** - Real interviewers ask questions mid-solution
4. **Manage time** - The clock pressure feels different with someone watching

---

## Mock Interview Structure

### Standard Format (45 minutes)

```
Time Breakdown:
0:00 - 0:03  Introduction (brief background)
0:03 - 0:08  Problem reading and clarification
0:08 - 0:15  Approach discussion
0:15 - 0:35  Implementation
0:35 - 0:40  Testing and debugging
0:40 - 0:45  Complexity analysis and questions
```

### Problem Selection Guidelines

| Mock # | Difficulty | Topic Selection |
|--------|------------|-----------------|
| 1 | 1 Medium | Familiar pattern (build confidence) |
| 2 | 1 Medium | Moderate difficulty |
| 3 | 2 Medium | Mixed topics, unknown to candidate |
| 4+ | 1 Medium + follow-up | Interview simulation |

### Recommended Mock Schedule

For a 2-week preparation:

| Day | Mock Type | Notes |
|-----|-----------|-------|
| Day 7 | Self-mock (recorded) | Review your own performance |
| Day 10 | Peer mock | Get external feedback |
| Day 13 | Self-mock (harder) | Build endurance |
| Day 14 | Peer mock (final) | Simulate real conditions |

---

## Types of Mock Interviews

### 1. Self-Mock (Solo Practice)

**How to conduct:**
1. Set a timer for 45 minutes
2. Select a problem you have NOT seen before
3. Record yourself (video or audio)
4. Pretend you are explaining to an interviewer
5. Review the recording afterward

**Benefits:**
- Available anytime
- No scheduling needed
- Can practice repeatedly

**Limitations:**
- No real-time feedback
- Cannot practice hint handling
- Easy to "cheat" on time

**Self-Mock Checklist:**
```
[ ] Selected unseen problem
[ ] Set up recording device
[ ] Removed all distractions
[ ] Timer set for 45 minutes
[ ] Spoke thoughts aloud throughout
[ ] Reviewed recording afterward
```

### 2. Peer Mock (With Partner)

**How to find partners:**
- Current colleagues preparing for interviews
- Online communities (Discord, Reddit r/cscareerquestions)
- Pramp (free peer matching)
- Former classmates

**How to conduct:**
1. One person is interviewer, one is candidate
2. Interviewer selects problem unknown to candidate
3. Interviewer stays silent initially, provides hints if needed
4. Switch roles for the next session
5. Provide detailed feedback to each other

**Interviewer Guidelines:**
```
DO:
- Stay silent during initial problem-solving
- Ask clarifying questions if approach is unclear
- Provide gentle hints if stuck (but wait 3-5 minutes first)
- Ask "what's the time complexity?" at the end
- Take notes for feedback

DON'T:
- Give away the solution
- Make discouraging facial expressions
- Interrupt unnecessarily
- Rush the candidate
- Be overly critical in feedback
```

### 3. Professional Mock

**Platforms:**
- Pramp (free, peer-based)
- Interviewing.io (paid, with experienced engineers)
- Prepfully (paid, with FAANG engineers)
- LeetCode Premium mock interviews

**When to use:**
- Final week before interview
- After multiple peer mocks
- When you need expert-level feedback

---

## The Mock Interview Rubric

Use this rubric to evaluate (and be evaluated on) mock interview performance.

### Evaluation Categories

#### 1. Problem Understanding (20 points)

| Score | Description |
|-------|-------------|
| 20 | Asked excellent clarifying questions, identified all edge cases |
| 15 | Good clarifications, caught most edge cases |
| 10 | Basic understanding, missed some edge cases |
| 5 | Rushed into coding, needed hints to understand |
| 0 | Did not understand the problem |

**Questions to ask:**
- What is the input size/range?
- Are there duplicates? How to handle them?
- What about empty input?
- Can values be negative?
- What is the expected output format?

#### 2. Approach Communication (20 points)

| Score | Description |
|-------|-------------|
| 20 | Clearly explained approach before coding, discussed tradeoffs |
| 15 | Good explanation of approach, some gaps |
| 10 | Explained while coding, not before |
| 5 | Minimal explanation, mostly silent |
| 0 | No communication of approach |

**What to communicate:**
- "I'm thinking of using [pattern] because..."
- "The brute force would be O(n^2), but we can optimize..."
- "Let me walk through an example first..."

#### 3. Code Quality (20 points)

| Score | Description |
|-------|-------------|
| 20 | Clean, readable code with good naming and structure |
| 15 | Good code, minor improvements possible |
| 10 | Working code but messy or hard to follow |
| 5 | Code has bugs or incomplete |
| 0 | Did not produce working code |

**Code quality checklist:**
- Meaningful variable names
- Proper indentation
- Modular functions when appropriate
- Comments for non-obvious logic

#### 4. Testing (20 points)

| Score | Description |
|-------|-------------|
| 20 | Systematically tested with examples and edge cases |
| 15 | Tested with provided example and one edge case |
| 10 | Basic testing, missed edge cases |
| 5 | Did not test, or testing was incorrect |
| 0 | No testing |

**Testing checklist:**
- Trace through with given example
- Empty input
- Single element
- Maximum size input (discuss, don't trace)
- Edge cases specific to problem

#### 5. Problem Solving & Debugging (20 points)

| Score | Description |
|-------|-------------|
| 20 | Solved independently, handled bugs gracefully |
| 15 | Solved with minimal hints, good debugging |
| 10 | Needed moderate hints, some struggle with bugs |
| 5 | Required significant hints, poor debugging |
| 0 | Could not solve even with hints |

**Total: 100 points**

### Score Interpretation

| Score | Interpretation | Action |
|-------|----------------|--------|
| 85-100 | Interview ready | Maintain confidence |
| 70-84 | Strong foundation | Minor refinements needed |
| 55-69 | Getting there | Focus on weak areas |
| 40-54 | More practice needed | Review patterns |
| <40 | Significant gaps | Return to fundamentals |

---

## Feedback Framework

### Giving Feedback (As Interviewer)

Use the "Sandwich" method:
1. **Start positive** - What did they do well?
2. **Constructive criticism** - What needs improvement?
3. **End encouraging** - Overall assessment and encouragement

**Feedback Template:**
```
Strengths:
- [What they did well]
- [Another strength]

Areas for Improvement:
- [Specific issue + how to fix it]
- [Another issue + suggestion]

Overall:
[Summary and encouragement]

Score: ___/100
```

**Example Feedback:**
```
Strengths:
- Great job clarifying the problem upfront
- Clean code with good variable names
- Correctly identified the two-pointer pattern

Areas for Improvement:
- Explain your approach BEFORE coding. You started
  coding within 2 minutes of reading the problem.
- Test edge cases more systematically. You missed
  the empty array case.
- When debugging, trace through your code step by
  step rather than staring at it.

Overall:
Solid problem-solving skills. Focus on slowing down
at the start to explain your approach. This will
also help you catch edge cases earlier.

Score: 72/100
```

### Receiving Feedback (As Candidate)

```
[ ] Listen without defending
[ ] Take notes on specific suggestions
[ ] Ask clarifying questions: "How would you have handled X?"
[ ] Thank the interviewer for their time
[ ] Review notes within 24 hours
[ ] Create action items for next mock
```

---

## Common Mock Interview Problems

### Warm-Up Problems (First Mock)

| Problem | Difficulty | Why It's Good for First Mock |
|---------|------------|------------------------------|
| Two Sum | Easy | Builds confidence, hash map pattern |
| Valid Parentheses | Easy | Stack pattern, clear constraints |
| Merge Two Sorted Lists | Easy | Pointer manipulation practice |

### Standard Mock Problems

| Problem | Difficulty | Patterns Tested |
|---------|------------|-----------------|
| 3Sum | Medium | Sorting, two pointers, duplicates |
| LRU Cache | Medium | Hash map, linked list, design |
| Word Search | Medium | Backtracking, grid traversal |
| Course Schedule | Medium | Graph, topological sort |
| Coin Change | Medium | Dynamic programming |

### Challenge Problems (Final Mocks)

| Problem | Difficulty | Why It's Challenging |
|---------|------------|----------------------|
| Merge K Sorted Lists | Hard | Heap, optimization thinking |
| Word Ladder | Hard | BFS, string manipulation |
| Serialize Binary Tree | Hard | Tree, multiple approaches |

---

## Post-Mock Analysis

### Self-Reflection Questions

After each mock, answer these:

```
1. What went well?
   -

2. What did I struggle with?
   -

3. At what point did I feel stuck? Why?
   -

4. Did I communicate my thoughts clearly?
   -

5. What would I do differently next time?
   -
```

### Performance Tracking

| Mock # | Date | Problem(s) | Score | Key Takeaways |
|--------|------|------------|-------|---------------|
| 1 | | | /100 | |
| 2 | | | /100 | |
| 3 | | | /100 | |
| 4 | | | /100 | |
| 5 | | | /100 | |

### Improvement Tracking

Track your scores across categories:

```
Mock 1: Understanding __/20, Approach __/20, Code __/20, Test __/20, Solve __/20
Mock 2: Understanding __/20, Approach __/20, Code __/20, Test __/20, Solve __/20
Mock 3: Understanding __/20, Approach __/20, Code __/20, Test __/20, Solve __/20
Mock 4: Understanding __/20, Approach __/20, Code __/20, Test __/20, Solve __/20
```

---

## Mock Interview Tips

### Before the Mock

```
[ ] Get good sleep the night before
[ ] Have water nearby
[ ] Ensure stable internet (for virtual mocks)
[ ] Test audio/video equipment
[ ] Have scratch paper ready
[ ] Clear your environment of distractions
[ ] Review patterns briefly (10 min)
```

### During the Mock

**First 5 Minutes:**
- Read problem carefully
- Ask clarifying questions
- Think about edge cases
- DO NOT start coding yet

**Minutes 5-15:**
- Explain your approach verbally
- Discuss time/space complexity
- Get confirmation before coding
- Consider alternative approaches

**Minutes 15-35:**
- Write clean code
- Think aloud while coding
- Handle bugs calmly
- Don't panic if stuck

**Last 10 Minutes:**
- Test with examples
- Check edge cases
- Discuss optimizations if time permits
- Ask thoughtful questions

### If You Get Stuck

**Wait 2-3 minutes before panicking:**
1. Re-read the problem
2. Think about simpler version of the problem
3. Consider brute force first
4. Ask for a clarifying hint (not solution)

**Phrases to use:**
- "Let me think about this for a moment..."
- "I'm considering approach X because..."
- "Could you give me a hint about the direction?"
- "I think I'm overcomplicating this. Let me simplify."

### Handling Hints Gracefully

When the interviewer gives a hint:
1. Say "Thank you, that's helpful"
2. Pause to process the hint
3. Explain how the hint changes your approach
4. Proceed with the new direction

**DO NOT:**
- Ignore the hint
- Say "I was about to do that" (even if true)
- Get defensive or frustrated

---

## Virtual Mock Interview Setup

### Technical Setup

```
Environment:
[ ] Quiet room with good lighting
[ ] Stable internet connection
[ ] Backup plan if connection fails
[ ] Screen sharing enabled
[ ] Coding environment ready (CoderPad, etc.)

Equipment:
[ ] Camera working (optional but recommended)
[ ] Microphone clear
[ ] Headphones to reduce echo
```

### Coding Environment Options

| Platform | Best For | Notes |
|----------|----------|-------|
| CoderPad | Professional feel | Industry standard |
| Google Docs | Simplicity | Works for any interview |
| LeetCode | Practice | Familiar interface |
| VS Code Live Share | Collaboration | Rich IDE features |

### Virtual Communication Tips

- Look at the camera occasionally (simulates eye contact)
- Speak clearly and at moderate pace
- Ask "Can you hear me okay?" at start
- Share your screen confidently
- Use a pointer/cursor to highlight code when explaining

---

## Sample Mock Interview Script

### For Interviewer

**Introduction (2 min):**
> "Hi, thanks for joining. Today we'll do a 45-minute coding session. I'll present a problem, you'll work through it while thinking aloud, and I'll ask some follow-up questions. Feel free to ask clarifying questions. Ready?"

**Problem Presentation:**
> "Here's the problem: [Read problem statement]. Take a moment to read through it."

**During Problem-Solving:**
- Stay mostly silent
- If candidate is stuck (5+ min): "What are you thinking right now?"
- If still stuck (10+ min): Give a gentle hint
- Note down observations for feedback

**Wrap-up (5 min):**
> "Good work. Before we wrap up, what's the time and space complexity of your solution?"
> "Is there anything you would optimize if you had more time?"
> "Do you have any questions for me?"

**Feedback (5-10 min after):**
> "Let me share some feedback..."

### For Candidate

**Introduction:**
> "Hi, thanks for doing this mock. I'm preparing for Google interviews and appreciate your time."

**Receiving Problem:**
> "Let me read through this carefully..."
> "Before I start, can I ask some clarifying questions?"

**Explaining Approach:**
> "So my understanding is [restate problem]. I'm thinking of using [pattern] because [reason]. The basic idea is..."

**During Implementation:**
> "I'm starting with [function name] which will..."
> "Here I'm checking for [edge case] because..."

**Testing:**
> "Let me trace through with this example..."
> "Now let me check edge cases: empty input, single element..."

**Wrap-up:**
> "The time complexity is O(n) because... Space is O(1) because..."

---

## Mock Interview Resources

### Free Platforms
- **Pramp** - Peer-to-peer mock interviews
- **LeetCode Discuss** - Find mock partners
- **Discord communities** - CS Career Hub, Blind 75

### Paid Platforms
- **Interviewing.io** - Anonymous mocks with engineers
- **Prepfully** - FAANG engineer mocks
- **LeetCode Premium** - Mock interview feature

### Finding Mock Partners

**Post template for finding partners:**
```
Looking for mock interview partner

Experience: [X years]
Target: Google SDE [Level]
Availability: [Days/Times with timezone]
Prep stage: [Week X of 2-week plan]
Languages: [Python/Java/etc.]

DM if interested in trading mocks!
```

---

## Mock Interview Checklist Summary

### Before
```
[ ] Scheduled with partner
[ ] Environment prepared
[ ] Technical setup tested
[ ] Problem selected (for interviewer)
[ ] Mentally prepared
```

### During
```
[ ] Clarified problem
[ ] Explained approach before coding
[ ] Thought aloud throughout
[ ] Handled hints gracefully
[ ] Tested systematically
```

### After
```
[ ] Received feedback
[ ] Recorded key takeaways
[ ] Updated performance tracker
[ ] Created action items
[ ] Scheduled next mock
```

---

*Complete at least 3 mock interviews before your real interview. Review [Common Mistakes](./common-mistakes.md) to learn from others' experiences.*

---

*Last updated: January 2026*
