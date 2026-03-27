# Study Schedule

::: info FLEXIBLE TIMELINE
This schedule assumes a **weekend sprint** --- Saturday and Sunday, roughly 8 hours each. If you have more time, spread it out. If you have less, prioritize Tier 1 (Graphs) and Tier 2 (Trees) above everything else.
:::

![Tier Priority — Where to Focus](/sde-coding/sprint/tier_distribution.png)

---

## Saturday --- 8 Hours

**Goal:** Master Graphs (all 8 problems) and start Trees.

| Time | Block | Focus | Problems | Goal |
|------|-------|-------|----------|------|
| 9:00 -- 10:00 | Morning 1 | Graphs: BFS/DFS | #1 Number of Islands, #2 Course Schedule | Master grid BFS and topological sort |
| 10:00 -- 11:00 | Morning 2 | Graphs: Topo Sort | #3 Course Schedule II, #4 Word Ladder | Lock in Kahn's algorithm and BFS shortest path |
| 11:00 -- 12:00 | Morning 3 | Graphs: Advanced | #5 Clone Graph, #6 Network Delay Time | DFS + HashMap cloning, Dijkstra's |
| 12:00 -- 1:00 | **Lunch** | **Break** | --- | Eat, walk, rest your brain |
| 1:00 -- 2:00 | Afternoon 1 | Graphs: Multi-source | #7 Rotting Oranges, #8 Pacific Atlantic | Multi-source BFS, reverse-direction DFS |
| 2:00 -- 3:00 | Afternoon 2 | Trees: Core | #9 Binary Tree Max Path Sum, #10 LCA of BST | Tree DP pattern, BST property exploitation |
| 3:00 -- 4:00 | Afternoon 3 | Trees: LCA + Serialize | #11 LCA of Binary Tree, #12 Serialize/Deserialize BT | Recursive LCA, BFS serialization |
| 4:00 -- 4:30 | **Break** | **Rest** | --- | Step away from the screen |
| 4:30 -- 5:30 | Evening 1 | Trees: Validate + Diameter | #13 Validate BST, #14 Diameter of Binary Tree | Inorder/bounds technique, post-order DFS |
| 5:30 -- 6:30 | Evening 2 | **Review** | Re-solve 2--3 problems you struggled with | Solidify weak spots before moving on |

::: warning END OF SATURDAY CHECKPOINT
By end of Saturday, you should be able to write BFS, DFS, Topological Sort, and Dijkstra's from memory. If you cannot, spend the first 30 minutes of Sunday morning re-doing Graph templates before moving on.
:::

---

## Sunday --- 8 Hours

**Goal:** Cover all remaining categories and run 2 timed mocks.

| Time | Block | Focus | Problems | Goal |
|------|-------|-------|----------|------|
| 9:00 -- 10:00 | Morning 1 | Sliding Window | #15 Longest Substring Without Repeating, #16 Minimum Window Substring | Master the expand/shrink template |
| 10:00 -- 11:00 | Morning 2 | Sliding Window + DP | #17 Sliding Window Maximum, #18 Longest Repeating Char Replacement | Monotonic deque, sliding window variant |
| 11:00 -- 12:00 | Morning 3 | DP: 1D | #19 Coin Change, #20 LIS, #21 House Robber | Lock in 1D DP pattern and transitions |
| 12:00 -- 1:00 | **Lunch** | **Break** | --- | Eat, walk, decompress |
| 1:00 -- 2:00 | Afternoon 1 | DP: 2D + Memo | #22 Unique Paths, #23 Longest Increasing Path in Matrix | Grid DP and DFS + memoization |
| 2:00 -- 3:00 | Afternoon 2 | HashMap + Heap | #24 Group Anagrams, #25 Top K Frequent, #26 Merge K Sorted Lists | Grouping pattern, min-heap usage |
| 3:00 -- 4:00 | Afternoon 3 | Binary Search | #27 Search in Rotated Sorted Array, #28 Find Peak Element, #29 Kth Largest | Binary search edge cases, quickselect |
| 4:00 -- 4:30 | **Break** | **Rest** | --- | Clear your head before strings and mocks |
| 4:30 -- 5:15 | Evening 1 | Strings | #30 Implement Trie, #31 Expressive Words, #32 Word Search | Trie implementation, two-pointer, backtracking |
| 5:15 -- 6:00 | Evening 2 | **Timed Mock 1** | Pick a problem you have not seen before | 45 minutes, plain text editor (no IDE), talk out loud |
| 6:00 -- 6:45 | Evening 3 | **Timed Mock 2** | Pick another unseen problem | Same rules. Practice the full interview flow |
| 6:45 -- 7:30 | Evening 4 | **Review** | Review both mocks, re-read [Templates](./templates) | Identify remaining gaps |

::: tip TIMED MOCK RULES
1. Use a plain text editor (no IDE) --- **no syntax highlighting, no autocomplete, no running code**
2. Set a 45-minute timer
3. Talk out loud as if an interviewer is listening
4. Follow the full flow: clarify, brute force, optimal, code, dry run, complexity
5. After time is up, check your solution against the LeetCode editorial
:::

---

## Day Before the Interview

::: danger PRE-INTERVIEW CHECKLIST
This is NOT a day for new problems. This is a day for consolidation and confidence.
:::

- [ ] **Re-solve 2--3 problems you struggled with** --- especially any where you needed to look at the walkthrough
- [ ] **Review all templates** --- read through the [Templates](./templates) page end to end. Can you write BFS, sliding window, and topo sort from memory?
- [ ] **Do 1 timed mock** --- 45 minutes, plain text editor, talk out loud
- [ ] **Review the [Interview Day](./interview-day) checklist** --- read the minute-by-minute breakdown and communication phrases
- [ ] **Prepare your environment** --- quiet room, stable internet, water bottle, pen and paper
- [ ] **Get 7--8 hours of sleep** --- a rested brain solves problems faster than a brain that crammed until 2 AM

---

## Hourly Approach for Each Problem

Use this flow for every single problem during your prep:

```
[0:00 - 0:02]  READ the problem statement carefully
                - What are the inputs? What are the outputs?
                - What are the constraints (array size, value range)?

[0:02 - 0:04]  BRUTE FORCE
                - What is the simplest approach, even if it is O(n^2) or worse?
                - State it clearly: "Brute force: check every pair, O(n^2)."

[0:04 - 0:09]  IDENTIFY PATTERN
                - Does this match a known template? (BFS, sliding window, DP, etc.)
                - What is the optimal time complexity I should aim for?
                - Plan your approach before writing code.

[0:09 - 0:24]  CODE THE SOLUTION
                - Write clean code with good variable names.
                - Use helper functions if logic is complex.
                - Handle edge cases (empty input, single element, etc.).

[0:24 - 0:29]  DRY RUN + DEBUG
                - Walk through a small example by hand.
                - Check off-by-one errors, boundary conditions.
                - Fix any bugs you find.

[0:29 - 0:30]  COMPLEXITY ANALYSIS
                - State time and space complexity.
```

::: info STUCK AFTER 10 MINUTES?
If you have been staring at a problem for 10 minutes with no progress, **look at the walkthrough on the tier page**. The goal is to learn the pattern, not to suffer. Read the approach, close it, and code the solution yourself from memory.
:::

---

## Progress Tracker

![Problem Difficulty Map](/sde-coding/sprint/problem_difficulty_map.png)

Check off each problem as you complete it. Be honest --- only check it off if you can solve it without looking at the solution.

### Tier 1: Graphs

| # | Problem | LC | Done | Confident? |
|---|---------|-----|------|------------|
| 1 | Number of Islands | 200 | [ ] | [ ] |
| 2 | Course Schedule | 207 | [ ] | [ ] |
| 3 | Course Schedule II | 210 | [ ] | [ ] |
| 4 | Word Ladder | 127 | [ ] | [ ] |
| 5 | Clone Graph | 133 | [ ] | [ ] |
| 6 | Network Delay Time | 743 | [ ] | [ ] |
| 7 | Rotting Oranges | 994 | [ ] | [ ] |
| 8 | Pacific Atlantic Water Flow | 417 | [ ] | [ ] |

### Tier 2: Trees

| # | Problem | LC | Done | Confident? |
|---|---------|-----|------|------------|
| 9 | Binary Tree Max Path Sum | 124 | [ ] | [ ] |
| 10 | Lowest Common Ancestor BST | 235 | [ ] | [ ] |
| 11 | Lowest Common Ancestor BT | 236 | [ ] | [ ] |
| 12 | Serialize/Deserialize BT | 297 | [ ] | [ ] |
| 13 | Validate BST | 98 | [ ] | [ ] |
| 14 | Diameter of Binary Tree | 543 | [ ] | [ ] |

### Tier 3: Sliding Window

| # | Problem | LC | Done | Confident? |
|---|---------|-----|------|------------|
| 15 | Longest Substring Without Repeating | 3 | [ ] | [ ] |
| 16 | Minimum Window Substring | 76 | [ ] | [ ] |
| 17 | Sliding Window Maximum | 239 | [ ] | [ ] |
| 18 | Longest Repeating Char Replacement | 424 | [ ] | [ ] |

### Tier 4: Dynamic Programming

| # | Problem | LC | Done | Confident? |
|---|---------|-----|------|------------|
| 19 | Coin Change | 322 | [ ] | [ ] |
| 20 | Longest Increasing Subsequence | 300 | [ ] | [ ] |
| 21 | House Robber | 198 | [ ] | [ ] |
| 22 | Unique Paths | 62 | [ ] | [ ] |
| 23 | Longest Increasing Path in Matrix | 329 | [ ] | [ ] |

### Tier 5: HashMap / Heap / Binary Search

| # | Problem | LC | Done | Confident? |
|---|---------|-----|------|------------|
| 24 | Group Anagrams | 49 | [ ] | [ ] |
| 25 | Top K Frequent Elements | 347 | [ ] | [ ] |
| 26 | Merge K Sorted Lists | 23 | [ ] | [ ] |
| 27 | Search in Rotated Sorted Array | 33 | [ ] | [ ] |
| 28 | Find Peak Element | 162 | [ ] | [ ] |
| 29 | Kth Largest Element | 215 | [ ] | [ ] |

### Tier 6: Strings

| # | Problem | LC | Done | Confident? |
|---|---------|-----|------|------------|
| 30 | Implement Trie | 208 | [ ] | [ ] |
| 31 | Expressive Words | 809 | [ ] | [ ] |
| 32 | Word Search | 79 | [ ] | [ ] |

---

::: tip PROGRESS MILESTONES
- **8 problems done** (all Graphs): You can handle the most common SDE interview question type.
- **14 problems done** (+ Trees): You have covered the top 2 tiers. Most candidates stop here and still do well.
- **23 problems done** (+ SW + DP): You are better prepared than 90% of candidates.
- **32 problems done** (all): Maximum coverage. Spend remaining time on mocks.
:::
