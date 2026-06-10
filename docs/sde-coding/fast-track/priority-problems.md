# Priority Problems: 50 Must-Solve Problems

> **The essential problem set for rapid interview preparation**

---

## Overview

This curated list of 50 problems represents the highest-value problems for SDE interview preparation. Each problem was selected based on:

- **Frequency**: How often similar problems appear in technical interviews
- **Pattern Coverage**: Ensures all essential patterns are practiced
- **Skill Building**: Problems build on each other progressively
- **Time Efficiency**: Maximum learning per hour invested

**Target Completion Time:** 25-30 hours total (30-40 minutes per problem)

---

## How to Use This List

### Recommended Approach

1. **Solve in order** - Problems are arranged by difficulty and dependency
2. **Time yourself** - Set a 35-40 minute limit per problem
3. **Review solutions** - After solving, study the optimal approach
4. **Document patterns** - Note which pattern each problem uses
5. **Revisit failures** - Re-solve any problem you could not complete

### Progress Tracking Legend

```
[ ] Not started
[~] Attempted but incomplete
[x] Solved independently
[!] Solved with hints
[*] Mastered (can solve in <20 min)
```

### Difficulty Ratings

| Rating | Description | Time Target |
|--------|-------------|-------------|
| E | Easy - Warm up | 15-20 min |
| M | Medium - Interview standard | 25-35 min |
| H | Hard - Stretch goal | 35-45 min |

---

## Week 1 Problems (1-25)

### Arrays and Strings (1-8)

| # | Problem | Difficulty | Pattern | LeetCode | Status |
|---|---------|------------|---------|----------|--------|
| 1 | Two Sum | E | Hash Map | [#1](https://leetcode.com/problems/two-sum/) | [ ] |
| 2 | Best Time to Buy and Sell Stock | E | Single Pass | [#121](https://leetcode.com/problems/best-time-to-buy-and-sell-stock/) | [ ] |
| 3 | Contains Duplicate | E | Hash Set | [#217](https://leetcode.com/problems/contains-duplicate/) | [ ] |
| 4 | Maximum Subarray | M | Kadane's | [#53](https://leetcode.com/problems/maximum-subarray/) | [ ] |
| 5 | Product of Array Except Self | M | Prefix/Suffix | [#238](https://leetcode.com/problems/product-of-array-except-self/) | [ ] |
| 6 | 3Sum | M | Two Pointers | [#15](https://leetcode.com/problems/3sum/) | [ ] |
| 7 | Container With Most Water | M | Two Pointers | [#11](https://leetcode.com/problems/container-with-most-water/) | [ ] |
| 8 | Longest Substring Without Repeating | M | Sliding Window | [#3](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | [ ] |

**Key Learning Goals:**
- Master two-pointer technique on sorted/unsorted arrays
- Understand sliding window for substring problems
- Recognize when to use hash maps vs sets

### Hash Tables (9-12)

| # | Problem | Difficulty | Pattern | LeetCode | Status |
|---|---------|------------|---------|----------|--------|
| 9 | Valid Anagram | E | Frequency Count | [#242](https://leetcode.com/problems/valid-anagram/) | [ ] |
| 10 | Group Anagrams | M | Hash + Sort | [#49](https://leetcode.com/problems/group-anagrams/) | [ ] |
| 11 | Top K Frequent Elements | M | Frequency + Heap | [#347](https://leetcode.com/problems/top-k-frequent-elements/) | [ ] |
| 12 | Longest Consecutive Sequence | M | Hash Set | [#128](https://leetcode.com/problems/longest-consecutive-sequence/) | [ ] |

**Key Learning Goals:**
- Use frequency maps for counting problems
- Combine hash tables with sorting/heaps
- Recognize O(n) solutions using hash sets

### Linked Lists (13-17)

| # | Problem | Difficulty | Pattern | LeetCode | Status |
|---|---------|------------|---------|----------|--------|
| 13 | Reverse Linked List | E | Pointer Manipulation | [#206](https://leetcode.com/problems/reverse-linked-list/) | [ ] |
| 14 | Merge Two Sorted Lists | E | Two Pointers | [#21](https://leetcode.com/problems/merge-two-sorted-lists/) | [ ] |
| 15 | Linked List Cycle | E | Fast/Slow Pointers | [#141](https://leetcode.com/problems/linked-list-cycle/) | [ ] |
| 16 | Remove Nth Node From End | M | Two Pointers | [#19](https://leetcode.com/problems/remove-nth-node-from-end-of-list/) | [ ] |
| 17 | Reorder List | M | Multiple Techniques | [#143](https://leetcode.com/problems/reorder-list/) | [ ] |

**Key Learning Goals:**
- Master in-place reversal
- Apply Floyd's cycle detection
- Combine multiple techniques (find middle + reverse + merge)

### Stacks and Queues (18-21)

| # | Problem | Difficulty | Pattern | LeetCode | Status |
|---|---------|------------|---------|----------|--------|
| 18 | Valid Parentheses | E | Stack Matching | [#20](https://leetcode.com/problems/valid-parentheses/) | [ ] |
| 19 | Min Stack | M | Auxiliary Stack | [#155](https://leetcode.com/problems/min-stack/) | [ ] |
| 20 | Daily Temperatures | M | Monotonic Stack | [#739](https://leetcode.com/problems/daily-temperatures/) | [ ] |
| 21 | Largest Rectangle in Histogram | H | Monotonic Stack | [#84](https://leetcode.com/problems/largest-rectangle-in-histogram/) | [ ] |

**Key Learning Goals:**
- Recognize monotonic stack patterns
- Track auxiliary information in stacks
- Handle "next greater/smaller" problems

### Binary Trees (22-25)

| # | Problem | Difficulty | Pattern | LeetCode | Status |
|---|---------|------------|---------|----------|--------|
| 22 | Invert Binary Tree | E | Recursive DFS | [#226](https://leetcode.com/problems/invert-binary-tree/) | [ ] |
| 23 | Maximum Depth of Binary Tree | E | DFS/BFS | [#104](https://leetcode.com/problems/maximum-depth-of-binary-tree/) | [ ] |
| 24 | Binary Tree Level Order Traversal | M | BFS | [#102](https://leetcode.com/problems/binary-tree-level-order-traversal/) | [ ] |
| 25 | Validate Binary Search Tree | M | Inorder/Range | [#98](https://leetcode.com/problems/validate-binary-search-tree/) | [ ] |

**Key Learning Goals:**
- Master recursive tree traversals
- Implement BFS for level-order
- Validate tree properties with constraints

---

## Week 2 Problems (26-50)

### Trees - Advanced (26-30)

| # | Problem | Difficulty | Pattern | LeetCode | Status |
|---|---------|------------|---------|----------|--------|
| 26 | Lowest Common Ancestor of BST | M | BST Properties | [#235](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) | [ ] |
| 27 | Lowest Common Ancestor of BT | M | Recursive DFS | [#236](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) | [ ] |
| 28 | Construct Binary Tree from Preorder and Inorder | M | Divide and Conquer | [#105](https://leetcode.com/problems/construct-binary-tree-from-preorder-and-inorder-traversal/) | [ ] |
| 29 | Serialize and Deserialize Binary Tree | H | BFS/DFS | [#297](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/) | [ ] |
| 30 | Binary Tree Maximum Path Sum | H | DFS + Global Max | [#124](https://leetcode.com/problems/binary-tree-maximum-path-sum/) | [ ] |

**Key Learning Goals:**
- Handle complex tree recursion
- Build trees from traversals
- Track global state during recursion

### Graphs (31-36)

| # | Problem | Difficulty | Pattern | LeetCode | Status |
|---|---------|------------|---------|----------|--------|
| 31 | Number of Islands | M | DFS/BFS Grid | [#200](https://leetcode.com/problems/number-of-islands/) | [ ] |
| 32 | Clone Graph | M | BFS + Hash Map | [#133](https://leetcode.com/problems/clone-graph/) | [ ] |
| 33 | Pacific Atlantic Water Flow | M | Multi-source BFS | [#417](https://leetcode.com/problems/pacific-atlantic-water-flow/) | [ ] |
| 34 | Course Schedule | M | Topological Sort | [#207](https://leetcode.com/problems/course-schedule/) | [ ] |
| 35 | Course Schedule II | M | Topological Sort | [#210](https://leetcode.com/problems/course-schedule-ii/) | [ ] |
| 36 | Graph Valid Tree | M | Union-Find | [#261](https://leetcode.com/problems/graph-valid-tree/) | [ ] |

**Key Learning Goals:**
- Apply BFS/DFS to grids and graphs
- Detect cycles using multiple methods
- Implement Union-Find structure

### Binary Search (37-39)

| # | Problem | Difficulty | Pattern | LeetCode | Status |
|---|---------|------------|---------|----------|--------|
| 37 | Search in Rotated Sorted Array | M | Modified Binary Search | [#33](https://leetcode.com/problems/search-in-rotated-sorted-array/) | [ ] |
| 38 | Find Minimum in Rotated Sorted Array | M | Binary Search | [#153](https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/) | [ ] |
| 39 | Koko Eating Bananas | M | Binary Search on Answer | [#875](https://leetcode.com/problems/koko-eating-bananas/) | [ ] |

**Key Learning Goals:**
- Handle rotated arrays
- Apply binary search to answer spaces
- Recognize monotonic properties

### Backtracking (40-42)

| # | Problem | Difficulty | Pattern | LeetCode | Status |
|---|---------|------------|---------|----------|--------|
| 40 | Subsets | M | Backtracking | [#78](https://leetcode.com/problems/subsets/) | [ ] |
| 41 | Permutations | M | Backtracking | [#46](https://leetcode.com/problems/permutations/) | [ ] |
| 42 | Combination Sum | M | Backtracking | [#39](https://leetcode.com/problems/combination-sum/) | [ ] |

**Key Learning Goals:**
- Master the backtracking template
- Handle duplicates and pruning
- Understand time complexity of exhaustive search

### Dynamic Programming (43-50)

| # | Problem | Difficulty | Pattern | LeetCode | Status |
|---|---------|------------|---------|----------|--------|
| 43 | Climbing Stairs | E | Fibonacci DP | [#70](https://leetcode.com/problems/climbing-stairs/) | [ ] |
| 44 | House Robber | M | 1D DP | [#198](https://leetcode.com/problems/house-robber/) | [ ] |
| 45 | Coin Change | M | Unbounded Knapsack | [#322](https://leetcode.com/problems/coin-change/) | [ ] |
| 46 | Longest Increasing Subsequence | M | LIS Pattern | [#300](https://leetcode.com/problems/longest-increasing-subsequence/) | [ ] |
| 47 | Word Break | M | String DP | [#139](https://leetcode.com/problems/word-break/) | [ ] |
| 48 | Unique Paths | M | Grid DP | [#62](https://leetcode.com/problems/unique-paths/) | [ ] |
| 49 | Longest Common Subsequence | M | 2D DP | [#1143](https://leetcode.com/problems/longest-common-subsequence/) | [ ] |
| 50 | Edit Distance | M | 2D DP | [#72](https://leetcode.com/problems/edit-distance/) | [ ] |

**Key Learning Goals:**
- Define DP states clearly
- Implement 1D and 2D DP tables
- Optimize space when possible

---

## Problem-Pattern Mapping

Quick reference for which pattern each problem uses:

### By Pattern

| Pattern | Problem Numbers |
|---------|----------------|
| **Two Pointers** | 6, 7, 14, 16 |
| **Sliding Window** | 8 |
| **Fast/Slow Pointers** | 15 |
| **Hash Map/Set** | 1, 3, 9, 10, 11, 12 |
| **Stack** | 18, 19, 20, 21 |
| **Binary Search** | 37, 38, 39 |
| **Tree DFS** | 22, 23, 25, 26, 27, 28, 30 |
| **Tree BFS** | 24, 29 |
| **Graph DFS/BFS** | 31, 32, 33 |
| **Topological Sort** | 34, 35 |
| **Union-Find** | 36 |
| **Backtracking** | 40, 41, 42 |
| **DP (1D)** | 43, 44, 45, 46, 47 |
| **DP (2D)** | 48, 49, 50 |

### By Difficulty

| Difficulty | Problems |
|------------|----------|
| **Easy (11)** | 1, 2, 3, 9, 13, 14, 15, 18, 22, 23, 43 |
| **Medium (36)** | 4, 5, 6, 7, 8, 10, 11, 12, 16, 17, 19, 20, 24, 25, 26, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 44, 45, 46, 47, 48, 49, 50 |
| **Hard (3)** | 21, 29, 30 |

---

## Daily Practice Schedule

### If You Have 2 Weeks

| Day | Problems | Focus Area |
|-----|----------|------------|
| Day 1 | 1-6 | Arrays and Two Pointers |
| Day 2 | 7-12 | Sliding Window and Hash |
| Day 3 | 13-17 | Linked Lists |
| Day 4 | 18-21 | Stacks |
| Day 5 | 22-25 | Binary Trees |
| Day 6 | 26-30 | Advanced Trees |
| Day 7 | Review + Mock | Consolidation |
| Day 8 | 31-36 | Graphs |
| Day 9 | 37-39 | Binary Search |
| Day 10 | 40-42 | Backtracking |
| Day 11 | 43-46 | DP Basics |
| Day 12 | 47-50 | DP Advanced |
| Day 13 | Mixed Practice | Random Selection |
| Day 14 | Review + Mock | Final Prep |

### If You Have 1 Week

Focus on starred problems (highest priority):

| Day | Priority Problems |
|-----|-------------------|
| Day 1 | 1*, 4*, 6*, 8* |
| Day 2 | 13*, 15*, 18*, 20* |
| Day 3 | 24*, 25*, 27* |
| Day 4 | 31*, 34* |
| Day 5 | 40*, 41*, 42* |
| Day 6 | 44*, 45*, 49* |
| Day 7 | Review + Mock |

---

## Solution Approach Notes

### Problem 1: Two Sum
```
Pattern: Hash Map
Key Insight: Store seen numbers with their indices
Time: O(n), Space: O(n)
Edge Cases: Same element used twice, no solution
```

### Problem 6: 3Sum
```
Pattern: Sort + Two Pointers
Key Insight: Fix one number, use two pointers for rest
Time: O(n^2), Space: O(1) or O(n) for sorting
Edge Cases: Duplicate triplets, all same numbers
```

### Problem 21: Largest Rectangle in Histogram
```
Pattern: Monotonic Stack (Increasing)
Key Insight: For each bar, find left and right boundaries
Time: O(n), Space: O(n)
Edge Cases: Single bar, all same heights
```

### Problem 34: Course Schedule
```
Pattern: Topological Sort (Kahn's Algorithm)
Key Insight: Build in-degree array, process zero-degree nodes
Time: O(V + E), Space: O(V + E)
Edge Cases: Cycle exists, single course, no prerequisites
```

### Problem 45: Coin Change
```
Pattern: Unbounded Knapsack DP
Key Insight: dp[amount] = min coins to make amount
Time: O(amount * coins), Space: O(amount)
Edge Cases: Amount = 0, impossible amount
```

### Problem 50: Edit Distance
```
Pattern: 2D String DP
Key Insight: dp[i][j] = min ops for word1[0:i] to word2[0:j]
Time: O(m*n), Space: O(m*n) or O(n) optimized
Edge Cases: Empty strings, identical strings
```

---

## Progress Tracker

### Week 1 Summary
```
Arrays/Strings:    ___/8  completed
Hash Tables:       ___/4  completed
Linked Lists:      ___/5  completed
Stacks/Queues:     ___/4  completed
Binary Trees:      ___/4  completed
Week 1 Total:      ___/25 completed
```

### Week 2 Summary
```
Trees Advanced:    ___/5  completed
Graphs:            ___/6  completed
Binary Search:     ___/3  completed
Backtracking:      ___/3  completed
Dynamic Programming: ___/8  completed
Week 2 Total:      ___/25 completed
```

### Overall Progress
```
Total Completed:   ___/50
Mastered (*):      ___/50
Need Review (~):   ___/50

Patterns Covered:
[ ] Two Pointers
[ ] Sliding Window
[ ] Fast/Slow Pointers
[ ] Hash Map/Set
[ ] Stack (including Monotonic)
[ ] Binary Search
[ ] Tree DFS/BFS
[ ] Graph DFS/BFS
[ ] Topological Sort
[ ] Union-Find
[ ] Backtracking
[ ] DP (1D and 2D)
```

---

## Tips for Problem Solving

### Before Coding
1. Read the problem twice
2. Identify the pattern (see [Pattern Selection Flowchart](./essential-patterns#pattern-selection-flowchart))
3. Think about edge cases
4. Explain your approach verbally

### During Coding
1. Start with a clear function signature
2. Write pseudocode comments first
3. Implement step by step
4. Handle edge cases explicitly

### After Coding
1. Trace through with an example
2. Test edge cases
3. Analyze time and space complexity
4. Consider optimization opportunities

### When Stuck
- 10 minutes: Look at the hint/pattern
- 20 minutes: Review the solution concept
- 30 minutes: Study the full solution, then implement yourself

---

## Additional Resources

### LeetCode Lists
- [Blind 75](https://leetcode.com/list/xi4ci4ig/) - Classic 75 problems
- [Grind 75](https://www.techinterviewhandbook.org/grind75) - Optimized progression
- [NeetCode 150](https://neetcode.io/practice) - With video explanations

### Video Explanations
- NeetCode YouTube channel
- Take U Forward (Striver's A2Z)
- Back to Back SWE

### Practice Platforms
- LeetCode Premium (recommended for company-tagged problems)
- Pramp (free mock interviews)
- Interviewing.io (paid mock interviews)

---

*Track your progress daily and review [Common Mistakes](./common-mistakes) to avoid pitfalls.*

---

*Last updated: January 2026*
