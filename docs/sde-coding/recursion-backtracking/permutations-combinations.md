# Permutations & Combination Sum

> **Classic backtracking with different constraints**

---

## Overview

Permutations and Combination Sum represent two fundamental backtracking patterns:
- **Permutations**: Arrange ALL elements in every possible order
- **Combination Sum**: Select elements (with possible reuse) to reach a target

Understanding these patterns unlocks solutions to dozens of related problems.

---

## Permutations

### Problem Statement

Given an array `nums` of distinct integers, return all possible permutations. You can return the answer in any order.

**Example:**
```
Input: nums = [1, 2, 3]
Output: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

**Key Insight**: For n distinct elements, there are n! (n factorial) possible permutations.

### Backtracking Tree

```
nums = [1, 2, 3]

                              []
               /              |              \
             [1]             [2]             [3]
            /   \           /   \           /   \
        [1,2] [1,3]     [2,1] [2,3]     [3,1] [3,2]
          |     |         |     |         |     |
      [1,2,3][1,3,2]  [2,1,3][2,3,1]  [3,1,2][3,2,1]
          ^     ^         ^     ^         ^     ^
       RESULT RESULT   RESULT RESULT   RESULT RESULT
```

### Detailed Decision Tree Visualization

```
Building permutation step-by-step:

Step 1: Choose first element
   Available: [1, 2, 3]

   Pick 1 -> remaining: [2, 3]
   Pick 2 -> remaining: [1, 3]
   Pick 3 -> remaining: [1, 2]

Step 2: Choose second element (example: picked 1 first)
   Current: [1], Available: [2, 3]

   Pick 2 -> current: [1, 2], remaining: [3]
   Pick 3 -> current: [1, 3], remaining: [2]

Step 3: Choose third element (example: picked 1, then 2)
   Current: [1, 2], Available: [3]

   Pick 3 -> current: [1, 2, 3], remaining: []

   Base case reached! Add [1, 2, 3] to results
```

### Solution Approach 1: Using Remaining Elements (Python)

```python
def permute(nums: list[int]) -> list[list[int]]:
    """
    Generate all permutations using backtracking with remaining list.

    Time: O(n! * n) - n! permutations, each takes O(n) to copy
    Space: O(n) - recursion depth + current path
    """
    result = []

    def backtrack(current: list[int], remaining: list[int]) -> None:
        # Base case: no more elements to add
        if not remaining:
            result.append(current[:])  # Add copy of current permutation
            return

        # Try each remaining element as the next choice
        for i in range(len(remaining)):
            # Choose: add element to current permutation
            current.append(remaining[i])

            # Explore: recurse with remaining elements (excluding chosen)
            new_remaining = remaining[:i] + remaining[i+1:]
            backtrack(current, new_remaining)

            # Unchoose: backtrack by removing the element
            current.pop()

    backtrack([], nums)
    return result


# Example usage
print(permute([1, 2, 3]))
# Output: [[1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]]
```

::: info Complexity: Time O(n! * n) · Space O(n)
- **Time:** O(n! * n) where n! is the number of permutations and O(n) to copy each permutation to the result
- **Space:** O(n) for recursion depth plus O(n) for the remaining list at each level
:::

### Solution Approach 2: In-Place Swapping (Python)

```python
def permute_swap(nums: list[int]) -> list[list[int]]:
    """
    Generate permutations using in-place swapping.
    More memory efficient - modifies array in place.

    Time: O(n! * n)
    Space: O(n) - only recursion stack
    """
    result = []

    def backtrack(start: int) -> None:
        # Base case: all positions filled
        if start == len(nums):
            result.append(nums[:])  # Add copy
            return

        # Try placing each element at position 'start'
        for i in range(start, len(nums)):
            # Choose: swap element to current position
            nums[start], nums[i] = nums[i], nums[start]

            # Explore: fix this position, recurse on rest
            backtrack(start + 1)

            # Unchoose: swap back (backtrack)
            nums[start], nums[i] = nums[i], nums[start]

    backtrack(0)
    return result
```

::: info Complexity: Time O(n! * n) · Space O(n)
- **Time:** O(n! * n) where n! permutations are generated and O(n) to copy each one
- **Space:** O(n) for recursion stack only - more memory efficient as swaps are done in-place
:::

### Swap-Based Visualization

```
nums = [1, 2, 3], start = 0

backtrack(0):
  i=0: swap(0,0) -> [1,2,3]
       backtrack(1):
         i=1: swap(1,1) -> [1,2,3]
              backtrack(2):
                i=2: swap(2,2) -> [1,2,3]
                     backtrack(3): ADD [1,2,3]
                     swap(2,2) -> [1,2,3]
              swap(1,1) -> [1,2,3]
         i=2: swap(1,2) -> [1,3,2]
              backtrack(2):
                i=2: swap(2,2) -> [1,3,2]
                     backtrack(3): ADD [1,3,2]
              swap(1,2) -> [1,2,3]
       swap(0,0) -> [1,2,3]

  i=1: swap(0,1) -> [2,1,3]
       backtrack(1): ... generates [2,1,3], [2,3,1]
       swap(0,1) -> [1,2,3]

  i=2: swap(0,2) -> [3,2,1]
       backtrack(1): ... generates [3,2,1], [3,1,2]
       swap(0,2) -> [1,2,3]
```

### Visual Guides

![Permutations Tree](/sde-coding/recursion/permutations_tree.png)

### Permutations II: With Duplicates

```python
def permuteUnique(nums: list[int]) -> list[list[int]]:
    """
    Generate unique permutations when input contains duplicates.

    Key insight: Sort array first, then skip duplicates at same level.

    Example: [1, 1, 2] -> [[1,1,2], [1,2,1], [2,1,1]]
    """
    nums.sort()  # Critical for duplicate detection!
    result = []
    used = [False] * len(nums)

    def backtrack(current: list[int]) -> None:
        if len(current) == len(nums):
            result.append(current[:])
            return

        for i in range(len(nums)):
            # Skip if already used in current permutation
            if used[i]:
                continue

            # Skip duplicates: if same as previous AND previous not used
            # This ensures we only use duplicates in order
            if i > 0 and nums[i] == nums[i-1] and not used[i-1]:
                continue

            used[i] = True
            current.append(nums[i])
            backtrack(current)
            current.pop()
            used[i] = False

    backtrack([])
    return result
```

::: info Complexity: Time O(n! * n) · Space O(n)
- **Time:** O(n! * n) worst case - sorting takes O(n log n), then generating permutations with duplicate skipping
- **Space:** O(n) for recursion depth plus O(n) for the used array
:::

### Duplicate Skipping Visualization

```
nums = [1, 1, 2] (sorted)

Without duplicate handling:
   [1a, 1b, 2] and [1b, 1a, 2] would both be generated

With duplicate handling:
   At position 0:
     - Pick 1a (index 0): OK
     - Pick 1b (index 1): SKIP! Same as 1a and 1a not used
     - Pick 2 (index 2): OK

   This ensures 1a is always used before 1b

Result: [[1,1,2], [1,2,1], [2,1,1]]
```

---

## Combination Sum

### Problem Statement

Given an array of **distinct** integers `candidates` and a target integer `target`, return a list of all **unique combinations** of candidates where the chosen numbers sum to target.

The **same number may be chosen unlimited times**.

**Example:**
```
Input: candidates = [2, 3, 6, 7], target = 7
Output: [[2, 2, 3], [7]]
Explanation:
  - 2 + 2 + 3 = 7
  - 7 = 7
```

### Key Insight: Preventing Duplicates

To avoid duplicate combinations like `[2,3,2]` and `[2,2,3]`:
- Only consider candidates **from current index forward**
- This ensures elements are chosen in **non-decreasing order of indices**

### Decision Tree Visualization

```
candidates = [2, 3, 6, 7], target = 7

Start: sum=0, path=[], start_index=0
|
+-- Choose 2 (idx=0): sum=2, path=[2]
|   |
|   +-- Choose 2 (idx=0): sum=4, path=[2,2]
|   |   |
|   |   +-- Choose 2 (idx=0): sum=6, path=[2,2,2]
|   |   |   |
|   |   |   +-- Choose 2: sum=8 > 7  PRUNE
|   |   |   +-- Choose 3: sum=9 > 7  PRUNE
|   |   |
|   |   +-- Choose 3 (idx=1): sum=7, path=[2,2,3]  FOUND!
|   |   +-- Choose 6 (idx=2): sum=10 > 7  PRUNE
|   |   +-- Choose 7 (idx=3): sum=11 > 7  PRUNE
|   |
|   +-- Choose 3 (idx=1): sum=5, path=[2,3]
|   |   |
|   |   +-- Choose 3: sum=8 > 7  PRUNE
|   |
|   +-- Choose 6 (idx=2): sum=8 > 7  PRUNE
|   +-- Choose 7 (idx=3): sum=9 > 7  PRUNE
|
+-- Choose 3 (idx=1): sum=3, path=[3]
|   |
|   +-- Choose 3: sum=6, path=[3,3]
|   |   +-- Choose 3: sum=9 > 7  PRUNE
|   +-- Choose 6: sum=9 > 7  PRUNE
|   +-- Choose 7: sum=10 > 7  PRUNE
|
+-- Choose 6 (idx=2): sum=6, path=[6]
|   +-- Choose 6: sum=12 > 7  PRUNE
|   +-- Choose 7: sum=13 > 7  PRUNE
|
+-- Choose 7 (idx=3): sum=7, path=[7]  FOUND!

Results: [[2,2,3], [7]]
```

### Solution (Python)

```python
def combinationSum(candidates: list[int], target: int) -> list[list[int]]:
    """
    Find all unique combinations that sum to target.
    Same number can be used unlimited times.

    Time: O(n^(t/m)) where t=target, m=min(candidates)
    Space: O(t/m) for recursion depth
    """
    result = []

    def backtrack(start: int, current: list[int], remaining: int) -> None:
        # Base case: found valid combination
        if remaining == 0:
            result.append(current[:])
            return

        # Base case: exceeded target
        if remaining < 0:
            return

        # Try each candidate from start index onward
        for i in range(start, len(candidates)):
            # Choose
            current.append(candidates[i])

            # Explore: pass i (not i+1) to allow reuse
            backtrack(i, current, remaining - candidates[i])

            # Unchoose
            current.pop()

    backtrack(0, [], target)
    return result


# Example
print(combinationSum([2, 3, 6, 7], 7))
# Output: [[2, 2, 3], [7]]
```

::: info Complexity: Time O(n^(t/m)) · Space O(t/m)
- **Time:** O(n^(t/m)) where t is the target sum and m is the minimum candidate value - represents max branching
- **Space:** O(t/m) for recursion depth in the worst case (using only minimum value repeatedly)
:::

### Optimized with Sorting and Pruning

```python
def combinationSum_optimized(candidates: list[int], target: int) -> list[list[int]]:
    """
    Optimized version with early termination.
    Sorting allows us to break early when candidate exceeds remaining.
    """
    candidates.sort()  # Sort for early termination
    result = []

    def backtrack(start: int, current: list[int], remaining: int) -> None:
        if remaining == 0:
            result.append(current[:])
            return

        for i in range(start, len(candidates)):
            # Early termination: all subsequent candidates are larger
            if candidates[i] > remaining:
                break

            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])
            current.pop()

    backtrack(0, [], target)
    return result
```

::: info Complexity: Time O(n^(t/m)) · Space O(t/m)
- **Time:** O(n^(t/m)) with better constants due to sorting and early termination pruning
- **Space:** O(t/m) for recursion depth plus O(n log n) for sorting
:::

---

## Combination Sum II: Each Element Used Once

### Problem Statement

Given a collection of candidate numbers (may contain duplicates) and a target, find all unique combinations that sum to target. **Each number may only be used once**.

### Solution

```python
def combinationSum2(candidates: list[int], target: int) -> list[list[int]]:
    """
    Find combinations where each element can only be used once.
    Input may contain duplicates.

    Key insight: Sort first, then skip duplicates at same recursion level.
    """
    candidates.sort()  # Critical for duplicate handling!
    result = []

    def backtrack(start: int, current: list[int], remaining: int) -> None:
        if remaining == 0:
            result.append(current[:])
            return
        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            # Skip duplicates at the same level
            if i > start and candidates[i] == candidates[i-1]:
                continue

            # Early termination (since sorted)
            if candidates[i] > remaining:
                break

            current.append(candidates[i])
            # Move to next index (i+1) since each element used once
            backtrack(i + 1, current, remaining - candidates[i])
            current.pop()

    backtrack(0, [], target)
    return result


# Example
print(combinationSum2([10, 1, 2, 7, 6, 1, 5], 8))
# Output: [[1,1,6], [1,2,5], [1,7], [2,6]]
```

::: info Complexity: Time O(2^n) · Space O(n)
- **Time:** O(2^n) where each element can be included or excluded - duplicate skipping reduces actual exploration
- **Space:** O(n) for recursion depth plus O(n log n) for sorting the candidates
:::

### Duplicate Skipping Visualization

```
candidates = [1, 1, 2, 5, 6, 7, 10] (sorted), target = 8

At start=0:
  i=0: Pick 1 (first)  -> Continue to explore
  i=1: Skip! 1 == previous 1, and i > start
  i=2: Pick 2          -> Continue to explore
  ...

Within the subtree of picking first 1:
  At start=1:
    i=1: Pick second 1 -> OK (i == start, not skipping)
    i=2: Pick 2        -> Continue
    ...

This ensures [1a, 1b, 6] is generated
but [1b, 1a, 6] is NOT (1b skipped at level 0)
```

---

## Combination Sum III: K Numbers Sum to N

### Problem Statement

Find all valid combinations of k numbers that sum up to n. Only numbers 1-9 may be used, and each number may be used at most once.

### Solution

```python
def combinationSum3(k: int, n: int) -> list[list[int]]:
    """
    Find k numbers from 1-9 that sum to n.
    Each number used at most once.
    """
    result = []

    def backtrack(start: int, current: list[int], remaining: int) -> None:
        # Found valid combination
        if len(current) == k and remaining == 0:
            result.append(current[:])
            return

        # Pruning: too many numbers or sum exceeded
        if len(current) >= k or remaining <= 0:
            return

        for num in range(start, 10):
            # Pruning: remaining numbers too small
            if num > remaining:
                break

            current.append(num)
            backtrack(num + 1, current, remaining - num)
            current.pop()

    backtrack(1, [], n)
    return result


# Example
print(combinationSum3(3, 7))   # Output: [[1,2,4]]
print(combinationSum3(3, 9))   # Output: [[1,2,6], [1,3,5], [2,3,4]]
```

::: info Complexity: Time O(C(9,k) * k) · Space O(k)
- **Time:** O(C(9,k) * k) where C(9,k) is the number of ways to choose k numbers from 1-9, and O(k) to copy each combination
- **Space:** O(k) for recursion depth and current combination being built
:::

---

## Pattern Comparison: Permutations vs Combinations

### Visual Comparison

```
PERMUTATIONS: Order matters, use all elements
[1, 2, 3] -> [1,2,3], [1,3,2], [2,1,3], [2,3,1], [3,1,2], [3,2,1]
              ^Different arrangements of same elements^

COMBINATIONS: Order doesn't matter, select subset
[1, 2, 3], k=2 -> [1,2], [1,3], [2,3]
                  ^Unique subsets, no [2,1] since same as [1,2]^

COMBINATION SUM: Find combinations that sum to target
[2, 3, 6, 7], target=7 -> [2,2,3], [7]
                          ^Can reuse elements^
```

### Code Pattern Comparison

```python
# PERMUTATIONS: Use all elements, track what's used
def permute(nums):
    def backtrack(current, remaining):
        if not remaining:           # Used all elements
            result.append(current[:])
        for i in range(len(remaining)):
            current.append(remaining[i])
            backtrack(current, remaining[:i] + remaining[i+1:])
            current.pop()

# COMBINATIONS: Select k elements
def combine(n, k):
    def backtrack(start, current):
        if len(current) == k:       # Selected k elements
            result.append(current[:])
        for i in range(start, n+1):
            current.append(i)
            backtrack(i + 1, current)  # i+1: no reuse
            current.pop()

# COMBINATION SUM: Select elements summing to target
def combinationSum(candidates, target):
    def backtrack(start, current, remaining):
        if remaining == 0:          # Sum equals target
            result.append(current[:])
        for i in range(start, len(candidates)):
            current.append(candidates[i])
            backtrack(i, current, remaining - candidates[i])  # i: allow reuse
            current.pop()
```

### Key Differences Table

| Aspect | Permutations | Combinations | Combination Sum |
|--------|--------------|--------------|-----------------|
| Order matters? | Yes | No | No |
| Use all elements? | Yes | Select k | Select any |
| Reuse allowed? | No | No | Yes (variant I) |
| Base case | len == n | len == k | sum == target |
| Next index | N/A (track used) | i + 1 | i (reuse) or i + 1 |

---

## Complexity Analysis

| Problem | Time Complexity | Space Complexity | Notes |
|---------|-----------------|------------------|-------|
| **Permutations** | O(n! * n) | O(n) | n! permutations, O(n) to copy each |
| **Permutations II** | O(n! * n) | O(n) | Same worst case, better with many duplicates |
| **Subsets** | O(2^n * n) | O(n) | 2^n subsets, O(n) to copy each |
| **Combinations** | O(C(n,k) * k) | O(k) | C(n,k) combinations |
| **Combination Sum** | O(n^(t/m)) | O(t/m) | t=target, m=min candidate |
| **Combination Sum II** | O(2^n) | O(n) | Each element included/excluded |
| **Combination Sum III** | O(C(9,k) * k) | O(k) | Limited to 9 numbers |

### Complexity Derivation for Combination Sum

```
Why O(n^(t/m))?

- Maximum recursion depth: t/m (using only minimum element)
- At each level: up to n choices
- Total nodes: approximately n^(t/m)

Example: candidates=[2,3], target=10
- Depth: 10/2 = 5
- Branching: 2 at each level
- Nodes: ~2^5 = 32
```

---

## Interview Applications

These patterns are foundational for many interview questions:

### 1. Phone Number Letter Combinations (LeetCode 17)
```
Similar to permutations: generate all possible letter combinations
"23" -> ["ad","ae","af","bd","be","bf","cd","ce","cf"]
```

### 2. Generate Parentheses (LeetCode 22)
```
Backtracking with constraints: balance open/close parentheses
n=3 -> ["((()))","(()())","(())()","()(())","()()()"]
```

### 3. Word Search (LeetCode 79)
```
Backtracking on 2D grid: explore all paths
Used in Google's web crawling and indexing problems
```

### 4. N-Queens (LeetCode 51)
```
Classic backtracking with constraint checking
Tests understanding of state space search
```

### 5. Sudoku Solver (LeetCode 37)
```
Constraint satisfaction via backtracking
Common in Google constraint optimization interviews
```

---

## Common Mistakes and Tips

### Mistake 1: Forgetting to Copy

```python
# WRONG - adds reference, gets modified later
result.append(current)

# CORRECT - adds copy
result.append(current[:])
# or
result.append(list(current))
```

### Mistake 2: Wrong Index in Combination Sum

```python
# Combination Sum I (reuse allowed)
backtrack(i, current, remaining - candidates[i])
#         ^ same index

# Combination Sum II (no reuse)
backtrack(i + 1, current, remaining - candidates[i])
#         ^ next index
```

### Mistake 3: Not Sorting Before Duplicate Handling

```python
# WRONG - duplicate detection fails without sorting
def combinationSum2(candidates, target):
    # candidates NOT sorted
    if i > start and candidates[i] == candidates[i-1]:
        continue  # Won't work!

# CORRECT
def combinationSum2(candidates, target):
    candidates.sort()  # Sort first!
    # Now duplicate detection works
```

### Mistake 4: Inefficient State Copying

```python
# INEFFICIENT - creates new list at each level
backtrack(current + [nums[i]], remaining[:i] + remaining[i+1:])

# BETTER - modify in place, backtrack
current.append(nums[i])
backtrack(current, ...)
current.pop()
```

---

## Practice Problems

### Essential Problems
1. [Permutations (LeetCode 46)](https://leetcode.com/problems/permutations/)
2. [Permutations II (LeetCode 47)](https://leetcode.com/problems/permutations-ii/)
3. [Combination Sum (LeetCode 39)](https://leetcode.com/problems/combination-sum/)
4. [Combination Sum II (LeetCode 40)](https://leetcode.com/problems/combination-sum-ii/)
5. [Combination Sum III (LeetCode 216)](https://leetcode.com/problems/combination-sum-iii/)

### Related Problems
6. [Subsets (LeetCode 78)](https://leetcode.com/problems/subsets/)
7. [Subsets II (LeetCode 90)](https://leetcode.com/problems/subsets-ii/)
8. [Combinations (LeetCode 77)](https://leetcode.com/problems/combinations/)
9. [Letter Combinations of Phone Number (LeetCode 17)](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)
10. [Generate Parentheses (LeetCode 22)](https://leetcode.com/problems/generate-parentheses/)

---

## Summary

### The Backtracking Template

```python
def backtrack_template(state):
    """
    Universal backtracking template for permutation/combination problems.
    """
    # 1. Base case: check if we found a valid solution
    if is_valid_solution(state):
        result.append(copy_of_current_solution())
        return

    # 2. Pruning: check if we should stop exploring this path
    if should_prune(state):
        return

    # 3. Explore: try all possible choices
    for choice in get_choices(state):
        # 3a. Make choice (modify state)
        make_choice(state, choice)

        # 3b. Recurse
        backtrack_template(new_state)

        # 3c. Undo choice (backtrack)
        undo_choice(state, choice)
```

### Key Takeaways

1. **Permutations**: Arrange all elements - track what's used/remaining
2. **Combinations**: Select subset - use start index to avoid duplicates
3. **Combination Sum**: Find sums - control reuse via index (i vs i+1)
4. **Duplicates**: Always sort first, then skip at same level
5. **Optimization**: Sort + early termination when possible

---

## References

- [LeetCode: A General Approach to Backtracking Questions](https://leetcode.com/problems/permutations/solutions/18239/A-general-approach-to-backtracking-questions-in-Java-(Subsets-Permutations-Combination-Sum-Palindrome-Partioning)/)
- [Backtracking Algorithm for Permutations, Combinations, Subsets](https://labuladong.online/en/algo/essential-technique/permutation-combination-subset-all-in-one/)
- [Combination Sum - LeetCode](https://leetcode.com/problems/combination-sum/)
- [Combination Sum II - LeetCode](https://leetcode.com/problems/combination-sum-ii/)
- [AlgoMap: Permutations Solution](https://algomap.io/problems/permutations)
- [AlgoMap: Combination Sum Solution](https://algomap.io/problems/combination-sum)
- [TakeUForward: Combination Sum Tutorial](https://takeuforward.org/data-structure/combination-sum-1)
