# Fibonacci, Generate Parentheses & Subsets

> **Foundation recursion and backtracking problems**

These three problems form the cornerstone of understanding recursion and backtracking. Fibonacci teaches memoization, Generate Parentheses introduces constrained backtracking, and Subsets demonstrates exhaustive enumeration.

---

## Fibonacci Numbers

### Problem Statement
The Fibonacci sequence is defined as:
- F(0) = 0
- F(1) = 1
- F(n) = F(n-1) + F(n-2) for n > 1

Sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, ...

### Visualization: Recursion Tree

**Without Memoization (Exponential Redundancy)**:
```text
                                    fib(5)
                                   /      \
                             fib(4)        fib(3)
                            /     \        /     \
                       fib(3)   fib(2)  fib(2)  fib(1)
                       /    \    /   \   /   \     |
                  fib(2) fib(1) f(1) f(0) f(1) f(0) 1
                  /   \    |     |    |    |    |
               f(1)  f(0)  1     0    1    0    0
                 |     |
                 1     0

Total calls for fib(5): 15 calls
Notice: fib(3) computed 2 times, fib(2) computed 3 times
```

**With Memoization (Linear - Pruned Tree)**:
```text
                                    fib(5)
                                   /      \
                             fib(4)        fib(3) [CACHED]
                            /     \
                       fib(3)   fib(2) [CACHED]
                       /    \
                  fib(2) fib(1) [CACHED]
                  /   \
               f(1)  f(0)

Total calls for fib(5): 9 calls (and only 6 unique computations)
Each subproblem solved exactly ONCE!
```

**Growth Comparison**:
```text
n     | Naive Calls | Memoized Calls | Speedup
------|-------------|----------------|--------
5     |      15     |       9        |  1.7x
10    |     177     |      19        |  9.3x
20    |   21,891    |      39        | 561x
30    | 2,692,537   |      59        | 45,636x
40    | 331,160,281 |      79        | 4.2M x
```

### Solutions Comparison

```python
# Naive recursive - O(2^n) time, O(n) space (call stack)
def fib_naive(n):
    """
    Simple but extremely inefficient.
    Time: O(2^n) - each call spawns 2 more calls
    Space: O(n) - maximum recursion depth
    """
    if n <= 1:
        return n
    return fib_naive(n-1) + fib_naive(n-2)


# Memoized - O(n) time, O(n) space
from functools import lru_cache

@lru_cache(maxsize=None)
def fib_memo(n):
    """
    Optimal recursive with caching.
    Time: O(n) - each subproblem computed once
    Space: O(n) - cache + call stack
    """
    if n <= 1:
        return n
    return fib_memo(n-1) + fib_memo(n-2)


# Manual memoization (interview-friendly)
def fib_memo_manual(n, memo=None):
    """
    Explicit memoization - good for interviews.
    """
    if memo is None:
        memo = {}
    if n <= 1:
        return n
    if n in memo:
        return memo[n]
    memo[n] = fib_memo_manual(n-1, memo) + fib_memo_manual(n-2, memo)
    return memo[n]


# Iterative (Bottom-up DP) - O(n) time, O(n) space
def fib_dp(n):
    """
    Bottom-up dynamic programming.
    Time: O(n)
    Space: O(n) - stores all values
    """
    if n <= 1:
        return n
    dp = [0] * (n + 1)
    dp[1] = 1
    for i in range(2, n + 1):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[n]


# Space-optimized iterative - O(n) time, O(1) space
def fib_iter(n):
    """
    Most efficient - only tracks last two values.
    Time: O(n)
    Space: O(1)
    """
    if n <= 1:
        return n
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr
    return curr


# Matrix exponentiation - O(log n) time, O(1) space
def fib_matrix(n):
    """
    Fastest theoretical approach using matrix exponentiation.
    | F(n+1)  F(n)   |   | 1 1 |^n
    | F(n)    F(n-1) | = | 1 0 |

    Time: O(log n)
    Space: O(1)
    """
    if n <= 1:
        return n

    def matrix_mult(A, B):
        return [
            [A[0][0]*B[0][0] + A[0][1]*B[1][0], A[0][0]*B[0][1] + A[0][1]*B[1][1]],
            [A[1][0]*B[0][0] + A[1][1]*B[1][0], A[1][0]*B[0][1] + A[1][1]*B[1][1]]
        ]

    def matrix_pow(M, p):
        if p == 1:
            return M
        if p % 2 == 0:
            half = matrix_pow(M, p // 2)
            return matrix_mult(half, half)
        else:
            return matrix_mult(M, matrix_pow(M, p - 1))

    base = [[1, 1], [1, 0]]
    result = matrix_pow(base, n)
    return result[0][1]
```

::: info Complexity: Time varies by approach · Space varies by approach
- **Time:** O(2^n) naive, O(n) memoized/iterative, O(log n) matrix exponentiation
- **Space:** O(n) for naive/memoized (call stack + cache), O(1) for space-optimized iterative and matrix approaches
:::

### Key Takeaways for Fibonacci

| Approach | Time | Space | When to Use |
|----------|------|-------|-------------|
| Naive | O(2^n) | O(n) | Never in production |
| Memoized | O(n) | O(n) | When recursion is natural |
| Iterative | O(n) | O(1) | Best for interviews |
| Matrix | O(log n) | O(1) | Very large n |

---

## Generate Parentheses

> **LeetCode 22** - [Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)

### Problem Statement
Given `n` pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

**Example**:
- Input: n = 3
- Output: ["((()))", "(()())", "(())()", "()(())", "()()()"]

### Backtracking Constraints
The key insight is understanding when we can add '(' or ')':
1. **Add '('**: Only if open_count < n (we have opening brackets left)
2. **Add ')'**: Only if close_count < open_count (there's an unmatched '(' to close)

### Backtracking Tree Visualization

```text
n = 2

                                ""
                               /
                             "("
                           /     \
                        "(("      "()"
                        /           \
                     "(()"         "()(
                      /               \
                   "(())"           "()()"
                   VALID!           VALID!

Legend:
  - At each node, we try adding '(' first (if possible)
  - Then we try adding ')' (if valid)
  - We prune branches that would create invalid sequences
```

**Detailed Decision Tree for n = 3**:
```text
Level 0:                            ""
                                    |
Level 1:                           "("
                                 /     \
Level 2:                      "(("     "()"
                             /    \       \
Level 3:                  "((("  "(()"   "()("
                           |     /   \      \
Level 4:                "((()" "(()(" "(()" "()(("
                           |      |      |      |
Level 5:              "((())" "(()()""(())(""()(()"
                          |      |       |      |
Level 6:              "((()))"  "(()())" "(())()" "()(())"  "()()()"
                       VALID!   VALID!   VALID!   VALID!    VALID!

Catalan number C(3) = 5 valid combinations
```

**Visual Representation of Pruning**:
```text
Why we can't start with ')':
"" -> ")" INVALID! (close_count > open_count)

Why "((" -> ")" is valid but "(" -> "))" is not:
"((" has open=2, close=0, so close < open, can add ')'
"()" has open=1, close=1, so close == open, CANNOT add ')'

Pruning saves exponential work:
Without pruning: 2^6 = 64 paths for n=3
With pruning: Only 5 valid paths (Catalan number)
```

### Solution with Detailed Comments

```python
from typing import List

def generateParenthesis(n: int) -> List[str]:
    """
    Generate all valid combinations of n pairs of parentheses.

    Time Complexity: O(4^n / sqrt(n)) - Catalan number bound
    Space Complexity: O(n) - recursion depth + current string

    Args:
        n: Number of pairs of parentheses
    Returns:
        List of all valid parentheses combinations
    """
    result = []

    def backtrack(current: str, open_count: int, close_count: int):
        """
        Recursively build valid parentheses combinations.

        Args:
            current: Current string being built
            open_count: Number of '(' used so far
            close_count: Number of ')' used so far
        """
        # Base case: we've used all parentheses
        if len(current) == 2 * n:
            result.append(current)
            return

        # Choice 1: Add opening parenthesis if we haven't used all
        if open_count < n:
            backtrack(current + '(', open_count + 1, close_count)

        # Choice 2: Add closing parenthesis if it won't create invalid sequence
        # We can only close if there's an unmatched opening parenthesis
        if close_count < open_count:
            backtrack(current + ')', open_count, close_count + 1)

    backtrack('', 0, 0)
    return result


# Alternative: Using list for better memory efficiency
def generateParenthesis_optimized(n: int) -> List[str]:
    """
    Memory-optimized version using list instead of string concatenation.
    """
    result = []

    def backtrack(path: list, open_count: int, close_count: int):
        if len(path) == 2 * n:
            result.append(''.join(path))
            return

        if open_count < n:
            path.append('(')
            backtrack(path, open_count + 1, close_count)
            path.pop()  # Backtrack

        if close_count < open_count:
            path.append(')')
            backtrack(path, open_count, close_count + 1)
            path.pop()  # Backtrack

    backtrack([], 0, 0)
    return result


# Iterative BFS approach
from collections import deque

def generateParenthesis_bfs(n: int) -> List[str]:
    """
    BFS approach - builds solutions level by level.
    """
    if n == 0:
        return []

    result = []
    # Queue entries: (current_string, open_count, close_count)
    queue = deque([('', 0, 0)])

    while queue:
        current, open_count, close_count = queue.popleft()

        if len(current) == 2 * n:
            result.append(current)
            continue

        if open_count < n:
            queue.append((current + '(', open_count + 1, close_count))

        if close_count < open_count:
            queue.append((current + ')', open_count, close_count + 1))

    return result
```

::: info Complexity: Time O(4^n / sqrt(n)) · Space O(n)
- **Time:** O(4^n / sqrt(n)) bounded by the nth Catalan number - represents the number of valid parentheses combinations
- **Space:** O(n) for recursion depth (building strings of length 2n) plus O(n) for storing each result string
:::

---

## Subsets (Power Set)

> **LeetCode 78** - [Subsets](https://leetcode.com/problems/subsets/)

### Problem Statement
Given an integer array `nums` of unique elements, return all possible subsets (the power set). The solution set must not contain duplicate subsets.

**Example**:
- Input: nums = [1, 2, 3]
- Output: [[], [1], [2], [3], [1,2], [1,3], [2,3], [1,2,3]]

### Key Insight
For an array of n elements, there are 2^n possible subsets. Each element can either be **included** or **excluded** from a subset.

### Backtracking Tree Visualization

```text
nums = [1, 2, 3]

                           []
                      /          \
                   [1]            []
                 /    \         /    \
             [1,2]   [1]      [2]    []
            /    \   /  \    /  \   /  \
       [1,2,3] [1,2] [1,3] [1] [2,3] [2] [3] []

                    OR (Include/Exclude View)

                           start
                             |
                    include 1? -----+
                   /                 \
                 yes                 no
                  |                   |
          include 2? ----+    include 2? ----+
          /              \     /              \
        yes             no   yes             no
         |               |    |               |
   include 3?     include 3? include 3?  include 3?
    /      \       /      \   /      \    /      \
  yes     no    yes     no  yes     no  yes     no
   |       |     |       |   |       |   |       |
[1,2,3] [1,2] [1,3]   [1] [2,3]   [2] [3]     []
```

**Alternative Visualization (Index-based)**:
```text
nums = [1, 2, 3]
Subsets built by choosing starting index:

                         backtrack(0, [])
                              |
              +---------------+---------------+
              |               |               |
         add nums[0]    add nums[1]    add nums[2]
         backtrack(1,[1]) backtrack(2,[2]) backtrack(3,[3])
              |               |               |
      +-------+-------+   +---+---+          done
      |       |       |   |       |
   [1,2]    [1,3]   done [2,3]  done
   /   \      |            |
[1,2,3] done done        done

All subsets collected: [[], [1], [1,2], [1,2,3], [1,3], [2], [2,3], [3]]
```

### Solutions

```python
from typing import List

# Solution 1: Backtracking (Most Intuitive)
def subsets(nums: List[int]) -> List[List[int]]:
    """
    Generate all subsets using backtracking.

    Time Complexity: O(n * 2^n) - 2^n subsets, each takes O(n) to copy
    Space Complexity: O(n) - recursion depth (excluding output)

    Key Insight: At each position, decide to include or skip each remaining element.
    """
    result = []

    def backtrack(start: int, current: List[int]):
        # Add current subset to result (captures all intermediate states)
        result.append(current[:])  # Make a copy!

        # Try adding each remaining element
        for i in range(start, len(nums)):
            current.append(nums[i])      # Include nums[i]
            backtrack(i + 1, current)    # Recurse with next index
            current.pop()                # Backtrack (exclude nums[i])

    backtrack(0, [])
    return result


# Solution 2: Cascading (Iterative)
def subsets_iterative(nums: List[int]) -> List[List[int]]:
    """
    Build subsets iteratively by adding each number to existing subsets.

    Process for [1, 2, 3]:
    Start: [[]]
    Add 1: [[], [1]]
    Add 2: [[], [1], [2], [1,2]]
    Add 3: [[], [1], [2], [1,2], [3], [1,3], [2,3], [1,2,3]]

    Time: O(n * 2^n)
    Space: O(1) excluding output
    """
    result = [[]]

    for num in nums:
        # For each existing subset, create a new one with num added
        result += [subset + [num] for subset in result]

    return result


# Solution 3: Bit Manipulation
def subsets_bits(nums: List[int]) -> List[List[int]]:
    """
    Use bit manipulation to generate all subsets.

    For n=3, we have masks 000 to 111 (0 to 7):
    000 -> []
    001 -> [1]
    010 -> [2]
    011 -> [1,2]
    100 -> [3]
    101 -> [1,3]
    110 -> [2,3]
    111 -> [1,2,3]

    Time: O(n * 2^n)
    Space: O(1) excluding output
    """
    n = len(nums)
    result = []

    # Iterate through all possible 2^n combinations
    for mask in range(2 ** n):
        subset = []
        for i in range(n):
            # Check if bit i is set in mask
            if mask & (1 << i):
                subset.append(nums[i])
        result.append(subset)

    return result


# Solution 4: Include/Exclude Pattern
def subsets_include_exclude(nums: List[int]) -> List[List[int]]:
    """
    At each index, make binary choice: include or exclude.

    More explicit decision tree representation.
    """
    result = []

    def backtrack(index: int, current: List[int]):
        if index == len(nums):
            result.append(current[:])
            return

        # Choice 1: Exclude nums[index]
        backtrack(index + 1, current)

        # Choice 2: Include nums[index]
        current.append(nums[index])
        backtrack(index + 1, current)
        current.pop()

    backtrack(0, [])
    return result
```

::: info Complexity: Time O(n * 2^n) · Space O(n)
- **Time:** O(n * 2^n) where 2^n is the number of subsets and O(n) to copy each subset to the result
- **Space:** O(n) for recursion depth (excluding output storage)
:::

### Handling Duplicates: Subsets II

> **LeetCode 90** - [Subsets II](https://leetcode.com/problems/subsets-ii/)

```python
def subsetsWithDup(nums: List[int]) -> List[List[int]]:
    """
    Generate subsets when nums may contain duplicates.

    Key: Sort first, then skip duplicates at the same level.

    Example: nums = [1, 2, 2]
    After sorting: [1, 2, 2]

    Without skip: [1,2], [1,2] <- duplicate!
    With skip: Only process first 2 at each level
    """
    result = []
    nums.sort()  # Critical: sort to group duplicates

    def backtrack(start: int, current: List[int]):
        result.append(current[:])

        for i in range(start, len(nums)):
            # Skip duplicates at the same level
            if i > start and nums[i] == nums[i-1]:
                continue
            current.append(nums[i])
            backtrack(i + 1, current)
            current.pop()

    backtrack(0, [])
    return result
```

::: info Complexity: Time O(n * 2^n) · Space O(n)
- **Time:** O(n * 2^n) worst case - sorting takes O(n log n), then generating up to 2^n subsets with O(n) copy each
- **Space:** O(n) for recursion depth plus O(n log n) for sorting
:::

---

## Visual Guides

### Fibonacci Recursion Tree
<!-- TODO: Add visualization ![Fibonacci Tree](/sde-coding/recursion/fibonacci_tree.png) -->

### Subsets Generation Tree
<!-- TODO: Add visualization ![Subsets Tree](/sde-coding/recursion/subsets_tree.png) -->

### Generate Parentheses Decision Tree
<!-- TODO: Add visualization ![Generate Parentheses](/sde-coding/recursion/generate_parentheses.png) -->

---

## Comparison of All Three Problems

| Problem | Pattern | Time Complexity | Space | Key Insight |
|---------|---------|-----------------|-------|-------------|
| Fibonacci | Recursion + Memoization | O(n) | O(n) or O(1) | Overlapping subproblems |
| Generate Parentheses | Constrained Backtracking | O(4^n/sqrt(n)) | O(n) | Balance constraint pruning |
| Subsets | Exhaustive Backtracking | O(n * 2^n) | O(n) | Include/exclude each element |

### Common Backtracking Template

```python
def backtrack_template(candidates, target):
    """
    Universal backtracking template.
    """
    result = []

    def backtrack(state, choices):
        # Base case: found a valid solution
        if is_solution(state):
            result.append(format_solution(state))
            return

        # Pruning: abandon invalid paths early
        if not is_valid(state):
            return

        # Try each choice
        for choice in choices:
            # Make choice
            state.add(choice)

            # Recurse with updated state
            backtrack(state, get_next_choices(choice))

            # Undo choice (backtrack)
            state.remove(choice)

    backtrack(initial_state(), candidates)
    return result
```

---

## Interview Applications

### Real Interview Problems

1. **Phone Number Letter Combinations** (LeetCode 17) - Similar to Generate Parentheses
2. **Combination Sum** (LeetCode 39) - Subsets variation with target sum
3. **Permutations** (LeetCode 46) - Related to subsets but order matters
4. **N-Queens** (LeetCode 51) - Complex constrained backtracking
5. **Word Search** (LeetCode 79) - Grid-based backtracking

### Interview Tips

1. **Fibonacci**: Always discuss trade-offs between approaches
2. **Generate Parentheses**: Clearly explain the constraint pruning
3. **Subsets**: Know all three approaches and when to use each

### Common Follow-ups

| Problem | Follow-up Question |
|---------|-------------------|
| Fibonacci | "How would you handle very large n?" (Matrix exponentiation) |
| Parentheses | "What if we need to generate strings with multiple bracket types?" |
| Subsets | "How do you handle duplicates?" (Sort + skip) |

---

## Practice Problems

### Fibonacci Extensions
- [ ] [509. Fibonacci Number](https://leetcode.com/problems/fibonacci-number/) - Basic
- [ ] [70. Climbing Stairs](https://leetcode.com/problems/climbing-stairs/) - Fibonacci variant
- [ ] [1137. N-th Tribonacci Number](https://leetcode.com/problems/n-th-tribonacci-number/) - Three-term recurrence

### Parentheses Extensions
- [ ] [22. Generate Parentheses](https://leetcode.com/problems/generate-parentheses/) - Core problem
- [ ] [20. Valid Parentheses](https://leetcode.com/problems/valid-parentheses/) - Stack validation
- [ ] [32. Longest Valid Parentheses](https://leetcode.com/problems/longest-valid-parentheses/) - DP

### Subsets Extensions
- [ ] [78. Subsets](https://leetcode.com/problems/subsets/) - Core problem
- [ ] [90. Subsets II](https://leetcode.com/problems/subsets-ii/) - With duplicates
- [ ] [46. Permutations](https://leetcode.com/problems/permutations/) - Order matters
- [ ] [39. Combination Sum](https://leetcode.com/problems/combination-sum/) - With target

---

## References

- [LeetCode 22 - Generate Parentheses](https://leetcode.com/problems/generate-parentheses/)
- [LeetCode 78 - Subsets](https://leetcode.com/problems/subsets/)
- [AlgoMonster - Generate Parentheses In-Depth](https://algo.monster/liteproblems/22)
- [AlgoMonster - Subsets In-Depth](https://algo.monster/liteproblems/78)
- [AlgoMap - Generate Parentheses Solution](https://algomap.io/problems/generate-parentheses)
- [AlgoMap - Subsets Solution](https://algomap.io/problems/subsets)
- [LeetCode Patterns - Backtracking](https://medium.com/leetcode-patterns/leetcode-pattern-3-backtracking-5d9e5a03dc26)
