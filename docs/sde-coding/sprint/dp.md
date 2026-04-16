# Tier 4: Dynamic Programming

::: tip PRIORITY: MEDIUM
1 DP question per interview loop is typical. You do not need to be a DP wizard -- you need to identify subproblems and state transitions clearly.
:::

---

## DP in Interviews

SDE interviewers test DP differently than most people practice it. They are not looking for you to memorize solutions. They want to see you:

1. **Identify that the problem has optimal substructure** -- can the answer be built from answers to smaller subproblems?
2. **Define the state clearly** -- what does `dp[i]` (or `dp[i][j]`) represent? Say this out loud before writing any code.
3. **Derive the recurrence** -- how does `dp[i]` relate to previous states? This is where the interview is won or lost.
4. **Handle base cases** -- what are the trivial subproblems?
5. **Optimize space** -- can you reduce from O(n^2) to O(n), or O(n) to O(1)?

::: warning INTERVIEW REALITY
DP solutions tend to be short (10-20 lines). The challenge is not writing the code -- it is explaining your reasoning. Spend more time talking through your state definition and recurrence than coding. The interviewer wants to hear your thought process.
:::

---

## Problem Table

| # | Problem | LC # | Pattern | Why |
|---|---------|------|---------|-----|
| 19 | [Coin Change](https://leetcode.com/problems/coin-change/) | 322 | 1D DP | Classic unbounded knapsack variant |
| 20 | [Longest Increasing Subsequence](https://leetcode.com/problems/longest-increasing-subsequence/) | 300 | 1D DP + binary search | O(n log n) variant is impressive |
| 21 | [House Robber](https://leetcode.com/problems/house-robber/) | 198 | 1D DP | Best warm-up problem |
| 22 | [Unique Paths](https://leetcode.com/problems/unique-paths/) | 62 | 2D DP | Grid DP |
| 23 | [Longest Increasing Path in a Matrix](https://leetcode.com/problems/longest-increasing-path-in-a-matrix/) | 329 | DFS + memo | Frequently asked in SDE interviews |

---

## The 5-Step DP Framework

Use this framework for every DP problem. Say each step out loud in your interview.

### Step 1: Define the State

> _"dp[i] represents..."_

This is the most important step. If you define the state correctly, the rest follows naturally. Ask yourself: what is the subproblem at position `i`?

| Problem | State Definition |
|---------|-----------------|
| Coin Change | `dp[amount]` = min coins to make `amount` |
| LIS | `dp[i]` = length of LIS ending at index `i` |
| House Robber | `dp[i]` = max money robbing houses `0..i` |
| Unique Paths | `dp[i][j]` = number of paths to reach cell `(i, j)` |
| Longest Path in Matrix | `memo[i][j]` = longest increasing path starting at `(i, j)` |

### Step 2: Find the Recurrence Relation

> _"dp[i] depends on dp[...] because..."_

This is where you connect the current subproblem to smaller ones.

| Problem | Recurrence |
|---------|-----------|
| Coin Change | `dp[a] = min(dp[a], dp[a - coin] + 1)` for each coin |
| LIS | `dp[i] = max(dp[j] + 1)` for all `j < i` where `nums[j] < nums[i]` |
| House Robber | `dp[i] = max(dp[i-1], dp[i-2] + nums[i])` |
| Unique Paths | `dp[i][j] = dp[i-1][j] + dp[i][j-1]` |
| Longest Path | `memo[i][j] = 1 + max(dfs(ni, nj))` for valid increasing neighbors |

### Step 3: Identify Base Cases

| Problem | Base Cases |
|---------|-----------|
| Coin Change | `dp[0] = 0` (0 coins for amount 0) |
| LIS | `dp[i] = 1` for all i (each element is a subsequence of length 1) |
| House Robber | `dp[0] = nums[0]`, `dp[1] = max(nums[0], nums[1])` |
| Unique Paths | `dp[0][j] = 1`, `dp[i][0] = 1` (only one way along edges) |
| Longest Path | A cell with no valid increasing neighbor returns 1 |

### Step 4: Determine Iteration Order

- **Bottom-up (tabulation):** Fill the table from smaller subproblems to larger. For Coin Change, iterate amounts from 1 to target. For 2D grid DP, iterate top-to-bottom, left-to-right.
- **Top-down (memoization):** Start from the final answer and recurse down. Best for problems where the iteration order is not obvious (like Longest Path in Matrix).

### Step 5: Optimize Space

| Problem | Before | After |
|---------|--------|-------|
| Coin Change | Already O(amount) | -- |
| LIS | O(n) | Cannot reduce (need all previous) |
| House Robber | O(n) array | O(1) with two variables |
| Unique Paths | O(m * n) grid | O(n) single row |
| Longest Path | O(m * n) memo | Cannot reduce (need random access) |

---

## Key DP Patterns

### Pattern 1: 1D DP (Linear)

The simplest pattern. `dp[i]` depends on one or more previous elements in a 1D array.

::: code-group

```python [Python]
# General 1D DP template
def solve(nums):
    n = len(nums)
    dp = [0] * n
    dp[0] = BASE_CASE

    for i in range(1, n):
        dp[i] = RECURRENCE(dp, i)

    return dp[n - 1]  # or max(dp), etc.
```

```java [Java]
// General 1D DP scaffold
int solve(int[] nums) {
    int n = nums.length;
    int[] dp = new int[n];
    dp[0] = 0; // BASE_CASE

    for (int i = 1; i < n; i++) {
        dp[i] = 0; // RECURRENCE(dp, i)
    }
    return dp[n - 1]; // or Arrays.stream(dp).max().getAsInt()
}
```

:::

**Problems that use this:** Coin Change, House Robber, LIS (O(n^2) version).

### Pattern 2: 2D Grid DP

`dp[i][j]` depends on neighboring cells, typically `dp[i-1][j]` and `dp[i][j-1]`.

::: code-group

```python [Python]
# General 2D Grid DP template
def solve(m, n):
    dp = [[0] * n for _ in range(m)]

    # base cases: first row and first column
    for j in range(n):
        dp[0][j] = BASE_ROW
    for i in range(m):
        dp[i][0] = BASE_COL

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = RECURRENCE(dp, i, j)

    return dp[m - 1][n - 1]
```

```java [Java]
// General 2D Grid DP scaffold
int solve(int m, int n) {
    int[][] dp = new int[m][n];

    // base cases: first row and first column
    for (int j = 0; j < n; j++) dp[0][j] = 0; // BASE_ROW
    for (int i = 0; i < m; i++) dp[i][0] = 0; // BASE_COL

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = 0; // RECURRENCE(dp, i, j)
        }
    }
    return dp[m - 1][n - 1];
}
```

:::

**Problems that use this:** Unique Paths, Edit Distance, Minimum Path Sum.

### Pattern 3: DFS + Memoization (Top-Down)

Best for problems where the subproblem structure is a DAG (like a grid where you can only move to strictly increasing neighbors).

::: code-group

```python [Python]
# General DFS + Memo template
from functools import lru_cache

def solve(matrix):
    m, n = len(matrix), len(matrix[0])

    @lru_cache(maxsize=None)
    def dfs(i, j):
        result = BASE_CASE
        for ni, nj in NEIGHBORS(i, j):
            if VALID(ni, nj):
                result = COMBINE(result, dfs(ni, nj))
        return result

    # try every starting point
    return max(dfs(i, j) for i in range(m) for j in range(n))
```

```java [Java]
// General DFS + Memo scaffold
int solve(int[][] matrix) {
    int m = matrix.length, n = matrix[0].length;
    int[][] memo = new int[m][n];
    // fill with -1 to denote "not computed yet"
    for (int[] row : memo) Arrays.fill(row, -1);

    int best = 0;
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            best = Math.max(best, dfs(matrix, memo, i, j));
    return best;
}

private int dfs(int[][] matrix, int[][] memo, int i, int j) {
    if (memo[i][j] != -1) return memo[i][j];
    int result = 1; // BASE_CASE
    int[][] dirs = {{0,1},{0,-1},{1,0},{-1,0}};
    for (int[] d : dirs) {
        int ni = i + d[0], nj = j + d[1];
        if (ni >= 0 && ni < matrix.length && nj >= 0 && nj < matrix[0].length
                && /* VALID condition */ true) {
            result = Math.max(result, 1 + dfs(matrix, memo, ni, nj));
        }
    }
    return memo[i][j] = result;
}
```

:::

**Problems that use this:** Longest Increasing Path in Matrix, Word Break (top-down variant).

::: info WHEN TO USE TOP-DOWN VS BOTTOM-UP
Use **top-down** (memoization) when: the iteration order is unclear, or not all subproblems need to be solved.
Use **bottom-up** (tabulation) when: the iteration order is obvious, and you want cleaner code with no recursion stack overhead.
In interviews, both are acceptable. Use whichever you can explain more clearly.
:::

---

## Problem Walkthroughs

::: details #19 -- Coin Change (LC 322)

**Problem:** Given coin denominations and a target amount, find the minimum number of coins to make that amount. Return -1 if impossible.

**Key insight:** This is unbounded knapsack -- you can reuse coins. `dp[amount]` = minimum coins to make `amount`. For each amount, try every coin and take the minimum.

**State:** `dp[a]` = minimum number of coins to make amount `a`
**Recurrence:** `dp[a] = min(dp[a - coin] + 1)` for each coin where `a - coin >= 0`
**Base case:** `dp[0] = 0`

::: code-group

```python [Python]
def coinChange(self, coins: list[int], amount: int) -> int:
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for a in range(1, amount + 1):
        for coin in coins:
            if coin <= a and dp[a - coin] != float('inf'):
                dp[a] = min(dp[a], dp[a - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1
```

```java [Java]
int coinChange(int[] coins, int amount) {
    int[] dp = new int[amount + 1];
    Arrays.fill(dp, amount + 1); // sentinel for "impossible"
    dp[0] = 0;

    for (int a = 1; a <= amount; a++) {
        for (int coin : coins) {
            if (coin <= a) {
                dp[a] = Math.min(dp[a], dp[a - coin] + 1);
            }
        }
    }
    return dp[amount] > amount ? -1 : dp[amount];
}
```

:::

| | Complexity |
|---|---|
| **Time** | O(amount * len(coins)) |
| **Space** | O(amount) |

![Coin Change DP Table](/sde-coding/sprint/dp_coin_change.png)

::: tip INTERVIEW TIP
State the brute force first: _"We could try all combinations recursively -- that is exponential. DP eliminates redundant subproblems by caching dp[amount]."_
:::

:::

::: details #20 -- Longest Increasing Subsequence (LC 300)

**Problem:** Given an array `nums`, find the length of the longest strictly increasing subsequence.

**Key insight:** There are two approaches. The O(n^2) DP solution is straightforward. The O(n log n) patience sorting approach is what impresses interviewers.

### Approach 1: O(n^2) DP

**State:** `dp[i]` = length of LIS ending at index `i`
**Recurrence:** `dp[i] = max(dp[j] + 1)` for all `j < i` where `nums[j] < nums[i]`
**Base case:** `dp[i] = 1` for all i

::: code-group

```python [Python]
def lengthOfLIS(self, nums: list[int]) -> int:
    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)
```

```java [Java]
int lengthOfLIS(int[] nums) {
    int n = nums.length;
    int[] dp = new int[n];
    Arrays.fill(dp, 1);
    int result = 1;

    for (int i = 1; i < n; i++) {
        for (int j = 0; j < i; j++) {
            if (nums[j] < nums[i]) {
                dp[i] = Math.max(dp[i], dp[j] + 1);
            }
        }
        result = Math.max(result, dp[i]);
    }
    return result;
}
```

:::

| | Complexity |
|---|---|
| **Time** | O(n^2) |
| **Space** | O(n) |

### Approach 2: O(n log n) with Binary Search (Patience Sorting)

**Key insight:** Maintain an array `tails` where `tails[i]` is the smallest tail element for an increasing subsequence of length `i + 1`. For each number, use binary search to find where it fits.

- If `num` is larger than all tails, extend the longest subsequence.
- Otherwise, replace the first tail that is >= `num` (this keeps tails as small as possible for future extensions).

::: code-group

```python [Python]
import bisect

def lengthOfLIS(self, nums: list[int]) -> int:
    tails = []

    for num in nums:
        pos = bisect.bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num

    return len(tails)
```

```java [Java]
int lengthOfLISOptimized(int[] nums) {
    List<Integer> tails = new ArrayList<>();

    for (int num : nums) {
        // binary search for first tail >= num
        int lo = 0, hi = tails.size();
        while (lo < hi) {
            int mid = lo + (hi - lo) / 2;
            if (tails.get(mid) < num) lo = mid + 1;
            else hi = mid;
        }
        if (lo == tails.size()) tails.add(num);
        else tails.set(lo, num);
    }
    return tails.size();
}
```

:::

| | Complexity |
|---|---|
| **Time** | O(n log n) -- binary search for each of n elements |
| **Space** | O(n) |

::: tip WHY THIS IMPRESSES
Most candidates only know the O(n^2) solution. Presenting the O(n log n) solution and explaining the patience sorting analogy shows depth. Say: _"We maintain the smallest possible tail for each subsequence length, allowing binary search to find the right position."_
:::

**Walkthrough of `tails` for `[10, 9, 2, 5, 3, 7, 101, 18]`:**

```
num=10:  tails=[10]
num=9:   tails=[9]         (replace 10)
num=2:   tails=[2]         (replace 9)
num=5:   tails=[2, 5]      (extend)
num=3:   tails=[2, 3]      (replace 5)
num=7:   tails=[2, 3, 7]   (extend)
num=101: tails=[2, 3, 7, 101] (extend)
num=18:  tails=[2, 3, 7, 18]  (replace 101)
Answer: len(tails) = 4
```

Note: `tails` is NOT the actual LIS -- it is a structure that tracks the length correctly. The actual LIS in this case is `[2, 3, 7, 101]` or `[2, 3, 7, 18]`.

:::

::: details #21 -- House Robber (LC 198)

**Problem:** Given an array where each element represents the money in a house, find the maximum money you can rob without robbing two adjacent houses.

**Key insight:** At each house, you have two choices: rob it (add its value to the best you could do two houses ago) or skip it (take the best from the previous house). This gives a clean two-variable recurrence.

**State:** `dp[i]` = max money from houses `0..i`
**Recurrence:** `dp[i] = max(dp[i-1], dp[i-2] + nums[i])`
**Base cases:** `dp[0] = nums[0]`, `dp[1] = max(nums[0], nums[1])`

::: code-group

```python [Python]
def rob(self, nums: list[int]) -> int:
    if len(nums) == 1:
        return nums[0]

    n = len(nums)
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])

    for i in range(2, n):
        dp[i] = max(dp[i - 1], dp[i - 2] + nums[i])

    return dp[n - 1]
```

```java [Java]
int rob(int[] nums) {
    if (nums.length == 1) return nums[0];

    int n = nums.length;
    int[] dp = new int[n];
    dp[0] = nums[0];
    dp[1] = Math.max(nums[0], nums[1]);

    for (int i = 2; i < n; i++) {
        dp[i] = Math.max(dp[i - 1], dp[i - 2] + nums[i]);
    }
    return dp[n - 1];
}
```

:::

**Space-optimized to O(1):**

::: code-group

```python [Python]
def rob(self, nums: list[int]) -> int:
    if len(nums) == 1:
        return nums[0]

    prev2 = nums[0]
    prev1 = max(nums[0], nums[1])

    for i in range(2, len(nums)):
        current = max(prev1, prev2 + nums[i])
        prev2 = prev1
        prev1 = current

    return prev1
```

```java [Java]
int robOptimized(int[] nums) {
    if (nums.length == 1) return nums[0];

    int prev2 = nums[0];
    int prev1 = Math.max(nums[0], nums[1]);

    for (int i = 2; i < nums.length; i++) {
        int current = Math.max(prev1, prev2 + nums[i]);
        prev2 = prev1;
        prev1 = current;
    }
    return prev1;
}
```

:::

| | Complexity |
|---|---|
| **Time** | O(n) |
| **Space** | O(1) with space optimization |

::: tip WARM-UP PROBLEM
Use this as your first DP practice. If you can derive the recurrence, base cases, and space optimization here, the framework transfers directly to harder problems.
:::

:::

::: details #22 -- Unique Paths (LC 62)

**Problem:** A robot starts at the top-left corner of an `m x n` grid and can only move right or down. Find the number of unique paths to the bottom-right corner.

**Key insight:** Each cell can only be reached from the cell above or the cell to the left. The number of paths to any cell is the sum of paths to those two cells.

**State:** `dp[i][j]` = number of unique paths to reach cell `(i, j)`
**Recurrence:** `dp[i][j] = dp[i-1][j] + dp[i][j-1]`
**Base cases:** `dp[0][j] = 1` (first row), `dp[i][0] = 1` (first column) -- only one way to reach any cell along the edges.

::: code-group

```python [Python]
def uniquePaths(self, m: int, n: int) -> int:
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1]

    return dp[m - 1][n - 1]
```

```java [Java]
int uniquePaths(int m, int n) {
    int[][] dp = new int[m][n];
    for (int[] row : dp) Arrays.fill(row, 1); // first row and col are all 1

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
        }
    }
    return dp[m - 1][n - 1];
}
```

:::

![Unique Paths Grid DP](/sde-coding/sprint/dp_unique_paths.png)

**Space-optimized to O(n):**

Since each row only depends on the current row and the row above, we can use a single row.

::: code-group

```python [Python]
def uniquePaths(self, m: int, n: int) -> int:
    row = [1] * n

    for i in range(1, m):
        for j in range(1, n):
            row[j] += row[j - 1]

    return row[n - 1]
```

```java [Java]
int uniquePathsOptimized(int m, int n) {
    int[] row = new int[n];
    Arrays.fill(row, 1);

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            row[j] += row[j - 1];
        }
    }
    return row[n - 1];
}
```

:::

| | Complexity |
|---|---|
| **Time** | O(m * n) |
| **Space** | O(n) with space optimization |

::: tip FOLLOW-UP AWARENESS
The interviewer may ask: _"What if some cells are blocked?"_ (Unique Paths II, LC 63). The answer: set `dp[i][j] = 0` for obstacle cells. Mention this proactively to show depth.
:::

:::

::: details #23 -- Longest Increasing Path in a Matrix (LC 329)

**Problem:** Given an `m x n` matrix of integers, find the length of the longest increasing path. You can move in 4 directions (up, down, left, right).

**Key insight:** This is DFS + memoization, not bottom-up DP, because the iteration order depends on the values in the matrix (you cannot simply iterate top-to-bottom). Each cell's longest path depends on its neighbors with strictly smaller values, forming a DAG.

**Why this is frequently asked in SDE interviews:** It combines graph traversal (DFS on a grid) with DP (memoization), two of the most common interview topics.

**State:** `memo[i][j]` = length of the longest increasing path starting at cell `(i, j)`
**Recurrence:** `memo[i][j] = 1 + max(dfs(ni, nj))` for each neighbor `(ni, nj)` where `matrix[ni][nj] > matrix[i][j]`
**Base case:** A cell with no valid increasing neighbor returns 1 (just itself).

::: code-group

```python [Python]
def longestIncreasingPath(self, matrix: list[list[int]]) -> int:
    if not matrix:
        return 0

    m, n = len(matrix), len(matrix[0])
    memo = {}

    def dfs(i, j):
        if (i, j) in memo:
            return memo[(i, j)]

        best = 1
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                best = max(best, 1 + dfs(ni, nj))

        memo[(i, j)] = best
        return best

    return max(dfs(i, j) for i in range(m) for j in range(n))
```

```java [Java]
int longestIncreasingPath(int[][] matrix) {
    if (matrix == null || matrix.length == 0) return 0;
    int m = matrix.length, n = matrix[0].length;
    int[][] memo = new int[m][n];
    int result = 0;

    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++)
            result = Math.max(result, lipDFS(matrix, memo, i, j));
    return result;
}

private int lipDFS(int[][] matrix, int[][] memo, int i, int j) {
    if (memo[i][j] != 0) return memo[i][j];
    int best = 1;
    int[][] dirs = {{0,1},{0,-1},{1,0},{-1,0}};
    for (int[] d : dirs) {
        int ni = i + d[0], nj = j + d[1];
        if (ni >= 0 && ni < matrix.length && nj >= 0 && nj < matrix[0].length
                && matrix[ni][nj] > matrix[i][j]) {
            best = Math.max(best, 1 + lipDFS(matrix, memo, ni, nj));
        }
    }
    return memo[i][j] = best;
}
```

:::

**Alternative using `@lru_cache`** (cleaner for interviews):

::: code-group

```python [Python]
from functools import lru_cache

def longestIncreasingPath(self, matrix: list[list[int]]) -> int:
    if not matrix:
        return 0

    m, n = len(matrix), len(matrix[0])

    @lru_cache(maxsize=None)
    def dfs(i, j):
        best = 1
        for di, dj in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            ni, nj = i + di, j + dj
            if 0 <= ni < m and 0 <= nj < n and matrix[ni][nj] > matrix[i][j]:
                best = max(best, 1 + dfs(ni, nj))
        return best

    return max(dfs(i, j) for i in range(m) for j in range(n))
```

```java [Java]
// Java equivalent: same lipDFS approach above — Java has no lru_cache decorator.
// Use the memo array approach as the canonical Java solution.
// (See implementation above.)
```

:::

| | Complexity |
|---|---|
| **Time** | O(m * n) -- each cell computed at most once |
| **Space** | O(m * n) -- memoization table + recursion stack |

::: tip WHY NOT BOTTOM-UP?
A candidate might ask: _"Can this be done bottom-up?"_ Yes -- sort all cells by value, then iterate in increasing order, computing the longest path for each cell using already-computed neighbors. But the DFS + memo approach is more natural and easier to write correctly in an interview.
:::

:::

---

## Common Mistakes

::: danger MISTAKES THAT COST OFFERS

**1. Jumping to code before defining the state.**
In an interview, the interviewer wants to hear: _"dp[i] represents the minimum coins to make amount i."_ If you start coding without saying this, you lose communication points even if your code is correct.

**2. Wrong base cases.**
The most common DP bug. For Coin Change, forgetting `dp[0] = 0` leads to all values staying at infinity. For House Robber, forgetting to handle `len(nums) == 1` causes an index error. Always enumerate base cases explicitly.

**3. Wrong iteration order.**
For bottom-up DP, you must fill smaller subproblems before larger ones. For Coin Change, iterate `a` from 1 to amount (not the reverse). For 2D grid DP, iterate top-to-bottom, left-to-right.

**4. Not considering space optimization.**
After presenting a working solution, say: _"We can optimize space from O(n) to O(1) because dp[i] only depends on dp[i-1] and dp[i-2]."_ This shows maturity and is often a follow-up question.

![Space Complexity Visualization](/sde-coding/sprint/space_complexity.png)

**5. Using DFS without memoization.**
For Longest Increasing Path in Matrix, plain DFS is exponential. The memoization is what makes it O(m * n). Forgetting `memo` is a critical bug that the interviewer will catch.

**6. Confusing subsequence with subarray.**
A **subsequence** does not need to be contiguous (`[2, 3, 7, 101]` from `[10, 9, 2, 5, 3, 7, 101, 18]`). A **subarray** must be contiguous. LIS is a subsequence problem -- do not assume elements are adjacent.

**7. Off-by-one in 2D DP grid dimensions.**
For Unique Paths, the grid is `m x n` but indices go from `0` to `m-1` and `0` to `n-1`. Initialize the DP table with the right dimensions and be careful with loop bounds.
:::

---

## Practice Checklist

Use this to track your progress. Practice in a plain text editor (no IDE) to simulate interview conditions.

- [ ] **#21 House Robber** -- Solve this first as a warm-up. Derive the recurrence on paper before coding. Then optimize to O(1) space.
- [ ] **#19 Coin Change** -- Solve bottom-up. Explain to yourself: _"Why do I initialize dp with infinity? Why is dp[0] = 0?"_
- [ ] **#22 Unique Paths** -- Solve with full 2D array first, then optimize to a single row. Can you explain the space optimization to a teammate?
- [ ] **#20 Longest Increasing Subsequence** -- Solve the O(n^2) version first. Then implement the O(n log n) patience sorting version. Trace through `tails` for `[10, 9, 2, 5, 3, 7, 101, 18]` on paper.
- [ ] **#23 Longest Increasing Path in Matrix** -- Solve with DFS + memo. Explain why bottom-up is harder here. Can you identify the DAG structure?
- [ ] **For each problem, practice the 5-step framework out loud:**
  1. Define the state
  2. State the recurrence
  3. List base cases
  4. Explain iteration order
  5. Describe space optimization
- [ ] **Time yourself** -- Can you explain (2 min) + code (12 min) + dry run (3 min) each problem in under 20 minutes?

::: tip NEXT UP
If you have finished DP, move on to [Tier 5: HashMap / Heap / Binary Search](./data-structures). If you are running low on time, stop here and spend remaining hours doing timed mocks in a plain text editor.
:::
