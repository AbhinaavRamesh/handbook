# Essential Patterns: The 10 Core Patterns for Rapid Prep

> **Master these patterns to solve 80% of coding interview problems**

---

## Overview

This guide covers the 10 most important algorithmic patterns that appear repeatedly in Google coding interviews. Rather than memorizing hundreds of solutions, focus on understanding these patterns deeply. Once mastered, you can apply them to solve new problems you have never seen before.

**Estimated Time:** 2-3 hours to review, ongoing practice to master

---

## Pattern Selection Flowchart

Use this decision tree when approaching a new problem:

```
START: Read the problem

Is input a sorted array or needs pair/triplet finding?
    YES --> Pattern 1: Two Pointers

Does it involve contiguous subarray/substring?
    YES --> Pattern 2: Sliding Window

Does it involve linked list or cycle detection?
    YES --> Pattern 3: Fast and Slow Pointers

Is it asking for merge or interval operations?
    YES --> Pattern 4: Merge Intervals

Can you eliminate half the search space each step?
    YES --> Pattern 5: Binary Search

Is it a tree problem?
    YES --> Pattern 6: Tree BFS/DFS

Is it a graph problem (nodes and edges)?
    YES --> Pattern 7: Graph BFS/DFS

Does it ask for all combinations/permutations?
    YES --> Pattern 8: Backtracking

Is it optimization (min/max) or counting ways?
    YES --> Pattern 9 or 10: Dynamic Programming

Otherwise --> Start with brute force, then optimize
```

---

## Pattern 1: Two Pointers

### When to Use
- Sorted arrays where you need to find pairs or triplets
- In-place array modifications
- Palindrome checking
- Merging sorted arrays

### Recognition Signals
- "Find two elements that sum to target"
- "Remove duplicates in-place"
- "Is this a palindrome?"
- "Merge two sorted..."

### Template: Opposite Direction

```python
def two_pointer_opposite(arr, target):
    """
    Use when: Finding pairs in a SORTED array
    Time: O(n), Space: O(1)
    """
    left, right = 0, len(arr) - 1

    while left < right:
        current = arr[left] + arr[right]

        if current == target:
            return [left, right]  # Found the pair
        elif current < target:
            left += 1   # Need larger sum
        else:
            right -= 1  # Need smaller sum

    return []  # No pair found
```

### Template: Same Direction

```python
def two_pointer_same(arr):
    """
    Use when: In-place modifications (remove duplicates, partition)
    Time: O(n), Space: O(1)
    """
    slow = 0  # Write pointer

    for fast in range(len(arr)):  # Read pointer
        if should_keep(arr[fast]):
            arr[slow] = arr[fast]
            slow += 1

    return slow  # New length
```

::: info Complexity: Time O(n) · Space O(1)
- **Time:** Each element visited at most once as pointers move
- **Space:** Only pointer variables used regardless of array size
:::

### Key Problems
| Problem | Variant | Key Insight |
|---------|---------|-------------|
| Two Sum II | Opposite | Sum comparison to move pointers |
| 3Sum | Opposite + Loop | Fix one, two-pointer for rest |
| Container With Most Water | Opposite | Move shorter height |
| Remove Duplicates | Same | Slow writes unique only |
| Valid Palindrome | Opposite | Skip non-alphanumeric |

---

## Pattern 2: Sliding Window

### When to Use
- Finding subarrays or substrings with certain properties
- Maximum/minimum of all subarrays of size k
- Longest/shortest substring with constraint

### Recognition Signals
- "Contiguous subarray..."
- "Substring with at most k..."
- "Maximum sum of size k..."
- "Smallest window containing..."

### Template: Fixed Window

```python
def fixed_window(arr, k):
    """
    Use when: Window size is fixed (k elements)
    Time: O(n), Space: O(1)
    """
    # Build initial window
    window_sum = sum(arr[:k])
    result = window_sum

    # Slide the window
    for i in range(k, len(arr)):
        window_sum += arr[i] - arr[i - k]  # Add right, remove left
        result = max(result, window_sum)

    return result
```

### Template: Variable Window

```python
def variable_window(arr, condition):
    """
    Use when: Finding min/max length with some condition
    Time: O(n), Space: O(1) or O(k) with hash
    """
    left = 0
    result = 0  # or float('inf') for minimum

    for right in range(len(arr)):
        # Expand: add arr[right] to window

        # Shrink: while window is invalid
        while not valid_window():
            # Remove arr[left] from window
            left += 1

        # Update result
        result = max(result, right - left + 1)

    return result
```

### Template: Sliding Window with HashMap

```python
def window_with_hashmap(s, k):
    """
    Use when: Tracking character frequencies
    Time: O(n), Space: O(k) where k = distinct chars
    """
    from collections import defaultdict

    char_count = defaultdict(int)
    left = 0
    result = 0

    for right in range(len(s)):
        char_count[s[right]] += 1

        # Shrink while window is invalid
        while len(char_count) > k:
            char_count[s[left]] -= 1
            if char_count[s[left]] == 0:
                del char_count[s[left]]
            left += 1

        result = max(result, right - left + 1)

    return result
```

::: info Complexity: Time O(n) · Space O(1) to O(k)
- **Time:** Each element added and removed from window at most once
- **Space:** O(1) for fixed window; O(k) for hashmap tracking k elements
:::

### Key Problems
| Problem | Type | Key Insight |
|---------|------|-------------|
| Max Sum Subarray of Size K | Fixed | Slide and track max |
| Minimum Size Subarray Sum | Variable | Shrink when sum >= target |
| Longest Substring Without Repeating | Variable | HashSet for uniqueness |
| Longest Substring with K Distinct | Variable | HashMap + count |
| Minimum Window Substring | Variable | Track "have" vs "need" |

---

## Pattern 3: Fast and Slow Pointers

### When to Use
- Detecting cycles in linked lists or arrays
- Finding the middle of a linked list
- Determining if a number is "happy"

### Recognition Signals
- "Detect cycle..."
- "Find middle element..."
- "Check if linked list has loop..."
- Numbers that form a cycle

### Template: Cycle Detection

```python
def has_cycle(head):
    """
    Floyd's Cycle Detection
    Time: O(n), Space: O(1)
    """
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

        if slow == fast:
            return True  # Cycle detected

    return False  # No cycle
```

### Template: Find Cycle Start

```python
def find_cycle_start(head):
    """
    After detecting cycle, find where it begins
    Time: O(n), Space: O(1)
    """
    slow = fast = head

    # Phase 1: Detect cycle
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            break
    else:
        return None  # No cycle

    # Phase 2: Find cycle start
    slow = head
    while slow != fast:
        slow = slow.next
        fast = fast.next

    return slow  # Start of cycle
```

### Template: Find Middle

```python
def find_middle(head):
    """
    Returns middle node (second middle if even length)
    Time: O(n), Space: O(1)
    """
    slow = fast = head

    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next

    return slow  # Middle node
```

::: info Complexity: Time O(n) · Space O(1)
- **Time:** Fast pointer traverses list once
- **Space:** Only two pointer variables used
:::

### Key Problems
| Problem | Variant | Key Insight |
|---------|---------|-------------|
| Linked List Cycle | Detection | Fast catches slow if cycle |
| Linked List Cycle II | Find start | Reset one pointer to head |
| Middle of Linked List | Find middle | Fast moves 2x speed |
| Happy Number | Number cycle | Treat digits as linked list |
| Palindrome Linked List | Middle + reverse | Find middle, reverse second half |

---

## Pattern 4: Merge Intervals

### When to Use
- Merging overlapping intervals
- Inserting into sorted intervals
- Meeting room scheduling

### Recognition Signals
- "Merge overlapping..."
- "Insert interval..."
- "Minimum meeting rooms..."
- Start and end times

### Template: Merge Overlapping

```python
def merge_intervals(intervals):
    """
    Merge all overlapping intervals
    Time: O(n log n), Space: O(n)
    """
    if not intervals:
        return []

    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for current in intervals[1:]:
        last = merged[-1]

        if current[0] <= last[1]:  # Overlapping
            last[1] = max(last[1], current[1])
        else:
            merged.append(current)

    return merged
```

### Template: Insert Interval

```python
def insert_interval(intervals, new):
    """
    Insert new interval and merge if necessary
    Time: O(n), Space: O(n)
    """
    result = []
    i = 0
    n = len(intervals)

    # Add all intervals that end before new starts
    while i < n and intervals[i][1] < new[0]:
        result.append(intervals[i])
        i += 1

    # Merge overlapping intervals
    while i < n and intervals[i][0] <= new[1]:
        new[0] = min(new[0], intervals[i][0])
        new[1] = max(new[1], intervals[i][1])
        i += 1
    result.append(new)

    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1

    return result
```

::: info Complexity: Time O(n log n) · Space O(n)
- **Time:** Dominated by sorting; merge/insert are O(n)
- **Space:** Result array stores up to n intervals
:::

### Key Problems
| Problem | Variant | Key Insight |
|---------|---------|-------------|
| Merge Intervals | Basic merge | Sort by start, extend end |
| Insert Interval | Insert + merge | Three phases: before, merge, after |
| Meeting Rooms | Check overlap | Sort and check adjacent |
| Meeting Rooms II | Count overlap | Min heap for end times |
| Non-overlapping Intervals | Remove minimum | Greedy - keep earlier end |

---

## Pattern 5: Binary Search Variations

### When to Use
- Searching in sorted arrays
- Finding boundaries (first/last occurrence)
- Search space reduction problems

### Recognition Signals
- Sorted array or matrix
- "Find position of..."
- O(log n) requirement
- Monotonic property

### Template: Classic Binary Search

```python
def binary_search(arr, target):
    """
    Standard binary search
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # Not found
```

### Template: Find Leftmost (First Occurrence)

```python
def find_leftmost(arr, target):
    """
    Find first occurrence of target
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            result = mid
            right = mid - 1  # Continue searching left
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result
```

### Template: Find Rightmost (Last Occurrence)

```python
def find_rightmost(arr, target):
    """
    Find last occurrence of target
    Time: O(log n), Space: O(1)
    """
    left, right = 0, len(arr) - 1
    result = -1

    while left <= right:
        mid = left + (right - left) // 2

        if arr[mid] == target:
            result = mid
            left = mid + 1  # Continue searching right
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return result
```

### Template: Search on Answer

```python
def binary_search_answer(low, high, condition):
    """
    Binary search on answer space
    Use when: Finding min/max that satisfies condition
    """
    while low < high:
        mid = low + (high - low) // 2

        if condition(mid):
            high = mid  # For minimum
            # low = mid + 1  # For maximum
        else:
            low = mid + 1  # For minimum
            # high = mid  # For maximum

    return low
```

::: info Complexity: Time O(log n) · Space O(1)
- **Time:** Halve search space each iteration
- **Space:** Only pointer variables used
:::

### Key Problems
| Problem | Variant | Key Insight |
|---------|---------|-------------|
| Binary Search | Classic | Standard template |
| Search Insert Position | Leftmost | Find insertion point |
| Search in Rotated Array | Modified | Find pivot, then search |
| Find Peak Element | Answer space | Compare with neighbors |
| Koko Eating Bananas | Answer space | Binary search on speed |

---

## Pattern 6: Tree BFS/DFS

### When to Use
- Level-order traversal
- Finding depth/height
- Path problems in trees
- Tree validation

### Recognition Signals
- Binary tree input
- "Level by level..."
- "Maximum depth..."
- "Path sum..."

### Template: BFS Level Order

```python
from collections import deque

def level_order_bfs(root):
    """
    Process tree level by level
    Time: O(n), Space: O(n)
    """
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        level_size = len(queue)

        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)

            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)

        result.append(level)

    return result
```

### Template: DFS Recursive

```python
def dfs_recursive(root):
    """
    DFS with recursion
    Time: O(n), Space: O(h) for call stack
    """
    def helper(node):
        if not node:
            return  # Base case

        # Preorder: process here
        helper(node.left)
        # Inorder: process here
        helper(node.right)
        # Postorder: process here

    helper(root)
```

### Template: DFS Iterative

```python
def dfs_iterative(root):
    """
    DFS with stack (preorder)
    Time: O(n), Space: O(h)
    """
    if not root:
        return []

    result = []
    stack = [root]

    while stack:
        node = stack.pop()
        result.append(node.val)

        # Right first so left is processed first
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)

    return result
```

::: info Complexity: Time O(n) · Space O(h) to O(n)
- **Time:** Visit each node exactly once
- **Space:** BFS uses O(n) for queue; DFS uses O(h) for stack/recursion
:::

### Key Problems
| Problem | Approach | Key Insight |
|---------|----------|-------------|
| Level Order Traversal | BFS | Queue per level |
| Maximum Depth | DFS | 1 + max(left, right) |
| Path Sum | DFS | Track remaining sum |
| Validate BST | DFS | Pass min/max bounds |
| Lowest Common Ancestor | DFS | Return when found |

---

## Pattern 7: Graph BFS/DFS

### When to Use
- Traversing graph nodes
- Finding shortest path (unweighted)
- Connected components
- Cycle detection

### Recognition Signals
- Grid-based problems (islands)
- Social network problems
- Shortest path (unweighted)
- "Connected components..."

### Template: BFS for Graphs

```python
from collections import deque

def bfs_graph(graph, start):
    """
    BFS from start node
    Time: O(V + E), Space: O(V)
    """
    visited = set([start])
    queue = deque([start])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return result
```

### Template: DFS for Graphs

```python
def dfs_graph(graph, start):
    """
    DFS from start node
    Time: O(V + E), Space: O(V)
    """
    visited = set()
    result = []

    def dfs(node):
        if node in visited:
            return
        visited.add(node)
        result.append(node)

        for neighbor in graph[node]:
            dfs(neighbor)

    dfs(start)
    return result
```

### Template: Grid BFS (Shortest Path)

```python
def bfs_grid(grid, start, end):
    """
    Shortest path in grid
    Time: O(m*n), Space: O(m*n)
    """
    rows, cols = len(grid), len(grid[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    queue = deque([(start[0], start[1], 0)])  # (row, col, distance)
    visited = set([start])

    while queue:
        r, c, dist = queue.popleft()

        if (r, c) == end:
            return dist

        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols and
                (nr, nc) not in visited and grid[nr][nc] != '#'):
                visited.add((nr, nc))
                queue.append((nr, nc, dist + 1))

    return -1  # No path found
```

::: info Complexity: Time O(V + E) or O(m * n) · Space O(V) or O(m * n)
- **Time:** Visit each vertex and edge once; grid problems are O(m*n)
- **Space:** Visited set stores all reachable nodes
:::

### Key Problems
| Problem | Approach | Key Insight |
|---------|----------|-------------|
| Number of Islands | DFS/BFS | Sink visited land |
| Clone Graph | BFS + HashMap | Map old to new nodes |
| Rotting Oranges | Multi-source BFS | All rotten oranges start |
| Course Schedule | Topological Sort | Check for cycle |
| Word Ladder | BFS | Each word is a node |

---

## Pattern 8: Backtracking

### When to Use
- Generate all combinations/permutations
- Solve constraint satisfaction problems
- Path finding with constraints
- Sudoku-like puzzles

### Recognition Signals
- "Find all possible..."
- "Generate all..."
- "All valid combinations..."
- "Solve the puzzle..."

### Template: Backtracking Framework

```python
def backtrack(candidates, path, result, start):
    """
    General backtracking template
    Time: O(2^n) or O(n!) depending on problem
    """
    # Base case: valid solution found
    if is_solution(path):
        result.append(path[:])  # Add copy of path
        return

    # Try all candidates
    for i in range(start, len(candidates)):
        # Pruning: skip invalid choices
        if not is_valid(candidates[i]):
            continue

        # Make choice
        path.append(candidates[i])

        # Explore further
        backtrack(candidates, path, result, i + 1)  # or i for reuse

        # Undo choice (backtrack)
        path.pop()
```

### Template: Subsets

```python
def subsets(nums):
    """
    Generate all subsets
    Time: O(n * 2^n), Space: O(n)
    """
    result = []

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result
```

### Template: Permutations

```python
def permutations(nums):
    """
    Generate all permutations
    Time: O(n * n!), Space: O(n)
    """
    result = []

    def backtrack(path, used):
        if len(path) == len(nums):
            result.append(path[:])
            return

        for i in range(len(nums)):
            if used[i]:
                continue
            used[i] = True
            path.append(nums[i])
            backtrack(path, used)
            path.pop()
            used[i] = False

    backtrack([], [False] * len(nums))
    return result
```

::: info Complexity: Time O(2^n) to O(n!) · Space O(n)
- **Time:** Subsets O(n*2^n); permutations O(n*n!)
- **Space:** Recursion depth and path storage proportional to n
:::

### Key Problems
| Problem | Variant | Key Insight |
|---------|---------|-------------|
| Subsets | Power set | Include or exclude each |
| Subsets II | With dups | Sort and skip duplicates |
| Permutations | All orders | Track used elements |
| Combination Sum | Target sum | Allow reuse (i, not i+1) |
| Word Search | Grid path | Mark visited, restore |
| N-Queens | Constraint | Check row, col, diagonals |

---

## Pattern 9: Dynamic Programming (1D)

### When to Use
- Optimization problems (min/max)
- Counting problems (number of ways)
- Decision problems (is it possible)
- Overlapping subproblems

### Recognition Signals
- "Minimum cost to..."
- "Maximum profit..."
- "Number of ways to..."
- "Can you reach..."

### Template: DP Framework

```python
def dp_template(input_data):
    """
    1. Define state: dp[i] represents...
    2. Find recurrence: dp[i] = f(dp[i-1], ...)
    3. Base cases: dp[0] = ...
    4. Build solution: iterate and fill
    5. Return answer: dp[n]
    """
    n = len(input_data)
    dp = [0] * (n + 1)

    # Base case
    dp[0] = base_value

    # Fill DP table
    for i in range(1, n + 1):
        dp[i] = recurrence(dp[i-1], input_data[i-1])

    return dp[n]
```

### Template: Space Optimized

```python
def dp_space_optimized(nums):
    """
    When dp[i] depends only on dp[i-1], dp[i-2]
    """
    if len(nums) <= 2:
        return base_case

    prev2, prev1 = initial_values

    for i in range(2, len(nums)):
        curr = compute(prev1, prev2, nums[i])
        prev2, prev1 = prev1, curr

    return prev1
```

### Common 1D DP Patterns

```python
# Fibonacci-like
# dp[i] = dp[i-1] + dp[i-2]
def climbing_stairs(n):
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1

# Max Non-Adjacent (House Robber)
# dp[i] = max(dp[i-1], dp[i-2] + nums[i])
def house_robber(nums):
    prev2, prev1 = 0, 0
    for num in nums:
        curr = max(prev1, prev2 + num)
        prev2, prev1 = prev1, curr
    return prev1

# Unbounded Knapsack (Coin Change)
# dp[amount] = min(dp[amount], dp[amount - coin] + 1)
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1
```

::: info Complexity: Time O(n) to O(n * m) · Space O(n) or O(1)
- **Time:** Linear for Fibonacci-like; O(n*m) for coin change
- **Space:** O(n) for dp array; O(1) when space-optimized
:::

### Key Problems
| Problem | Pattern | State Definition |
|---------|---------|------------------|
| Climbing Stairs | Fibonacci | dp[i] = ways to reach step i |
| House Robber | Max non-adjacent | dp[i] = max money up to i |
| Coin Change | Unbounded knapsack | dp[i] = min coins for amount i |
| LIS | Subsequence | dp[i] = LIS ending at i |
| Word Break | String segmentation | dp[i] = can segment s[0:i] |

---

## Pattern 10: Dynamic Programming (2D)

### When to Use
- Two sequence problems (LCS, Edit Distance)
- Grid traversal with optimization
- Interval DP
- 0/1 Knapsack

### Recognition Signals
- Two strings/sequences
- Grid with costs/paths
- "Transform X into Y..."
- Select items with constraints

### Template: 2D DP

```python
def dp_2d_template(seq1, seq2):
    """
    State: dp[i][j] depends on previous states
    """
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = base_case_row
    for j in range(n + 1):
        dp[0][j] = base_case_col

    # Fill table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if condition(seq1[i-1], seq2[j-1]):
                dp[i][j] = dp[i-1][j-1] + value
            else:
                dp[i][j] = func(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

    return dp[m][n]
```

### Common 2D DP Patterns

```python
# Longest Common Subsequence
def lcs(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

# Edit Distance
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j],      # Delete
                                   dp[i][j-1],      # Insert
                                   dp[i-1][j-1])    # Replace

    return dp[m][n]

# Unique Paths (Grid)
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]

    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]

    return dp[m-1][n-1]

# 0/1 Knapsack
def knapsack(weights, values, capacity):
    n = len(weights)
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for w in range(capacity + 1):
            dp[i][w] = dp[i-1][w]  # Don't take
            if weights[i-1] <= w:
                dp[i][w] = max(dp[i][w],
                               dp[i-1][w - weights[i-1]] + values[i-1])

    return dp[n][capacity]
```

::: info Complexity: Time O(m * n) · Space O(m * n) or O(n)
- **Time:** Fill m*n table for two-sequence problems
- **Space:** O(m*n) for 2D table; can reduce to O(n) with space optimization
:::

### Key Problems
| Problem | Pattern | State Definition |
|---------|---------|------------------|
| LCS | Two sequences | dp[i][j] = LCS of s1[0:i], s2[0:j] |
| Edit Distance | Transformation | dp[i][j] = min ops for s1[0:i] to s2[0:j] |
| Unique Paths | Grid | dp[i][j] = paths to (i,j) |
| 0/1 Knapsack | Item selection | dp[i][w] = max value with items 0..i-1, capacity w |
| Longest Palindromic Subsequence | Single sequence | dp[i][j] = LPS of s[i:j+1] |

---

## Quick Reference Card

```
PATTERN CHEAT SHEET
-------------------

Two Pointers: left, right = 0, len-1; move based on condition
Sliding Window: expand right, shrink left when invalid
Fast/Slow: slow = slow.next, fast = fast.next.next
Binary Search: left, right, mid; eliminate half each time
Tree DFS: base case (null), recurse left/right
Tree BFS: queue, process level by level
Graph DFS: visited set, explore all neighbors
Graph BFS: queue + visited, shortest path
Backtracking: choose, explore, unchoose
DP 1D: dp[i] = f(dp[i-1], dp[i-2], ...)
DP 2D: dp[i][j] = f(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])

COMPLEXITY QUICK REFERENCE
--------------------------
Two Pointers: O(n) time, O(1) space
Sliding Window: O(n) time, O(k) space
Binary Search: O(log n) time, O(1) space
BFS/DFS Tree: O(n) time, O(h) space
BFS/DFS Graph: O(V+E) time, O(V) space
Backtracking: O(2^n) or O(n!) time
DP: O(n) or O(n*m) time, reducible space
```

---

## Pattern Mastery Checklist

Track your comfort level with each pattern (1-5):

```
[ ] Pattern 1: Two Pointers          ___/5
[ ] Pattern 2: Sliding Window        ___/5
[ ] Pattern 3: Fast and Slow         ___/5
[ ] Pattern 4: Merge Intervals       ___/5
[ ] Pattern 5: Binary Search         ___/5
[ ] Pattern 6: Tree BFS/DFS          ___/5
[ ] Pattern 7: Graph BFS/DFS         ___/5
[ ] Pattern 8: Backtracking          ___/5
[ ] Pattern 9: DP (1D)               ___/5
[ ] Pattern 10: DP (2D)              ___/5

Total: ___/50

Score Interpretation:
40-50: Interview ready
30-39: Good foundation, need more practice
20-29: Focus on weak patterns
<20:   Start from Week 1 basics
```

---

*Practice these patterns with the [Priority Problems](./priority-problems) list to solidify your understanding.*

---

*Last updated: January 2026*
