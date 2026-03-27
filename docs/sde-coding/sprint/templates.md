# Code Templates --- Your Interview Arsenal

Everything you need, nothing you don't. Memorize these templates and you can solve 80%+ of SDE coding questions by pattern-matching.

## Which Pattern Should I Use?

```mermaid
flowchart TD
    START(["What type of problem?"])

    START --> Q1{"Shortest path /<br>minimum steps?"}
    START --> Q2{"Dependencies /<br>ordering?"}
    START --> Q3{"Connected components /<br>islands?"}
    START --> Q4{"Subarray / substring<br>with condition?"}
    START --> Q5{"Find target in<br>sorted data?"}
    START --> Q6{"Top K / Kth<br>element?"}
    START --> Q7{"Grouping / counting<br>/ lookup?"}
    START --> Q8{"Tree<br>problem?"}
    START --> Q9{"All combinations /<br>permutations?"}
    START --> Q10{"Optimal substructure<br>+ overlapping?"}
    START --> Q11{"Prefix<br>matching?"}

    Q1 -->|Yes| Q1a{"Weighted<br>graph?"}
    Q1a -->|Yes| A1["Dijkstra's Algorithm"]
    Q1a -->|No| A2["BFS"]

    Q2 -->|Yes| A3["Topological Sort<br>Kahn's BFS"]

    Q3 -->|Yes| A4["DFS / Union-Find"]

    Q4 -->|Yes| Q4a{"Fixed<br>size?"}
    Q4a -->|Yes| A5["Fixed Sliding Window"]
    Q4a -->|No| A6["Variable Sliding Window"]

    Q5 -->|Yes| A7["Binary Search"]

    Q6 -->|Yes| A8["Heap — heapq"]

    Q7 -->|Yes| A9["HashMap"]

    Q8 -->|Yes| Q8a{"Need level<br>info?"}
    Q8a -->|Yes| A10["BFS — level-order"]
    Q8a -->|No| A11["DFS — recursion"]

    Q9 -->|Yes| A12["Backtracking"]

    Q10 -->|Yes| A13["Dynamic Programming"]

    Q11 -->|Yes| A14["Trie"]

    %% Styling
    classDef question fill:#313244,stroke:#585b70,color:#cdd6f4
    classDef graphAns fill:#6c2f4a,stroke:#f38ba8,color:#f38ba8,font-weight:bold
    classDef treeAns fill:#6c4a2e,stroke:#fab387,color:#fab387,font-weight:bold
    classDef windowAns fill:#5c5530,stroke:#f9e2af,color:#f9e2af,font-weight:bold
    classDef dpAns fill:#3a5c42,stroke:#a6e3a1,color:#a6e3a1,font-weight:bold
    classDef dsAns fill:#2e4a6c,stroke:#89b4fa,color:#89b4fa,font-weight:bold
    classDef stringAns fill:#4a2e6c,stroke:#cba6f7,color:#cba6f7,font-weight:bold
    classDef searchAns fill:#2e4a5c,stroke:#89dceb,color:#89dceb,font-weight:bold
    classDef btAns fill:#5c2e5c,stroke:#f5c2e7,color:#f5c2e7,font-weight:bold
    classDef startNode fill:#585b70,stroke:#cdd6f4,color:#cdd6f4,font-weight:bold

    class START startNode
    class Q1,Q1a,Q2,Q3,Q4,Q4a,Q5,Q6,Q7,Q8,Q8a,Q9,Q10,Q11 question
    class A1,A2,A3,A4 graphAns
    class A10,A11 treeAns
    class A5,A6 windowAns
    class A13 dpAns
    class A7 searchAns
    class A8,A9 dsAns
    class A14 stringAns
    class A12 btAns
```

::: tip WHEN STUCK
Many problems combine patterns. If nothing fits cleanly, try **BFS/DFS + HashMap** as a starting point.
:::

---

## Quick-Reference Table

| Pattern | Template | When to Use |
|---------|----------|-------------|
| BFS (grid) | `deque` + 4-direction loop | Shortest path on unweighted grid, flood fill, connected components |
| BFS (adj list) | `deque` + visited set | Shortest path on unweighted graph, word ladder |
| DFS (recursive) | Call stack + visited | Tree traversal, connected components, backtracking |
| DFS (iterative) | Explicit stack + visited | Same as recursive DFS, avoids stack overflow |
| Topological Sort | Kahn's BFS + indegree array | Course schedule, build order, dependency resolution |
| Dijkstra's | Min-heap + dist array | Shortest path on weighted graph (non-negative weights) |
| Tree DFS | Recursive pre/in/post order | Path sums, diameter, max depth, validate BST |
| Tree BFS | `deque` level-order | Level-order traversal, zigzag, right side view |
| Sliding Window | Two pointers + window state | Longest/shortest substring, subarray with constraint |
| Monotonic Deque | `deque` maintaining order | Sliding window maximum/minimum |
| 1D DP | `dp[i]` array | Coin change, house robber, LIS |
| 2D Grid DP | `dp[i][j]` matrix | Unique paths, min path sum |
| DFS + Memo | `@cache` or dict | Longest increasing path, word break, tree DP on grid |
| Binary Search | `lo, hi` + midpoint | Search rotated array, find peak, left/right bound |
| Heap (Top-K) | `heapq` min-heap of size K | Top-K frequent, merge K sorted, Kth largest |
| HashMap Grouping | `defaultdict(list)` | Group anagrams, group by pattern |
| Trie | Nested dicts or TrieNode | Prefix search, autocomplete, IP matching |
| Backtracking | DFS + mark/unmark visited | Word search, permutations, combinations |

---

## Graph Templates

### BFS on a Grid

```python
from collections import deque

def bfs_grid(grid, start_r, start_c):
    """BFS on 2D grid. Use for shortest path, flood fill, connected components."""
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start_r, start_c)])
    visited = set()
    visited.add((start_r, start_c))
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    while queue:
        r, c = queue.popleft()
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and (nr, nc) not in visited and grid[nr][nc] != 0:
                visited.add((nr, nc))
                queue.append((nr, nc))

    return visited
```

### BFS on Adjacency List

```python
from collections import deque

def bfs(graph, start):
    """BFS on adjacency list. Returns visited set. O(V + E)."""
    queue = deque([start])
    visited = {start}

    while queue:
        node = queue.popleft()
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)

    return visited
```

### DFS --- Recursive

```python
def dfs_recursive(graph, node, visited):
    """DFS on adjacency list. Modify the body for your problem."""
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)
```

### DFS --- Iterative

```python
def dfs_iterative(graph, start):
    """Iterative DFS using explicit stack. Avoids recursion limit."""
    stack = [start]
    visited = set()

    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)

    return visited
```

### Topological Sort --- Kahn's Algorithm (BFS)

```python
from collections import deque, defaultdict

def topo_sort(num_nodes, edges):
    """Kahn's algorithm. Returns topological order or [] if cycle exists. O(V + E)."""
    indegree = [0] * num_nodes
    adj = defaultdict(list)

    for u, v in edges:
        adj[u].append(v)
        indegree[v] += 1

    queue = deque([i for i in range(num_nodes) if indegree[i] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for nei in adj[node]:
            indegree[nei] -= 1
            if indegree[nei] == 0:
                queue.append(nei)

    return order if len(order) == num_nodes else []  # empty = cycle detected
```

### Dijkstra's Algorithm

```python
import heapq
from collections import defaultdict

def dijkstra(graph, start, n):
    """
    Shortest path from start to all nodes. graph = {u: [(v, weight), ...]}.
    Returns dist array. O((V + E) log V).
    """
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]  # (distance, node)

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # skip stale entries
        for v, w in graph[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))

    return dist
```

---

## Tree Templates

### TreeNode Class

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### DFS Traversals --- Recursive

```python
def preorder(root):
    """Root -> Left -> Right. Use for: serialization, copying tree."""
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def inorder(root):
    """Left -> Root -> Right. Use for: BST gives sorted order."""
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def postorder(root):
    """Left -> Right -> Root. Use for: delete tree, evaluate expression tree."""
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]
```

### BFS Level-Order Traversal

```python
from collections import deque

def level_order(root):
    """Returns list of lists, one per level. O(n)."""
    if not root:
        return []
    result = []
    queue = deque([root])

    while queue:
        level_size = len(queue)
        level = []
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

### BST Search and Insert

```python
def bst_search(root, target):
    """O(h) search in BST. Returns node or None."""
    while root:
        if target == root.val:
            return root
        elif target < root.val:
            root = root.left
        else:
            root = root.right
    return None

def bst_insert(root, val):
    """O(h) insert into BST. Returns root."""
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = bst_insert(root.left, val)
    else:
        root.right = bst_insert(root.right, val)
    return root
```

---

## Sliding Window Templates

### Universal Sliding Window

```python
def sliding_window(s):
    """
    Expand right pointer, shrink left pointer when window is invalid.
    Modify INVALID_CONDITION and result tracking for your problem.
    """
    left = 0
    window = {}  # track character counts or other window state
    result = 0

    for right in range(len(s)):
        # --- EXPAND: add s[right] to window ---
        window[s[right]] = window.get(s[right], 0) + 1

        # --- SHRINK: while window violates the constraint ---
        while INVALID_CONDITION:  # e.g., len(window) > k, or duplicates exist
            window[s[left]] -= 1
            if window[s[left]] == 0:
                del window[s[left]]
            left += 1

        # --- UPDATE RESULT ---
        result = max(result, right - left + 1)

    return result
```

### Monotonic Deque --- Sliding Window Maximum

```python
from collections import deque

def max_sliding_window(nums, k):
    """
    Returns max of each window of size k. O(n).
    Deque stores indices. Front = index of current max.
    """
    dq = deque()  # stores indices, values are decreasing
    result = []

    for i in range(len(nums)):
        # remove indices outside the window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # remove smaller elements from back (they'll never be the max)
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()

        dq.append(i)

        # window is full, record the max
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

---

## DP Templates

### 1D DP

```python
def dp_1d(nums):
    """
    Generic 1D DP. Modify base case and transition for your problem.
    Examples: Coin Change, House Robber, LIS.
    """
    n = len(nums)
    dp = [0] * (n + 1)  # dp[i] = answer considering first i elements

    # base case
    dp[0] = BASE_VALUE  # e.g., 0 for sums, 1 for counts

    for i in range(1, n + 1):
        # transition: dp[i] depends on previous states
        dp[i] = TRANSITION(dp, nums, i)  # e.g., max(dp[i-1], dp[i-2] + nums[i-1])

    return dp[n]
```

**Coin Change (concrete example):**

```python
def coin_change(coins, amount):
    """Minimum coins to make amount. O(amount * len(coins))."""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0

    for a in range(1, amount + 1):
        for coin in coins:
            if coin <= a:
                dp[a] = min(dp[a], dp[a - coin] + 1)

    return dp[amount] if dp[amount] != float('inf') else -1
```

### 2D Grid DP

```python
def grid_dp(grid):
    """
    Generic 2D grid DP. Modify for min path sum, unique paths, etc.
    """
    rows, cols = len(grid), len(grid[0])
    dp = [[0] * cols for _ in range(rows)]

    # base case: first cell
    dp[0][0] = grid[0][0]

    # fill first row
    for c in range(1, cols):
        dp[0][c] = dp[0][c - 1] + grid[0][c]

    # fill first column
    for r in range(1, rows):
        dp[r][0] = dp[r - 1][0] + grid[r][0]

    # fill rest
    for r in range(1, rows):
        for c in range(1, cols):
            dp[r][c] = grid[r][c] + min(dp[r - 1][c], dp[r][c - 1])

    return dp[rows - 1][cols - 1]
```

### Top-Down DFS + Memoization

```python
from functools import lru_cache

def dfs_memo(matrix):
    """
    DFS with memoization on a grid. Used for: Longest Increasing Path in Matrix.
    O(m * n), each cell computed once.
    """
    rows, cols = len(matrix), len(matrix[0])
    directions = [(0, 1), (0, -1), (1, 0), (-1, 0)]

    @lru_cache(maxsize=None)
    def dfs(r, c):
        best = 1  # at minimum, the cell itself
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and matrix[nr][nc] > matrix[r][c]:
                best = max(best, 1 + dfs(nr, nc))
        return best

    return max(dfs(r, c) for r in range(rows) for c in range(cols))
```

---

## Data Structure Templates

### Binary Search --- Left Bound and Right Bound

```python
def binary_search_left(nums, target):
    """
    Find the leftmost index where target could be inserted (bisect_left).
    Returns first index i where nums[i] >= target.
    """
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

def binary_search_right(nums, target):
    """
    Find the rightmost index where target could be inserted (bisect_right).
    Returns first index i where nums[i] > target.
    """
    lo, hi = 0, len(nums)
    while lo < hi:
        mid = (lo + hi) // 2
        if nums[mid] <= target:
            lo = mid + 1
        else:
            hi = mid
    return lo
```

### Heap --- Top-K Pattern

```python
import heapq

def top_k_frequent(nums, k):
    """
    Return the k most frequent elements. O(n log k).
    Use a min-heap of size k to track the top-k.
    """
    from collections import Counter
    count = Counter(nums)

    # min-heap of size k: (frequency, element)
    return heapq.nlargest(k, count.keys(), key=count.get)

# Manual min-heap approach (more general):
def top_k_manual(nums, k):
    from collections import Counter
    count = Counter(nums)
    heap = []

    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)  # remove smallest frequency

    return [num for freq, num in heap]
```

### HashMap Grouping Pattern

```python
from collections import defaultdict

def group_by_pattern(items):
    """
    Group items by a computed key. Used for: Group Anagrams, group by frequency, etc.
    """
    groups = defaultdict(list)

    for item in items:
        key = compute_key(item)  # e.g., tuple(sorted(item)) for anagrams
        groups[key].append(item)

    return list(groups.values())

# Concrete example: Group Anagrams
def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)
    return list(groups.values())
```

---

## String Templates

### Trie (Prefix Tree)

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word):
        """Insert a word into the trie. O(m) where m = len(word)."""
        node = self.root
        for ch in word:
            if ch not in node.children:
                node.children[ch] = TrieNode()
            node = node.children[ch]
        node.is_end = True

    def search(self, word):
        """Return True if word is in the trie. O(m)."""
        node = self.root
        for ch in word:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return node.is_end

    def starts_with(self, prefix):
        """Return True if any word starts with prefix. O(m)."""
        node = self.root
        for ch in prefix:
            if ch not in node.children:
                return False
            node = node.children[ch]
        return True
```

### Backtracking on Grid

```python
def word_search(board, word):
    """
    Search for word in grid using backtracking. O(m * n * 4^L).
    Mark cells visited by modifying in-place, then restore.
    """
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, idx):
        if idx == len(word):
            return True
        if r < 0 or r >= rows or c < 0 or c >= cols:
            return False
        if board[r][c] != word[idx]:
            return False

        # mark visited
        temp = board[r][c]
        board[r][c] = '#'

        # explore 4 directions
        found = (backtrack(r + 1, c, idx + 1) or
                 backtrack(r - 1, c, idx + 1) or
                 backtrack(r, c + 1, idx + 1) or
                 backtrack(r, c - 1, idx + 1))

        # restore
        board[r][c] = temp
        return found

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False
```

---

## How to Use These in a Plain Text Editor

::: tip WRITING CODE WITHOUT AN IDE
1. **Indent with spaces, not tabs** --- plain text editors handle spaces more predictably. Use 4 spaces per level.

2. **Write helper functions** --- Break your solution into small functions. It shows design sense and makes dry-running easier.

3. **Use descriptive variable names** --- `left`, `right`, `queue`, `visited` are clear. Avoid `i`, `j`, `k` unless iterating simple loops.

4. **Skip imports in the interview** --- Your interviewer knows you need `deque`. Focus on logic, mention imports verbally.

5. **Leave blank lines between logical blocks** --- It makes your code readable when there is no syntax highlighting.

6. **Comment sparingly but strategically** --- A one-line comment above tricky logic shows you know it is tricky. Do not comment obvious lines.

7. **Write the function signature first** --- Before filling in the body, write `def solve(grid):` and the return statement. This anchors your solution.

8. **Practice typing your templates from memory** --- Open a plain text editor, write BFS from scratch, then sliding window, then topo sort. Repeat until it is muscle memory.
:::

---

## Complexity Quick Reference

![Data Structure Complexity Table](/sde-coding/sprint/ds_complexity_table.png)

| Data Structure | Access | Search | Insert | Delete | Notes |
|---------------|--------|--------|--------|--------|-------|
| **Array** | O(1) | O(n) | O(n) | O(n) | O(1) insert/delete at end |
| **Hash Map** | -- | O(1) avg | O(1) avg | O(1) avg | O(n) worst case |
| **Hash Set** | -- | O(1) avg | O(1) avg | O(1) avg | O(n) worst case |
| **Heap (Min/Max)** | O(1) top | O(n) | O(log n) | O(log n) | `heapq` in Python is min-heap |
| **BST (balanced)** | -- | O(log n) | O(log n) | O(log n) | Worst case O(n) if unbalanced |
| **Stack** | O(1) top | O(n) | O(1) | O(1) | LIFO |
| **Queue** | O(1) front | O(n) | O(1) | O(1) | FIFO, use `deque` not `list` |
| **Deque** | O(1) ends | O(n) | O(1) | O(1) | Double-ended, use for BFS + monotonic |
| **Trie** | -- | O(m) | O(m) | O(m) | m = word length |

| Algorithm | Time | Space | When to Use |
|-----------|------|-------|-------------|
| **BFS** | O(V + E) | O(V) | Shortest path (unweighted) |
| **DFS** | O(V + E) | O(V) | Connected components, cycle detection |
| **Dijkstra's** | O((V + E) log V) | O(V) | Shortest path (weighted, non-negative) |
| **Topological Sort** | O(V + E) | O(V) | DAG ordering, dependency resolution |
| **Binary Search** | O(log n) | O(1) | Sorted array, monotonic condition |
| **Merge Sort** | O(n log n) | O(n) | Stable sort, linked list sort |
| **Quick Sort** | O(n log n) avg | O(log n) | In-place sort, quickselect |
| **Sliding Window** | O(n) | O(k) | Substring/subarray with constraint |
| **DP (1D)** | O(n) | O(n) | Optimal substructure, overlapping subproblems |
| **DP (2D)** | O(m * n) | O(m * n) | Grid problems, two-sequence problems |

![Time Complexity Growth Rates](/sde-coding/sprint/complexity_comparison.png)
