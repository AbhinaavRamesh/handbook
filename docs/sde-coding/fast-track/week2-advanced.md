# Week 2: Advanced + Practice - Day-by-Day Breakdown

> **Master complex patterns and simulate real interview conditions**

---

## Overview

Week 2 builds on Week 1 foundations with advanced topics: graphs, recursion, backtracking, and dynamic programming. The week culminates with multiple mock interviews to build confidence and refine your interview performance.

**Weekly Goals:**
- Solve 30-35 additional problems (total: 60-70)
- Complete 3 mock interviews total (at least 1 with a human)
- Master the art of thinking out loud
- Build confidence handling unfamiliar problems

**Daily Time Commitment:** 3-4 hours

---

## Day 8: Graphs (BFS/DFS)

### Learning Objectives
- Implement graph traversals (BFS and DFS)
- Identify connected components
- Detect cycles in graphs

### Study Plan (3-4 hours)

| Time Block | Activity | Details |
|------------|----------|---------|
| 0:00-0:30 | **Concept Review** | Graph representations, BFS vs DFS |
| 0:30-2:00 | **Problem Solving** | Solve 5-6 problems |
| 2:00-2:30 | **Pattern Review** | When to use BFS vs DFS |
| 2:30-3:00 | **Edge Cases** | Disconnected graphs, self-loops |
| 3:00-3:30 | **Verbal Practice** | Trace through a graph traversal |

### Must-Solve Problems

| # | Problem | Difficulty | Pattern | Time Limit |
|---|---------|------------|---------|------------|
| 1 | Number of Islands | Medium | DFS/BFS Grid | 20 min |
| 2 | Clone Graph | Medium | BFS + Hash Map | 25 min |
| 3 | Max Area of Island | Medium | DFS | 20 min |
| 4 | Pacific Atlantic Water Flow | Medium | Multi-source BFS | 30 min |
| 5 | Rotting Oranges | Medium | BFS Level Order | 25 min |
| 6 | Surrounded Regions | Medium | DFS from Boundary | 25 min |

### Code Templates

::: code-group

```python [Python]
from collections import deque

# Edge List to Adjacency List
def build_graph(edges, n, directed=False):
    graph = {i: [] for i in range(n)}
    for u, v in edges:
        graph[u].append(v)
        if not directed:
            graph[v].append(u)
    return graph

# BFS Template
def bfs(graph, start):
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

# DFS Template (Iterative)
def dfs_iterative(graph, start):
    visited = set()
    stack = [start]
    result = []

    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            result.append(node)

            for neighbor in graph[node]:
                if neighbor not in visited:
                    stack.append(neighbor)

    return result

# Number of Islands (Grid DFS)
def num_islands(grid):
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    count = 0

    def dfs(r, c):
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            grid[r][c] == '0'):
            return

        grid[r][c] = '0'  # Mark visited
        dfs(r + 1, c)
        dfs(r - 1, c)
        dfs(r, c + 1)
        dfs(r, c - 1)

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == '1':
                count += 1
                dfs(r, c)

    return count
```

```java [Java]
// Edge List to Adjacency List
Map<Integer, List<Integer>> buildGraph(int[][] edges, int n, boolean directed) {
    Map<Integer, List<Integer>> graph = new HashMap<>();
    for (int i = 0; i < n; i++) graph.put(i, new ArrayList<>());
    for (int[] e : edges) {
        graph.get(e[0]).add(e[1]);
        if (!directed) graph.get(e[1]).add(e[0]);
    }
    return graph;
}

// BFS Template
List<Integer> bfs(Map<Integer, List<Integer>> graph, int start) {
    List<Integer> result = new ArrayList<>();
    Set<Integer> visited = new HashSet<>();
    Deque<Integer> queue = new ArrayDeque<>();
    visited.add(start); queue.offer(start);
    while (!queue.isEmpty()) {
        int node = queue.poll(); result.add(node);
        for (int nb : graph.getOrDefault(node, Collections.emptyList()))
            if (visited.add(nb)) queue.offer(nb);
    }
    return result;
}

// DFS Template (Iterative)
List<Integer> dfsIterative(Map<Integer, List<Integer>> graph, int start) {
    List<Integer> result = new ArrayList<>();
    Set<Integer> visited = new HashSet<>();
    Deque<Integer> stack = new ArrayDeque<>();
    stack.push(start);
    while (!stack.isEmpty()) {
        int node = stack.pop();
        if (visited.add(node)) {
            result.add(node);
            for (int nb : graph.getOrDefault(node, Collections.emptyList()))
                if (!visited.contains(nb)) stack.push(nb);
        }
    }
    return result;
}

// Number of Islands (Grid DFS)
int numIslands(char[][] grid) {
    int rows = grid.length, cols = grid[0].length, count = 0;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (grid[r][c] == '1') { islandDFS(grid, r, c, rows, cols); count++; }
    return count;
}
void islandDFS(char[][] grid, int r, int c, int rows, int cols) {
    if (r < 0 || r >= rows || c < 0 || c >= cols || grid[r][c] != '1') return;
    grid[r][c] = '0';
    islandDFS(grid, r+1, c, rows, cols); islandDFS(grid, r-1, c, rows, cols);
    islandDFS(grid, r, c+1, rows, cols); islandDFS(grid, r, c-1, rows, cols);
}
```

:::

::: info Complexity: Time O(V + E) or O(m * n) · Space O(V) or O(m * n)
- **Time:** BFS/DFS visit each vertex and edge once; grid problems are O(m*n)
- **Space:** Visited set and recursion/queue can store all nodes
:::

### Day 8 Checklist

```
[ ] Reviewed graph representations (15 min)
[ ] Understood BFS vs DFS use cases
[ ] Solved Number of Islands
[ ] Solved Clone Graph
[ ] Solved Max Area of Island
[ ] Attempted Pacific Atlantic Water Flow
[ ] Solved Rotting Oranges
[ ] Practiced tracing BFS/DFS on paper
```

---

## Day 9: Graphs (Advanced)

### Learning Objectives
- Implement shortest path algorithms
- Perform topological sorting
- Use Union-Find for connectivity

### Study Plan (3-4 hours)

| Time Block | Activity | Details |
|------------|----------|---------|
| 0:00-0:45 | **Concept Review** | Dijkstra, Topological Sort, Union-Find |
| 0:45-2:15 | **Problem Solving** | Solve 4-5 problems |
| 2:15-2:45 | **Pattern Review** | When to use each algorithm |
| 2:45-3:15 | **Edge Cases** | Cycles, disconnected components |
| 3:15-3:45 | **Verbal Practice** | Explain topological sort |

### Must-Solve Problems

| # | Problem | Difficulty | Pattern | Time Limit |
|---|---------|------------|---------|------------|
| 1 | Course Schedule | Medium | Topological Sort | 25 min |
| 2 | Course Schedule II | Medium | Topological Sort | 25 min |
| 3 | Number of Connected Components | Medium | Union-Find | 20 min |
| 4 | Graph Valid Tree | Medium | Union-Find + Cycle | 25 min |
| 5 | Network Delay Time | Medium | Dijkstra | 30 min |

### Code Templates

::: code-group

```python [Python]
from collections import deque, defaultdict
import heapq

# Topological Sort (Kahn's Algorithm - BFS)
def topological_sort_bfs(num_nodes, edges):
    graph = defaultdict(list)
    in_degree = [0] * num_nodes

    for u, v in edges:
        graph[u].append(v)
        in_degree[v] += 1

    queue = deque([i for i in range(num_nodes) if in_degree[i] == 0])
    result = []

    while queue:
        node = queue.popleft()
        result.append(node)

        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    return result if len(result) == num_nodes else []

# Union-Find (Disjoint Set Union)
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.components = n

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # Path compression
        return self.parent[x]

    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False

        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1

        self.components -= 1
        return True

    def connected(self, x, y):
        return self.find(x) == self.find(y)

# Dijkstra's Shortest Path
def dijkstra(graph, start, n):
    dist = [float('inf')] * n
    dist[start] = 0
    heap = [(0, start)]

    while heap:
        d, u = heapq.heappop(heap)

        if d > dist[u]:
            continue

        for v, weight in graph[u]:
            if dist[u] + weight < dist[v]:
                dist[v] = dist[u] + weight
                heapq.heappush(heap, (dist[v], v))

    return dist
```

```java [Java]
// Topological Sort (Kahn's Algorithm - BFS)
int[] topologicalSortBFS(int numNodes, int[][] edges) {
    List<List<Integer>> graph = new ArrayList<>();
    int[] inDegree = new int[numNodes];
    for (int i = 0; i < numNodes; i++) graph.add(new ArrayList<>());
    for (int[] e : edges) { graph.get(e[0]).add(e[1]); inDegree[e[1]]++; }
    Deque<Integer> queue = new ArrayDeque<>();
    for (int i = 0; i < numNodes; i++) if (inDegree[i] == 0) queue.offer(i);
    int[] result = new int[numNodes]; int idx = 0;
    while (!queue.isEmpty()) {
        int node = queue.poll(); result[idx++] = node;
        for (int nb : graph.get(node)) if (--inDegree[nb] == 0) queue.offer(nb);
    }
    return idx == numNodes ? result : new int[]{};
}

// Union-Find (Disjoint Set Union)
class UnionFind {
    int[] parent, rank; int components;
    UnionFind(int n) {
        parent = new int[n]; rank = new int[n]; components = n;
        for (int i = 0; i < n; i++) parent[i] = i;
    }
    int find(int x) {
        if (parent[x] != x) parent[x] = find(parent[x]);
        return parent[x];
    }
    boolean union(int x, int y) {
        int px = find(x), py = find(y);
        if (px == py) return false;
        if (rank[px] < rank[py]) { int t = px; px = py; py = t; }
        parent[py] = px;
        if (rank[px] == rank[py]) rank[px]++;
        components--; return true;
    }
    boolean connected(int x, int y) { return find(x) == find(y); }
}

// Dijkstra's Shortest Path
int[] dijkstra(List<int[]>[] graph, int start, int n) {
    // graph[u] = [[v, weight], ...]
    int[] dist = new int[n]; Arrays.fill(dist, Integer.MAX_VALUE); dist[start] = 0;
    PriorityQueue<int[]> heap = new PriorityQueue<>(Comparator.comparingInt(a -> a[0]));
    heap.offer(new int[]{0, start});
    while (!heap.isEmpty()) {
        int[] cur = heap.poll(); int d = cur[0], u = cur[1];
        if (d > dist[u]) continue;
        for (int[] edge : graph[u]) {
            int v = edge[0], w = edge[1];
            if (dist[u] + w < dist[v]) { dist[v] = dist[u] + w; heap.offer(new int[]{dist[v], v}); }
        }
    }
    return dist;
}
```

:::

::: info Complexity: Topo Sort O(V + E) · Union-Find O(alpha(n)) · Dijkstra O((V + E) log V)
- **Time:** Topological sort is linear; Union-Find is near-constant per operation; Dijkstra uses heap
- **Space:** All algorithms use O(V) for storage structures
:::

### Day 9 Checklist

```
[ ] Reviewed topological sort theory (20 min)
[ ] Reviewed Union-Find structure (15 min)
[ ] Solved Course Schedule
[ ] Solved Course Schedule II
[ ] Solved Number of Connected Components
[ ] Solved Graph Valid Tree
[ ] Attempted Network Delay Time with Dijkstra
[ ] Practiced explaining Union-Find operations
```

---

## Day 10: Recursion & Backtracking

### Learning Objectives
- Master recursive problem decomposition
- Generate subsets and permutations
- Implement constraint-based backtracking

### Study Plan (3-4 hours)

| Time Block | Activity | Details |
|------------|----------|---------|
| 0:00-0:30 | **Concept Review** | Backtracking framework |
| 0:30-2:00 | **Problem Solving** | Solve 5-6 problems |
| 2:00-2:30 | **Pattern Review** | Pruning strategies |
| 2:30-3:00 | **Edge Cases** | Empty input, duplicates |
| 3:00-3:30 | **Verbal Practice** | Trace recursion tree |

### Must-Solve Problems

| # | Problem | Difficulty | Pattern | Time Limit |
|---|---------|------------|---------|------------|
| 1 | Subsets | Medium | Backtracking | 20 min |
| 2 | Subsets II | Medium | Backtracking + Skip Dups | 25 min |
| 3 | Permutations | Medium | Backtracking | 20 min |
| 4 | Combination Sum | Medium | Backtracking | 25 min |
| 5 | Word Search | Medium | Grid Backtracking | 30 min |
| 6 | N-Queens | Hard | Constraint Backtracking | 35 min |

### Code Templates

::: code-group

```python [Python]
# Subsets
def subsets(nums):
    result = []

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

# Subsets II (with duplicates)
def subsets_with_dup(nums):
    result = []
    nums.sort()

    def backtrack(start, path):
        result.append(path[:])

        for i in range(start, len(nums)):
            if i > start and nums[i] == nums[i-1]:
                continue
            path.append(nums[i])
            backtrack(i + 1, path)
            path.pop()

    backtrack(0, [])
    return result

# Permutations
def permutations(nums):
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

# Combination Sum
def combination_sum(candidates, target):
    result = []

    def backtrack(start, path, remaining):
        if remaining == 0:
            result.append(path[:])
            return
        if remaining < 0:
            return

        for i in range(start, len(candidates)):
            path.append(candidates[i])
            backtrack(i, path, remaining - candidates[i])  # i, not i+1 (reuse allowed)
            path.pop()

    backtrack(0, [], target)
    return result

# Word Search
def word_search(board, word):
    rows, cols = len(board), len(board[0])

    def backtrack(r, c, idx):
        if idx == len(word):
            return True
        if (r < 0 or r >= rows or c < 0 or c >= cols or
            board[r][c] != word[idx]):
            return False

        temp = board[r][c]
        board[r][c] = '#'

        found = (backtrack(r + 1, c, idx + 1) or
                 backtrack(r - 1, c, idx + 1) or
                 backtrack(r, c + 1, idx + 1) or
                 backtrack(r, c - 1, idx + 1))

        board[r][c] = temp
        return found

    for r in range(rows):
        for c in range(cols):
            if backtrack(r, c, 0):
                return True
    return False
```

```java [Java]
// Subsets
List<List<Integer>> subsets(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    subsetsHelper(nums, 0, new ArrayList<>(), result); return result;
}
void subsetsHelper(int[] nums, int start, List<Integer> path, List<List<Integer>> result) {
    result.add(new ArrayList<>(path));
    for (int i = start; i < nums.length; i++) {
        path.add(nums[i]); subsetsHelper(nums, i + 1, path, result); path.remove(path.size() - 1);
    }
}

// Subsets II (with duplicates)
List<List<Integer>> subsetsWithDup(int[] nums) {
    Arrays.sort(nums); List<List<Integer>> result = new ArrayList<>();
    subsetsWithDupHelper(nums, 0, new ArrayList<>(), result); return result;
}
void subsetsWithDupHelper(int[] nums, int start, List<Integer> path, List<List<Integer>> result) {
    result.add(new ArrayList<>(path));
    for (int i = start; i < nums.length; i++) {
        if (i > start && nums[i] == nums[i - 1]) continue;
        path.add(nums[i]); subsetsWithDupHelper(nums, i + 1, path, result); path.remove(path.size() - 1);
    }
}

// Permutations
List<List<Integer>> permutations(int[] nums) {
    List<List<Integer>> result = new ArrayList<>();
    permsHelper(nums, new ArrayList<>(), new boolean[nums.length], result); return result;
}
void permsHelper(int[] nums, List<Integer> path, boolean[] used, List<List<Integer>> result) {
    if (path.size() == nums.length) { result.add(new ArrayList<>(path)); return; }
    for (int i = 0; i < nums.length; i++) {
        if (used[i]) continue;
        used[i] = true; path.add(nums[i]);
        permsHelper(nums, path, used, result);
        path.remove(path.size() - 1); used[i] = false;
    }
}

// Combination Sum
List<List<Integer>> combinationSum(int[] candidates, int target) {
    List<List<Integer>> result = new ArrayList<>();
    combHelper(candidates, target, 0, new ArrayList<>(), result); return result;
}
void combHelper(int[] cands, int rem, int start, List<Integer> path, List<List<Integer>> result) {
    if (rem == 0) { result.add(new ArrayList<>(path)); return; }
    if (rem < 0) return;
    for (int i = start; i < cands.length; i++) {
        path.add(cands[i]); combHelper(cands, rem - cands[i], i, path, result); path.remove(path.size() - 1);
    }
}

// Word Search
boolean wordSearch(char[][] board, String word) {
    int rows = board.length, cols = board[0].length;
    for (int r = 0; r < rows; r++)
        for (int c = 0; c < cols; c++)
            if (wordDFS(board, word, r, c, 0, rows, cols)) return true;
    return false;
}
boolean wordDFS(char[][] board, String word, int r, int c, int idx, int rows, int cols) {
    if (idx == word.length()) return true;
    if (r < 0 || r >= rows || c < 0 || c >= cols || board[r][c] != word.charAt(idx)) return false;
    char tmp = board[r][c]; board[r][c] = '#';
    boolean found = wordDFS(board, word, r+1, c, idx+1, rows, cols)
                 || wordDFS(board, word, r-1, c, idx+1, rows, cols)
                 || wordDFS(board, word, r, c+1, idx+1, rows, cols)
                 || wordDFS(board, word, r, c-1, idx+1, rows, cols);
    board[r][c] = tmp;
    return found;
}
```

:::

::: info Complexity: Time O(2^n) to O(n!) · Space O(n)
- **Time:** Subsets O(n*2^n); permutations O(n*n!); word search O(m*n*4^L)
- **Space:** Recursion depth and path storage proportional to input size
:::

### Day 10 Checklist

```
[ ] Reviewed backtracking framework (15 min)
[ ] Drew recursion trees for understanding
[ ] Solved Subsets
[ ] Solved Subsets II with duplicate handling
[ ] Solved Permutations
[ ] Solved Combination Sum
[ ] Solved Word Search
[ ] Attempted N-Queens (optional)
[ ] Practiced tracing backtracking verbally
```

---

## Day 11: Dynamic Programming (Basic)

### Learning Objectives
- Recognize DP problem patterns
- Implement 1D DP solutions
- Convert recursion to tabulation

### Study Plan (4 hours)

| Time Block | Activity | Details |
|------------|----------|---------|
| 0:00-0:45 | **Concept Review** | DP fundamentals, memoization vs tabulation |
| 0:45-2:15 | **Problem Solving** | Solve 5-6 problems |
| 2:15-2:45 | **Pattern Review** | Identify optimal substructure |
| 2:45-3:15 | **Optimization** | Space optimization techniques |
| 3:15-3:45 | **Verbal Practice** | Explain state transition |

### Must-Solve Problems

| # | Problem | Difficulty | Pattern | Time Limit |
|---|---------|------------|---------|------------|
| 1 | Climbing Stairs | Easy | Fibonacci-like | 10 min |
| 2 | House Robber | Medium | Max Non-Adjacent | 20 min |
| 3 | House Robber II | Medium | Circular DP | 25 min |
| 4 | Coin Change | Medium | Unbounded Knapsack | 25 min |
| 5 | Longest Increasing Subsequence | Medium | LIS Pattern | 25 min |
| 6 | Word Break | Medium | String DP | 25 min |

### Code Templates

::: code-group

```python [Python]
# Climbing Stairs (Fibonacci)
def climb_stairs(n):
    if n <= 2:
        return n
    prev2, prev1 = 1, 2
    for _ in range(3, n + 1):
        curr = prev1 + prev2
        prev2, prev1 = prev1, curr
    return prev1

# House Robber
def house_robber(nums):
    prev2, prev1 = 0, 0
    for num in nums:
        curr = max(prev1, prev2 + num)
        prev2, prev1 = prev1, curr
    return prev1

# Coin Change (Unbounded Knapsack)
def coin_change(coins, amount):
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for i in range(1, amount + 1):
        for coin in coins:
            if coin <= i and dp[i - coin] != float('inf'):
                dp[i] = min(dp[i], dp[i - coin] + 1)
    return dp[amount] if dp[amount] != float('inf') else -1

# Longest Increasing Subsequence O(n^2)
def length_of_lis(nums):
    dp = [1] * len(nums)
    for i in range(1, len(nums)):
        for j in range(i):
            if nums[j] < nums[i]:
                dp[i] = max(dp[i], dp[j] + 1)
    return max(dp)

# LIS with Binary Search O(n log n)
def length_of_lis_optimized(nums):
    from bisect import bisect_left
    tails = []
    for num in nums:
        pos = bisect_left(tails, num)
        if pos == len(tails):
            tails.append(num)
        else:
            tails[pos] = num
    return len(tails)

# Word Break
def word_break(s, word_dict):
    word_set = set(word_dict)
    n = len(s)
    dp = [False] * (n + 1)
    dp[0] = True
    for i in range(1, n + 1):
        for j in range(i):
            if dp[j] and s[j:i] in word_set:
                dp[i] = True
                break
    return dp[n]
```

```java [Java]
// Climbing Stairs
int climbStairs(int n) {
    if (n <= 2) return n;
    int prev2 = 1, prev1 = 2;
    for (int i = 3; i <= n; i++) { int curr = prev1 + prev2; prev2 = prev1; prev1 = curr; }
    return prev1;
}

// House Robber
int houseRobber(int[] nums) {
    int prev2 = 0, prev1 = 0;
    for (int num : nums) { int curr = Math.max(prev1, prev2 + num); prev2 = prev1; prev1 = curr; }
    return prev1;
}

// Coin Change
int coinChange(int[] coins, int amount) {
    int[] dp = new int[amount + 1]; Arrays.fill(dp, Integer.MAX_VALUE); dp[0] = 0;
    for (int i = 1; i <= amount; i++)
        for (int coin : coins)
            if (coin <= i && dp[i - coin] != Integer.MAX_VALUE)
                dp[i] = Math.min(dp[i], dp[i - coin] + 1);
    return dp[amount] == Integer.MAX_VALUE ? -1 : dp[amount];
}

// Longest Increasing Subsequence O(n^2)
int lengthOfLIS(int[] nums) {
    int[] dp = new int[nums.length]; Arrays.fill(dp, 1);
    int max = 1;
    for (int i = 1; i < nums.length; i++) {
        for (int j = 0; j < i; j++)
            if (nums[j] < nums[i]) dp[i] = Math.max(dp[i], dp[j] + 1);
        max = Math.max(max, dp[i]);
    }
    return max;
}

// Word Break
boolean wordBreak(String s, List<String> wordDict) {
    Set<String> wordSet = new HashSet<>(wordDict);
    int n = s.length(); boolean[] dp = new boolean[n + 1]; dp[0] = true;
    for (int i = 1; i <= n; i++)
        for (int j = 0; j < i; j++)
            if (dp[j] && wordSet.contains(s.substring(j, i))) { dp[i] = true; break; }
    return dp[n];
}
```

:::

::: info Complexity: Time O(n) to O(n^2) · Space O(n) or O(1)
- **Time:** Fibonacci O(n); Coin Change O(n*m); LIS O(n^2) or O(n log n)
- **Space:** O(n) for dp array; O(1) for space-optimized versions
:::

### Day 11 Checklist

```
[ ] Reviewed DP fundamentals (30 min)
[ ] Understood memoization vs tabulation
[ ] Solved Climbing Stairs
[ ] Solved House Robber
[ ] Solved House Robber II
[ ] Solved Coin Change
[ ] Solved Longest Increasing Subsequence
[ ] Solved Word Break
[ ] Practiced explaining state transitions verbally
```

---

## Day 12: Dynamic Programming (Advanced)

### Learning Objectives
- Implement 2D DP solutions
- Handle string matching DP
- Master grid-based DP

### Study Plan (4 hours)

| Time Block | Activity | Details |
|------------|----------|---------|
| 0:00-0:45 | **Concept Review** | 2D DP, string DP patterns |
| 0:45-2:30 | **Problem Solving** | Solve 4-5 problems |
| 2:30-3:00 | **Pattern Review** | Common 2D state transitions |
| 3:00-3:30 | **Optimization** | Space reduction to 1D |
| 3:30-4:00 | **Verbal Practice** | Explain 2D DP table filling |

### Must-Solve Problems

| # | Problem | Difficulty | Pattern | Time Limit |
|---|---------|------------|---------|------------|
| 1 | Unique Paths | Medium | Grid DP | 20 min |
| 2 | Unique Paths II | Medium | Grid DP + Obstacles | 20 min |
| 3 | Longest Common Subsequence | Medium | String DP | 25 min |
| 4 | Edit Distance | Medium | String DP | 30 min |
| 5 | 0/1 Knapsack | Medium | Classic DP | 30 min |

### Code Templates

::: code-group

```python [Python]
# Unique Paths
def unique_paths(m, n):
    dp = [[1] * n for _ in range(m)]
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[m-1][n-1]

# Space Optimized Unique Paths
def unique_paths_optimized(m, n):
    dp = [1] * n
    for i in range(1, m):
        for j in range(1, n):
            dp[j] += dp[j-1]
    return dp[n-1]

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

# Edit Distance (Levenshtein)
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1): dp[i][0] = i
    for j in range(n + 1): dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
    return dp[m][n]

# 0/1 Knapsack (Space Optimized)
def knapsack_optimized(weights, values, capacity):
    dp = [0] * (capacity + 1)
    for i in range(len(weights)):
        for w in range(capacity, weights[i] - 1, -1):
            dp[w] = max(dp[w], dp[w - weights[i]] + values[i])
    return dp[capacity]
```

```java [Java]
// Unique Paths (Space Optimized)
int uniquePaths(int m, int n) {
    int[] dp = new int[n]; Arrays.fill(dp, 1);
    for (int i = 1; i < m; i++)
        for (int j = 1; j < n; j++) dp[j] += dp[j - 1];
    return dp[n - 1];
}

// Longest Common Subsequence
int lcs(String text1, String text2) {
    int m = text1.length(), n = text2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            dp[i][j] = text1.charAt(i-1) == text2.charAt(j-1)
                ? dp[i-1][j-1] + 1 : Math.max(dp[i-1][j], dp[i][j-1]);
    return dp[m][n];
}

// Edit Distance
int editDistance(String word1, String word2) {
    int m = word1.length(), n = word2.length();
    int[][] dp = new int[m + 1][n + 1];
    for (int i = 0; i <= m; i++) dp[i][0] = i;
    for (int j = 0; j <= n; j++) dp[0][j] = j;
    for (int i = 1; i <= m; i++)
        for (int j = 1; j <= n; j++)
            dp[i][j] = word1.charAt(i-1) == word2.charAt(j-1)
                ? dp[i-1][j-1]
                : 1 + Math.min(dp[i-1][j], Math.min(dp[i][j-1], dp[i-1][j-1]));
    return dp[m][n];
}

// 0/1 Knapsack (Space Optimized - traverse backwards)
int knapsackOptimized(int[] weights, int[] values, int capacity) {
    int[] dp = new int[capacity + 1];
    for (int i = 0; i < weights.length; i++)
        for (int w = capacity; w >= weights[i]; w--)
            dp[w] = Math.max(dp[w], dp[w - weights[i]] + values[i]);
    return dp[capacity];
}
```

:::

::: info Complexity: Time O(m * n) · Space O(m * n) or O(n)
- **Time:** Fill m*n table for 2D DP problems
- **Space:** O(m*n) for 2D table; O(n) when space-optimized to 1D
:::

### Day 12 Checklist

```
[ ] Reviewed 2D DP patterns (30 min)
[ ] Drew DP tables for visualization
[ ] Solved Unique Paths
[ ] Solved Unique Paths II
[ ] Solved Longest Common Subsequence
[ ] Solved Edit Distance
[ ] Implemented 0/1 Knapsack
[ ] Practiced space optimization
[ ] Practiced explaining table filling verbally
```

---

## Day 13: Mixed Practice + Mock Interview #2

### Learning Objectives
- Apply patterns to random problems
- Improve problem identification speed
- Build interview endurance

### Study Plan (4 hours)

| Time Block | Activity | Details |
|------------|----------|---------|
| 0:00-0:30 | **Pattern Review** | Quick review of all 10 patterns |
| 0:30-1:30 | **Random Problems** | Solve 3 problems without topic hints |
| 1:30-2:30 | **Mock Interview #2** | 45-60 min timed interview |
| 2:30-3:15 | **Mock Review** | Detailed analysis of performance |
| 3:15-3:45 | **Weak Area Study** | Focus on identified gaps |

### Random Practice Problems

Solve these without looking at which topic they belong to:

| # | Problem | Difficulty | Time Limit |
|---|---------|------------|------------|
| 1 | Merge Intervals | Medium | 25 min |
| 2 | Search in Rotated Sorted Array | Medium | 25 min |
| 3 | Decode Ways | Medium | 25 min |

### Mock Interview #2 Format

**Recommended Problems:**

| Problem | Topics | Expected Time |
|---------|--------|---------------|
| Word Ladder | BFS + String | 25 min |
| Coin Change | DP | 20 min |

**Focus Areas for Mock #2:**
- Clear problem clarification
- Structured approach explanation
- Time complexity analysis
- Edge case handling

### Mock Evaluation Form

```
Rate 1-5:

UNDERSTANDING
[ ] Clarified problem constraints
[ ] Asked good clarifying questions
[ ] Identified edge cases early

APPROACH
[ ] Explained strategy before coding
[ ] Considered multiple approaches
[ ] Chose appropriate data structures

IMPLEMENTATION
[ ] Wrote clean, readable code
[ ] Good variable naming
[ ] Appropriate comments

TESTING
[ ] Traced through example
[ ] Tested edge cases
[ ] Fixed bugs methodically

COMMUNICATION
[ ] Thought out loud continuously
[ ] Explained tradeoffs
[ ] Managed time well

Total: ___/25
```

### Day 13 Checklist

```
[ ] Reviewed all patterns (30 min)
[ ] Solved 3 random problems
[ ] Completed Mock Interview #2
[ ] Analyzed mock performance
[ ] Identified 2-3 areas to improve
[ ] Made notes for tomorrow's review
```

---

## Day 14: Final Review + Mock Interview #3

### Learning Objectives
- Consolidate all Week 2 learning
- Build peak confidence
- Final mock interview performance

### Study Plan (4 hours)

| Time Block | Activity | Details |
|------------|----------|---------|
| 0:00-0:45 | **Comprehensive Review** | All patterns and templates |
| 0:45-1:15 | **Weak Areas** | Re-solve problem types you struggled with |
| 1:15-2:15 | **Mock Interview #3** | Final confidence-building mock |
| 2:15-2:45 | **Mock Review** | Final performance analysis |
| 2:45-3:30 | **Interview Prep** | Review day-of checklist |
| 3:30-4:00 | **Mental Preparation** | Confidence building, relaxation |

### Pattern Quick Reference

| Pattern | Recognition Signal | Key Template |
|---------|-------------------|--------------|
| Two Pointers | Sorted array, pairs | opposite/same direction |
| Sliding Window | Subarray/substring | fixed/variable window |
| Fast/Slow | Cycle, middle element | Floyd's algorithm |
| Binary Search | Sorted, O(log n) | left, right, mid |
| BFS | Level order, shortest | queue + visited |
| DFS | Explore paths | stack/recursion |
| Backtracking | All combinations | choose, explore, unchoose |
| DP (1D) | Optimization, counting | state transition |
| DP (2D) | Two sequences, grid | table filling |
| Topological Sort | Dependencies | Kahn's or DFS |

### Final Mock Interview #3

**Recommended Problems:**

| Problem | Topics | Expected Time |
|---------|--------|---------------|
| Meeting Rooms II | Heap + Intervals | 25 min |
| Serialize/Deserialize Binary Tree | Tree + BFS | 25 min |

**Mock #3 Goals:**
- Demonstrate improvement from Mock #1 and #2
- Simulate actual interview pressure
- Practice graceful handling of hints

### Week 2 Summary Template

```markdown
## Week 2 Summary

### Problems Solved This Week: ___/35
### Total Problems Solved: ___/70

### Strongest Topics:
1.
2.
3.

### Areas Still Needing Work:
1.
2.

### Mock Interview Scores:
- Mock #1: ___/30
- Mock #2: ___/25
- Mock #3: ___/25

### Improvement Areas from Mocks:
1.
2.
3.

### Patterns Mastered:
- [ ] BFS/DFS on Graphs
- [ ] Topological Sort
- [ ] Union-Find
- [ ] Backtracking
- [ ] 1D Dynamic Programming
- [ ] 2D Dynamic Programming

### Confidence Level (1-10): ___

### Final Notes Before Interview:
-
-
-
```

### Day 14 Checklist

```
[ ] Completed comprehensive pattern review
[ ] Re-solved 2-3 weak area problems
[ ] Completed Mock Interview #3
[ ] Analyzed final mock performance
[ ] Reviewed day-of-checklist
[ ] Prepared interview logistics
[ ] Got good sleep planned for tonight
```

---

## Week 2 Problem Checklist

### Day 8: Graphs (BFS/DFS)
- [ ] Number of Islands
- [ ] Clone Graph
- [ ] Max Area of Island
- [ ] Pacific Atlantic Water Flow
- [ ] Rotting Oranges
- [ ] Surrounded Regions

### Day 9: Graphs (Advanced)
- [ ] Course Schedule
- [ ] Course Schedule II
- [ ] Number of Connected Components
- [ ] Graph Valid Tree
- [ ] Network Delay Time

### Day 10: Recursion and Backtracking
- [ ] Subsets
- [ ] Subsets II
- [ ] Permutations
- [ ] Combination Sum
- [ ] Word Search
- [ ] N-Queens (optional)

### Day 11: DP (Basic)
- [ ] Climbing Stairs
- [ ] House Robber
- [ ] House Robber II
- [ ] Coin Change
- [ ] Longest Increasing Subsequence
- [ ] Word Break

### Day 12: DP (Advanced)
- [ ] Unique Paths
- [ ] Unique Paths II
- [ ] Longest Common Subsequence
- [ ] Edit Distance
- [ ] 0/1 Knapsack

### Day 13: Mixed Practice + Mock
- [ ] 3 random problems solved
- [ ] Mock Interview #2 completed
- [ ] Performance analyzed

### Day 14: Final Review + Mock
- [ ] Comprehensive review completed
- [ ] Mock Interview #3 completed
- [ ] Ready for real interview

---

## Tips for Week 2 Success

### Handling Advanced Topics
- DP is hard - spend extra time on state definitions
- Draw recursion trees for backtracking
- Practice graph problems on paper first

### Mock Interview Mindset
- Treat mocks as seriously as real interviews
- If possible, do one mock with a human (friend, colleague)
- Record yourself to review communication style

### Energy Management
- Week 2 is intense - take proper breaks
- Day before interview: light review only
- Prioritize sleep over extra practice

### Interview Ready Signals
You are ready when you can:
- Identify the pattern within 5 minutes of seeing a problem
- Explain your approach before coding
- Handle hints gracefully
- Test your code methodically
- Stay calm when stuck

---

*After completing this week, review the [Day-of Checklist](./day-of-checklist) before your interview.*

---

*Last updated: January 2026*
