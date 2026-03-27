# Tier 1: Graphs {#top}

<Badge type="danger" text="HIGHEST FREQUENCY" /> <Badge type="tip" text="8 Problems" /> <Badge type="info" text="~3 hours prep" />

> Graphs appeared in **nearly every SDE onsite** in 2024-2025. If you only study one topic, make it this one.

---

## Why Graphs?

Modern tech infrastructure _is_ a graph. Maps, search ranking, network routing, dependency resolution, knowledge graphs -- every core product touches graph algorithms daily. Interviewers pick graph problems because they:

1. **Test multiple skills at once** -- data structure choice, traversal strategy, complexity analysis, and edge-case handling.
2. **Have natural follow-ups** -- "Now what if the graph is weighted?" or "What if we need the shortest path?" lets interviewers calibrate depth.
3. **Expose weak fundamentals quickly** -- if you cannot reason about BFS vs DFS in the first 5 minutes, the round is effectively over.

::: tip Interview Insight
Candidates who received offers report that graph questions often come disguised as grid problems (islands, mazes, rotten oranges) or as dependency problems (course scheduling, build systems). Recognizing the underlying graph is half the battle.
:::

![BFS vs DFS Grid Traversal](/sde-coding/sprint/bfs_vs_dfs.png)

---

## Problem Table

| # | Problem | LC | Pattern | Why This Problem |
|---|---------|:--:|---------|------------------|
| 1 | [Number of Islands](https://leetcode.com/problems/number-of-islands/) | 200 | BFS/DFS grid | Frequently asked in SDE interviews (Jan 2025) |
| 2 | [Course Schedule](https://leetcode.com/problems/course-schedule/) | 207 | Topological Sort | Multiple SDE onsites |
| 3 | [Course Schedule II](https://leetcode.com/problems/course-schedule-ii/) | 210 | Topo Sort (return order) | Follow-up pattern |
| 4 | [Word Ladder](https://leetcode.com/problems/word-ladder/) | 127 | BFS shortest path | Frequently asked in SDE interviews |
| 5 | [Clone Graph](https://leetcode.com/problems/clone-graph/) | 133 | DFS + HashMap | Classic pattern |
| 6 | [Network Delay Time](https://leetcode.com/problems/network-delay-time/) | 743 | Dijkstra's | Teleporter variant asked |
| 7 | [Rotting Oranges](https://leetcode.com/problems/rotting-oranges/) | 994 | Multi-source BFS | Common follow-up |
| 8 | [Pacific Atlantic Water Flow](https://leetcode.com/problems/pacific-atlantic-water-flow/) | 417 | Multi-source DFS | Tests reverse thinking |

---

## Pattern Recognition

Choosing the right algorithm in the first 2 minutes is what separates "Strong Hire" from "Lean No Hire." Use this decision framework:

![Graph Representations](/sde-coding/sprint/graph_representations.png)

### Algorithm Decision Flowchart

| Question to Ask Yourself | If YES | If NO |
|--------------------------|--------|-------|
| Is the graph **unweighted** and I need the **shortest path**? | **BFS** | Keep going |
| Is the graph **weighted with positive edges** and I need shortest path? | **Dijkstra's** | Keep going |
| Do I need to detect **cycles** or find a **valid ordering** of dependencies? | **Topological Sort** (Kahn's BFS) | Keep going |
| Am I counting **connected components** or **flood-filling** a grid? | **BFS or DFS** (either works) | Keep going |
| Do I need to explore **all paths** or do **backtracking**? | **DFS** (recursive) | Keep going |
| Are there **multiple starting points** simultaneously? | **Multi-source BFS** (enqueue all sources first) | Single-source BFS/DFS |

::: info When BFS vs DFS doesn't matter
For problems like Number of Islands where you just need to visit all connected cells, BFS and DFS produce the same result. Pick whichever you can write faster in a plain text editor. BFS is generally safer (no stack overflow on large grids), but DFS recursive is fewer lines of code.
:::

::: warning When it DOES matter
- **Shortest path in unweighted graph** --> Must use BFS (DFS does NOT guarantee shortest path).
- **Topological ordering** --> Use BFS-based Kahn's algorithm (easier to code than DFS-based).
- **Dijkstra's** --> Only for weighted graphs with non-negative edges.
:::

---

## Core Templates

These four templates cover every graph problem in this tier. Memorize them well enough to write them in a plain text editor without hesitation.

### BFS Template (Adjacency List)

```python{4,7-8}
from collections import deque

def bfs(graph, start):
    queue = deque([start])       # FIFO queue
    visited = {start}
    while queue:
        node = queue.popleft()   # O(1) pop from left
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)  # mark BEFORE enqueue
                queue.append(neighbor)
    return visited
```

::: tip Critical Detail
Always add to `visited` **before** enqueueing, not after dequeueing. Adding after dequeueing causes duplicate processing and can turn O(V+E) into O(V^2) or worse.
:::

### BFS on a Grid

```python{5-6,10,14}
from collections import deque

def bfs_grid(grid, start_r, start_c):
    rows, cols = len(grid), len(grid[0])
    queue = deque([(start_r, start_c)])
    visited = {(start_r, start_c)}

    while queue:
        r, c = queue.popleft()
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
            nr, nc = r + dr, c + dc
            if (0 <= nr < rows and 0 <= nc < cols
                    and (nr, nc) not in visited
                    and grid[nr][nc] == "1"):  # condition varies
                visited.add((nr, nc))
                queue.append((nr, nc))
    return visited
```

### DFS Template (Recursive + Iterative)

```python
# ---- Recursive DFS ----
def dfs_recursive(graph, node, visited):
    visited.add(node)
    for neighbor in graph[node]:
        if neighbor not in visited:
            dfs_recursive(graph, neighbor, visited)

# ---- Iterative DFS (uses a stack) ----
def dfs_iterative(graph, start):
    stack = [start]
    visited = set()
    while stack:
        node = stack.pop()        # LIFO -- last in, first out
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                stack.append(neighbor)
    return visited
```

::: info Recursive vs Iterative
Recursive DFS is cleaner for interviews (fewer lines), but iterative DFS avoids stack overflow on large inputs. For plain text editor interviews, recursive is usually fine -- just mention the stack overflow risk if asked.
:::

### Topological Sort (Kahn's Algorithm -- BFS-based)

```python{6-8,12-15,17}
from collections import deque, defaultdict

def topo_sort(num_courses, prerequisites):
    indegree = [0] * num_courses
    adj = defaultdict(list)
    for course, prereq in prerequisites:
        adj[prereq].append(course)
        indegree[course] += 1                   # build indegree

    queue = deque(i for i in range(num_courses)
                  if indegree[i] == 0)           # start with 0-indegree
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:          # newly free
                queue.append(neighbor)

    return order if len(order) == num_courses else []  # cycle check
```

![Topological Sort — Kahn's Algorithm](/sde-coding/sprint/topological_sort.png)

### Dijkstra's Algorithm (Min-Heap)

```python{5-6,10,13-14}
import heapq
from collections import defaultdict

def dijkstra(n, edges, source):
    adj = defaultdict(list)
    for u, v, w in edges:
        adj[u].append((v, w))

    dist = [float('inf')] * (n + 1)
    dist[source] = 0
    heap = [(0, source)]         # (distance, node)

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:          # skip stale entries
            continue
        for v, w in adj[u]:
            if dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                heapq.heappush(heap, (dist[v], v))

    return dist
```

::: warning Dijkstra Pitfall
The `if d > dist[u]: continue` line is essential. Without it, you process stale heap entries and the algorithm degrades from O(E log V) to much worse. This is the #1 Dijkstra bug in interviews.
:::

![Dijkstra's Algorithm — Shortest Path](/sde-coding/sprint/dijkstra.png)

---

## Problem Walkthroughs

Work through each problem in order. For each one, try solving it yourself for 20 minutes before expanding the solution.

### Problem 1: Number of Islands (LC 200)

::: details Click to expand solution

**Pattern:** BFS/DFS grid traversal -- count connected components.

**Approach:** Scan the grid cell by cell. When you find a `'1'` that hasn't been visited, that's a new island. Run BFS/DFS from that cell to mark all connected land cells as visited. Increment your island count.

```python{8-9,15,18}
from collections import deque

def numIslands(grid):
    if not grid:
        return 0

    rows, cols = len(grid), len(grid[0])
    visited = set()
    islands = 0

    def bfs(r, c):
        queue = deque([(r, c)])
        visited.add((r, c))
        while queue:
            row, col = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = row + dr, col + dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and (nr, nc) not in visited
                        and grid[nr][nc] == "1"):
                    visited.add((nr, nc))
                    queue.append((nr, nc))

    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == "1" and (r, c) not in visited:
                bfs(r, c)
                islands += 1

    return islands
```

**Complexity:**
- Time: O(M x N) -- every cell visited at most once.
- Space: O(M x N) -- visited set in worst case (all land).

**Key Insight:** You can also modify the grid in-place (set `'1'` to `'0'` after visiting) to avoid the visited set entirely, but mention the trade-off: mutating input data. In an interview, ask the interviewer if modifying the input is acceptable.

:::

### Problem 2: Course Schedule (LC 207)

::: details Click to expand solution

**Pattern:** Topological Sort -- cycle detection in a directed graph.

**Approach:** Model courses as nodes and prerequisites as directed edges. If we can produce a valid topological ordering that includes all courses, there is no cycle and we can finish all courses. Use Kahn's algorithm (BFS-based topo sort).

```python{9,18}
from collections import deque, defaultdict

def canFinish(numCourses, prerequisites):
    indegree = [0] * numCourses
    adj = defaultdict(list)

    for course, prereq in prerequisites:
        adj[prereq].append(course)
        indegree[course] += 1

    queue = deque(i for i in range(numCourses) if indegree[i] == 0)
    completed = 0

    while queue:
        node = queue.popleft()
        completed += 1
        for neighbor in adj[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return completed == numCourses
```

**Complexity:**
- Time: O(V + E) where V = numCourses, E = len(prerequisites).
- Space: O(V + E) for adjacency list and indegree array.

**Key Insight:** If there's a cycle, some nodes will never reach indegree 0, so `completed` will be less than `numCourses`. This is cleaner than DFS-based cycle detection with three-color marking.

:::

### Problem 3: Course Schedule II (LC 210)

::: details Click to expand solution

**Pattern:** Topological Sort -- same as Course Schedule but return the actual ordering.

**Approach:** Identical to Course Schedule, but instead of just counting, collect the order in a list. If the list length doesn't equal `numCourses`, return an empty list (cycle exists).

```python{12,22}
from collections import deque, defaultdict

def findOrder(numCourses, prerequisites):
    indegree = [0] * numCourses
    adj = defaultdict(list)

    for course, prereq in prerequisites:
        adj[prereq].append(course)
        indegree[course] += 1

    queue = deque(i for i in range(numCourses) if indegree[i] == 0)
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for neighbor in adj[node]:
            indegree[neighbor] -= 1
            if indegree[neighbor] == 0:
                queue.append(neighbor)

    return order if len(order) == numCourses else []
```

**Complexity:**
- Time: O(V + E).
- Space: O(V + E).

**Key Insight:** This is the natural follow-up to LC 207. In an interview, if you solve Course Schedule quickly, expect the interviewer to ask "Now return the actual order." Having Kahn's memorized makes this a 2-line change.

:::

### Problem 4: Word Ladder (LC 127)

::: details Click to expand solution

**Pattern:** BFS shortest path in an unweighted implicit graph.

**Approach:** Each word is a node. Two words are connected if they differ by exactly one character. BFS from `beginWord` finds the shortest transformation sequence. The key optimization: instead of comparing all word pairs O(N^2), for each word generate all possible one-character mutations and check if they exist in the word set.

```python{9-10,14-15,20-23}
from collections import deque

def ladderLength(beginWord, endWord, wordList):
    word_set = set(wordList)
    if endWord not in word_set:
        return 0

    queue = deque([(beginWord, 1)])  # (word, depth)
    visited = {beginWord}

    while queue:
        word, depth = queue.popleft()
        if word == endWord:
            return depth

        for i in range(len(word)):
            for c in 'abcdefghijklmnopqrstuvwxyz':
                next_word = word[:i] + c + word[i+1:]
                if next_word in word_set and next_word not in visited:
                    visited.add(next_word)
                    queue.append((next_word, depth + 1))

    return 0
```

**Complexity:**
- Time: O(M^2 x N) where M = word length, N = number of words. For each word, we try 26 x M mutations, and string slicing is O(M).
- Space: O(M x N) for the word set and visited set.

**Key Insight:** The implicit graph construction is what makes this tricky. Do NOT build an explicit adjacency list by comparing all pairs -- that's O(N^2 x M). The wildcard pattern approach (generating `*ot`, `h*t`, `ho*` for "hot") is an alternative that uses a dictionary mapping patterns to words, which can be faster for very large word lists.

:::

### Problem 5: Clone Graph (LC 133)

::: details Click to expand solution

**Pattern:** DFS + HashMap for deep copying a graph structure.

**Approach:** Use a hashmap to map each original node to its clone. DFS through the graph: for each node, create a clone if it doesn't exist yet, then recursively clone all neighbors. The hashmap serves double duty as both a visited set and a lookup table.

```python{7,10-11,14-15}
class Node:
    def __init__(self, val=0, neighbors=None):
        self.val = val
        self.neighbors = neighbors if neighbors is not None else []

def cloneGraph(node):
    if not node:
        return None

    cloned = {}  # original node -> cloned node

    def dfs(n):
        if n in cloned:
            return cloned[n]
        clone = Node(n.val)
        cloned[n] = clone           # map BEFORE recursing (handles cycles)
        for neighbor in n.neighbors:
            clone.neighbors.append(dfs(neighbor))
        return clone

    return dfs(node)
```

**Complexity:**
- Time: O(V + E) -- visit each node and edge once.
- Space: O(V) for the hashmap and recursion stack.

**Key Insight:** You MUST add the clone to the hashmap BEFORE recursing into neighbors. Otherwise, cycles in the graph cause infinite recursion. This is the single most common bug in this problem.

:::

### Problem 6: Network Delay Time (LC 743)

::: details Click to expand solution

**Pattern:** Dijkstra's algorithm -- single-source shortest path in a weighted directed graph.

**Approach:** Run Dijkstra's from the source node `k`. After processing, the answer is the maximum distance among all nodes (the time for the signal to reach the farthest node). If any node is unreachable (`inf`), return -1.

```python{11,14-15,22}
import heapq
from collections import defaultdict

def networkDelayTime(times, n, k):
    adj = defaultdict(list)
    for u, v, w in times:
        adj[u].append((v, w))

    dist = [float('inf')] * (n + 1)
    dist[k] = 0
    heap = [(0, k)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue                # skip stale entries
        for v, w in adj[u]:
            new_dist = dist[u] + w
            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    result = max(dist[1:])          # nodes are 1-indexed
    return result if result < float('inf') else -1
```

**Complexity:**
- Time: O(E log V) with a binary heap.
- Space: O(V + E).

**Key Insight:** The "teleporter" variant interviewers ask is essentially: "what if certain edges have weight 0?" This is a 0-1 BFS problem (use a deque, push weight-0 edges to front, weight-1 edges to back). Mentioning this optimization unprompted shows depth.

:::

### Problem 7: Rotting Oranges (LC 994)

::: details Click to expand solution

**Pattern:** Multi-source BFS -- BFS from multiple starting points simultaneously.

**Approach:** Enqueue ALL rotten oranges at the start (not just one). Each BFS "level" represents one minute passing. After BFS completes, check if any fresh oranges remain.

```python{7-12,15,21}
from collections import deque

def orangesRotting(grid):
    rows, cols = len(grid), len(grid[0])
    queue = deque()
    fresh = 0

    # Enqueue ALL rotten oranges as starting points
    for r in range(rows):
        for c in range(cols):
            if grid[r][c] == 2:
                queue.append((r, c))
            elif grid[r][c] == 1:
                fresh += 1

    if fresh == 0:
        return 0

    minutes = 0
    while queue:
        minutes += 1
        for _ in range(len(queue)):    # process level by level
            r, c = queue.popleft()
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < rows and 0 <= nc < cols
                        and grid[nr][nc] == 1):
                    grid[nr][nc] = 2   # mark rotten (acts as visited)
                    fresh -= 1
                    queue.append((nr, nc))

    return minutes - 1 if fresh == 0 else -1
```

**Complexity:**
- Time: O(M x N) -- each cell processed at most once.
- Space: O(M x N) -- queue size in worst case.

**Key Insight:** The "minus 1" at the end (`minutes - 1`) trips up many candidates. The last BFS level finds no new oranges to rot, so it over-counts by 1. Alternatively, initialize `minutes = -1` before the while loop. Also, note we modify the grid in-place (setting fresh to 2) to avoid a separate visited set -- ask the interviewer if this is acceptable.

:::

### Problem 8: Pacific Atlantic Water Flow (LC 417)

::: details Click to expand solution

**Pattern:** Multi-source DFS -- reverse the problem direction.

**Approach:** Instead of asking "can water flow FROM this cell TO the ocean?" (hard), ask "can water flow FROM the ocean TO this cell?" (easier). Run DFS from all Pacific-border cells and all Atlantic-border cells separately. The answer is the intersection of cells reachable from both oceans.

```python{8-10,17-18,27}
class Solution:
    def pacificAtlantic(self, heights: List[List[int]]) -> List[List[int]]:
        if not heights:
            return []
        m, n = len(heights), len(heights[0])
        directions = [(1,0), (-1,0), (0,1), (0,-1)]
        
        def dfs(i, j, visited):
            visited.add((i, j))
            for dx, dy in directions:
                x, y = i + dx, j + dy
                if 0 <= x < m and 0 <= y < n:
                    if (x, y) not in visited and heights[x][y] >= heights[i][j]:
                        dfs(x, y, visited)

        pacific, atlantic = set(), set()
        for j in range(n): dfs(0, j, pacific)
        for i in range(m): dfs(i, 0, pacific)
        for j in range(n): dfs(m-1, j, atlantic)
        for i in range(m): dfs(i, n-1, atlantic)
        return list(pacific & atlantic)
```

**Complexity:**
- Time: O(M x N) -- each cell visited at most twice (once per ocean).
- Space: O(M x N) for the two reachable sets and recursion stack.

**Key Insight:** The "reverse thinking" is the entire difficulty. Water flows from high to low, but we DFS from the ocean uphill (we only move to cells with height >= current). This transforms a problem with M x N starting points into one with only 2(M + N) starting points. Mention this inversion technique in the interview -- it shows algorithmic maturity.

:::

---

## Common Mistakes

::: danger Mistakes That Cost Offers

**1. Forgetting to mark visited BEFORE enqueueing in BFS**
```python
# WRONG -- causes duplicates in queue
node = queue.popleft()
visited.add(node)        # too late! other paths already enqueued this node

# CORRECT -- mark when adding to queue
if neighbor not in visited:
    visited.add(neighbor)    # mark immediately
    queue.append(neighbor)
```

**2. Using DFS for shortest path problems**

DFS does NOT guarantee shortest path in an unweighted graph. If the problem says "minimum number of steps/moves/transformations," you MUST use BFS.

**3. Off-by-one in level-order BFS**

When counting levels (minutes, steps), remember that the last BFS iteration where nothing new is discovered still increments your counter. Either start at -1 or subtract 1 at the end.

**4. Wrong edge direction in topological sort**

For Course Schedule, `[a, b]` means "b is a prerequisite for a" (b -> a). Many candidates reverse this and get wrong answers. Always clarify edge direction with the interviewer.

**5. Skipping the stale-entry check in Dijkstra's**
```python
d, u = heapq.heappop(heap)
if d > dist[u]:
    continue    # THIS LINE IS ESSENTIAL
```
Without this, you reprocess nodes with outdated distances, which is both slower and can cause incorrect results in certain variants.

**6. Not handling disconnected components**

For "Number of Islands" or any connected-component problem, you need an outer loop that starts BFS/DFS from every unvisited node -- not just the first one you find.

**7. Mutating input without permission**

Some solutions mark visited cells by modifying the grid (e.g., setting `'1'` to `'0'`). In an interview, always ask: "Is it okay to modify the input?" Some interviewers specifically disallow it.

:::

---

## Practice Checklist

Track your progress. Solve each problem in a plain text editor (no IDE) without running the code.

| Done | # | Problem | Solved Clean? | Under 25 min? |
|:----:|---|---------|:-------------:|:--------------:|
| <input type="checkbox" /> | 1 | Number of Islands | <input type="checkbox" /> | <input type="checkbox" /> |
| <input type="checkbox" /> | 2 | Course Schedule | <input type="checkbox" /> | <input type="checkbox" /> |
| <input type="checkbox" /> | 3 | Course Schedule II | <input type="checkbox" /> | <input type="checkbox" /> |
| <input type="checkbox" /> | 4 | Word Ladder | <input type="checkbox" /> | <input type="checkbox" /> |
| <input type="checkbox" /> | 5 | Clone Graph | <input type="checkbox" /> | <input type="checkbox" /> |
| <input type="checkbox" /> | 6 | Network Delay Time | <input type="checkbox" /> | <input type="checkbox" /> |
| <input type="checkbox" /> | 7 | Rotting Oranges | <input type="checkbox" /> | <input type="checkbox" /> |
| <input type="checkbox" /> | 8 | Pacific Atlantic Water Flow | <input type="checkbox" /> | <input type="checkbox" /> |

::: tip Definition of "Done"
A problem is truly done when you can:
1. Write the full solution from memory in a plain text editor.
2. Explain the pattern and complexity out loud in under 60 seconds.
3. Handle at least one follow-up variant (e.g., "What if the grid wraps around?" or "What if edges have weights?").
:::

---

<div style="text-align: center; margin-top: 2rem; color: var(--vp-c-text-2);">

**8 problems. 4 patterns. You've got this.**

[Back to Battle Plan](./) | [Next: Tier 2 -- Trees](./trees)

</div>
