# Alien Dictionary

> **Deriving character ordering using topological sort**

---

## Problem Statement

**LeetCode 269: Alien Dictionary** (Premium)

There is a new alien language that uses the English alphabet. However, the order of the letters is unknown to you.

You are given a list of strings `words` from the alien language's dictionary, where the strings in `words` are **sorted lexicographically** by the rules of this new language.

Derive the order of letters in this language, and return it. If no valid ordering exists, return an empty string. If there are multiple valid solutions, return **any** of them.

---

## Visual Overview

![Alien Dictionary Visualization](./assets/alien_dictionary.png)

---

## Examples

### Example 1

```
Input: words = ["wrt", "wrf", "er", "ett", "rftt"]
Output: "wertf"

Explanation:
Compare adjacent words to find ordering:
  wrt vs wrf -> t comes before f (t < f)
  wrf vs er  -> w comes before e (w < e)
  er vs ett  -> r comes before t (r < t)
  ett vs rftt -> e comes before r (e < r)

Order relations: t->f, w->e, r->t, e->r
Topological sort: w -> e -> r -> t -> f
```

### Example 2

```
Input: words = ["z", "x"]
Output: "zx"

Explanation: z comes before x in alien order
```

### Example 3

```
Input: words = ["z", "x", "z"]
Output: ""

Explanation: z < x and x < z is a contradiction (cycle)
```

---

## Key Insight

This is a **topological sort** problem in disguise:

1. **Extract ordering rules** by comparing adjacent words
2. **Build a directed graph** where edge a->b means "a comes before b"
3. **Topological sort** gives a valid ordering (if no cycle exists)

---

## Approach

### Step 1: Extract Ordering Relations

Compare adjacent words character by character:
- Find first position where characters differ
- That gives us an ordering rule

```
"wrt" vs "wrf"
  w == w (continue)
  r == r (continue)
  t != f -> rule: t < f (t comes before f)
```

**Important**: If word1 is a prefix of word2, that's fine. But if word2 is a prefix of word1, it's invalid!

```
["abc", "ab"] -> INVALID! (longer word can't come before its prefix)
```

### Step 2: Build Graph and Topological Sort

Use Kahn's algorithm (BFS) or DFS for topological sort.

---

## Solution 1: Kahn's Algorithm (BFS)

```python
from collections import defaultdict, deque

def alienOrder(words: list[str]) -> str:
    # Build adjacency list and in-degree count
    graph = defaultdict(set)
    in_degree = {c: 0 for word in words for c in word}

    # Step 1: Extract ordering rules from adjacent words
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]

        # Check for invalid case: "abc" before "ab"
        if len(word1) > len(word2) and word1.startswith(word2):
            return ""

        # Find first differing character
        for c1, c2 in zip(word1, word2):
            if c1 != c2:
                # c1 comes before c2
                if c2 not in graph[c1]:
                    graph[c1].add(c2)
                    in_degree[c2] += 1
                break  # Only first difference matters

    # Step 2: Kahn's algorithm (BFS topological sort)
    # Start with characters having in-degree 0
    queue = deque([c for c in in_degree if in_degree[c] == 0])
    result = []

    while queue:
        char = queue.popleft()
        result.append(char)

        for neighbor in graph[char]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    # If not all characters processed, there's a cycle
    if len(result) != len(in_degree):
        return ""

    return ''.join(result)
```

::: info Complexity: Time O(C) · Space O(U + min(U^2, N))
- **Time:** O(C) to extract rules from total characters, O(U + E) for topological sort where E <= min(U^2, N-1)
- **Space:** Graph adjacency uses O(E), in-degree map O(U), queue O(U) where U = unique characters
:::

---

## Solution 2: DFS Topological Sort

```python
from collections import defaultdict

def alienOrder(words: list[str]) -> str:
    # Build adjacency list
    graph = defaultdict(set)
    all_chars = set(c for word in words for c in word)

    # Extract ordering rules
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]

        if len(word1) > len(word2) and word1.startswith(word2):
            return ""

        for c1, c2 in zip(word1, word2):
            if c1 != c2:
                graph[c1].add(c2)
                break

    # DFS with cycle detection
    # State: 0 = unvisited, 1 = visiting (in current path), 2 = visited
    state = {c: 0 for c in all_chars}
    result = []

    def dfs(char: str) -> bool:
        """Returns True if no cycle detected."""
        if state[char] == 1:
            return False  # Cycle detected!
        if state[char] == 2:
            return True  # Already processed

        state[char] = 1  # Mark as visiting

        for neighbor in graph[char]:
            if not dfs(neighbor):
                return False

        state[char] = 2  # Mark as visited
        result.append(char)  # Add in reverse post-order
        return True

    for char in all_chars:
        if not dfs(char):
            return ""

    return ''.join(reversed(result))
```

::: info Complexity: Time O(C) · Space O(U + min(U^2, N))
- **Time:** Rule extraction O(C), DFS visits each character once and traverses edges once
- **Space:** Graph O(E), state dictionary O(U), result list O(U), recursion stack O(U)
:::

---

## Complexity Analysis

| Aspect | Complexity | Explanation |
|--------|------------|-------------|
| Time | O(C) | C = total characters across all words |
| Space | O(U + min(U^2, N)) | U = unique chars, N = number of words |

- Extracting rules: O(C) where C = sum of word lengths
- Topological sort: O(V + E) where V = unique chars, E = edges

---

## Process Visualization

```mermaid
graph TD
    subgraph "Step 1: Compare Adjacent Words"
        A["wrt vs wrf"] --> A1["t -> f"]
        B["wrf vs er"] --> B1["w -> e"]
        C["er vs ett"] --> C1["r -> t"]
        D["ett vs rftt"] --> D1["e -> r"]
    end

    subgraph "Step 2: Build Graph"
        G1["w"] --> G2["e"]
        G2 --> G3["r"]
        G3 --> G4["t"]
        G4 --> G5["f"]
    end

    subgraph "Step 3: Topological Sort"
        T1["in_degree: {w:0, e:1, r:1, t:1, f:1}"]
        T2["Start: w (in_degree=0)"]
        T3["Process: w->e, e->r, r->t, t->f"]
        T4["Result: wertf"]
    end
```

---

## Common Mistakes

### 1. Missing the Prefix Check

```python
# WRONG - doesn't handle "abc" before "ab"
for c1, c2 in zip(word1, word2):
    if c1 != c2:
        graph[c1].add(c2)
        break

# CORRECT - add prefix check
if len(word1) > len(word2) and word1.startswith(word2):
    return ""
for c1, c2 in zip(word1, word2):
    if c1 != c2:
        graph[c1].add(c2)
        break
```

### 2. Adding Duplicate Edges

```python
# WRONG - can inflate in-degree
if c1 != c2:
    graph[c1].append(c2)
    in_degree[c2] += 1

# CORRECT - use set or check
if c1 != c2:
    if c2 not in graph[c1]:
        graph[c1].add(c2)
        in_degree[c2] += 1
```

### 3. Missing Characters with No Relations

```python
# WRONG - only includes characters with edges
in_degree = defaultdict(int)

# CORRECT - include ALL characters from words
in_degree = {c: 0 for word in words for c in word}
```

### 4. Not Detecting Cycle

```python
# WRONG - doesn't check if all chars processed
return ''.join(result)

# CORRECT - verify no cycle
if len(result) != len(in_degree):
    return ""  # Cycle exists
return ''.join(result)
```

---

## Edge Cases

1. **Single word**: All characters are valid, any order works
2. **All same words**: No ordering constraints, any order works
3. **Contradicting order**: `["z", "x", "z"]` - return ""
4. **Longer word before prefix**: `["abc", "ab"]` - return ""
5. **Characters with no constraints**: Include them anywhere in result

---

## Variations

### Return All Valid Orderings

Instead of returning one order, return all valid topological orderings:

```python
def allAlienOrders(words: list[str]) -> list[str]:
    # Build graph...
    # Use backtracking to find all valid orders
    pass
```

### Validate Given Ordering

Check if a proposed ordering is valid for the given words:

```python
def isValidOrder(words: list[str], order: str) -> bool:
    char_rank = {c: i for i, c in enumerate(order)}
    for i in range(len(words) - 1):
        word1, word2 = words[i], words[i + 1]
        for c1, c2 in zip(word1, word2):
            if c1 != c2:
                if char_rank.get(c1, float('inf')) > char_rank.get(c2, float('inf')):
                    return False
                break
        else:
            if len(word1) > len(word2):
                return False
    return True
```

---

## Related Problems

::: details Course Schedule (LeetCode 207)
**Problem:** Determine if you can finish all courses given prerequisite pairs (detect cycle in directed graph).

**Key Insight:** Topological sort - if cycle exists, cannot complete all courses.

```mermaid
graph LR
    A[Course 0] --> B[Course 1]
    B --> C[Course 2]
    C -.->|cycle?| A
    style A fill:#90EE90
```

**Approach:** BFS (Kahn's algorithm) counting processed nodes, or DFS with 3-state coloring for cycle detection.

**Complexity:** O(V+E) time, O(V+E) space
:::

::: details Course Schedule II (LeetCode 210)
**Problem:** Return a valid course order to finish all courses, or empty array if impossible.

**Key Insight:** Same as Course Schedule but output the topological ordering.

```mermaid
graph LR
    A[0] --> B[1]
    A --> C[2]
    B --> D[3]
    C --> D
```

**Approach:** Kahn's algorithm - track processing order. Output queue processing order.

**Complexity:** O(V+E) time, O(V+E) space
:::

::: details Verifying an Alien Dictionary (LeetCode 953)
**Problem:** Given words sorted in alien dictionary order and the order string, verify if words are sorted correctly.

**Key Insight:** Reverse of Alien Dictionary - given the order, validate the input.

```mermaid
graph TD
    A[Build char rank map] --> B[Compare adjacent words]
    B --> C{First diff char?}
    C -->|order[c1] < order[c2]| D[Valid]
    C -->|order[c1] > order[c2]| E[Invalid]
```

**Approach:** Map each character to its rank, then compare adjacent word pairs lexicographically.

**Complexity:** O(C) time where C = total characters, O(1) space
:::

::: details Find All Possible Recipes from Given Supplies (LeetCode 2115)
**Problem:** Find which recipes can be made given supplies and recipe dependencies.

**Key Insight:** Topological sort on recipe dependency graph with external supplies as sources.

```mermaid
graph TD
    A[Supplies] --> B[Recipe 1]
    A --> C[Recipe 2]
    B --> D[Recipe 3]
    C --> D
```

**Approach:** Build dependency graph, use Kahn's algorithm starting from supplies and already-makeable recipes.

**Complexity:** O(V+E) time, O(V+E) space
:::

---

## Interview Tips

1. **Clarify the input**: Words are already sorted in alien order
2. **Explain the graph model**: Characters are nodes, ordering rules are edges
3. **Choose BFS or DFS**: Both work, BFS (Kahn's) is often cleaner
4. **Handle edge cases explicitly**: Prefix check, cycle detection

### Follow-up Questions

- **Q: What if some characters don't appear in any word?**
  - A: Can place them anywhere in the order (no constraints)

- **Q: What if we want to verify an ordering is correct?**
  - A: Check each adjacent pair follows the given order

- **Q: Can there be multiple valid orderings?**
  - A: Yes, if multiple characters have the same in-degree at some point

---

## References

- [LeetCode 269: Alien Dictionary](https://leetcode.com/problems/alien-dictionary/)
- [NeetCode Solution](https://neetcode.io/solutions/alien-dictionary)
- [Alien Dictionary - In-Depth Explanation](https://algo.monster/liteproblems/269)
- [Topological Sort - Khan's Algorithm](https://www.geeksforgeeks.org/topological-sorting-indegree-based-solution/)
