---
title: Java-Specific Pointers
description: Idioms, gotchas, and performance tips for using Java in coding interviews. Covers collections, boxing, concurrency-free data structures, and modern Java features aligned to interview timing.
---

# Java-Specific Pointers

> **Execute cleanly in Java when the interview clock is running**

---

## Overview

This page is a deep-dive companion to [Choosing a Language](./choosing-language.md). That page helps you decide which language to use; this page helps you write idiomatic, bug-free Java fast once you have decided. Every section focuses on patterns that recur in coding interviews (LeetCode-style algorithms), not general-purpose Java engineering.

Target baseline: **Java 17 LTS**. Every snippet here compiles with `javac 17.0.x` and runs on the HotSpot JVM.

::: tip Audience
You already know Java fundamentals (classes, generics, collections). You want to stop wasting minutes on `Integer` cache surprises, `Arrays.sort` descending on primitives, or `StringBuilder` vs `String.concat`. Skim the table of contents, bookmark sections you forget, and revisit the cheat sheet the night before an onsite.
:::

---

## Essential Imports and Boilerplate

Most interview solutions need a small, predictable set of imports. Paste this at the top and delete unused lines at the end:

```java
import java.util.*;
import java.util.stream.*;

class Solution {
    // your method goes here
}
```

`java.util.*` covers `List`, `ArrayList`, `Map`, `HashMap`, `Set`, `HashSet`, `Deque`, `ArrayDeque`, `PriorityQueue`, `TreeMap`, `TreeSet`, `Arrays`, `Collections`, `Comparator`. `java.util.stream.*` covers `Stream`, `IntStream`, `Collectors`.

For competitive programming (less common in interviews), you may also need:

```java
import java.io.*;

BufferedReader br = new BufferedReader(new InputStreamReader(System.in));
// br.readLine() ~3-5x faster than Scanner.nextLine() for large inputs
```

Interview reality: `Scanner` is fine unless the problem explicitly flags I/O-heavy input.

---

## Primitives vs Boxed Types

Java has eight primitives (`int`, `long`, `double`, `boolean`, `char`, `byte`, `short`, `float`) and corresponding boxed types (`Integer`, `Long`, ...). This distinction is the source of the most common Java interview bugs.

### The Integer Cache Trap

Java caches boxed `Integer` values in the range `[-128, 127]`. Inside this range, `==` on two `Integer` references happens to return `true`. Outside, it silently returns `false`:

```java
Integer a = 127, b = 127;
Integer c = 128, d = 128;
System.out.println(a == b); // true  (cached reference)
System.out.println(c == d); // false (separate heap objects)
System.out.println(c.equals(d)); // true (always safe)
```

**Rule:** Use `.equals()` for object comparison. Reserve `==` for primitive types and null checks.

### Autoboxing Cost in Hot Loops

Autoboxing allocates on every conversion. Inside a tight inner loop, this can turn an O(n) algorithm into a heap-thrashing mess:

```java
List<Integer> nums = List.of(1, 2, 3, 4, 5);
int sum = 0;
for (int n : nums) sum += n; // enhanced-for unboxes once per element — cheap
```

```java
// anti-pattern: boxed accumulator
Integer boxedSum = 0;
for (int n : new int[]{1, 2, 3}) boxedSum += n; // allocates Integer each iteration
```

Keep accumulators as primitives: `int` or `long`.

### Overflow

`int` overflows silently at `Integer.MAX_VALUE` (`2^31 - 1`). Use `long` when multiplying or summing values that could exceed `~2.1 * 10^9`:

```java
int n = 1_000_000;
long product = (long) n * n; // cast before multiply, not after
```

A frequent interview bug: `(low + high) / 2` overflows for large `int` indices. Use `low + (high - low) / 2` instead:

```java
int mid = low + (high - low) / 2;
```

---

## Collections Framework Cheat Sheet

Problem pattern → concrete class. Memorize this table:

| Pattern | Abstract type | Concrete class | Why |
|---|---|---|---|
| Dynamic array | `List<T>` | `ArrayList<>` | O(1) amortized append, O(1) index |
| Stack (LIFO) | `Deque<T>` | `ArrayDeque<>` | Faster than legacy `Stack` |
| Queue (FIFO) | `Deque<T>` or `Queue<T>` | `ArrayDeque<>` | Faster than `LinkedList` |
| Double-ended queue | `Deque<T>` | `ArrayDeque<>` | Constant-time both ends |
| Hash map | `Map<K,V>` | `HashMap<>` | Average O(1) lookup |
| Sorted map (floor/ceiling) | `NavigableMap<K,V>` | `TreeMap<>` | O(log n) ordered ops |
| Hash set | `Set<T>` | `HashSet<>` | Average O(1) lookup |
| Sorted set | `NavigableSet<T>` | `TreeSet<>` | O(log n) ordered ops |
| Priority queue (min-heap) | `Queue<T>` | `PriorityQueue<>` | O(log n) insert/poll |
| Insertion-order map (LRU) | `Map<K,V>` | `LinkedHashMap<>` | Maintains insertion order |

---

## Why ArrayDeque over Stack and LinkedList

The `Stack` class is a legacy synchronized descendant of `Vector`. Every `push`/`pop` takes a lock. `LinkedList` as a queue adds pointer-chasing overhead per element.

```java
Deque<Integer> stack = new ArrayDeque<>();
stack.push(1);                 // preferred over Stack.push
stack.push(2);
int top = stack.pop();         // returns 2
int peek = stack.peek();       // returns 1

Deque<Integer> queue = new ArrayDeque<>();
queue.offer(1);                // enqueue at tail
queue.offer(2);
int front = queue.poll();      // returns 1
```

One caveat: `ArrayDeque` does **not** permit `null` elements. If you genuinely need nulls in a deque (rare in interviews), fall back to `LinkedList`.

---

## PriorityQueue Idioms

`PriorityQueue` is a binary min-heap by default (smallest element at head).

```java
PriorityQueue<Integer> minHeap = new PriorityQueue<>();
minHeap.offer(3); minHeap.offer(1); minHeap.offer(2);
int smallest = minHeap.poll(); // returns 1
```

### Max-heap

```java
PriorityQueue<Integer> maxHeap = new PriorityQueue<>(Comparator.reverseOrder());
```

### Custom comparator

Sort by a derived field — for example, closest point to origin:

```java
int[][] points = {{1, 3}, {-2, 2}, {5, 8}};
PriorityQueue<int[]> pq = new PriorityQueue<>(
    (a, b) -> Integer.compare(a[0]*a[0] + a[1]*a[1], b[0]*b[0] + b[1]*b[1])
);
for (int[] p : points) pq.offer(p);
int[] closest = pq.poll(); // {1, 3}
```

### Top-K pattern

Keep a **min-heap of size K** to find the K largest elements. The heap's root is the "smallest of the large" — evict it whenever a bigger value arrives:

```java
PriorityQueue<Integer> heap = new PriorityQueue<>();
for (int n : nums) {
    heap.offer(n);
    if (heap.size() > k) heap.poll();
}
// heap now contains the k largest values
```

---

## HashMap Idioms

### getOrDefault

Replaces the verbose `if (map.containsKey(k)) ... else ...` pattern:

```java
Map<Character, Integer> freq = new HashMap<>();
for (char c : "aabbc".toCharArray()) {
    freq.put(c, freq.getOrDefault(c, 0) + 1);
}
```

### merge (cleaner for counters)

```java
Map<Character, Integer> freq = new HashMap<>();
for (char c : "aabbc".toCharArray()) {
    freq.merge(c, 1, Integer::sum);
}
```

### computeIfAbsent (grouping pattern)

Building an adjacency list or groups-by-key:

```java
Map<String, List<String>> groups = new HashMap<>();
String[] words = {"eat", "tea", "tan", "ate", "nat", "bat"};
for (String w : words) {
    char[] chars = w.toCharArray();
    Arrays.sort(chars);
    String key = new String(chars);
    groups.computeIfAbsent(key, k -> new ArrayList<>()).add(w);
}
```

`computeIfAbsent` returns the existing or newly-created value, so the chained `.add(w)` works either way.

---

## String Handling

Strings in Java are immutable. Every `+=` inside a loop allocates a new `char[]`, which is O(n²) across n concatenations:

```java
// anti-pattern
String out = "";
for (int i = 0; i < n; i++) out += "x"; // O(n^2)
```

```java
StringBuilder sb = new StringBuilder();
for (int i = 0; i < n; i++) sb.append('x'); // O(n)
String out = sb.toString();
```

### Char-array trick for index access

`charAt(i)` involves a bounds check. For hot loops, convert once and reuse:

```java
char[] chars = s.toCharArray();
for (int i = 0; i < chars.length; i++) {
    // direct char[] access — slightly faster and cleaner for sliding window
}
```

### String.split has regex semantics

`"a.b.c".split(".")` returns `[]` because `.` is the regex "any char". Use `Pattern.quote` or a different delimiter:

```java
String[] parts = "a.b.c".split("\\.");           // ["a", "b", "c"]
String[] alt   = "a.b.c".split(Pattern.quote(".")); // same
```

`Pattern` lives in `java.util.regex.Pattern` — add the import or use the escape form.

---

## Array and List Conversions

Java's array/list bridge is surprisingly rough. Memorize these four conversions:

```java
// int[] -> List<Integer>
int[] arr = {1, 2, 3};
List<Integer> list = Arrays.stream(arr).boxed().collect(Collectors.toList());

// List<Integer> -> int[]
int[] back = list.stream().mapToInt(Integer::intValue).toArray();

// Object array (including Integer[]) -> List
Integer[] boxed = {1, 2, 3};
List<Integer> list2 = Arrays.asList(boxed); // fixed-size view (!)

// List<List<Integer>> -> int[][]
List<List<Integer>> matrix = new ArrayList<>();
int[][] array2d = matrix.stream()
    .map(row -> row.stream().mapToInt(Integer::intValue).toArray())
    .toArray(int[][]::new);
```

::: warning Arrays.asList pitfall
`Arrays.asList(arr)` returns a fixed-size list backed by the array. Calling `.add()` or `.remove()` throws `UnsupportedOperationException`. Wrap in `new ArrayList<>(Arrays.asList(arr))` if you need a mutable list.
:::

---

## 2D Arrays

Java's 2D arrays are arrays of arrays (jagged by default):

```java
int[][] grid = new int[3][4];     // 3 rows, 4 cols, all zeros
int[][] fixed = {{1, 2}, {3, 4}, {5, 6}};
int rows = grid.length;
int cols = grid[0].length;
```

Fill with a non-zero default:

```java
int[][] dp = new int[n][m];
for (int[] row : dp) Arrays.fill(row, -1); // memoization sentinel
```

Iterate:

```java
for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
        // grid[i][j]
    }
}
```

For BFS on a grid, the 4-direction offset array is a frequent idiom:

```java
int[][] DIRS = {{0, 1}, {0, -1}, {1, 0}, {-1, 0}};
for (int[] d : DIRS) {
    int ni = i + d[0], nj = j + d[1];
    if (ni >= 0 && ni < rows && nj >= 0 && nj < cols) {
        // visit grid[ni][nj]
    }
}
```

---

## Recursion and Stack Depth

The HotSpot JVM defaults to ~512KB stack per thread (varies by OS/architecture). Deep recursion on large inputs (e.g., a linked list of 100,000 nodes or a degenerate BST) will throw `StackOverflowError`.

**Interview guidance:**
- Recursion is fine for balanced trees (depth ~log n) and backtracking (depth bounded by input).
- Convert to iterative with an explicit `Deque` stack when the call tree can reach depth in the tens of thousands.
- For DFS on a linked list with 10^5 nodes, iterate.

```java
// Iterative inorder traversal — no recursion
List<Integer> result = new ArrayList<>();
Deque<TreeNode> stack = new ArrayDeque<>();
TreeNode curr = root;
while (curr != null || !stack.isEmpty()) {
    while (curr != null) { stack.push(curr); curr = curr.left; }
    curr = stack.pop();
    result.add(curr.val);
    curr = curr.right;
}
```

If you must keep recursion in a system with deep input, tell the interviewer you would run with `-Xss8m` to raise the stack size, then explain the iterative fallback.

---

## Common Gotchas

| Gotcha | Symptom | Fix |
|---|---|---|
| `int` overflow in `(low + high) / 2` | Wrong midpoint on large indices | `low + (high - low) / 2` |
| `Arrays.sort(int[])` can't take `Comparator` | Compile error with lambda | Box: `Integer[] arr = ...; Arrays.sort(arr, cmp)` |
| Descending primitive sort | No direct API | Sort asc, then reverse, or box first |
| `Integer[] cmp` subtraction | Overflow for `MIN_VALUE - MAX_VALUE` | `Integer.compare(a, b)` |
| Integer division truncation | `5 / 2 == 2`, not 2.5 | Cast to `double` when needed |
| `HashMap` iteration order | Non-deterministic | Use `LinkedHashMap` for stable order |
| `List.of(...)` is immutable | `add()` throws | `new ArrayList<>(List.of(...))` |

### Safe comparator idioms

```java
// Primitive-safe numeric comparator
Arrays.sort(array, (a, b) -> Integer.compare(a.val, b.val));

// Multi-key (lexicographic)
Arrays.sort(array, Comparator
    .comparingInt((int[] a) -> a[0])
    .thenComparingInt(a -> a[1]));

// Reverse primitive sort via boxing
Integer[] boxed = Arrays.stream(arr).boxed().toArray(Integer[]::new);
Arrays.sort(boxed, Comparator.reverseOrder());
```

---

## Streams: When to Use and When to Avoid

Streams are expressive but carry three costs in interview settings:

1. **Reviewability.** An interviewer watching you type prefers line-by-line imperative code they can step through.
2. **Debuggability.** A stream pipeline with a bug is harder to bisect than a for-loop.
3. **Boxing.** Object streams autobox; only `IntStream`, `LongStream`, `DoubleStream` avoid it.

**Use streams when:**
- One-line aggregates: `Arrays.stream(nums).sum()`, `Arrays.stream(nums).max().getAsInt()`.
- Conversions already covered above (`mapToInt`, `boxed`, `toArray`).
- Final collection step with `Collectors.toList()`.

**Avoid streams when:**
- You have early exit / `break` logic.
- You mutate shared state.
- You are nested three levels deep — the imperative version will read better.

---

## Modern Java Features Worth Knowing

Java 17 LTS is the assumed baseline. Features worth having in muscle memory:

```java
// var (Java 10+) — local variable type inference
var map = new HashMap<String, List<Integer>>();

// Records (Java 14+) — concise immutable data classes
record Point(int x, int y) {}
Point p = new Point(3, 4);
System.out.println(p.x()); // accessor

// Enhanced switch (Java 14+) — expression form with arrows
String day = switch (n) {
    case 1, 2, 3, 4, 5 -> "weekday";
    case 6, 7 -> "weekend";
    default -> throw new IllegalArgumentException();
};

// Text blocks (Java 15+)
String json = """
    {"name": "alice", "age": 30}
    """;
```

`var` is nice for interview speed but can hide types from the interviewer — prefer explicit types for method parameters, return values, and any variable whose type is not obvious from the RHS.

---

## Idiomatic Patterns for Interview Problems

| Pattern | Java idiom |
|---|---|
| Sliding window | `int left = 0; for (int right = 0; right < n; right++) { ... while (shrink) left++; }` |
| Two pointers | `int l = 0, r = n - 1; while (l < r) { ... }` |
| Hash frequency | `Map<Character, Integer> freq = new HashMap<>(); for (char c : s.toCharArray()) freq.merge(c, 1, Integer::sum);` |
| BFS (grid) | `Deque<int[]> q = new ArrayDeque<>(); q.offer(new int[]{r, c}); while (!q.isEmpty()) { int[] cur = q.poll(); ... }` |
| DFS (tree) | Recursive method on `TreeNode root` |
| Backtracking | `void backtrack(List<Integer> path, int start) { if (done) result.add(new ArrayList<>(path)); for (...) { path.add(x); backtrack(...); path.remove(path.size()-1); } }` |
| Top-K | Min-heap of size K (see PriorityQueue section) |
| Union-Find | `int[] parent; int find(int x) { return parent[x] == x ? x : (parent[x] = find(parent[x])); }` |
| Prefix sum | `long[] pre = new long[n+1]; for (int i = 0; i < n; i++) pre[i+1] = pre[i] + nums[i];` |

---

## Performance Tips

### ArrayList vs LinkedList

In practice `ArrayList` beats `LinkedList` for nearly every interview use case — including frequent middle insertions, because the array's cache locality dominates the theoretical O(n) shift cost.

### Initial capacity

If you know the final size, size the collection at construction to avoid resize copies:

```java
Map<Integer, Integer> map = new HashMap<>(nums.length * 4 / 3);
List<String> list = new ArrayList<>(n);
```

The `* 4 / 3` is because `HashMap` default load factor is 0.75.

### Avoid autoboxing in hot loops

Prefer `int[]` and `long[]` over `List<Integer>` for inner DP tables, prefix sums, and BFS queues keyed by index. Only box at collection boundaries (return value, final answer).

### String.intern

Rarely useful in interviews — mention it only if asked about memory-dedup scenarios.

---

## Minimal Java Skeleton for a LeetCode-Style Problem

```java
import java.util.*;

class Solution {
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> seen = new HashMap<>();
        for (int i = 0; i < nums.length; i++) {
            int complement = target - nums[i];
            if (seen.containsKey(complement)) {
                return new int[]{seen.get(complement), i};
            }
            seen.put(nums[i], i);
        }
        return new int[0];
    }
}
```

Note: method signature follows LeetCode Java conventions — `public` method, array return. Tests drive through `new Solution().twoSum(...)`.

---

## Further Reading

- [Choosing a Language](./choosing-language.md) — decide if Java is the right tool for your interview.
- [Tips for Acing Technical Interviews](./interview-tips.md) — general strategy across all languages.
- [Code Templates](../sprint/templates.md) — copy-paste starter code for common patterns in Python and Java.
- Oracle documentation: [The Java Tutorials — Collections](https://docs.oracle.com/javase/tutorial/collections/).
- Joshua Bloch, *Effective Java* (3rd ed.) — the canonical reference for idiomatic Java.

::: tip One-night-before revision
If you have 20 minutes the night before an interview, reread: Integer cache, `HashMap` idioms (`merge`, `computeIfAbsent`), `PriorityQueue` custom comparator, the 2D array DIRS pattern, and the common-gotchas table.
:::
