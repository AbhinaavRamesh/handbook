# Tier 5: HashMap + Heap + Binary Search

> **6 problems** | Priority: **Medium** | Time estimate: **2-3 hours**

---

## Three Tools, One Tier

HashMap, Heap, and Binary Search are your optimization swiss army knives. They rarely define a problem category on their own -- instead, they show up as the key ingredient that takes a brute-force solution from O(n^2) down to O(n) or O(n log n). If graphs and trees test your ability to model problems, this tier tests your ability to **recognize which tool shaves off a factor of n**.

- **HashMap** turns "find something" from O(n) to O(1). Grouping, counting, caching -- it is the workhorse of optimization.
- **Heap** answers "give me the best/worst K" without fully sorting. Any time you hear "top K", "Kth largest", or "merge sorted things", reach for a heap.
- **Binary Search** halves the search space every step. It works on sorted arrays, but also on **answer spaces** -- any monotonic function can be binary searched.

Master these three and you will have a tool for nearly every optimization question interviewers throw at you.

---

## Problem Table

| # | Problem | LC # | Pattern | Why |
|---|---------|------|---------|-----|
| 24 | [Group Anagrams](https://leetcode.com/problems/group-anagrams) | 49 | HashMap | "Group rotations" variant |
| 25 | [Top K Frequent Elements](https://leetcode.com/problems/top-k-frequent-elements) | 347 | Heap / Bucket sort | "Most talkative users" variant |
| 26 | [Merge K Sorted Lists](https://leetcode.com/problems/merge-k-sorted-lists) | 23 | Min-heap | Classic |
| 27 | [Search in Rotated Sorted Array](https://leetcode.com/problems/search-in-rotated-sorted-array) | 33 | Binary search | Tricky edge cases |
| 28 | [Find Peak Element](https://leetcode.com/problems/find-peak-element) | 162 | Binary search | Quick pattern |
| 29 | [Kth Largest Element in an Array](https://leetcode.com/problems/kth-largest-element-in-an-array) | 215 | Quickselect / Heap | Common |

---

## HashMap Patterns

HashMaps solve three families of problems in interviews:

### 1. Grouping

Given a collection of items, group them by some computed key. The trick is choosing the right key function.

```python
# Grouping pattern
from collections import defaultdict

def group_by_key(items):
    groups = defaultdict(list)
    for item in items:
        key = compute_key(item)  # the creative part
        groups[key].append(item)
    return list(groups.values())
```

**Group Anagrams** uses `tuple(sorted(word))` or a character-count tuple as the key. Interviewers have asked variants like "group rotations" (cyclic permutations of a string) -- same pattern, different key function.

### 2. Frequency Counting

Count occurrences, then act on the counts. Often paired with a heap or bucket sort for "top K" style questions.

```python
from collections import Counter
freq = Counter(nums)
# Now you can sort by frequency, filter, bucket, etc.
```

### 3. Two-Sum Style (Complement Lookup)

Store what you have seen so far, and for each new element check if its complement exists.

```python
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
```

::: tip Interview Insight
Interviewers often disguise grouping problems. "Group rotations", "group shifted strings", "group by frequency signature" -- they all reduce to: pick a canonical key, dump into a `defaultdict(list)`.
:::

---

## Heap Patterns

![Heap Operations — Min-Heap Structure](/sde-coding/sprint/heap_operations.png)

Python's `heapq` module gives you a **min-heap only**. This is the single most common source of bugs in interviews.

### Python heapq Gotchas

```python
import heapq

# MIN-HEAP (default) -- smallest element comes out first
heap = []
heapq.heappush(heap, 5)
heapq.heappush(heap, 2)
heapq.heappush(heap, 8)
print(heapq.heappop(heap))  # 2 (smallest)

# MAX-HEAP trick -- negate all values
max_heap = []
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -2)
heapq.heappush(max_heap, -8)
print(-heapq.heappop(max_heap))  # 8 (largest)

# TUPLE COMPARISON -- heapq compares by first element, then second
# Use this for (priority, value) pairs
heapq.heappush(heap, (priority, tiebreaker, value))
```

::: danger HEAPQ IS MIN-HEAP ONLY
Every year, candidates lose points by forgetting to negate values when they need a max-heap. If you need the K **largest**, either:
1. Use a min-heap of size K (pop when size exceeds K -- the smallest falls out, leaving K largest), or
2. Negate all values and use min-heap as max-heap.

Option 1 is cleaner. Option 2 is simpler when you need to repeatedly extract the max.
:::

### Pattern: Top-K Elements

```python
import heapq

def top_k(nums, k):
    """Keep a min-heap of size k. The root is the kth largest."""
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)  # remove smallest -- not in top K
    return heap
```

### Pattern: Merge K Sorted Things

```python
import heapq

def merge_k_sorted(lists):
    """Push the head of each list, pop smallest, push its successor."""
    heap = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))  # (value, list_idx, elem_idx)

    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    return result
```

### Pattern: Kth Element

Two approaches:
1. **Min-heap of size K** -- O(n log k) time, O(k) space. Push all, pop when size > k. Root is the answer.
2. **Quickselect** -- O(n) average, O(n^2) worst. Partition like quicksort, recurse on the side containing index k.

---

## Binary Search Patterns

![Binary Search — Halving the Search Space](/sde-coding/sprint/binary_search.png)

Binary search is not just "find element in sorted array." It is a general technique for any scenario where you can split a search space in half.

### The Template

```python
def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2  # avoid overflow (matters in Java/C++)
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1  # not found (lo is insertion point)
```

::: warning OFF-BY-ONE
The `lo <= hi` vs `lo < hi` distinction is critical:
- `lo <= hi`: use when searching for an exact value. Loop ends when `lo > hi` (empty range).
- `lo < hi`: use when narrowing to a single candidate. Loop ends when `lo == hi` (one element left).

Pick one convention and stick with it.
:::

### Pattern: Rotated Sorted Array

The array has two sorted halves. At each step, determine which half is sorted and whether the target is in that half.

```python
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] == target:
            return mid
        # Left half is sorted
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        # Right half is sorted
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1
```

### Pattern: Answer Space Binary Search

When the answer itself can be binary searched (the function is monotonic):

```python
def answer_space_bs():
    lo, hi = MIN_ANSWER, MAX_ANSWER
    while lo < hi:
        mid = lo + (hi - lo) // 2
        if feasible(mid):
            hi = mid       # mid might be the answer, search left
        else:
            lo = mid + 1   # mid is not feasible, search right
    return lo  # lo == hi, the smallest feasible answer
```

### Pattern: Find Peak Element

If `nums[mid] < nums[mid+1]`, a peak must exist to the right. Otherwise, a peak exists at `mid` or to its left. This works even on unsorted arrays.

---

## Problem Walkthroughs

::: details Problem 24: Group Anagrams (LC 49) -- HashMap

**Key Insight:** Two strings are anagrams if and only if they have the same character frequency. Use a canonical form as the hash key.

```python
from collections import defaultdict

def groupAnagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        # Option 1: sorted string as key -- O(k log k) per word
        key = tuple(sorted(s))

        # Option 2: character count tuple as key -- O(k) per word
        # count = [0] * 26
        # for c in s:
        #     count[ord(c) - ord('a')] += 1
        # key = tuple(count)

        groups[key].append(s)
    return list(groups.values())
```

**Complexity:**
- Time: O(n * k log k) where n = number of strings, k = max string length. O(n * k) with the count-tuple approach.
- Space: O(n * k) for storing all strings in the hashmap.

**Variant -- "Group Rotations":** Given strings, group those that are cyclic rotations of each other (e.g., "abc", "bca", "cab"). Key function: find the lexicographically smallest rotation and use that as the key. Alternatively, double the string (`s + s`) and check if one is a substring of the doubled version of another.
:::

::: details Problem 25: Top K Frequent Elements (LC 347) -- Heap / Bucket Sort

**Key Insight:** Count frequencies, then extract the top K. Three approaches, each with different trade-offs.

```python
from collections import Counter
import heapq

def topKFrequent(nums, k):
    freq = Counter(nums)

    # Approach 1: Min-heap of size k -- O(n log k)
    return heapq.nlargest(k, freq.keys(), key=freq.get)

    # Approach 2: Bucket sort -- O(n) time, O(n) space
    # buckets = [[] for _ in range(len(nums) + 1)]
    # for num, count in freq.items():
    #     buckets[count].append(num)
    # result = []
    # for i in range(len(buckets) - 1, -1, -1):
    #     for num in buckets[i]:
    #         result.append(num)
    #         if len(result) == k:
    #             return result

def topKFrequent_bucket(nums, k):
    """Bucket sort approach -- O(n) time."""
    freq = Counter(nums)
    buckets = [[] for _ in range(len(nums) + 1)]
    for num, count in freq.items():
        buckets[count].append(num)

    result = []
    for i in range(len(buckets) - 1, -1, -1):
        for num in buckets[i]:
            result.append(num)
            if len(result) == k:
                return result
    return result
```

**Complexity:**
- Heap approach: O(n log k) time, O(n) space.
- Bucket sort approach: O(n) time, O(n) space.

**When to use which:** Bucket sort is O(n) but uses more memory. In an interview, mention both and let the interviewer pick. The bucket sort approach impresses because it avoids the heap entirely.
:::

::: details Problem 26: Merge K Sorted Lists (LC 23) -- Min-Heap

**Key Insight:** Maintain a min-heap of size K holding one node from each list. Pop the smallest, push its successor. The heap ensures you always pick the globally smallest element.

```python
import heapq

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def mergeKLists(lists):
    heap = []
    # Push the head of each non-empty list
    for i, node in enumerate(lists):
        if node:
            heapq.heappush(heap, (node.val, i, node))

    dummy = ListNode(0)
    curr = dummy

    while heap:
        val, i, node = heapq.heappop(heap)
        curr.next = node
        curr = curr.next
        if node.next:
            heapq.heappush(heap, (node.next.val, i, node.next))

    return dummy.next
```

**Complexity:**
- Time: O(N log K) where N = total number of nodes, K = number of lists.
- Space: O(K) for the heap.

**Why the index `i` in the tuple?** Python's `heapq` compares tuples element by element. If two nodes have the same value, it tries to compare the `ListNode` objects, which will crash. The index `i` serves as a guaranteed-unique tiebreaker.

::: tip
If the interviewer asks "can you do this without a heap?" -- yes, divide and conquer. Merge pairs of lists recursively, like merge sort. Same O(N log K) time, O(log K) space for recursion stack.
:::

:::

::: details Problem 27: Search in Rotated Sorted Array (LC 33) -- Binary Search

**Key Insight:** At every midpoint, at least one half of the array is sorted. Determine which half is sorted, then check if the target falls within that sorted half.

```python
def search(nums, target):
    lo, hi = 0, len(nums) - 1

    while lo <= hi:
        mid = lo + (hi - lo) // 2

        if nums[mid] == target:
            return mid

        # Left half [lo..mid] is sorted
        if nums[lo] <= nums[mid]:
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1   # target is in the sorted left half
            else:
                lo = mid + 1   # target is in the right half
        # Right half [mid..hi] is sorted
        else:
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1   # target is in the sorted right half
            else:
                hi = mid - 1   # target is in the left half

    return -1
```

**Complexity:**
- Time: O(log n)
- Space: O(1)

**Edge Case:** `nums[lo] <= nums[mid]` uses `<=` (not `<`) because when `lo == mid` (two-element subarray), the left "half" is a single element and it is trivially sorted. Getting this comparison wrong is the #1 source of bugs.

**Follow-up -- with duplicates (LC 81):** When `nums[lo] == nums[mid] == nums[hi]`, you cannot determine which half is sorted. Fall back to `lo += 1` and `hi -= 1`. Worst case becomes O(n).
:::

::: details Problem 28: Find Peak Element (LC 162) -- Binary Search

**Key Insight:** If `nums[mid] < nums[mid + 1]`, then a peak must exist somewhere to the right (including `mid + 1`). Otherwise, a peak exists at `mid` or to its left. This gives us a binary search on an unsorted array.

```python
def findPeakElement(nums):
    lo, hi = 0, len(nums) - 1

    while lo < hi:
        mid = lo + (hi - lo) // 2
        if nums[mid] < nums[mid + 1]:
            lo = mid + 1    # peak is to the right
        else:
            hi = mid        # peak is at mid or to the left

    return lo  # lo == hi, this is a peak
```

**Complexity:**
- Time: O(log n)
- Space: O(1)

**Why this works:** The problem states `nums[-1] = nums[n] = -infinity`. So if you are climbing uphill (going right), you must eventually hit a peak before falling off the right edge. Same logic in reverse for going left.

**Note the template difference:** This uses `lo < hi` (not `lo <= hi`) and `hi = mid` (not `hi = mid - 1`). This is the "narrow to one candidate" template -- the loop exits when `lo == hi`, which is the answer.
:::

::: details Problem 29: Kth Largest Element in an Array (LC 215) -- Quickselect / Heap

**Key Insight:** You do not need to fully sort the array. Quickselect partitions the array so that the pivot is in its final sorted position, then recurses only on the side that contains index k.

**Approach 1: Min-heap of size K**

```python
import heapq

def findKthLargest_heap(nums, k):
    """O(n log k) time, O(k) space."""
    heap = []
    for num in nums:
        heapq.heappush(heap, num)
        if len(heap) > k:
            heapq.heappop(heap)  # remove smallest
    return heap[0]  # root is the kth largest
```

**Approach 2: Quickselect**

```python
import random

def findKthLargest(nums, k):
    """O(n) average time, O(1) extra space."""
    target = len(nums) - k  # kth largest = (n-k)th smallest

    def quickselect(lo, hi):
        # Randomize pivot to avoid O(n^2) worst case
        pivot_idx = random.randint(lo, hi)
        nums[pivot_idx], nums[hi] = nums[hi], nums[pivot_idx]
        pivot = nums[hi]

        # Partition: elements < pivot go to the left
        store = lo
        for i in range(lo, hi):
            if nums[i] < pivot:
                nums[store], nums[i] = nums[i], nums[store]
                store += 1
        nums[store], nums[hi] = nums[hi], nums[store]

        if store == target:
            return nums[store]
        elif store < target:
            return quickselect(store + 1, hi)
        else:
            return quickselect(lo, store - 1)

    return quickselect(0, len(nums) - 1)
```

**Complexity:**
- Heap: O(n log k) time, O(k) space.
- Quickselect: O(n) average time, O(n^2) worst case, O(1) extra space (in-place).

**Which to use in an interview?** Start with the heap approach -- it is clean, correct, and takes 5 lines. Then mention Quickselect as the O(n) optimization. If the interviewer wants to see it, code it up. Randomizing the pivot is essential to avoid worst-case behavior on sorted input.
:::

---

## Common Mistakes

::: danger MISTAKES THAT COST OFFERS

**1. Forgetting heapq is min-heap**

You want the K largest elements and you write:
```python
# WRONG -- this keeps the K smallest!
heap = nums[:k]
heapq.heapify(heap)
for num in nums[k:]:
    if num > heap[0]:
        heapq.heapreplace(heap, num)
```
Wait -- this is actually correct! The min-heap's root is the smallest of the K largest. `heapreplace` swaps out the smallest when something bigger comes along. The confusion arises when you try to negate values unnecessarily. **Think carefully about what the heap root represents before negating.**

**2. Off-by-one in binary search**

The classic bug: using `hi = mid - 1` when you should use `hi = mid`, or `lo <= hi` when you should use `lo < hi`. The rule of thumb:
- Searching for exact match? Use `lo <= hi` with `hi = mid - 1` and `lo = mid + 1`.
- Narrowing to one candidate? Use `lo < hi` with `hi = mid` and `lo = mid + 1`.

Never mix conventions -- that is where infinite loops come from.

**3. Not handling duplicates in HashMap grouping**

When using `sorted(s)` as a key, remember that `sorted()` returns a list, which is unhashable. You must convert to `tuple(sorted(s))`.

**4. Comparing non-comparable objects in heapq**

If two heap elements have the same priority, Python tries to compare the next tuple element. If that element is a `ListNode` or some other non-comparable object, you get a `TypeError`. Always include a unique tiebreaker (like an index) in your heap tuples:
```python
heapq.heappush(heap, (priority, unique_index, actual_object))
```

**5. Integer overflow in `mid` calculation**

In Python this does not matter (arbitrary precision integers), but if you are asked to code in Java or C++:
```python
# Safe in all languages:
mid = lo + (hi - lo) // 2
# Unsafe in Java/C++ for large values:
mid = (lo + hi) // 2  # can overflow!
```
Mention this in your interview even if coding in Python -- it shows awareness.
:::

---

## Practice Checklist

Use this to track your progress. Each problem should be solvable in under 20 minutes without looking at the solution.

- [ ] **Group Anagrams** (LC 49) -- Can you write the `defaultdict` + sorted-key approach in 5 minutes?
- [ ] **Top K Frequent Elements** (LC 347) -- Can you explain both the heap and bucket sort approaches?
- [ ] **Merge K Sorted Lists** (LC 23) -- Can you handle the tiebreaker tuple correctly without looking it up?
- [ ] **Search in Rotated Sorted Array** (LC 33) -- Can you get the `<=` edge case right on the first try?
- [ ] **Find Peak Element** (LC 162) -- Can you explain WHY binary search works on an unsorted array?
- [ ] **Kth Largest Element** (LC 215) -- Can you write Quickselect from scratch with a random pivot?

::: tip SELF-TEST
For each problem, close the solution panel and rewrite it on paper or in a plain text editor. If you cannot write it cleanly without referencing anything, you do not know it well enough for a plain text editor interview.
:::
