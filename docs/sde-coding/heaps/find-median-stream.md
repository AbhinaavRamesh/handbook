# Find Median from Data Stream

> **Maintain running median using two heaps**

---

## Problem Statement

The **median** is the middle value in an ordered integer list. If the size of the list is even, there is no middle value, and the median is the mean of the two middle values.

Implement the `MedianFinder` class:

- `MedianFinder()` initializes the MedianFinder object.
- `void addNum(int num)` adds the integer `num` from the data stream to the data structure.
- `double findMedian()` returns the median of all elements so far.

**LeetCode:** [Problem 295 - Find Median from Data Stream](https://leetcode.com/problems/find-median-from-data-stream/)

---

## Examples

### Example 1
```
Input:
["MedianFinder", "addNum", "addNum", "findMedian", "addNum", "findMedian"]
[[], [1], [2], [], [3], []]

Output:
[null, null, null, 1.5, null, 2.0]

Explanation:
MedianFinder medianFinder = new MedianFinder();
medianFinder.addNum(1);    // arr = [1]
medianFinder.addNum(2);    // arr = [1, 2]
medianFinder.findMedian(); // return 1.5 (i.e., (1 + 2) / 2)
medianFinder.addNum(3);    // arr = [1, 2, 3]
medianFinder.findMedian(); // return 2.0
```

---

## Constraints

- `-10^5 <= num <= 10^5`
- There will be at least one element in the data structure before calling `findMedian`
- At most `5 * 10^4` calls will be made to `addNum` and `findMedian`

---

## Visual Guides

![Median Stream Visualization](/sde-coding/heaps/median_stream.png)

---

## The Two Heaps Pattern

This is a classic **Two Heaps** pattern problem. The key insight is to split the data into two halves:

1. **Max-Heap (small)**: Contains the **smaller half** of numbers
   - Root = maximum of smaller half
2. **Min-Heap (large)**: Contains the **larger half** of numbers
   - Root = minimum of larger half

### Why This Works

```
Sorted data:    [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                 ^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^
                  smaller half       larger half

Max-Heap (small):     Min-Heap (large):
      (5)                   (6)
     / \                   / \
   (3) (4)               (7) (8)
   / \                   / \
 (1) (2)               (9) (10)

Median is at the boundary:
- If equal sizes: average of roots = (5 + 6) / 2 = 5.5
- If max-heap larger: root of max-heap = 5
```

### Invariants

1. **All elements in max_heap <= All elements in min_heap**
2. **|max_heap| == |min_heap|** OR **|max_heap| == |min_heap| + 1**
3. The median is always accessible from the heap roots in O(1)

---

## Solution

```python
import heapq

class MedianFinder:
    """
    Find median from data stream using two heaps.

    - small (max-heap): stores the smaller half of numbers
    - large (min-heap): stores the larger half of numbers

    Time Complexity:
        - addNum: O(log n) for heap operations
        - findMedian: O(1) to access heap roots

    Space Complexity: O(n) to store all numbers
    """

    def __init__(self):
        # Python only has min-heap, so negate values for max-heap
        self.small = []  # Max heap (stores negated values)
        self.large = []  # Min heap

    def addNum(self, num: int) -> None:
        """Add a number to the data structure."""
        # Step 1: Add to max heap (smaller half)
        heapq.heappush(self.small, -num)

        # Step 2: Balance - move largest from small to large
        # This ensures all elements in small <= all elements in large
        heapq.heappush(self.large, -heapq.heappop(self.small))

        # Step 3: Ensure small has >= elements than large
        # This keeps the median accessible from small's root when odd count
        if len(self.small) < len(self.large):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        """Return the median of all elements."""
        if len(self.small) > len(self.large):
            # Odd number of elements - median is root of max-heap
            return -self.small[0]
        # Even number of elements - median is average of two roots
        return (-self.small[0] + self.large[0]) / 2
```

::: info Complexity: Time O(log n) for addNum, O(1) for findMedian · Space O(n)
- **Time:** addNum performs a constant number of heap operations, each O(log n) where n is total elements added. findMedian only accesses heap roots in O(1)
- **Space:** Both heaps together store all n elements added to the stream
:::

---

## Step-by-Step Walkthrough

```
Stream: [5, 2, 3, 4, 1]

=== Step 1: addNum(5) ===
1. Push -5 to small: small = [-5]
2. Move to large: pop -5 from small, push 5 to large
   small = [], large = [5]
3. Balance sizes: large > small
   Pop 5 from large, push -5 to small
   small = [-5], large = []

State: small = [-5], large = []
Median: -small[0] = 5

=== Step 2: addNum(2) ===
1. Push -2 to small: small = [-5, -2]
   (After heapify: [-2, -5], but root is still max = -(-5) = 5... wait)
   Actually: heappush maintains min-heap, so -2 > -5, root is -5
   small = [-5, -2] means root is -5, which represents 5

   Hmm, let me reconsider. With negation:
   - We push -2
   - Min-heap puts smaller values at root
   - -5 < -2, so -5 stays at root
   - small[0] = -5, meaning max value is 5

   Wait, that's wrong for a max-heap!
   For max-heap behavior: we want the largest value accessible.
   With negation: largest original = smallest negated = at min-heap root

   small = [-5], push -2:
   -2 > -5, so -5 stays at root as the smaller negated value
   But -5 represents the larger original value 5. This is what we want!

   Actually: after heappush, we need to check the heap structure.
   heappush([-5], -2) -> [-5, -2] (parent -5 < child -2, valid min-heap)
   small[0] = -5 means max of original values is 5. Correct!

2. Move to large: pop -5, push 5
   small = [-2], large = [5]

3. Balance: len(small) = 1, len(large) = 1, equal. No action.

State: small = [-2], large = [5]
Median: (-(-2) + 5) / 2 = (2 + 5) / 2 = 3.5

=== Step 3: addNum(3) ===
1. Push -3 to small: small = [-3, -2]
   -3 < -2, so -3 is at root (representing max = 3)

2. Move to large: pop -3, push 3
   small = [-2], large = [3, 5]

3. Balance: len(small) = 1, len(large) = 2
   len(small) < len(large), so move from large to small
   Pop 3 from large, push -3 to small
   small = [-3, -2], large = [5]

State: small = [-3, -2], large = [5]
Median: -small[0] = 3

=== Step 4: addNum(4) ===
1. Push -4 to small: small = [-4, -2, -3]
   After heapify: root should be min, which is -4

2. Move to large: pop -4, push 4
   small = [-3, -2], large = [4, 5]

3. Balance: sizes are equal (2, 2). No action.

State: small = [-3, -2], large = [4, 5]
Median: (-(-3) + 4) / 2 = (3 + 4) / 2 = 3.5

=== Step 5: addNum(1) ===
1. Push -1 to small: small = [-3, -2, -1]
2. Move to large: pop -3, push 3
   small = [-2, -1], large = [3, 4, 5]

   Wait, that doesn't seem right. Let me reconsider.
   After push -1: small could be [-3, -2, -1] or restructured
   Min of [-3, -2, -1] is -3 (at root)
   Pop -3 from small, push 3 to large
   small = [-2, -1], large = [3, 4, 5]

3. Balance: len(small) = 2, len(large) = 3
   len(small) < len(large), move from large to small
   Pop 3, push -3 to small
   small = [-3, -1, -2], large = [4, 5]

State: small = [-3, ...], large = [4, 5]
Median: -small[0] = 3

Final sorted data: [1, 2, 3, 4, 5]
Median = 3. Correct!
```

---

## Alternative Implementation: Cleaner Logic

```python
import heapq

class MedianFinder:
    def __init__(self):
        self.small = []  # max heap (negated)
        self.large = []  # min heap

    def addNum(self, num: int) -> None:
        # Always add to small first, then balance
        if not self.small or num <= -self.small[0]:
            heapq.heappush(self.small, -num)
        else:
            heapq.heappush(self.large, num)

        # Balance: ensure |small| - |large| is 0 or 1
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        elif len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self) -> float:
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

::: info Complexity: Time O(log n) for addNum, O(1) for findMedian · Space O(n)
- **Time:** addNum performs at most 3 heap operations, each O(log n). findMedian accesses heap roots in O(1)
- **Space:** Both heaps together store all n elements
:::

---

## Complexity Analysis

| Operation | Time | Space |
|-----------|------|-------|
| addNum | O(log n) | O(1) per call |
| findMedian | O(1) | O(1) |
| Overall for n operations | O(n log n) | O(n) |

---

## Common Mistakes

1. **Forgetting to negate for max-heap**
   ```python
   # Wrong
   heapq.heappush(self.small, num)

   # Correct
   heapq.heappush(self.small, -num)
   ```

2. **Incorrect balance condition**
   ```python
   # Wrong: allows large to have more elements
   if len(self.small) < len(self.large):
       pass  # Should rebalance!

   # Correct: maintain invariant
   if len(self.small) < len(self.large):
       heapq.heappush(self.small, -heapq.heappop(self.large))
   ```

3. **Not negating when accessing max-heap root**
   ```python
   # Wrong
   return self.small[0]

   # Correct
   return -self.small[0]
   ```

---

## Follow-up Questions

### Q1: What if 99% of all numbers fall between 0 and 100?

Use **bucket counting** for the common range:
- Array of size 101 for counts of 0-100
- Two heaps for outliers (< 0 and > 100)
- Track total count and outlier counts
- findMedian: binary search in buckets + outlier handling

### Q2: How to handle element removal?

Use **lazy deletion** with a deletion map:

```python
class MedianFinderWithDelete:
    def __init__(self):
        self.small = []
        self.large = []
        self.small_size = 0
        self.large_size = 0
        self.deleted = {}

    def remove(self, num: int) -> None:
        self.deleted[num] = self.deleted.get(num, 0) + 1
        if num <= -self.small[0]:
            self.small_size -= 1
        else:
            self.large_size -= 1
        self._balance()
        self._prune()

    def _prune(self) -> None:
        # Remove deleted elements from heap tops
        while self.small and self.deleted.get(-self.small[0], 0) > 0:
            self.deleted[-self.small[0]] -= 1
            heapq.heappop(self.small)
        while self.large and self.deleted.get(self.large[0], 0) > 0:
            self.deleted[self.large[0]] -= 1
            heapq.heappop(self.large)
```

---

## Visual ASCII Diagram

```
Stream: [5, 2, 3, 4, 1]

After all insertions:

     MAX HEAP (small)              MIN HEAP (large)
     ===============              ================
           [3]                          [4]
          /   \                         /
        [2]   [1]                     [5]

     Contains: {1, 2, 3}           Contains: {4, 5}
     Root: 3 (max of small)        Root: 4 (min of large)

     Sorted view: [1, 2, 3] | [4, 5]
                         ^
                       Median = 3 (odd count)
```

---

## Related Problems

::: details Sliding Window Median (LeetCode 480)
**Problem:** Find the median of each sliding window of size k as it moves through an array.

**Key Insight:** Extends two-heap pattern with lazy deletion to handle element removal as window slides.

**Approach:** Use two heaps like Find Median, but track deleted elements in a hash map. Remove lazily when elements reach heap top.

**Complexity:** Time O(n log k); Space O(n) for deletion tracking
:::

::: details Kth Largest Element in an Array (LeetCode 215)
**Problem:** Find the kth largest element in an unsorted array.

**Key Insight:** Use min-heap of size k, or QuickSelect for average O(n) time.

**Approach (Heap):** Maintain min-heap of k elements. For each element, if larger than heap root, replace root. Final root is kth largest.

**Complexity:** Time O(n log k) for heap, O(n) average for QuickSelect; Space O(k) or O(1)
:::

::: details Find K Pairs with Smallest Sums (LeetCode 373)
**Problem:** Given two sorted arrays, find k pairs with the smallest sums.

**Key Insight:** Use min-heap to process pairs in order of sum without generating all combinations.

**Approach:** Initialize heap with pairs (nums1[i], nums2[0]). Pop smallest, push pair with next element from nums2.

**Complexity:** Time O(k log k); Space O(k)
:::

---

## Interview Tips

1. **Start with the insight**
   - "Split data into two halves using two heaps"
   - "Max-heap for smaller half, min-heap for larger half"

2. **Explain the invariants**
   - Size balance: |small| = |large| or |small| = |large| + 1
   - Value separation: max(small) <= min(large)

3. **Walk through an example**
   - Show how elements flow between heaps
   - Demonstrate findMedian for odd and even counts

4. **Discuss the Python max-heap trick**
   - Mention that Python only has min-heap
   - Explain negation strategy

---

## Sources

- [Find Median from Data Stream - LeetCode](https://leetcode.com/problems/find-median-from-data-stream/)
- [Two Heaps Pattern - Design Gurus](https://www.designgurus.io/course-play/grokking-the-coding-interview/doc/639db019d7c9c2e74be0b50e)
- [NeetCode Solution](https://neetcode.io/solutions/find-median-from-data-stream)
