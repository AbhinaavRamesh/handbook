# Binary Search, Heaps & Intervals

> **Logarithmic efficiency and priority-based processing**

---

## Binary Search Mastery

Binary search is a fundamental algorithm that reduces the search space by half with each iteration. Understanding the three core templates will help you solve the majority of binary search problems.

### The 3 Templates

**Template 1: Exact Match**
```python
def binary_search_exact(arr, target):
    """Find exact target position, return -1 if not found"""
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = left + (right - left) // 2  # Avoid overflow
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return -1  # Not found
```

::: info Complexity: Time O(log n) · Space O(1)
- **Time:** Halve search space each iteration
- **Space:** Only three pointer variables used
:::

**Template 2: Left Bound (First Occurrence / Lower Bound)**
```python
def binary_search_left(arr, target):
    """Find first position where arr[i] >= target"""
    left, right = 0, len(arr)  # Note: right = len(arr)

    while left < right:  # Note: < not <=
        mid = left + (right - left) // 2
        if arr[mid] < target:
            left = mid + 1
        else:
            right = mid  # Note: not mid - 1

    return left  # Returns insertion point if not found
```

::: info Complexity: Time O(log n) · Space O(1)
- **Time:** Halve search space each iteration
- **Space:** Only three pointer variables used
:::

**Template 3: Right Bound (Last Occurrence / Upper Bound)**
```python
def binary_search_right(arr, target):
    """Find first position where arr[i] > target"""
    left, right = 0, len(arr)

    while left < right:
        mid = left + (right - left) // 2
        if arr[mid] <= target:  # Note: <= instead of <
            left = mid + 1
        else:
            right = mid

    return left  # Points to first element > target
```

::: info Complexity: Time O(log n) · Space O(1)
- **Time:** Halve search space each iteration
- **Space:** Only three pointer variables used
:::

**Finding First and Last Position (LeetCode 34)**
```python
def search_range(arr, target):
    """Find [first, last] occurrence of target"""
    def find_left():
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] < target:
                left = mid + 1
            else:
                right = mid
        return left

    def find_right():
        left, right = 0, len(arr)
        while left < right:
            mid = (left + right) // 2
            if arr[mid] <= target:
                left = mid + 1
            else:
                right = mid
        return left - 1  # -1 because we want last occurrence

    first = find_left()
    if first >= len(arr) or arr[first] != target:
        return [-1, -1]
    return [first, find_right()]
```

::: info Complexity: Time O(log n) · Space O(1)
- **Time:** Two binary searches, each O(log n)
- **Space:** Only pointer variables used
:::

### When Binary Search Applies

| Scenario | Example Problems |
|----------|-----------------|
| **Sorted Array** | Classic search, Find First/Last Position |
| **Rotated Sorted Array** | Search in Rotated Array, Find Minimum |
| **Monotonic Function** | Sqrt(x), Valid Perfect Square |
| **Search Space Reduction** | Capacity to Ship, Koko Eating Bananas |
| **2D Matrix** | Search 2D Matrix, Search Row-Col Sorted |
| **Answer Optimization** | Minimize Maximum, Split Array Largest Sum |

**Key Insight**: Binary search works when there's a **monotonic condition** - if condition(x) is true, then condition(x+1) is also true (or vice versa).

### Binary Search on Answer Space

```python
def binary_search_on_answer(check_feasible, low, high):
    """
    Template for 'minimize maximum' or 'maximize minimum' problems
    check_feasible(x) returns True if x is a feasible answer
    """
    while low < high:
        mid = (low + high) // 2
        if check_feasible(mid):
            high = mid  # Try to find smaller feasible answer
        else:
            low = mid + 1
    return low

# Example: Koko Eating Bananas
def min_eating_speed(piles, h):
    def can_finish(speed):
        hours = sum((pile + speed - 1) // speed for pile in piles)
        return hours <= h

    return binary_search_on_answer(can_finish, 1, max(piles))
```

::: info Complexity: Time O(n log m) · Space O(1)
- **Time:** O(log m) binary search iterations, each with O(n) feasibility check
- **Space:** Only tracking search bounds and counters
:::

### Mermaid Diagram: Search Space Reduction

```mermaid
flowchart TD
    A["Initial Array: [1,3,5,7,9,11,13,15]<br/>Target: 7"] --> B["left=0, right=7<br/>mid=3, arr[3]=7"]
    B -->|"arr[mid] == target"| C["Found at index 3!"]

    D["Search Space Halving"] --> E["Iteration 1: 8 elements"]
    E --> F["Iteration 2: 4 elements"]
    F --> G["Iteration 3: 2 elements"]
    G --> H["Iteration 4: 1 element"]
    H --> I["O(log n) = O(log 8) = 3 iterations"]
```

```mermaid
flowchart LR
    subgraph "Left Bound Search for target=5"
        A1["[1,2,5,5,5,8,9]"] --> B1["Find FIRST 5"]
        B1 --> C1["Return index 2"]
    end

    subgraph "Right Bound Search for target=5"
        A2["[1,2,5,5,5,8,9]"] --> B2["Find LAST 5"]
        B2 --> C2["Return index 4"]
    end
```

---

## Heap Essentials

### Min Heap vs Max Heap

Python's `heapq` module implements a **min-heap** by default. For a max-heap, negate the values.

```python
import heapq

# MIN HEAP (default)
min_heap = []
heapq.heappush(min_heap, 5)
heapq.heappush(min_heap, 3)
heapq.heappush(min_heap, 7)
print(heapq.heappop(min_heap))  # Output: 3 (smallest)

# MAX HEAP (negate values)
max_heap = []
heapq.heappush(max_heap, -5)
heapq.heappush(max_heap, -3)
heapq.heappush(max_heap, -7)
print(-heapq.heappop(max_heap))  # Output: 7 (largest)

# Heapify existing list - O(n)
arr = [5, 3, 7, 1, 9]
heapq.heapify(arr)  # Now arr is a min-heap
```

::: info Complexity: Time O(log n) push/pop · Space O(n)
- **Time:** Push and pop are O(log n); heapify is O(n)
- **Space:** Heap stores n elements
:::

**Key Operations Complexity:**
| Operation | Time Complexity |
|-----------|----------------|
| heappush | O(log n) |
| heappop | O(log n) |
| heapify | O(n) |
| heap[0] (peek) | O(1) |
| nlargest(k, arr) | O(n log k) |
| nsmallest(k, arr) | O(n log k) |

### Top-K Pattern

The Top-K pattern is one of the most common heap applications. **Key insight**: Use a min-heap of size K to find K largest, and max-heap of size K to find K smallest.

```python
import heapq

def top_k_largest(arr, k):
    """Find k largest elements using min-heap"""
    # Method 1: Using nlargest (simple but creates new list)
    return heapq.nlargest(k, arr)

def top_k_largest_manual(arr, k):
    """Manual implementation with O(n log k) complexity"""
    min_heap = []

    for num in arr:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)  # Remove smallest

    return sorted(min_heap, reverse=True)

def kth_largest_element(arr, k):
    """Find the kth largest element (LeetCode 215)"""
    min_heap = []

    for num in arr:
        heapq.heappush(min_heap, num)
        if len(min_heap) > k:
            heapq.heappop(min_heap)

    return min_heap[0]  # kth largest is at root

# Top K Frequent Elements (LeetCode 347)
def top_k_frequent(nums, k):
    from collections import Counter
    count = Counter(nums)
    # Use min-heap with (frequency, num) pairs
    return heapq.nlargest(k, count.keys(), key=count.get)
```

::: info Complexity: Time O(n log k) · Space O(n)
- **Time:** O(n) to process elements; heap operations O(log k) for k-sized heap
- **Space:** O(n) for frequency counter; O(k) for heap
:::

### Two Heaps Pattern

Used when you need to track both smallest and largest elements, commonly for finding median.

```python
import heapq

class MedianFinder:
    """Find Median from Data Stream (LeetCode 295)"""

    def __init__(self):
        self.small = []  # Max-heap (negated) for smaller half
        self.large = []  # Min-heap for larger half

    def addNum(self, num):
        # Always add to max-heap first
        heapq.heappush(self.small, -num)

        # Balance: largest of small <= smallest of large
        if self.small and self.large and (-self.small[0] > self.large[0]):
            heapq.heappush(self.large, -heapq.heappop(self.small))

        # Size balance: small can have at most 1 more than large
        if len(self.small) > len(self.large) + 1:
            heapq.heappush(self.large, -heapq.heappop(self.small))
        if len(self.large) > len(self.small):
            heapq.heappush(self.small, -heapq.heappop(self.large))

    def findMedian(self):
        if len(self.small) > len(self.large):
            return -self.small[0]
        return (-self.small[0] + self.large[0]) / 2
```

::: info Complexity: Time O(log n) add · O(1) median · Space O(n)
- **Time:** Each add involves heap operations O(log n); median is O(1) peek
- **Space:** Both heaps together store all n elements
:::

### Streaming/Online Problems

Heaps excel when processing data streams where you need to maintain order without storing everything.

```python
import heapq

class KthLargestStream:
    """Kth Largest Element in a Stream (LeetCode 703)"""

    def __init__(self, k, nums):
        self.k = k
        self.heap = nums[:]
        heapq.heapify(self.heap)
        while len(self.heap) > k:
            heapq.heappop(self.heap)

    def add(self, val):
        heapq.heappush(self.heap, val)
        if len(self.heap) > self.k:
            heapq.heappop(self.heap)
        return self.heap[0]

def merge_k_sorted_lists(lists):
    """Merge K Sorted Lists (LeetCode 23)"""
    result = []
    heap = []

    # Add first element from each list with (val, list_idx, elem_idx)
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))

    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)

        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))

    return result
```

::: info Complexity: Time O(n log k) · Space O(k)
- **Time:** Process n total elements; each heap operation O(log k) for k lists
- **Space:** Heap stores at most k elements (one from each list)
:::

### Heap with Custom Objects

```python
import heapq
from dataclasses import dataclass, field

@dataclass(order=True)
class Task:
    priority: int
    name: str = field(compare=False)

# Using custom comparison
tasks = []
heapq.heappush(tasks, Task(2, "Low priority"))
heapq.heappush(tasks, Task(1, "High priority"))
print(heapq.heappop(tasks).name)  # "High priority"

# Alternative: tuple approach
heap = []
heapq.heappush(heap, (priority, index, item))  # index breaks ties
```

::: info Complexity: Time O(log n) · Space O(n)
- **Time:** Heap operations O(log n) with custom comparison
- **Space:** Heap stores n custom objects
:::

---

## Interval Problems

### The Sorting Trick

**Always sort intervals first!** The sorting key depends on the problem:
- **Merge intervals**: Sort by start time
- **Meeting rooms**: Sort by start time
- **Non-overlapping intervals**: Sort by end time (greedy)

### Merge vs Insert vs Intersection

```python
def merge_intervals(intervals):
    """Merge Overlapping Intervals (LeetCode 56)"""
    if not intervals:
        return []

    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    merged = [intervals[0]]

    for start, end in intervals[1:]:
        if start <= merged[-1][1]:  # Overlapping
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    return merged

def insert_interval(intervals, new_interval):
    """Insert Interval (LeetCode 57)"""
    result = []
    i = 0
    n = len(intervals)

    # Add all intervals before new_interval
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1

    # Merge overlapping intervals
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    result.append(new_interval)

    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1

    return result

def interval_intersection(A, B):
    """Interval List Intersections (LeetCode 986)"""
    result = []
    i = j = 0

    while i < len(A) and j < len(B):
        # Find intersection
        start = max(A[i][0], B[j][0])
        end = min(A[i][1], B[j][1])

        if start <= end:  # Valid intersection
            result.append([start, end])

        # Move pointer with smaller end
        if A[i][1] < B[j][1]:
            i += 1
        else:
            j += 1

    return result
```

::: info Complexity: Merge O(n log n) · Insert O(n) · Intersection O(n + m)
- **Time:** Merge requires sorting; insert/intersection are linear scans
- **Space:** O(n) for result arrays
:::

### Meeting Rooms Problems

```python
def can_attend_all_meetings(intervals):
    """Meeting Rooms (LeetCode 252)"""
    intervals.sort(key=lambda x: x[0])

    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i-1][1]:
            return False
    return True

def min_meeting_rooms(intervals):
    """Meeting Rooms II (LeetCode 253) - Using Heap"""
    if not intervals:
        return 0

    intervals.sort(key=lambda x: x[0])
    rooms = []  # Min-heap of end times

    for start, end in intervals:
        if rooms and rooms[0] <= start:
            heapq.heappop(rooms)  # Reuse room
        heapq.heappush(rooms, end)

    return len(rooms)

def min_meeting_rooms_sweep_line(intervals):
    """Alternative: Sweep Line Algorithm"""
    events = []
    for start, end in intervals:
        events.append((start, 1))   # Meeting starts
        events.append((end, -1))    # Meeting ends

    events.sort()
    max_rooms = current_rooms = 0

    for time, delta in events:
        current_rooms += delta
        max_rooms = max(max_rooms, current_rooms)

    return max_rooms
```

::: info Complexity: Time O(n log n) · Space O(n)
- **Time:** Dominated by sorting; processing is O(n)
- **Space:** Heap or events array stores up to n elements
:::

### Non-Overlapping Intervals

```python
def erase_overlap_intervals(intervals):
    """Non-overlapping Intervals (LeetCode 435)
    Find minimum number of intervals to remove"""
    if not intervals:
        return 0

    # Sort by END time (greedy: keep intervals that end earliest)
    intervals.sort(key=lambda x: x[1])

    count = 0
    prev_end = float('-inf')

    for start, end in intervals:
        if start >= prev_end:
            prev_end = end  # Keep this interval
        else:
            count += 1  # Remove this interval

    return count
```

::: info Complexity: Time O(n log n) · Space O(1)
- **Time:** Sorting dominates; greedy scan is O(n)
- **Space:** Sorting may use O(log n) stack; otherwise constant
:::

### Interval Diagram

```mermaid
flowchart TD
    subgraph "Merge Intervals Example"
        A["Input: [[1,3],[2,6],[8,10],[15,18]]"]
        B["Sort by start: Already sorted"]
        C["[1,3] and [2,6] overlap -> [1,6]"]
        D["[1,6] and [8,10] no overlap"]
        E["Output: [[1,6],[8,10],[15,18]]"]
        A --> B --> C --> D --> E
    end
```

---

## Quick Problem Matrix

| Problem | Technique | Time | Space | Key Insight |
|---------|-----------|------|-------|-------------|
| Binary Search | Template 1 | O(log n) | O(1) | Exact match in sorted array |
| First/Last Position | Template 2+3 | O(log n) | O(1) | Use both bounds |
| Search Rotated Array | Modified BS | O(log n) | O(1) | Check which half is sorted |
| Find Peak Element | BS | O(log n) | O(1) | Go towards larger neighbor |
| Sqrt(x) | BS on answer | O(log n) | O(1) | Search space is [0, x] |
| Koko Eating Bananas | BS on answer | O(n log m) | O(1) | Binary search on speed |
| Kth Largest | Min-heap size K | O(n log k) | O(k) | Maintain K largest |
| Top K Frequent | Heap + HashMap | O(n log k) | O(n) | Count then heap |
| Find Median Stream | Two Heaps | O(log n) | O(n) | Balance max/min heaps |
| Merge K Lists | Min-heap | O(n log k) | O(k) | Track heads of lists |
| Merge Intervals | Sort + Sweep | O(n log n) | O(n) | Sort by start |
| Meeting Rooms II | Sort + Heap | O(n log n) | O(n) | Track end times |
| Non-Overlapping | Sort by end | O(n log n) | O(1) | Greedy: earliest end |

---

## Interview Applications

Based on actual interview questions and patterns from top companies:

### Classic Google Questions

**1. Top 100 Google Searches (Design + Heap)**
> "Design a system to find the top 100 unique Google searches from a billion searches per day."

```python
from collections import Counter
import heapq

def top_100_searches(searches):
    # Count frequencies
    count = Counter(searches)
    # Use heap to get top 100
    return heapq.nlargest(100, count.items(), key=lambda x: x[1])
```

**2. Search in 2D Matrix (Multi-dimensional Binary Search)**
```python
def search_matrix(matrix, target):
    """Google-style 2D matrix search"""
    if not matrix:
        return False

    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1

    while left <= right:
        mid = (left + right) // 2
        row, col = mid // cols, mid % cols

        if matrix[row][col] == target:
            return True
        elif matrix[row][col] < target:
            left = mid + 1
        else:
            right = mid - 1

    return False
```

**3. Find Peak Element (Common Interview)**
```python
def find_peak_element(nums):
    """O(log n) solution using binary search"""
    left, right = 0, len(nums) - 1

    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[mid + 1]:
            right = mid  # Peak is on left side (including mid)
        else:
            left = mid + 1  # Peak is on right side

    return left
```

**4. Capacity to Ship Packages (Binary Search on Answer)**
```python
def ship_within_days(weights, days):
    """Minimize maximum capacity needed"""
    def can_ship(capacity):
        day_count = 1
        current_load = 0
        for weight in weights:
            if current_load + weight > capacity:
                day_count += 1
                current_load = weight
            else:
                current_load += weight
        return day_count <= days

    left = max(weights)  # Minimum capacity
    right = sum(weights)  # Maximum capacity

    while left < right:
        mid = (left + right) // 2
        if can_ship(mid):
            right = mid
        else:
            left = mid + 1

    return left
```

### Problem Recognition Cheat Sheet

| If you see... | Think... | Pattern |
|--------------|----------|---------|
| "Sorted array" | Binary Search | Template 1/2/3 |
| "Find first/last" | Binary Search | Left/Right bound |
| "Minimize maximum" | Binary Search on answer | Feasibility check |
| "K largest/smallest" | Heap | Top-K pattern |
| "Median of stream" | Two Heaps | Max-heap + Min-heap |
| "K sorted lists" | Heap | Merge with min-heap |
| "Overlapping intervals" | Sort + Merge | Sort by start |
| "Meeting rooms" | Sort + Heap | Track end times |
| "Maximum concurrent" | Sweep Line | Events array |

---

## Common Mistakes to Avoid

### Binary Search Pitfalls
1. **Off-by-one errors**: Know when to use `<` vs `<=` and `mid` vs `mid+1`
2. **Overflow**: Use `mid = left + (right - left) // 2` instead of `(left + right) // 2`
3. **Wrong boundary**: `right = len(arr)` for bounds, `right = len(arr) - 1` for exact

### Heap Pitfalls
1. **Forgetting heapq is min-heap**: Negate for max-heap
2. **Modifying heap directly**: Only use heapq functions
3. **Tuple comparison**: Include index to break ties

### Interval Pitfalls
1. **Not sorting first**: Always sort before processing
2. **Wrong sort key**: End for greedy selection, start for merging
3. **Edge cases**: Empty list, single interval, all overlapping

---

## References

- [LeetCode Binary Search Template](https://leetcode.com/discuss/post/786126/Python-Powerful-Ultimate-Binary-Search-Template.-Solved-many-problems/)
- [Tech Interview Handbook - Heap](https://www.techinterviewhandbook.org/algorithms/heap/)
- [Top K Elements Pattern](https://www.designgurus.io/answers/detail/what-is-the-top-k-elements-pattern-for-coding-interviews)
- [Google Coding Interview Questions](https://www.educative.io/blog/google-coding-interview-questions)
- [Binary Search Interview Questions](https://interviewing.io/binary-search-interview-questions)
