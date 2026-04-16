# Tier 3: Sliding Window / Two Pointers

::: tip PRIORITY: HIGH
Phone screen favorites that also appeared in onsites. Master the expand/shrink template and you cover 4 problems in one pattern.
:::

---

## The Sliding Window Mental Model

Every sliding window problem follows the same core idea: maintain a **window** defined by two pointers (`left` and `right`) that slides across the input. The algorithm has two phases that repeat:

1. **Expand** -- move `right` forward, adding the new element to your window state.
2. **Shrink** -- while the window is invalid (violates the constraint), move `left` forward and remove elements from your window state.

Think of it like an inchworm crawling across a string. The right end reaches forward to explore, and the left end catches up to maintain a valid window.

**Why this works:** By never moving `left` or `right` backward, every element enters and exits the window at most once, giving you O(n) time instead of the O(n^2) brute force of checking every substring.

**The key question to ask yourself:** _"What makes a window valid or invalid?"_ Once you answer that, the template writes itself.

![Sliding Window Animation — Minimum Window Substring](/sde-coding/sprint/sliding_window.gif)

---

## Problem Table

| # | Problem | LC # | Pattern | Why |
|---|---------|------|---------|-----|
| 15 | [Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/) | 3 | Sliding window | Phone screen classic |
| 16 | [Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/) | 76 | Sliding window (hard) | Shows mastery |
| 17 | [Sliding Window Maximum](https://leetcode.com/problems/sliding-window-maximum/) | 239 | Monotonic deque | Advanced pattern |
| 18 | [Longest Repeating Character Replacement](https://leetcode.com/problems/longest-repeating-character-replacement/) | 424 | Sliding window | Common variant |

---

## Template: The Universal Sliding Window

This single template covers problems 15, 16, and 18. Memorize it.

:::code-group

```python [Python]
def sliding_window(s):
    left = 0
    window = {}          # tracks window state (char counts, etc.)
    result = 0           # or float('inf') for minimization

    for right in range(len(s)):
        # --- EXPAND: add s[right] to window ---
        char = s[right]
        window[char] = window.get(char, 0) + 1

        # --- SHRINK: while window is INVALID, contract from left ---
        while WINDOW_IS_INVALID:
            window[s[left]] -= 1
            if window[s[left]] == 0:
                del window[s[left]]
            left += 1

        # --- UPDATE: record best valid window ---
        result = max(result, right - left + 1)

    return result
```

```java [Java]
int slidingWindow(String s) {
    int left = 0, result = 0;
    Map<Character, Integer> window = new HashMap<>();

    for (int right = 0; right < s.length(); right++) {
        // --- EXPAND: add s[right] to window ---
        char c = s.charAt(right);
        window.merge(c, 1, Integer::sum);

        // --- SHRINK: while window is INVALID, contract from left ---
        while (/* WINDOW_IS_INVALID */ false) {
            char l = s.charAt(left);
            window.merge(l, -1, Integer::sum);
            if (window.get(l) == 0) window.remove(l);
            left++;
        }

        // --- UPDATE: record best valid window ---
        result = Math.max(result, right - left + 1);
    }
    return result;
}
```

:::

**Adapt by changing three things:**
1. What you track in `window` (character counts, sum, frequency map)
2. The `WINDOW_IS_INVALID` condition (duplicates exist, sum exceeds target, missing required chars)
3. The `result` update (max length, min length, specific indices)

---

## When to Use Sliding Window vs Two Pointers

These are related but distinct patterns. Use this decision guide:

| Signal | Technique | Example |
|--------|-----------|---------|
| Contiguous subarray/substring | **Sliding Window** | Longest substring without repeating chars |
| Sorted array + pair condition | **Two Pointers (inward)** | Two Sum II (sorted) |
| Linked list cycle/middle | **Two Pointers (fast/slow)** | Linked List Cycle |
| Fixed-size window (size k) | **Sliding Window (fixed)** | Max sum subarray of size k |
| Variable-size window with constraint | **Sliding Window (variable)** | Min window substring |
| Merging/comparing two sequences | **Two Pointers (parallel)** | Merge sorted arrays |

::: info RULE OF THUMB
If the problem says "subarray" or "substring" and asks for a length, sum, or count -- it is almost certainly sliding window. If it says "sorted array" and asks about pairs or partitions -- think two pointers.
:::

---

## Problem Walkthroughs

::: details #15 -- Longest Substring Without Repeating Characters (LC 3)

**Problem:** Given a string `s`, find the length of the longest substring without repeating characters.

**Key insight:** The window is valid when every character in it is unique. As soon as a character repeats, shrink from the left until it is unique again.

**Why it shows up on phone screens:** It is the purest application of the sliding window template. If you cannot solve this cleanly, the interviewer will not proceed.

:::code-group

```python [Python]
def lengthOfLongestSubstring(self, s: str) -> int:
    left = 0
    window = {}
    result = 0

    for right in range(len(s)):
        char = s[right]
        window[char] = window.get(char, 0) + 1

        # shrink until no duplicates
        while window[char] > 1:
            window[s[left]] -= 1
            if window[s[left]] == 0:
                del window[s[left]]
            left += 1

        result = max(result, right - left + 1)

    return result
```

```java [Java]
int lengthOfLongestSubstring(String s) {
    int left = 0, result = 0;
    Map<Character, Integer> window = new HashMap<>();

    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        window.merge(c, 1, Integer::sum);

        // shrink until no duplicates
        while (window.get(c) > 1) {
            char l = s.charAt(left);
            window.merge(l, -1, Integer::sum);
            if (window.get(l) == 0) window.remove(l);
            left++;
        }

        result = Math.max(result, right - left + 1);
    }
    return result;
}
```

:::

**Optimized variant** (jump `left` directly using last-seen index):

:::code-group

```python [Python]
def lengthOfLongestSubstring(self, s: str) -> int:
    last_seen = {}
    left = 0
    result = 0

    for right in range(len(s)):
        char = s[right]
        if char in last_seen and last_seen[char] >= left:
            left = last_seen[char] + 1
        last_seen[char] = right
        result = max(result, right - left + 1)

    return result
```

```java [Java]
int lengthOfLongestSubstringOptimized(String s) {
    Map<Character, Integer> lastSeen = new HashMap<>();
    int left = 0, result = 0;

    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        if (lastSeen.containsKey(c) && lastSeen.get(c) >= left) {
            left = lastSeen.get(c) + 1;
        }
        lastSeen.put(c, right);
        result = Math.max(result, right - left + 1);
    }
    return result;
}
```

:::

| | Complexity |
|---|---|
| **Time** | O(n) -- each character visited at most twice |
| **Space** | O(min(n, m)) -- m is the character set size |

:::

::: details #16 -- Minimum Window Substring (LC 76)

**Problem:** Given strings `s` and `t`, find the minimum window in `s` that contains all characters of `t` (including duplicates).

**Key insight:** Use two frequency maps. Track how many distinct characters in `t` are fully satisfied in the current window (`have` vs `need`). Expand until all characters are satisfied, then shrink to minimize the window.

**Why this shows mastery:** It combines the sliding window template with frequency counting and a "formed" tracker. Clean code here signals strong fundamentals to the interviewer.

:::code-group

```python [Python]
from collections import Counter

def minWindow(self, s: str, t: str) -> str:
    if not t or not s:
        return ""

    need = Counter(t)
    required = len(need)  # number of distinct chars we need

    left = 0
    have = 0              # how many distinct chars are fully satisfied
    window = {}
    result = (float('inf'), 0, 0)  # (length, left, right)

    for right in range(len(s)):
        char = s[right]
        window[char] = window.get(char, 0) + 1

        # check if this char is now fully satisfied
        if char in need and window[char] == need[char]:
            have += 1

        # shrink while window is valid (contains all of t)
        while have == required:
            # update result if this window is smaller
            if (right - left + 1) < result[0]:
                result = (right - left + 1, left, right)

            # remove s[left] and shrink
            window[s[left]] -= 1
            if s[left] in need and window[s[left]] < need[s[left]]:
                have -= 1
            left += 1

    length, lo, hi = result
    return s[lo:hi + 1] if length != float('inf') else ""
```

```java [Java]
String minWindow(String s, String t) {
    if (t.isEmpty() || s.isEmpty()) return "";

    Map<Character, Integer> need = new HashMap<>();
    for (char c : t.toCharArray()) need.merge(c, 1, Integer::sum);
    int required = need.size();

    int left = 0, have = 0;
    int resLen = Integer.MAX_VALUE, resL = 0, resR = 0;
    Map<Character, Integer> window = new HashMap<>();

    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        window.merge(c, 1, Integer::sum);

        // check if this char is now fully satisfied
        if (need.containsKey(c) && window.get(c).equals(need.get(c))) have++;

        // shrink while window is valid (contains all of t)
        while (have == required) {
            if (right - left + 1 < resLen) {
                resLen = right - left + 1;
                resL = left;
                resR = right;
            }
            char l = s.charAt(left);
            window.merge(l, -1, Integer::sum);
            if (need.containsKey(l) && window.get(l) < need.get(l)) have--;
            left++;
        }
    }
    return resLen == Integer.MAX_VALUE ? "" : s.substring(resL, resR + 1);
}
```

:::

| | Complexity |
|---|---|
| **Time** | O(n + m) -- n = len(s), m = len(t) |
| **Space** | O(n + m) -- two frequency maps |

::: tip INTERVIEW TIP
In a real interview, define `need` and `have` clearly before coding. Say: _"I will track how many distinct characters from t are fully satisfied in my window."_ This shows structured thinking.
:::

:::

::: details #17 -- Sliding Window Maximum (LC 239)

**Problem:** Given an array `nums` and a window size `k`, return the maximum value in each window as it slides from left to right.

**Key insight:** This is NOT a standard sliding window problem. You need a **monotonic deque** -- a deque that keeps elements in decreasing order. The front of the deque is always the maximum. See the [Monotonic Deque Pattern](#the-monotonic-deque-pattern) section below.

:::code-group

```python [Python]
from collections import deque

def maxSlidingWindow(self, nums: list[int], k: int) -> list[int]:
    dq = deque()   # stores INDICES, not values
    result = []

    for i in range(len(nums)):
        # remove indices that are out of window
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # maintain decreasing order: remove smaller elements from back
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()

        dq.append(i)

        # window is fully formed once i >= k - 1
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

```java [Java]
int[] maxSlidingWindow(int[] nums, int k) {
    int n = nums.length;
    int[] result = new int[n - k + 1];
    Deque<Integer> dq = new ArrayDeque<>();  // stores INDICES, not values

    for (int i = 0; i < n; i++) {
        // remove indices that are out of window
        while (!dq.isEmpty() && dq.peekFirst() < i - k + 1) dq.pollFirst();

        // maintain decreasing order: remove smaller elements from back
        while (!dq.isEmpty() && nums[dq.peekLast()] <= nums[i]) dq.pollLast();

        dq.offerLast(i);

        // window is fully formed once i >= k - 1
        if (i >= k - 1) result[i - k + 1] = nums[dq.peekFirst()];
    }
    return result;
}
```

:::

| | Complexity |
|---|---|
| **Time** | O(n) -- each element enters and exits the deque at most once |
| **Space** | O(k) -- deque holds at most k elements |

:::

::: details #18 -- Longest Repeating Character Replacement (LC 424)

**Problem:** Given a string `s` and an integer `k`, find the length of the longest substring where you can replace at most `k` characters to make all characters the same.

**Key insight:** In any valid window of length `window_len`, the number of characters you need to replace equals `window_len - max_freq` (where `max_freq` is the count of the most frequent character in the window). The window is valid when `window_len - max_freq <= k`.

**The subtle trick:** You do not need to decrease `max_freq` when shrinking. Since we are maximizing, keeping the historical maximum only causes us to keep the window size (it never incorrectly shrinks). This is a common interview discussion point.

:::code-group

```python [Python]
def characterReplacement(self, s: str, k: int) -> int:
    count = {}
    left = 0
    max_freq = 0
    result = 0

    for right in range(len(s)):
        char = s[right]
        count[char] = count.get(char, 0) + 1
        max_freq = max(max_freq, count[char])

        # window length - max_freq = chars to replace
        # if that exceeds k, shrink
        while (right - left + 1) - max_freq > k:
            count[s[left]] -= 1
            left += 1

        result = max(result, right - left + 1)

    return result
```

```java [Java]
int characterReplacement(String s, int k) {
    int[] count = new int[26];
    int left = 0, maxFreq = 0, result = 0;

    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        count[c - 'A']++;
        maxFreq = Math.max(maxFreq, count[c - 'A']);

        // window length - maxFreq = chars to replace
        // if that exceeds k, shrink
        while ((right - left + 1) - maxFreq > k) {
            count[s.charAt(left) - 'A']--;
            left++;
        }

        result = Math.max(result, right - left + 1);
    }
    return result;
}
```

:::

| | Complexity |
|---|---|
| **Time** | O(n) |
| **Space** | O(1) -- at most 26 characters tracked |

::: tip INTERVIEW TALKING POINT
If the interviewer asks _"Is it correct to not update max_freq when shrinking?"_ -- the answer is yes. Since we only care about the maximum window length ever seen, a stale `max_freq` only prevents the window from growing (it never causes a wrong answer). The window effectively slides without shrinking below its historical best size.
:::

:::

---

## The Monotonic Deque Pattern

Problem #17 (Sliding Window Maximum) uses a different technique from the standard sliding window template. Here is when and why you need it.

### When to Use a Monotonic Deque

Use a monotonic deque when you need the **minimum or maximum within a sliding window of fixed size**. The standard sliding window template tracks aggregates (sums, counts, sets) but cannot efficiently track min/max -- a heap gives O(n log k) but a monotonic deque gives O(n).

### How It Works

A **monotonic decreasing deque** maintains this invariant: elements from front to back are in decreasing order.

```
Window: [1, 3, -1, -3, 5, 3, 6, 7], k = 3

Step by step (deque stores indices):
i=0: nums[0]=1   -> deque=[0]
i=1: nums[1]=3   -> pop 0 (1<3), deque=[1]
i=2: nums[2]=-1  -> deque=[1,2], window full, max=nums[1]=3
i=3: nums[3]=-3  -> deque=[1,2,3], evict 1 (out of window), max=nums[2]=-1
i=4: nums[4]=5   -> pop all (5>everything), deque=[4], max=nums[4]=5
...
```

### The Template

:::code-group

```python [Python]
from collections import deque

def monotonic_deque_max(nums, k):
    """Returns max in each sliding window of size k."""
    dq = deque()  # stores indices in decreasing order of values
    result = []

    for i in range(len(nums)):
        # 1. Remove expired indices from front
        while dq and dq[0] < i - k + 1:
            dq.popleft()

        # 2. Remove smaller elements from back (maintain decreasing order)
        while dq and nums[dq[-1]] <= nums[i]:
            dq.pop()

        # 3. Add current index
        dq.append(i)

        # 4. Record result once first window is complete
        if i >= k - 1:
            result.append(nums[dq[0]])

    return result
```

```java [Java]
int[] monotonicDequeMax(int[] nums, int k) {
    int n = nums.length;
    int[] result = new int[n - k + 1];
    Deque<Integer> dq = new ArrayDeque<>(); // stores indices, decreasing values

    for (int i = 0; i < n; i++) {
        // 1. Remove expired indices from front
        while (!dq.isEmpty() && dq.peekFirst() < i - k + 1) dq.pollFirst();

        // 2. Maintain decreasing order: remove smaller elements from back
        while (!dq.isEmpty() && nums[dq.peekLast()] <= nums[i]) dq.pollLast();

        // 3. Add current index
        dq.offerLast(i);

        // 4. Record result once first window is complete
        if (i >= k - 1) result[i - k + 1] = nums[dq.peekFirst()];
    }
    return result;
}
```

:::

::: info FOR MINIMUM INSTEAD OF MAXIMUM
Flip the comparison in step 2 to `>=` and use a monotonic **increasing** deque. The front of the deque will be the minimum.
:::

---

## Common Mistakes

::: danger MISTAKES THAT COST OFFERS

**1. Forgetting to remove from window when shrinking.**
When you increment `left`, you must update your window state (decrement counts, remove from set, etc.). Forgetting this is the #1 bug in sliding window problems.

**2. Off-by-one on window length.**
Window length is `right - left + 1`, not `right - left`. Draw it out: indices [2, 3, 4] have length 3 = 4 - 2 + 1.

**3. Using a heap instead of a deque for Sliding Window Maximum.**
A max-heap gives O(n log k). The interviewer expects O(n) with a monotonic deque. Using a heap signals you do not know the optimal approach.

**4. Not handling empty strings / edge cases.**
Always check: what if `s` is empty? What if `k` is 0? What if `t` is longer than `s`? State these edge cases out loud before coding.

**5. Storing values instead of indices in the deque.**
For Sliding Window Maximum, store **indices** in the deque so you can check whether elements have expired (left the window). If you store values, you cannot determine when to evict.

**6. Confusing "while" with "if" for the shrink phase.**
The shrink step is a `while` loop, not an `if`. The window might need to shrink by multiple positions to become valid again.
:::

---

## Practice Checklist

Use this to track your progress. Practice each problem in a plain text editor (no IDE, no autocomplete) to simulate interview conditions.

- [ ] **#15 Longest Substring Without Repeating Characters** -- Solve in under 10 minutes. This is your warm-up.
- [ ] **#18 Longest Repeating Character Replacement** -- Explain the `max_freq` trick out loud. Can you justify why it is correct without updating `max_freq` on shrink?
- [ ] **#16 Minimum Window Substring** -- Solve with the `have`/`need` pattern. Practice explaining your approach for 2 minutes before writing code.
- [ ] **#17 Sliding Window Maximum** -- Draw the deque state for `[1, 3, -1, -3, 5, 3, 6, 7]` with k=3. Write the solution without looking at the template.
- [ ] **Dry run all 4** -- Pick one problem, write the solution on paper, and trace through a test case step by step. This is exactly what your interviewer expects.
- [ ] **Time yourself** -- Can you explain your approach (2 min) + code (12 min) + dry run (3 min) for any of these in under 20 minutes?

::: tip NEXT UP
If you have finished Sliding Window, move on to [Tier 4: Dynamic Programming](./dp). If you still have time after DP, continue to [Tier 5: HashMap / Heap / Binary Search](./data-structures).
:::
