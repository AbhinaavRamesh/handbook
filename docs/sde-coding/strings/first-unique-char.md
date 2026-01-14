# First Unique Character in a String

> **Hash map frequency counting with index tracking**

Finding the first unique character is a classic problem that tests your understanding of hash maps, character counting, and efficient traversal strategies.

---

## Problem Statement

Given a string `s`, find the **first non-repeating character** in it and return its index. If it does not exist, return `-1`.

This is [LeetCode Problem #387](https://leetcode.com/problems/first-unique-character-in-a-string/) - an Easy difficulty problem.

### Examples

| Input | Output | Explanation |
|-------|--------|-------------|
| `"leetcode"` | `0` | 'l' is the first unique character |
| `"loveleetcode"` | `2` | 'v' is the first unique character |
| `"aabb"` | `-1` | No unique characters exist |
| `"z"` | `0` | Single character is unique |
| `""` | `-1` | Empty string has no characters |

### Constraints

- `1 <= s.length <= 10^5`
- `s` consists of only lowercase English letters

---

## Approach 1: Two-Pass with Counter

The most straightforward approach:

1. **First pass**: Count frequency of each character
2. **Second pass**: Find first character with count = 1

### Mermaid Diagram

```mermaid
flowchart TD
    A[Start] --> B[Pass 1: Count all characters]
    B --> C[Pass 2: Scan from left]
    C --> D{count[char] == 1?}
    D -->|Yes| E[Return index]
    D -->|No| F[Move to next char]
    F --> G{More characters?}
    G -->|Yes| D
    G -->|No| H[Return -1]

    style E fill:#c8e6c9
    style H fill:#ffcdd2
```

### Solution

```python
from collections import Counter

def firstUniqChar(s: str) -> int:
    """
    Find index of first unique character using Counter.

    Args:
        s: Input string of lowercase letters

    Returns:
        Index of first unique character, or -1 if none exists

    Time: O(n) - two passes through string
    Space: O(1) - at most 26 characters in count
    """
    # Pass 1: Count all characters
    count = Counter(s)

    # Pass 2: Find first with count 1
    for i, char in enumerate(s):
        if count[char] == 1:
            return i

    return -1
```

::: info Complexity: Time O(n) - Space O(1)
- **Time:** O(n) for two passes through the string - Counter creation and linear scan for first unique.
- **Space:** O(1) because at most 26 lowercase letters are stored, which is constant regardless of input size.
:::

### Visual Walkthrough

For input `"loveleetcode"`:

```
Pass 1 - Count characters:
l:1, o:2, v:1, e:4, l:2, t:1, c:1, d:1

Wait, let me recount:
l: 1 (index 0) + 1 (index 4) = 2
o: 1 (index 1) + 1 (index 6) = 2
v: 1 (index 2)                = 1  <-- First unique!
e: 1 (index 3) + 1 (index 5) + 1 (index 7) + 1 (index 11) = 4
t: 1 (index 8)                = 1
c: 1 (index 9)                = 1
d: 1 (index 10)               = 1

Pass 2 - Scan left to right:
Index 0: 'l' -> count=2 (skip)
Index 1: 'o' -> count=2 (skip)
Index 2: 'v' -> count=1 (FOUND!)

Return 2
```

---

## Approach 2: Array-Based (Optimized)

For lowercase letters only, use a fixed-size array instead of a hash map:

```python
def firstUniqChar_array(s: str) -> int:
    """
    Find first unique character using array.

    Time: O(n)
    Space: O(1) - fixed 26-element array
    """
    # Count frequencies
    count = [0] * 26
    for char in s:
        count[ord(char) - ord('a')] += 1

    # Find first unique
    for i, char in enumerate(s):
        if count[ord(char) - ord('a')] == 1:
            return i

    return -1
```

::: info Complexity: Time O(n) - Space O(1)
- **Time:** O(n) for two passes through the string.
- **Space:** O(1) - fixed 26-element array regardless of input size. Slightly faster than hash map due to no hashing overhead.
:::

---

## Approach 3: Single Pass with Index Tracking

Store both count AND first occurrence index:

```python
def firstUniqChar_single_structure(s: str) -> int:
    """
    Track both count and first index for each character.

    Time: O(n)
    Space: O(1) - at most 26 entries
    """
    # Store (count, first_index) for each character
    char_info = {}

    for i, char in enumerate(s):
        if char in char_info:
            char_info[char] = (char_info[char][0] + 1, char_info[char][1])
        else:
            char_info[char] = (1, i)

    # Find minimum index among unique characters
    min_index = float('inf')
    for count, first_idx in char_info.values():
        if count == 1:
            min_index = min(min_index, first_idx)

    return min_index if min_index != float('inf') else -1
```

::: info Complexity: Time O(n) - Space O(1)
- **Time:** O(n) for single pass plus O(k) to find minimum where k is unique characters.
- **Space:** O(1) - at most 26 entries storing (count, first_index) tuples.
:::

---

## Approach 4: Using OrderedDict (Python 3.7+)

Leverage insertion order preservation:

```python
from collections import OrderedDict

def firstUniqChar_ordered(s: str) -> int:
    """
    Use OrderedDict to maintain character order.

    Time: O(n)
    Space: O(k) where k = unique characters
    """
    # In Python 3.7+, regular dict maintains order too
    char_count = {}

    for char in s:
        char_count[char] = char_count.get(char, 0) + 1

    # First key with count 1 is our answer
    for char, count in char_count.items():
        if count == 1:
            return s.index(char)  # O(n) but only called once

    return -1
```

::: info Complexity: Time O(n) - Space O(1)
- **Time:** O(n) for counting pass plus O(n) worst case for s.index() call - still O(n) overall.
- **Space:** O(1) - at most 26 unique lowercase letters stored in dict.
:::

---

## Complexity Analysis

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Counter (2-pass) | O(n) | O(1)* | Most readable |
| Array-based | O(n) | O(1) | Fastest for lowercase |
| Index tracking | O(n) | O(1)* | Single structure |
| OrderedDict | O(n) | O(1)* | Leverages dict order |

*O(1) because at most 26 lowercase letters

---

## Variations

### 1. First Unique Character in a Stream

For a stream of characters, return the first unique at each step:

```python
from collections import OrderedDict

class FirstUnique:
    """
    Find first unique character in a stream.

    Example:
    Stream: a, b, a, c
    After 'a': first unique = 'a'
    After 'b': first unique = 'a'
    After 'a': first unique = 'b' (a is now duplicate)
    After 'c': first unique = 'b'
    """

    def __init__(self):
        self.char_count = {}
        self.unique_order = []  # Maintain order of first appearances

    def add(self, char: str) -> None:
        self.char_count[char] = self.char_count.get(char, 0) + 1
        if self.char_count[char] == 1:
            self.unique_order.append(char)

    def getFirstUnique(self) -> str:
        # Remove duplicates from front of list
        while self.unique_order and self.char_count[self.unique_order[0]] > 1:
            self.unique_order.pop(0)

        return self.unique_order[0] if self.unique_order else ""
```

::: info Complexity: Time O(1) amortized - Space O(k)
- **Time:** O(1) amortized for add() and getFirstUnique() - duplicates are lazily removed from front.
- **Space:** O(k) where k is number of unique characters seen so far.
:::

### 2. Last Unique Character

Find the last character that appears exactly once:

```python
def lastUniqChar(s: str) -> int:
    """Find index of last unique character."""
    count = Counter(s)

    for i in range(len(s) - 1, -1, -1):
        if count[s[i]] == 1:
            return i

    return -1
```

::: info Complexity: Time O(n) - Space O(1)
- **Time:** O(n) for counting plus O(n) worst case for reverse scan.
- **Space:** O(1) - at most 26 characters stored in Counter.
:::

### 3. All Unique Characters

Return all unique characters in order of appearance:

```python
def allUniqChars(s: str) -> str:
    """Return all unique characters maintaining order."""
    count = Counter(s)
    return ''.join(char for char in s if count[char] == 1)

# Example: "aabbc" -> "c"
# Example: "abcabd" -> "cd"
```

::: info Complexity: Time O(n) - Space O(n)
- **Time:** O(n) for counting plus O(n) to build result string.
- **Space:** O(n) worst case for the output string when all characters are unique.
:::

---

## Edge Cases

| Case | Input | Output | Handling |
|------|-------|--------|----------|
| Empty string | `""` | `-1` | No characters |
| Single char | `"a"` | `0` | Only char is unique |
| All duplicates | `"aabb"` | `-1` | No unique exists |
| All unique | `"abc"` | `0` | First char |
| Unique at end | `"aabbcd"` | `4` | Last unique |

---

## Interview Tips

1. **Start with the simple two-pass solution**: It's clean and O(n)
2. **Mention space optimization**: For lowercase only, 26-element array is O(1)
3. **Discuss streaming variant**: Shows deeper understanding
4. **Handle edge cases explicitly**: Empty string, all duplicates

### Follow-up Questions

- **"What if we need to update the string?"** - Use a more dynamic data structure
- **"What if it's a stream?"** - Use OrderedDict or linked list + hash map
- **"What about case-insensitive?"** - Convert to lowercase first

---

## Related Problems

::: details First Unique Number (LeetCode 1429)
**Problem:** Design a data structure that supports adding numbers and finding the first unique number in a stream.

**Key Insight:** Maintain both a count map and an ordered list/queue of unique candidates.

**Approach:** Use hash map for counts and linked list or OrderedDict to maintain insertion order. Remove from candidates when count exceeds 1.

**Complexity:** O(1) amortized for add, O(1) for getFirstUnique
:::

::: details Single Number (LeetCode 136)
**Problem:** Given a non-empty array where every element appears twice except one, find the single element.

**Key Insight:** XOR of two same numbers is 0. XOR is associative and commutative.

**Approach:** XOR all numbers together. Pairs cancel out, leaving only the single number.

**Complexity:** O(n) time, O(1) space
:::

::: details Find the Difference (LeetCode 389)
**Problem:** Given two strings s and t where t is s with one extra character added, find that character.

**Key Insight:** Can use XOR (like Single Number) or character counting.

**Approach:** XOR all characters from both strings. Or sum ASCII values and subtract. Or use Counter subtraction.

**Complexity:** O(n) time, O(1) space with XOR
:::

---

## Summary

| Key Point | Details |
|-----------|---------|
| Pattern | Character frequency counting |
| Best Approach | Two-pass with Counter |
| Time Complexity | O(n) |
| Space Complexity | O(1) for lowercase only |

**Key Insight:** The problem requires both frequency information (which characters are unique) and order information (which unique character comes first). The two-pass approach elegantly handles both by first counting, then scanning in order.

---

## References

- [LeetCode - First Unique Character in a String](https://leetcode.com/problems/first-unique-character-in-a-string/)
- [GeeksforGeeks - First Non-Repeating Character](https://www.geeksforgeeks.org/given-a-string-find-its-first-non-repeating-character/)
