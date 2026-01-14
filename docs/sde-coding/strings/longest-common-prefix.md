# Longest Common Prefix

> **Vertical scanning and divide-and-conquer approaches**

Finding the longest common prefix among an array of strings is a classic problem that tests your understanding of string comparison techniques and algorithm design patterns.

---

## Problem Statement

Write a function to find the **longest common prefix** string amongst an array of strings.

If there is no common prefix, return an empty string `""`.

This is [LeetCode Problem #14](https://leetcode.com/problems/longest-common-prefix/) - an Easy difficulty problem.

### Examples

| Input | Output | Explanation |
|-------|--------|-------------|
| `["flower","flow","flight"]` | `"fl"` | Common prefix is "fl" |
| `["dog","racecar","car"]` | `""` | No common prefix |
| `["interspecies","interstellar","interstate"]` | `"inters"` | All start with "inters" |
| `["a"]` | `"a"` | Single string is its own prefix |
| `[]` | `""` | Empty array has no prefix |

### Constraints

- `1 <= strs.length <= 200`
- `0 <= strs[i].length <= 200`
- `strs[i]` consists of only lowercase English letters

---

## Approach 1: Vertical Scanning

Compare characters column by column across all strings.

### How It Works

1. Take the first string as reference
2. For each character position (column):
   - Check if all strings have this character at this position
   - If any string is shorter or has a different character, stop
3. Return the prefix found

### Mermaid Diagram

```mermaid
flowchart TD
    A[Start with first string] --> B[For each character position i]
    B --> C[For each string in array]
    C --> D{string length > i?}
    D -->|No| E[Return prefix so far]
    D -->|Yes| F{char matches first[i]?}
    F -->|No| E
    F -->|Yes| G[Check next string]
    G --> C
    C --> H{All strings checked?}
    H -->|Yes| I[Include char in prefix]
    I --> B
    B --> J{More chars in first?}
    J -->|No| K[Return full first string]

    style E fill:#c8e6c9
    style K fill:#c8e6c9
```

### Visual Walkthrough

For `["flower", "flow", "flight"]`:

```
Position 0:  f  f  f  -> All match 'f' -> prefix = "f"
Position 1:  l  l  l  -> All match 'l' -> prefix = "fl"
Position 2:  o  o  i  -> Mismatch! 'o' != 'i'

Return "fl"
```

### Solution

```python
def longestCommonPrefix(strs: list[str]) -> str:
    """
    Find longest common prefix using vertical scanning.

    Args:
        strs: List of strings to compare

    Returns:
        Longest common prefix string

    Time: O(S) where S = sum of all characters in all strings
    Space: O(1) - only using a few variables
    """
    if not strs:
        return ""

    # Use first string as reference
    for i in range(len(strs[0])):
        char = strs[0][i]

        # Check this position in all other strings
        for j in range(1, len(strs)):
            # If string is shorter or character doesn't match
            if i >= len(strs[j]) or strs[j][i] != char:
                return strs[0][:i]

    return strs[0]
```

---

## Approach 2: Horizontal Scanning

Compare strings pairwise, reducing the prefix each time.

### How It Works

1. Start with the first string as the prefix
2. Compare with each subsequent string
3. Reduce prefix until it matches the beginning of each string

```python
def longestCommonPrefix_horizontal(strs: list[str]) -> str:
    """
    Find LCP using horizontal scanning (pairwise comparison).

    Time: O(S) where S = sum of all characters
    Space: O(1)
    """
    if not strs:
        return ""

    prefix = strs[0]

    for i in range(1, len(strs)):
        # Reduce prefix until it matches start of current string
        while not strs[i].startswith(prefix):
            prefix = prefix[:-1]
            if not prefix:
                return ""

    return prefix
```

### Visual Walkthrough

For `["flower", "flow", "flight"]`:

```
Start: prefix = "flower"

Compare with "flow":
  "flow".startswith("flower")? No
  prefix = "flowe"
  "flow".startswith("flowe")? No
  prefix = "flow"
  "flow".startswith("flow")? Yes!

Compare with "flight":
  "flight".startswith("flow")? No
  prefix = "flo"
  "flight".startswith("flo")? No
  prefix = "fl"
  "flight".startswith("fl")? Yes!

Return "fl"
```

---

## Approach 3: Divide and Conquer

Recursively split the array and find LCP of each half.

```python
def longestCommonPrefix_divideConquer(strs: list[str]) -> str:
    """
    Find LCP using divide and conquer.

    Time: O(S) where S = sum of all characters
    Space: O(m * log n) where m = length of shortest string
           (for recursion stack and intermediate strings)
    """
    if not strs:
        return ""

    def commonPrefix(left: str, right: str) -> str:
        """Find common prefix of two strings."""
        min_len = min(len(left), len(right))
        for i in range(min_len):
            if left[i] != right[i]:
                return left[:i]
        return left[:min_len]

    def divideAndConquer(left: int, right: int) -> str:
        if left == right:
            return strs[left]

        mid = (left + right) // 2
        lcp_left = divideAndConquer(left, mid)
        lcp_right = divideAndConquer(mid + 1, right)

        return commonPrefix(lcp_left, lcp_right)

    return divideAndConquer(0, len(strs) - 1)
```

---

## Approach 4: Binary Search

Use binary search on the length of the prefix.

```python
def longestCommonPrefix_binarySearch(strs: list[str]) -> str:
    """
    Find LCP using binary search on prefix length.

    Time: O(S * log m) where m = length of shortest string
    Space: O(1)
    """
    if not strs:
        return ""

    def isCommonPrefix(length: int) -> bool:
        """Check if prefix of given length is common to all strings."""
        prefix = strs[0][:length]
        return all(s.startswith(prefix) for s in strs)

    # Binary search on the length
    min_len = min(len(s) for s in strs)
    low, high = 0, min_len

    while low < high:
        mid = (low + high + 1) // 2  # Upper middle
        if isCommonPrefix(mid):
            low = mid
        else:
            high = mid - 1

    return strs[0][:low]
```

---

## Approach 5: Using Python's Built-in (Pythonic)

```python
import os

def longestCommonPrefix_pythonic(strs: list[str]) -> str:
    """
    Pythonic solution using os.path.commonprefix.

    Note: This works but may not be accepted in interviews
    as it uses a library function.
    """
    return os.path.commonprefix(strs)


def longestCommonPrefix_zip(strs: list[str]) -> str:
    """
    Using zip to compare characters column-wise.

    Time: O(S)
    Space: O(m) for the zip objects
    """
    if not strs:
        return ""

    prefix = []
    for chars in zip(*strs):
        if len(set(chars)) == 1:  # All same
            prefix.append(chars[0])
        else:
            break

    return ''.join(prefix)
```

---

## Approach 6: Sort and Compare

After sorting, compare only the first and last strings.

```python
def longestCommonPrefix_sort(strs: list[str]) -> str:
    """
    Sort strings, then compare first and last only.

    Key insight: After lexicographic sorting, the common prefix
    of all strings must be the common prefix of first and last.

    Time: O(n log n * m) for sorting, O(m) for comparison
    Space: O(n) for sorted copy or O(1) if in-place
    """
    if not strs:
        return ""

    strs.sort()
    first, last = strs[0], strs[-1]

    i = 0
    while i < len(first) and i < len(last) and first[i] == last[i]:
        i += 1

    return first[:i]
```

---

## Complexity Comparison

| Approach | Time | Space | Best For |
|----------|------|-------|----------|
| Vertical Scanning | O(S) | O(1) | General use |
| Horizontal Scanning | O(S) | O(1) | Simple implementation |
| Divide & Conquer | O(S) | O(m log n) | Parallel processing |
| Binary Search | O(S log m) | O(1) | Very long strings |
| Sort + Compare | O(n log n * m) | O(n) | Already sorted input |
| Zip (Pythonic) | O(S) | O(m) | Quick coding |

Where:
- S = sum of all character counts
- n = number of strings
- m = length of shortest string

---

## Edge Cases

| Case | Input | Output | Handling |
|------|-------|--------|----------|
| Empty array | `[]` | `""` | Return empty |
| Single string | `["abc"]` | `"abc"` | Return the string |
| Empty string in array | `["", "abc"]` | `""` | LCP is empty |
| No common prefix | `["dog", "cat"]` | `""` | First chars differ |
| All identical | `["abc", "abc"]` | `"abc"` | Return full string |

---

## Interview Tips

1. **Start with vertical scanning**: Most intuitive and efficient
2. **Ask about input size**: Binary search for very long strings
3. **Handle edge cases first**: Empty array, single string
4. **Mention optimization**: Early termination when prefix becomes empty

### Common Follow-up Questions

- **"What if strings are very long?"** - Binary search on length
- **"What if we have many strings?"** - Divide and conquer for parallelization
- **"What if strings change frequently?"** - Use a Trie data structure

---

## Trie-Based Solution (Advanced)

For multiple queries or dynamic updates, a Trie is optimal:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.is_end = False

class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def longestCommonPrefix(self) -> str:
        """
        Find LCP by traversing until a branch point.
        """
        prefix = []
        node = self.root

        while node and len(node.children) == 1 and not node.is_end:
            char = next(iter(node.children))
            prefix.append(char)
            node = node.children[char]

        return ''.join(prefix)


def longestCommonPrefix_trie(strs: list[str]) -> str:
    """
    Find LCP using a Trie.

    Time: O(S) to build trie, O(m) to find LCP
    Space: O(S) for trie

    Best for: Multiple queries after initial build
    """
    if not strs:
        return ""

    trie = Trie()
    for s in strs:
        if not s:  # Empty string means no common prefix
            return ""
        trie.insert(s)

    return trie.longestCommonPrefix()
```

---

## Related Problems

::: details Implement Trie - Prefix Tree (LeetCode 208)
**Problem:** Implement a trie with insert, search, and startsWith methods.

**Key Insight:** Trie nodes contain children map and end-of-word flag. Each path from root represents a prefix.

**Approach:** Insert by creating nodes for each character. Search follows path and checks end flag. startsWith only checks path existence.

**Complexity:** O(m) for all operations where m is key length, O(alphabet_size * n * m) space
:::

::: details Search Suggestions System (LeetCode 1268)
**Problem:** Given products array and searchWord, return list of suggested products after each character of searchWord is typed. Suggest at most 3 products with common prefix, sorted lexicographically.

**Key Insight:** Can use Trie for prefix matching or binary search on sorted array.

**Approach:** Sort products. For each prefix, binary search for leftmost match, take up to 3 products starting there.

**Complexity:** O(n log n + m * n) time for sorting + queries, or O(sum of lengths) for Trie
:::

::: details Longest Common Subsequence (LeetCode 1143)
**Problem:** Given two strings, return the length of their longest common subsequence (not necessarily contiguous).

**Key Insight:** Classic 2D DP. Different from prefix - subsequence can skip characters.

**Approach:** dp[i][j] = LCS length for first i chars of text1 and first j chars of text2. Match adds 1, mismatch takes max of excluding either.

**Complexity:** O(m * n) time, O(m * n) space (can optimize to O(n))
:::

---

## Summary

| Key Point | Details |
|-----------|---------|
| Best Approach | Vertical scanning for simplicity |
| Time Complexity | O(S) for most approaches |
| Space Complexity | O(1) for vertical/horizontal scanning |
| Key Insight | Only need to check until first mismatch |

**Takeaway:** The vertical scanning approach is the most intuitive and efficient for typical inputs. For specialized cases (very long strings, multiple queries), consider binary search or Trie-based solutions.

---

## References

- [LeetCode - Longest Common Prefix](https://leetcode.com/problems/longest-common-prefix/)
- [GeeksforGeeks - Longest Common Prefix](https://www.geeksforgeeks.org/longest-common-prefix-using-sorting/)
