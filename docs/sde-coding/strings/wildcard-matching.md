# Wildcard Matching

> **2D dynamic programming with greedy optimization**

Wildcard Matching is a classic hard DP problem that's often compared to Regular Expression Matching. While similar, the `*` operator works differently here, matching any sequence of characters (including empty).

---

## Problem Statement

Given an input string `s` and a pattern `p`, implement wildcard pattern matching with support for `'?'` and `'*'` where:

- `'?'` Matches any single character.
- `'*'` Matches any sequence of characters (including the empty sequence).

The matching should cover the **entire** input string (not partial).

This is [LeetCode Problem #44](https://leetcode.com/problems/wildcard-matching/) - a Hard difficulty problem.

### Examples

| s | p | Output | Explanation |
|---|---|--------|-------------|
| `"aa"` | `"a"` | `false` | "a" doesn't match "aa" |
| `"aa"` | `"*"` | `true` | `*` matches any sequence |
| `"cb"` | `"?a"` | `false` | `?` matches 'c', but 'b' != 'a' |
| `"adceb"` | `"*a*b"` | `true` | First `*`="", `a`='a', `*`="dce", `b`='b' |
| `"acdcb"` | `"a*c?b"` | `false` | Can't match |

### Constraints

- `0 <= s.length, p.length <= 2000`
- `s` contains only lowercase English letters
- `p` contains only lowercase English letters, `'?'` or `'*'`

---

## Key Difference from Regex Matching

| Feature | Regex (`*`) | Wildcard (`*`) |
|---------|-------------|----------------|
| What `*` matches | Zero or more of **preceding element** | Any sequence (independent) |
| `a*` meaning | Zero or more 'a' | 'a' followed by anything |
| `*` alone | Invalid (needs preceding) | Matches any string |
| `.*` meaning | Any sequence | '.' then any sequence |

---

## DP Table Visualization

![Wildcard Matching DP Table](./assets/wildcard-matching-dp-table.png)

---

## Approach 1: Dynamic Programming

### Key Insight

Define `dp[i][j]` = True if `s[0:i]` matches `p[0:j]`.

### Transitions

| Pattern char | Condition | Transition |
|--------------|-----------|------------|
| Letter | Must match | `dp[i][j] = dp[i-1][j-1]` if `s[i-1] == p[j-1]` |
| `?` | Any single char | `dp[i][j] = dp[i-1][j-1]` |
| `*` | Empty sequence | `dp[i][j] = dp[i][j-1]` |
| `*` | One+ characters | `dp[i][j] = dp[i-1][j]` |

### Mermaid Diagram

```mermaid
flowchart TD
    A["dp[i][j] = ?"] --> B{p[j-1] == '*'?}
    B -->|Yes| C["Match empty: dp[i][j-1]<br/>OR<br/>Match one+ chars: dp[i-1][j]"]
    B -->|No| D{p[j-1] == '?' OR<br/>p[j-1] == s[i-1]?}
    D -->|Yes| E["dp[i][j] = dp[i-1][j-1]"]
    D -->|No| F["dp[i][j] = False"]

    style C fill:#e1f5fe
    style E fill:#c8e6c9
    style F fill:#ffcdd2
```

### Solution: 2D DP

::: code-group

```python [Python]
def isMatch(s: str, p: str) -> bool:
    """
    Wildcard pattern matching using 2D DP.

    Args:
        s: Input string
        p: Pattern with '?' (single char) and '*' (any sequence)

    Returns:
        True if pattern matches entire string

    Time: O(m * n) where m = len(s), n = len(p)
    Space: O(m * n) for DP table
    """
    m, n = len(s), len(p)

    # dp[i][j] = True if s[0:i] matches p[0:j]
    dp = [[False] * (n + 1) for _ in range(m + 1)]

    # Base case: empty string matches empty pattern
    dp[0][0] = True

    # Base case: pattern with leading '*' can match empty string
    # e.g., "*", "**", "***" all match ""
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
        else:
            break  # Non-'*' character breaks the chain

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # '*' matches empty (dp[i][j-1]) or one+ chars (dp[i-1][j])
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]

            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                # '?' matches any char, or exact character match
                dp[i][j] = dp[i - 1][j - 1]

            # else: dp[i][j] remains False

    return dp[m][n]
```

```java [Java]
public boolean isMatch(String s, String p) {
    int m = s.length(), n = p.length();
    boolean[][] dp = new boolean[m + 1][n + 1];
    dp[0][0] = true;

    for (int j = 1; j <= n; j++) {
        if (p.charAt(j - 1) == '*') dp[0][j] = dp[0][j - 1];
        else break;
    }

    for (int i = 1; i <= m; i++) {
        for (int j = 1; j <= n; j++) {
            char pc = p.charAt(j - 1);
            if (pc == '*') {
                dp[i][j] = dp[i][j - 1] || dp[i - 1][j];
            } else if (pc == '?' || pc == s.charAt(i - 1)) {
                dp[i][j] = dp[i - 1][j - 1];
            }
        }
    }
    return dp[m][n];
}
```

:::


::: info Complexity: Time O(m * n) · Space O(m * n)
- **Time:** O(m * n) where m = len(s), n = len(p) - filling each cell of the DP table once
- **Space:** O(m * n) for the 2D DP table storing match states for all (string prefix, pattern prefix) combinations
:::

---

## Visual Walkthrough

For `s = "adceb"`, `p = "*a*b"`:

```
Pattern "*a*b" breakdown:
- * : matches "" (or could match more)
- a : must match 'a'
- * : matches "dce"
- b : must match 'b'

DP Table:
          ""    *    a    *    b
    ""     T    T    F    F    F
     a     F    T    T    T    F
     d     F    T    F    T    F
     c     F    T    F    T    F
     e     F    T    F    T    F
     b     F    T    F    T    T

Key transitions:
- dp[0][1] = T because * matches empty
- dp[1][1] = T because * matches "a"
- dp[1][2] = T because 'a' matches 'a' (dp[0][1] = T)
- dp[1][3] = T because * matches empty after "a"
- dp[5][3] = T because * matches "dce"
- dp[5][4] = T because 'b' matches 'b' (dp[4][3] = T)

Answer: dp[5][4] = True
```

---

## Space-Optimized Solution: O(n) Space

```python
def isMatch_optimized(s: str, p: str) -> bool:
    """
    Space-optimized wildcard matching using single row.

    Time: O(m * n)
    Space: O(n)
    """
    m, n = len(s), len(p)

    # Single row DP
    dp = [False] * (n + 1)
    dp[0] = True

    # Initialize for empty string
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[j] = dp[j - 1]
        else:
            break

    for i in range(1, m + 1):
        prev_diagonal = dp[0]  # dp[i-1][j-1]
        dp[0] = False  # Non-empty string can't match empty pattern

        for j in range(1, n + 1):
            temp = dp[j]  # Save for next iteration's diagonal

            if p[j - 1] == '*':
                # dp[i][j-1] (empty) or dp[i-1][j] (one+ chars)
                dp[j] = dp[j - 1] or dp[j]
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                dp[j] = prev_diagonal
            else:
                dp[j] = False

            prev_diagonal = temp

    return dp[n]
```

::: info Complexity: Time O(m * n) · Space O(n)
- **Time:** O(m * n) - same time complexity as 2D DP, processing all subproblems
- **Space:** O(n) - only storing a single row of size n+1, reusing it for each row of the original table
:::

---

## Approach 2: Greedy with Backtracking

A more efficient approach for average cases:

```python
def isMatch_greedy(s: str, p: str) -> bool:
    """
    Greedy approach with backtracking for '*'.

    Key idea: When we see '*', remember its position.
    If matching fails later, backtrack to '*' and try
    matching one more character.

    Time: O(m * n) worst case, but faster on average
    Space: O(1)
    """
    m, n = len(s), len(p)
    s_idx, p_idx = 0, 0
    star_idx = -1  # Position of last '*' in pattern
    s_temp_idx = -1  # Position in s when we matched last '*'

    while s_idx < m:
        # Case 1: Characters match or '?'
        if p_idx < n and (p[p_idx] == '?' or p[p_idx] == s[s_idx]):
            s_idx += 1
            p_idx += 1

        # Case 2: '*' found - save position for backtracking
        elif p_idx < n and p[p_idx] == '*':
            star_idx = p_idx
            s_temp_idx = s_idx
            p_idx += 1  # Try matching '*' with empty string first

        # Case 3: Mismatch - backtrack to last '*'
        elif star_idx != -1:
            p_idx = star_idx + 1
            s_temp_idx += 1
            s_idx = s_temp_idx  # '*' matches one more character

        # Case 4: No '*' to backtrack to - fail
        else:
            return False

    # Check remaining pattern characters (must all be '*')
    while p_idx < n and p[p_idx] == '*':
        p_idx += 1

    return p_idx == n
```

::: info Complexity: Time O(m * n) worst · Space O(1)
- **Time:** O(m * n) in worst case (e.g., pattern "*a*a*a...") but O(m + n) in average case with fewer '*' characters
- **Space:** O(1) - only using a constant number of index variables for tracking positions
:::

### Why Greedy Works

1. When we see `*`, we first try matching it with empty string
2. If later matching fails, we backtrack and let `*` consume one more char
3. We only need to remember the **last** `*` position (not all of them)

```
Example: s="adceb", p="*a*b"

s_idx=0, p_idx=0: '*' found, star_idx=0, try empty match
s_idx=0, p_idx=1: 'a'=='a', match!
s_idx=1, p_idx=2: '*' found, star_idx=2, try empty match
s_idx=1, p_idx=3: 'd'!='b', backtrack to star_idx=2
s_idx=2, p_idx=3: 'c'!='b', backtrack, s_temp_idx=3
s_idx=3, p_idx=3: 'e'!='b', backtrack, s_temp_idx=4
s_idx=4, p_idx=3: 'b'=='b', match!
Pattern exhausted, string exhausted -> True
```

---

## Complexity Analysis

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| 2D DP | O(m * n) | O(m * n) | Straightforward |
| 1D DP | O(m * n) | O(n) | Space optimized |
| Greedy | O(m * n) worst | O(1) | Faster average case |

---

## Edge Cases

| s | p | Output | Note |
|---|---|--------|------|
| `""` | `""` | `true` | Empty matches empty |
| `""` | `"*"` | `true` | `*` matches empty |
| `""` | `"?"` | `false` | `?` needs a character |
| `"a"` | `"*"` | `true` | `*` matches anything |
| `"abc"` | `"*****"` | `true` | Multiple `*` same as one |
| `"a"` | `"*?*"` | `true` | `?` matches 'a' |

---

## Common Mistakes

| Mistake | Problem | Fix |
|---------|---------|-----|
| Treating `*` like regex | `*` doesn't need preceding char | Independent operator |
| Not handling multiple `*` | `***` should work | Each `*` can be empty |
| Wrong base case | Leading `*` | Check all leading `*` in pattern |
| Greedy without backtrack | Gets stuck | Must save `*` position |

---

## Interview Tips

1. **Distinguish from regex**: `*` here is independent, not tied to preceding char
2. **Start with DP**: Easier to reason about correctly
3. **Mention greedy**: Shows you know optimization
4. **Handle `**` case**: Multiple stars collapse to one

### Follow-up Questions

- **"What's the difference from regex?"** - `*` matches any sequence independently
- **"Can you do O(1) space?"** - Yes, greedy approach
- **"What about very long patterns with many `*`?"** - Greedy is faster

---

## Related Problems

::: details Regular Expression Matching (LeetCode 10)
**Problem:** Implement regex matching with '.' (any single char) and '*' (zero or more of preceding element).

**Key Insight:** Key difference: '*' in regex modifies the preceding character (a* = zero or more 'a'), while wildcard '*' is independent.

**Approach:** 2D DP where dp[i][j] means s[0:i] matches p[0:j]. For '*', handle zero occurrences (dp[i][j-2]) and one+ occurrences if preceding matches.

**Complexity:** O(m * n) time, O(m * n) space
:::

::: details Edit Distance (LeetCode 72)
**Problem:** Find minimum insert, delete, or replace operations to transform word1 into word2.

**Key Insight:** Similar 2D DP framework but with different semantics - counting operations vs. boolean matching.

**Approach:** dp[i][j] = min operations for word1[0:i] to word2[0:j]. Three transitions: replace (diagonal+1), delete (above+1), insert (left+1).

**Complexity:** O(m * n) time, O(m * n) space (can optimize to O(n))
:::

---

## Complete Solution with Tests

```python
def isMatch(s: str, p: str) -> bool:
    """
    Wildcard pattern matching with '?' and '*'.

    Time: O(m * n)
    Space: O(m * n)
    """
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]

    dp[0][0] = True

    for j in range(1, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 1]
        else:
            break

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 1] or dp[i - 1][j]
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                dp[i][j] = dp[i - 1][j - 1]

    return dp[m][n]


def isMatch_greedy(s: str, p: str) -> bool:
    """Greedy approach with O(1) space."""
    m, n = len(s), len(p)
    s_idx, p_idx = 0, 0
    star_idx, s_temp = -1, -1

    while s_idx < m:
        if p_idx < n and (p[p_idx] == '?' or p[p_idx] == s[s_idx]):
            s_idx += 1
            p_idx += 1
        elif p_idx < n and p[p_idx] == '*':
            star_idx = p_idx
            s_temp = s_idx
            p_idx += 1
        elif star_idx != -1:
            p_idx = star_idx + 1
            s_temp += 1
            s_idx = s_temp
        else:
            return False

    while p_idx < n and p[p_idx] == '*':
        p_idx += 1

    return p_idx == n


# Test cases
if __name__ == "__main__":
    test_cases = [
        ("aa", "a", False),
        ("aa", "*", True),
        ("cb", "?a", False),
        ("adceb", "*a*b", True),
        ("acdcb", "a*c?b", False),
        ("", "", True),
        ("", "*", True),
        ("", "?", False),
        ("abc", "*****", True),
        ("abcde", "a*e", True),
        ("abcde", "*?*?*?*?*?*", True),
    ]

    print("Testing DP approach:")
    for s, p, expected in test_cases:
        result = isMatch(s, p)
        status = "PASS" if result == expected else "FAIL"
        print(f'{status}: isMatch("{s}", "{p}") = {result}')

    print("\nTesting Greedy approach:")
    for s, p, expected in test_cases:
        result = isMatch_greedy(s, p)
        status = "PASS" if result == expected else "FAIL"
        print(f'{status}: isMatch_greedy("{s}", "{p}") = {result}')
```

---

## Summary

| Key Point | Details |
|-----------|---------|
| Pattern | 2D String DP or Greedy with Backtracking |
| `*` Meaning | Any sequence (including empty) |
| `?` Meaning | Any single character |
| Time | O(m * n) |
| Space | O(m * n) DP or O(1) Greedy |

**Takeaway:** Wildcard Matching differs from Regex in that `*` independently matches any sequence. The DP solution is straightforward, while the greedy approach with backtracking is more space-efficient. Know both approaches for interviews.

---

## References

- [LeetCode - Wildcard Matching](https://leetcode.com/problems/wildcard-matching/)
- [GeeksforGeeks - Wildcard Pattern Matching](https://www.geeksforgeeks.org/wildcard-pattern-matching/)
