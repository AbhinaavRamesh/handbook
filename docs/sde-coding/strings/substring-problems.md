# Substring Problems

> **Classic sliding window and expansion techniques**

Substring problems are among the most frequently asked questions in technical interviews at companies like Google, Meta, and Amazon. These problems test your understanding of sliding window techniques, hash maps, and dynamic programming. This guide covers three essential substring problems that every software engineer should master.

---

## Longest Substring Without Repeating Characters

### Problem Statement

Given a string `s`, find the length of the **longest substring** without repeating characters.

This is [LeetCode Problem #3](https://leetcode.com/problems/longest-substring-without-repeating-characters/) - a classic Medium difficulty problem.

### Examples

| Input | Output | Explanation |
|-------|--------|-------------|
| `"abcabcbb"` | `3` | The answer is `"abc"`, with length 3. Note that `"bca"` and `"cab"` are also valid answers. |
| `"bbbbb"` | `1` | The answer is `"b"`, with length 1. |
| `"pwwkew"` | `3` | The answer is `"wke"`, with length 3. Note: The answer must be a **substring**, `"pwke"` is a subsequence, not a substring. |
| `""` | `0` | Empty string has no characters. |

### Approach: Sliding Window with Hash Map

The key insight is to maintain a **sliding window** that contains no repeating characters. We use a hash map to track the most recent index of each character.

**Algorithm:**
1. Initialize two pointers: `left` (window start) and `right` (window end)
2. Use a hash map to store each character's most recent index
3. For each character at position `right`:
   - If the character was seen before and its previous index is within the current window (`>= left`), move `left` to `previous_index + 1`
   - Update the character's index in the hash map
   - Calculate the current window length and update max if needed
4. Return the maximum length found

### Mermaid Diagram

```mermaid
flowchart TD
    A[Start: left=0, max_length=0] --> B[Iterate right from 0 to n-1]
    B --> C{char in map AND<br/>map[char] >= left?}
    C -->|Yes| D[Move left to map[char] + 1]
    C -->|No| E[Keep left unchanged]
    D --> F[Update map[char] = right]
    E --> F
    F --> G[max_length = max(max_length, right - left + 1)]
    G --> H{More characters?}
    H -->|Yes| B
    H -->|No| I[Return max_length]
```

### Visual Walkthrough

For input `"abcabcbb"`:

```
Step 1: [a]bcabcbb    left=0, right=0, window="a", max=1
Step 2: [a b]cabcbb   left=0, right=1, window="ab", max=2
Step 3: [a b c]abcbb  left=0, right=2, window="abc", max=3
Step 4: a[b c a]bcbb  left=1, right=3, window="bca", max=3 (a seen at 0, move left to 1)
Step 5: a b[c a b]cbb left=2, right=4, window="cab", max=3 (b seen at 1, move left to 2)
Step 6: a b c[a b c]bb left=3, right=5, window="abc", max=3 (c seen at 2, move left to 3)
Step 7: a b c a[b c b]b left=4, right=6, window="bcb"->skip (b seen at 4, move left to 5)
        a b c a b[c b]b  left=5, right=6, window="cb", max=3
Step 8: a b c a b c[b]b  left=7, right=7, window="b", max=3
```

### Solution

::: code-group

```python [Python]
def lengthOfLongestSubstring(s: str) -> int:
    """
    Find the length of the longest substring without repeating characters.

    Args:
        s: Input string

    Returns:
        Length of the longest substring without repeating characters
    """
    char_index = {}  # Maps character to its most recent index
    max_length = 0
    left = 0  # Left boundary of the sliding window

    for right, char in enumerate(s):
        # If character was seen and is within current window
        if char in char_index and char_index[char] >= left:
            # Move left boundary past the previous occurrence
            left = char_index[char] + 1

        # Update the character's most recent index
        char_index[char] = right

        # Update maximum length
        max_length = max(max_length, right - left + 1)

    return max_length
```

```java [Java]
public int lengthOfLongestSubstring(String s) {
    Map<Character, Integer> charIndex = new HashMap<>();
    int maxLen = 0, left = 0;
    for (int right = 0; right < s.length(); right++) {
        char c = s.charAt(right);
        if (charIndex.containsKey(c) && charIndex.get(c) >= left) {
            left = charIndex.get(c) + 1;
        }
        charIndex.put(c, right);
        maxLen = Math.max(maxLen, right - left + 1);
    }
    return maxLen;
}
```

:::


::: info Complexity: Time O(n) · Space O(min(n, m))
- **Time:** O(n) - single pass through the string, each character processed once with O(1) hash map operations
- **Space:** O(min(n, m)) where m is the character set size (26 for lowercase, 128 for ASCII) - hash map stores at most min(n, m) entries
:::

### Alternative: Using Set (Two Pointer)

```python
def lengthOfLongestSubstring_set(s: str) -> int:
    """
    Alternative approach using a set to track characters in current window.
    """
    char_set = set()
    max_length = 0
    left = 0

    for right in range(len(s)):
        # Shrink window from left until no duplicate
        while s[right] in char_set:
            char_set.remove(s[left])
            left += 1

        char_set.add(s[right])
        max_length = max(max_length, right - left + 1)

    return max_length
```

::: info Complexity: Time O(n) · Space O(min(n, m))
- **Time:** O(n) - each character is visited at most twice (once by right pointer, once when left shrinks window)
- **Space:** O(min(n, m)) where m is the character set size - the set stores at most min(n, m) unique characters
:::

### Complexity Analysis

| Metric | Complexity | Explanation |
|--------|------------|-------------|
| **Time** | O(n) | Each character is visited at most twice (once by right pointer, once when left pointer passes it) |
| **Space** | O(min(n, m)) | Where m is the size of the character set (e.g., 26 for lowercase letters, 128 for ASCII) |

### Edge Cases to Consider

- Empty string: Return 0
- Single character: Return 1
- All same characters: Return 1
- All unique characters: Return length of string

---

## Remove Duplicates in String (Smallest Lexicographical Order)

### Problem Statement

Given a string `s`, remove duplicate letters so that every letter appears **once and only once**. You must make sure your result is the **smallest in lexicographical order** among all possible results.

This is [LeetCode Problem #316](https://leetcode.com/problems/remove-duplicate-letters/) - a Medium difficulty problem.

### Examples

| Input | Output | Explanation |
|-------|--------|-------------|
| `"bcabc"` | `"abc"` | Remove duplicates: result must contain b, c, a exactly once. "abc" < "acb" < "bac" < "bca" < "cab" < "cba" |
| `"cbacdcbc"` | `"acdb"` | Need all unique letters {a, b, c, d}. "acdb" is the smallest possible |
| `"abcd"` | `"abcd"` | Already all unique, return as is |
| `"aaaa"` | `"a"` | Only one unique character |

### Approach: Monotonic Stack with Last Occurrence Tracking

**Key Insights:**
1. We need to include each character exactly once
2. We want the smallest lexicographical result
3. We can only remove a character if it appears again later

**Algorithm:**
1. Precompute the last occurrence index of each character
2. Use a stack to build the result and a set to track what's already included
3. For each character:
   - Skip if already in the result
   - While the stack top is greater than current char AND the stack top appears later in the string, pop it
   - Push current character onto stack

### Mermaid Diagram

```mermaid
flowchart TD
    A[Precompute last_occurrence for each char] --> B[Initialize empty stack and seen set]
    B --> C[Iterate through each char at index i]
    C --> D{char in seen?}
    D -->|Yes| E[Skip to next char]
    D -->|No| F{stack not empty AND<br/>char < stack[-1] AND<br/>i < last_occurrence[stack[-1]]?}
    F -->|Yes| G[Pop from stack<br/>Remove from seen]
    G --> F
    F -->|No| H[Push char to stack<br/>Add to seen]
    H --> I{More characters?}
    E --> I
    I -->|Yes| C
    I -->|No| J[Return stack as string]
```

### Visual Walkthrough

For input `"bcabc"`:

```
last_occurrence: {b: 3, c: 4, a: 2}

Step 1: char='b', stack=[], seen={}
        Push 'b': stack=['b'], seen={b}

Step 2: char='c', stack=['b'], seen={b}
        'c' > 'b', just push
        stack=['b','c'], seen={b,c}

Step 3: char='a', stack=['b','c'], seen={b,c}
        'a' < 'c' and c appears later (index 4)? Yes! Pop 'c'
        stack=['b'], seen={b}
        'a' < 'b' and b appears later (index 3)? Yes! Pop 'b'
        stack=[], seen={}
        Push 'a': stack=['a'], seen={a}

Step 4: char='b', stack=['a'], seen={a}
        Push 'b': stack=['a','b'], seen={a,b}

Step 5: char='c', stack=['a','b'], seen={a,b}
        Push 'c': stack=['a','b','c'], seen={a,b,c}

Result: "abc"
```

### Solution

::: code-group

```python [Python]
def removeDuplicateLetters(s: str) -> str:
    """
    Remove duplicate letters to get smallest lexicographical result.

    Args:
        s: Input string with possible duplicates

    Returns:
        Smallest lexicographical string with each letter appearing once
    """
    # Record the last occurrence of each character
    last_occurrence = {c: i for i, c in enumerate(s)}

    stack = []  # Monotonic stack to build result
    seen = set()  # Track characters already in stack

    for i, char in enumerate(s):
        # Skip if already in result
        if char in seen:
            continue

        # Remove larger characters that appear later in the string
        while stack and char < stack[-1] and i < last_occurrence[stack[-1]]:
            seen.remove(stack.pop())

        stack.append(char)
        seen.add(char)

    return ''.join(stack)
```

```java [Java]
public String removeDuplicateLetters(String s) {
    int[] lastIdx = new int[26];
    for (int i = 0; i < s.length(); i++) lastIdx[s.charAt(i) - 'a'] = i;

    Deque<Character> stack = new ArrayDeque<>();
    boolean[] inStack = new boolean[26];

    for (int i = 0; i < s.length(); i++) {
        char c = s.charAt(i);
        if (inStack[c - 'a']) continue;
        while (!stack.isEmpty() && c < stack.peek() && lastIdx[stack.peek() - 'a'] > i) {
            inStack[stack.pop() - 'a'] = false;
        }
        stack.push(c);
        inStack[c - 'a'] = true;
    }

    StringBuilder sb = new StringBuilder();
    for (char c : stack) sb.append(c);
    return sb.reverse().toString();
}
```

:::


::: info Complexity: Time O(n) · Space O(1)
- **Time:** O(n) - single pass through string; each character pushed and popped at most once
- **Space:** O(1) - stack and set contain at most 26 lowercase letters (constant regardless of input size)
:::

### Alternative: Using Counter

```python
from collections import Counter

def removeDuplicateLetters_counter(s: str) -> str:
    """
    Alternative using Counter to track remaining occurrences.
    """
    # Count remaining occurrences of each character
    remaining = Counter(s)
    stack = []
    seen = set()

    for char in s:
        remaining[char] -= 1

        if char in seen:
            continue

        # Pop larger characters that still have occurrences remaining
        while stack and char < stack[-1] and remaining[stack[-1]] > 0:
            seen.remove(stack.pop())

        stack.append(char)
        seen.add(char)

    return ''.join(stack)
```

::: info Complexity: Time O(n) · Space O(1)
- **Time:** O(n) - single pass with O(1) counter updates; each character pushed/popped at most once
- **Space:** O(1) - counter, stack, and set all bounded by 26 lowercase letters
:::

### Complexity Analysis

| Metric | Complexity | Explanation |
|--------|------------|-------------|
| **Time** | O(n) | Each character is pushed and popped at most once |
| **Space** | O(1) | Stack and set contain at most 26 lowercase letters |

### Key Insight

The "monotonic stack" pattern is crucial here: we maintain a stack where we can pop elements if:
1. The current element is smaller (for lexicographical order)
2. The popped element will appear again later (so we can include it later)

---

## Longest Palindromic Substring

### Problem Statement

Given a string `s`, return the **longest palindromic substring** in `s`.

A palindrome reads the same forwards and backwards.

This is [LeetCode Problem #5](https://leetcode.com/problems/longest-palindromic-substring/) - a Medium difficulty problem.

### Examples

| Input | Output | Explanation |
|-------|--------|-------------|
| `"babad"` | `"bab"` or `"aba"` | Both are valid longest palindromes of length 3 |
| `"cbbd"` | `"bb"` | Longest palindrome has length 2 |
| `"a"` | `"a"` | Single character is a palindrome |
| `"forgeeksskeegfor"` | `"geeksskeeg"` | Length 10 palindrome |

### Approach 1: Expand Around Center

**Key Insight:** Every palindrome has a center. For odd-length palindromes, the center is a single character. For even-length palindromes, the center is between two characters.

**Algorithm:**
1. For each position i (0 to n-1):
   - Expand around i for odd-length palindromes
   - Expand around (i, i+1) for even-length palindromes
2. Track the longest palindrome found
3. Return the longest substring

### Mermaid Diagram: Expand Around Center

```mermaid
flowchart TD
    A[Start: result = empty string] --> B[For each index i in string]
    B --> C[Expand odd: center at i]
    C --> D[Expand while s[left] == s[right]]
    D --> E{odd > result?}
    E -->|Yes| F[result = odd palindrome]
    E -->|No| G[Keep current result]
    F --> H[Expand even: center between i and i+1]
    G --> H
    H --> I[Expand while s[left] == s[right]]
    I --> J{even > result?}
    J -->|Yes| K[result = even palindrome]
    J -->|No| L[Keep current result]
    K --> M{More indices?}
    L --> M
    M -->|Yes| B
    M -->|No| N[Return result]
```

### Visual Walkthrough: Expand Around Center

For input `"babad"`:

```
Index 0 (b):
  Odd expand from (0,0): "b" - length 1
  Even expand from (0,1): b!=a - length 0

Index 1 (a):
  Odd expand from (1,1): "a"
    -> expand (0,2): b!=b? No, b==b! "bab" - length 3 [NEW MAX]
    -> expand (-1,3): out of bounds
  Even expand from (1,2): a!=b - length 0

Index 2 (b):
  Odd expand from (2,2): "b"
    -> expand (1,3): a==a! "aba" - length 3 [TIES MAX]
    -> expand (0,4): b!=d - stop
  Even expand from (2,3): b!=a - length 0

Index 3 (a):
  Odd expand from (3,3): "a" - length 1
  Even expand from (3,4): a!=d - length 0

Index 4 (d):
  Odd expand from (4,4): "d" - length 1
  Even expand from (4,5): out of bounds

Result: "bab" (or "aba")
```

### Solution: Expand Around Center (Python)

::: code-group

```python [Python]
def longestPalindrome(s: str) -> str:
    """
    Find the longest palindromic substring using expand around center.

    Args:
        s: Input string

    Returns:
        Longest palindromic substring
    """
    if not s:
        return ""

    def expand(left: int, right: int) -> str:
        """Expand around center and return the palindrome."""
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        # Return the palindrome (left+1 because we went one step too far)
        return s[left + 1:right]

    result = ""

    for i in range(len(s)):
        # Odd length palindrome (single center)
        odd = expand(i, i)
        if len(odd) > len(result):
            result = odd

        # Even length palindrome (center between i and i+1)
        even = expand(i, i + 1)
        if len(even) > len(result):
            result = even

    return result
```

```java [Java]
public String longestPalindrome(String s) {
    if (s == null || s.isEmpty()) return "";
    int start = 0, maxLen = 1;
    for (int i = 0; i < s.length(); i++) {
        int len = Math.max(expandSub(s, i, i), expandSub(s, i, i + 1));
        if (len > maxLen) {
            maxLen = len;
            start = i - (len - 1) / 2;
        }
    }
    return s.substring(start, start + maxLen);
}

private int expandSub(String s, int l, int r) {
    while (l >= 0 && r < s.length() && s.charAt(l) == s.charAt(r)) { l--; r++; }
    return r - l - 1;
}
```

:::


::: info Complexity: Time O(n^2) · Space O(1)
- **Time:** O(n^2) - for each of n centers, expansion can take up to O(n) in the worst case (all same characters)
- **Space:** O(1) - only storing pointers and indices; the returned substring references the original string
:::

### Complexity: Expand Around Center

| Metric | Complexity | Explanation |
|--------|------------|-------------|
| **Time** | O(n^2) | For each of n centers, we may expand up to n/2 times |
| **Space** | O(1) | Only storing pointers (excluding output string) |

---

### Approach 2: Dynamic Programming

**Key Insight:** A substring `s[i:j+1]` is a palindrome if:
1. `s[i] == s[j]`, AND
2. `s[i+1:j]` is also a palindrome (or j - i <= 1)

**Algorithm:**
1. Create a 2D DP table where `dp[i][j]` = True if `s[i:j+1]` is a palindrome
2. Base cases: all single characters are palindromes
3. Fill the table by increasing substring length
4. Track the longest palindrome found

### Mermaid Diagram: DP Approach

```mermaid
flowchart TD
    A[Create n x n DP table] --> B[Initialize: dp[i][i] = True for all i]
    B --> C[Check length 2 substrings]
    C --> D[For length 3 to n]
    D --> E[For each starting index i]
    E --> F[Calculate j = i + length - 1]
    F --> G{s[i] == s[j] AND<br/>dp[i+1][j-1]?}
    G -->|Yes| H[dp[i][j] = True<br/>Update start, max_len]
    G -->|No| I[dp[i][j] = False]
    H --> J{More substrings?}
    I --> J
    J -->|Yes| E
    J -->|No| K[Return s[start:start+max_len]]
```

### Solution: Dynamic Programming (Python)

```python
def longestPalindrome_dp(s: str) -> str:
    """
    Find the longest palindromic substring using dynamic programming.

    Args:
        s: Input string

    Returns:
        Longest palindromic substring
    """
    n = len(s)
    if n == 0:
        return ""

    # dp[i][j] = True if s[i:j+1] is a palindrome
    dp = [[False] * n for _ in range(n)]

    start = 0  # Starting index of longest palindrome
    max_len = 1  # Length of longest palindrome

    # All substrings of length 1 are palindromes
    for i in range(n):
        dp[i][i] = True

    # Check substrings of length 2
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2

    # Check substrings of length 3 and greater
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1  # Ending index

            # s[i:j+1] is palindrome if s[i]==s[j] and s[i+1:j] is palindrome
            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                start = i
                max_len = length

    return s[start:start + max_len]
```

::: info Complexity: Time O(n^2) · Space O(n^2)
- **Time:** O(n^2) - filling the n x n DP table, each cell computed in O(1)
- **Space:** O(n^2) - storing the boolean DP table for all (i, j) substring combinations
:::

### Complexity: Dynamic Programming

| Metric | Complexity | Explanation |
|--------|------------|-------------|
| **Time** | O(n^2) | Fill n x n DP table |
| **Space** | O(n^2) | Store n x n DP table |

### Comparison of Approaches

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Brute Force | O(n^3) | O(1) | Check all substrings |
| Expand Around Center | O(n^2) | O(1) | Recommended for interviews |
| Dynamic Programming | O(n^2) | O(n^2) | Good for understanding, more space |
| Manacher's Algorithm | O(n) | O(n) | Optimal but complex |

**Interview Tip:** The Expand Around Center approach is usually preferred in interviews because it has O(1) space complexity and is easier to implement correctly under pressure.

---

## Interview Applications

These substring problems frequently appear in technical interviews and are foundational for more complex problems:

### Common Variations

1. **Longest Substring with K Distinct Characters** - Extension of sliding window
2. **Minimum Window Substring** - Sliding window with character frequency
3. **Palindrome Partitioning** - DP with palindrome checking
4. **Count Palindromic Substrings** - Similar to longest palindrome

### Pattern Recognition

| Pattern | Problems | Key Technique |
|---------|----------|---------------|
| Sliding Window | Longest substring without repeat, Minimum window substring | Two pointers + hash map |
| Monotonic Stack | Remove duplicates, Next greater element | Stack + greedy decisions |
| Expand Around Center | Longest palindrome, Count palindromes | Two-pointer expansion |
| 2D DP | Palindrome problems, Edit distance | Subproblem decomposition |

### Interview Tips

1. **Clarify constraints**: Ask about character set (ASCII? Unicode?), string length limits
2. **Start with brute force**: Explain the naive O(n^2) or O(n^3) approach first
3. **Optimize incrementally**: Show how sliding window or DP improves complexity
4. **Handle edge cases**: Empty string, single character, all same characters
5. **Test your solution**: Walk through an example before coding

### Related LeetCode Problems

- [3. Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- [5. Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)
- [316. Remove Duplicate Letters](https://leetcode.com/problems/remove-duplicate-letters/)
- [76. Minimum Window Substring](https://leetcode.com/problems/minimum-window-substring/)
- [647. Palindromic Substrings](https://leetcode.com/problems/palindromic-substrings/)
- [159. Longest Substring with At Most Two Distinct Characters](https://leetcode.com/problems/longest-substring-with-at-most-two-distinct-characters/)

---

## Summary

| Problem | Technique | Time | Space |
|---------|-----------|------|-------|
| Longest Substring Without Repeating | Sliding Window + Hash Map | O(n) | O(min(n,m)) |
| Remove Duplicate Letters | Monotonic Stack | O(n) | O(1) |
| Longest Palindromic Substring | Expand Around Center | O(n^2) | O(1) |

**Key Takeaways:**
- Sliding window is the go-to technique for substring problems with constraints
- Monotonic stacks help when you need to make greedy decisions based on future elements
- For palindromes, expand around center is the most space-efficient O(n^2) solution
- Always consider both time AND space complexity when choosing an approach

---

## Repeated DNA Sequences

### Problem Statement (LeetCode 187)

The DNA sequence is composed of a series of nucleotides abbreviated as `'A'`, `'C'`, `'G'`, and `'T'`.

Given a string `s` that represents a DNA sequence, return all the **10-letter-long** sequences (substrings) that occur more than once in a DNA molecule. You may return the answer in any order.

### Examples

```
Example 1:
Input: s = "AAAAACCCCCAAAAACCCCCCAAAAAGGGTTT"
Output: ["AAAAACCCCC","CCCCCAAAAA"]
Explanation: These two 10-letter sequences appear more than once.

Example 2:
Input: s = "AAAAAAAAAAAAA"
Output: ["AAAAAAAAAA"]
Explanation: The sequence "AAAAAAAAAA" appears twice (at index 0 and 1).
```

### Approach 1: HashSet with Substrings

The simplest approach uses two sets:
1. **seen**: All 10-letter substrings we've encountered
2. **repeated**: Substrings that appear more than once

::: code-group

```python [Python]
def findRepeatedDnaSequences(s: str) -> list[str]:
    """
    Find all 10-letter DNA sequences that appear more than once.

    Time: O(n * 10) = O(n) - extracting each substring takes O(10)
    Space: O(n * 10) = O(n) - storing substrings of length 10
    """
    if len(s) < 10:
        return []

    seen = set()
    repeated = set()

    for i in range(len(s) - 9):  # -9 because we need 10 characters
        substring = s[i:i + 10]

        if substring in seen:
            repeated.add(substring)
        else:
            seen.add(substring)

    return list(repeated)
```

```java [Java]
public List<String> findRepeatedDnaSequences(String s) {
    if (s.length() < 10) return new ArrayList<>();
    Set<String> seen = new HashSet<>();
    Set<String> repeated = new HashSet<>();
    for (int i = 0; i <= s.length() - 10; i++) {
        String sub = s.substring(i, i + 10);
        if (!seen.add(sub)) repeated.add(sub);
    }
    return new ArrayList<>(repeated);
}
```

:::


::: info Complexity: Time O(n) · Space O(n)
- **Time:** O(n * 10) = O(n) - extracting each 10-character substring takes O(10), iterating through n-9 positions
- **Space:** O(n * 10) = O(n) - storing up to n-9 substrings of length 10 in the hash sets
:::

### Why Two Sets?

```
Using a single counter would also work, but two sets are cleaner:

seen:     Tracks all sequences we've encountered at least once
repeated: Tracks sequences we've encountered more than once

When we see a sequence:
- If NOT in seen: add to seen
- If in seen AND not in repeated: add to repeated
- If already in repeated: do nothing (already recorded)

This ensures each repeated sequence appears exactly once in output.
```

### Approach 2: Rolling Hash (Rabin-Karp)

For better space efficiency, we can encode each 10-letter sequence as an integer using a rolling hash. Since DNA has only 4 characters, we can represent each as 2 bits.

```python
def findRepeatedDnaSequences_rolling(s: str) -> list[str]:
    """
    Find repeated DNA sequences using rolling hash.

    Encoding: A=0, C=1, G=2, T=3 (2 bits each)
    10 characters = 20 bits = fits in an integer

    Time: O(n) - rolling hash update is O(1)
    Space: O(n) - but storing integers instead of strings (more efficient)
    """
    if len(s) < 10:
        return []

    # Character to 2-bit encoding
    char_to_bits = {'A': 0, 'C': 1, 'G': 2, 'T': 3}

    # Build initial hash for first 10 characters
    current_hash = 0
    for i in range(10):
        current_hash = (current_hash << 2) | char_to_bits[s[i]]

    seen = {current_hash}
    repeated = set()

    # Rolling hash for remaining substrings
    # Mask to keep only 20 bits (10 chars * 2 bits each)
    mask = (1 << 20) - 1

    for i in range(10, len(s)):
        # Remove leftmost char, add new rightmost char
        current_hash = ((current_hash << 2) & mask) | char_to_bits[s[i]]

        if current_hash in seen:
            repeated.add(current_hash)
        else:
            seen.add(current_hash)

    # Convert hashes back to strings
    bits_to_char = {0: 'A', 1: 'C', 2: 'G', 3: 'T'}
    result = []

    for h in repeated:
        sequence = []
        for _ in range(10):
            sequence.append(bits_to_char[h & 3])
            h >>= 2
        result.append(''.join(reversed(sequence)))

    return result
```

::: info Complexity: Time O(n) · Space O(n)
- **Time:** O(n) - rolling hash update is O(1) per position, converting hashes back to strings is O(result size * 10)
- **Space:** O(n) - storing integer hashes (20 bits each) instead of strings, more memory efficient than string approach
:::

### Visual Walkthrough: Rolling Hash

```
String: "AAAAACCCCC AAAAACCCCC" (spaces added for clarity)

Encoding: A=00, C=01, G=10, T=11

First window "AAAAACCCCC":
  A  A  A  A  A  C  C  C  C  C
  00 00 00 00 00 01 01 01 01 01
  Hash = 0b00000000000101010101 = 341

Slide window by 1 (add 'A', remove 'A'):
  A  A  A  A  C  C  C  C  C  A
  00 00 00 00 01 01 01 01 01 00

  New hash = ((old_hash << 2) & mask) | new_char
           = ((341 << 2) & 0xFFFFF) | 0
           = (1364 & 0xFFFFF) | 0
           = 1364 (but actually we need to mask properly)

The rolling hash allows O(1) updates instead of O(10) substring creation.
```

### Approach 3: Using Counter (Cleaner but Less Efficient)

```python
from collections import Counter

def findRepeatedDnaSequences_counter(s: str) -> list[str]:
    """
    Using Counter for cleaner code.

    Time: O(n * 10) = O(n)
    Space: O(n * 10) = O(n)
    """
    if len(s) < 10:
        return []

    # Count all 10-letter substrings
    counts = Counter(s[i:i + 10] for i in range(len(s) - 9))

    # Return those appearing more than once
    return [seq for seq, count in counts.items() if count > 1]
```

::: info Complexity: Time O(n) · Space O(n)
- **Time:** O(n * 10) = O(n) - Counter iterates through all substrings, each extraction is O(10)
- **Space:** O(n * 10) = O(n) - Counter stores all unique substrings with their counts
:::

### Complexity Comparison

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Two HashSets | O(n) | O(n * 10) | Stores full substrings |
| Rolling Hash | O(n) | O(n) | Stores integers (20 bits each) |
| Counter | O(n) | O(n * 10) | Cleaner code, same complexity |

### Mermaid Diagram: Two-Set Approach

```mermaid
flowchart TD
    A[Start: i = 0] --> B[Extract substring s[i:i+10]]
    B --> C{substring in seen?}
    C -->|No| D[Add to seen set]
    C -->|Yes| E{substring in repeated?}
    E -->|No| F[Add to repeated set]
    E -->|Yes| G[Skip - already recorded]
    D --> H{i < len(s) - 9?}
    F --> H
    G --> H
    H -->|Yes| I[i = i + 1]
    I --> B
    H -->|No| J[Return list(repeated)]
```

### Edge Cases

```python
# Edge case 1: String too short
s = "ACGT"  # Length 4 < 10
# Output: [] (no 10-letter sequences possible)

# Edge case 2: Exactly 10 characters
s = "AAAAAAAAAA"  # Length 10
# Output: [] (only one 10-letter sequence, can't repeat)

# Edge case 3: All same character
s = "AAAAAAAAAAAAAAAA"  # Length 16
# Output: ["AAAAAAAAAA"] (positions 0-9, 1-10, 2-11, etc. all same)

# Edge case 4: No repeats
s = "ACGTACGTACGT"  # Length 12
# Output: [] (each 10-letter sequence is unique)
```

### Why This Problem Matters

1. **Bioinformatics**: Finding repeated DNA sequences is fundamental in genome analysis
2. **Rolling Hash**: Demonstrates the Rabin-Karp pattern useful in many string problems
3. **Bit Manipulation**: Shows how to use bits for efficient encoding (2 bits per character)
4. **Set Operations**: Clean pattern for finding duplicates with O(1) lookup

### Related Problems

::: details Repeated DNA Sequences (LeetCode 187)
**Problem:** Find all 10-letter-long sequences that occur more than once in a DNA string (containing only A, C, G, T).

**Key Insight:** Use HashSet to track seen sequences. Can use rolling hash or bit encoding for efficiency.

**Approach:** Slide 10-character window. Add each sequence to "seen" set. If already seen, add to "result" set.

**Complexity:** O(n) time, O(n) space
:::

::: details Longest Duplicate Substring (LeetCode 1044)
**Problem:** Given a string s, find the longest substring that appears at least twice. Return any such substring.

**Key Insight:** Binary search on length combined with rolling hash for O(1) substring comparison.

**Approach:** Binary search for maximum length. For each candidate length, use rolling hash to find duplicates.

**Complexity:** O(n log n) average time, O(n) space
:::

::: details Find the Index of the First Occurrence (LeetCode 28)
**Problem:** Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

**Key Insight:** Classic string matching - can use brute force, Rabin-Karp, or KMP.

**Approach:** For interview: brute force O(nm) is acceptable. Mention Rabin-Karp (rolling hash) or KMP for O(n+m).

**Complexity:** O(n*m) brute force, O(n+m) with KMP
:::

::: details Longest Happy Prefix (LeetCode 1392)
**Problem:** Find the longest prefix of string s which is also a suffix (excluding the string itself).

**Key Insight:** This is exactly what the KMP failure function computes.

**Approach:** Build KMP failure array. The last value gives the length of longest proper prefix which is also suffix.

**Complexity:** O(n) time, O(n) space
:::

::: details Shortest Palindrome (LeetCode 214)
**Problem:** Add characters to the front of string s to make it a palindrome. Return the shortest such palindrome.

**Key Insight:** Find longest palindrome starting at index 0, then prepend reverse of remaining suffix.

**Approach:** Use KMP on s + "#" + reverse(s) to find longest palindromic prefix. Or use rolling hash.

**Complexity:** O(n) time, O(n) space
:::

### Interview Tips

1. **Start simple**: Use the HashSet approach first - it's O(n) and easy to implement
2. **Mention optimization**: Discuss rolling hash as a space optimization
3. **Know the encoding**: A=00, C=01, G=10, T=11 (or any consistent 2-bit encoding)
4. **Handle edge cases**: Check `len(s) < 10` at the start
5. **Explain the "two sets" pattern**: This is reusable for many "find duplicates" problems

---

## References

- [LeetCode - Longest Substring Without Repeating Characters](https://leetcode.com/problems/longest-substring-without-repeating-characters/)
- [LeetCode - Longest Palindromic Substring](https://leetcode.com/problems/longest-palindromic-substring/)
- [GeeksforGeeks - Longest Substring Without Repeating Characters](https://www.geeksforgeeks.org/dsa/length-of-the-longest-substring-without-repeating-characters/)
- [GeeksforGeeks - Longest Palindromic Substring](https://www.geeksforgeeks.org/dsa/longest-palindromic-substring/)
- [AlgoMonster - Longest Substring Without Repeating Characters](https://algo.monster/liteproblems/3)
- [AlgoMonster - Longest Palindromic Substring](https://algo.monster/liteproblems/5)
- [NeetCode - Longest Palindromic Substring](https://neetcode.io/problems/longest-palindromic-substring?list=blind75)
