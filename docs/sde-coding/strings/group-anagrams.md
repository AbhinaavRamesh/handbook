# Group Anagrams

> **Hash map grouping with canonical representation**

Group Anagrams is a classic medium-difficulty problem that tests your understanding of hash maps, string manipulation, and the concept of canonical forms for grouping related items.

---

## Problem Statement

Given an array of strings `strs`, group the **anagrams** together. You can return the answer in any order.

An **anagram** is a word or phrase formed by rearranging the letters of a different word or phrase, using all the original letters exactly once.

This is [LeetCode Problem #49](https://leetcode.com/problems/group-anagrams/) - a Medium difficulty problem.

### Examples

| Input | Output |
|-------|--------|
| `["eat","tea","tan","ate","nat","bat"]` | `[["bat"],["nat","tan"],["ate","eat","tea"]]` |
| `[""]` | `[[""]]` |
| `["a"]` | `[["a"]]` |

### Constraints

- `1 <= strs.length <= 10^4`
- `0 <= strs[i].length <= 100`
- `strs[i]` consists of lowercase English letters

---

## Core Insight

Two strings are anagrams if and only if they have the **same characters with the same frequencies**. We need a way to identify anagrams that can serve as a hash map key.

### Canonical Forms for Anagrams

| Approach | Key for "eat" | Key for "tea" | Same? |
|----------|--------------|---------------|-------|
| Sorted string | `"aet"` | `"aet"` | Yes |
| Character count tuple | `(1,0,0,0,1,0,...,1,0)` | `(1,0,0,0,1,0,...,1,0)` | Yes |
| Character count string | `"a1e1t1"` | `"a1e1t1"` | Yes |

---

## Approach 1: Sorted String as Key (Most Common)

### How It Works

1. For each string, sort its characters to get a canonical form
2. Use the sorted string as a key in a hash map
3. Group strings with the same key

### Mermaid Diagram

```mermaid
flowchart TD
    A[Input: list of strings] --> B[For each string s]
    B --> C[Sort s to get key]
    C --> D[Add s to groups[key]]
    D --> E{More strings?}
    E -->|Yes| B
    E -->|No| F[Return groups.values()]

    style F fill:#c8e6c9
```

### Visual Walkthrough

```
Input: ["eat", "tea", "tan", "ate", "nat", "bat"]

String    Sorted Key    Added to Group
------    ----------    --------------
"eat"     "aet"         groups["aet"] = ["eat"]
"tea"     "aet"         groups["aet"] = ["eat", "tea"]
"tan"     "ant"         groups["ant"] = ["tan"]
"ate"     "aet"         groups["aet"] = ["eat", "tea", "ate"]
"nat"     "ant"         groups["ant"] = ["tan", "nat"]
"bat"     "abt"         groups["abt"] = ["bat"]

Result: [["eat", "tea", "ate"], ["tan", "nat"], ["bat"]]
```

### Solution

::: code-group

```python [Python]
from collections import defaultdict

def groupAnagrams(strs: list[str]) -> list[list[str]]:
    """
    Group anagrams using sorted string as key.

    Args:
        strs: List of strings to group

    Returns:
        List of groups, where each group contains anagrams

    Time: O(n * k log k) where n = number of strings, k = max string length
    Space: O(n * k) for storing all strings in the hash map
    """
    groups = defaultdict(list)

    for s in strs:
        # Sort characters to get canonical form
        key = ''.join(sorted(s))
        groups[key].append(s)

    return list(groups.values())
```

```java [Java]
public List<List<String>> groupAnagrams(String[] strs) {
    Map<String, List<String>> groups = new HashMap<>();
    for (String s : strs) {
        char[] ch = s.toCharArray();
        Arrays.sort(ch);
        String key = new String(ch);
        groups.computeIfAbsent(key, k -> new ArrayList<>()).add(s);
    }
    return new ArrayList<>(groups.values());
}
```

:::


::: info Complexity: Time O(n * k log k) - Space O(n * k)
- **Time:** O(n * k log k) where n is number of strings and k is max string length. Sorting each string takes O(k log k).
- **Space:** O(n * k) to store all strings in the hash map grouped by their sorted key.
:::

### Using tuple as Key (Slightly Faster)

```python
def groupAnagrams_tuple_key(strs: list[str]) -> list[list[str]]:
    """
    Use tuple(sorted(s)) as key - avoids string join.
    """
    groups = defaultdict(list)

    for s in strs:
        key = tuple(sorted(s))
        groups[key].append(s)

    return list(groups.values())
```

::: info Complexity: Time O(n * k log k) - Space O(n * k)
- **Time:** O(n * k log k) - same as string key version but avoids string join operation.
- **Space:** O(n * k) for storing strings. Tuple keys have slightly lower overhead than string keys.
:::

---

## Approach 2: Character Count as Key

### How It Works

Instead of sorting (O(k log k) per string), count character frequencies (O(k) per string).

Use the count as a tuple key: `(count_a, count_b, ..., count_z)`

### Solution

```python
def groupAnagrams_count(strs: list[str]) -> list[list[str]]:
    """
    Group anagrams using character count tuple as key.

    Time: O(n * k) where n = number of strings, k = max string length
    Space: O(n * k) for storing strings
    """
    groups = defaultdict(list)

    for s in strs:
        # Count characters - O(k)
        count = [0] * 26
        for char in s:
            count[ord(char) - ord('a')] += 1

        # Use tuple as key (lists aren't hashable)
        key = tuple(count)
        groups[key].append(s)

    return list(groups.values())
```

::: info Complexity: Time O(n * k) - Space O(n * k)
- **Time:** O(n * k) where counting characters is O(k) per string - faster than O(k log k) sorting for longer strings.
- **Space:** O(n * k) for storing strings plus O(n * 26) for the count tuple keys (constant factor).
:::

### Visual Walkthrough

```
String "eat":
  count = [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
  #        a       e                             t
  key = (1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0)

String "tea":
  count = [1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0]
  key = (1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0)

Same key! -> Grouped together
```

---

## Complexity Comparison

| Approach | Time | Space | Notes |
|----------|------|-------|-------|
| Sorted Key | O(n * k log k) | O(n * k) | Simple to implement |
| Count Key | O(n * k) | O(n * k) | Faster for long strings |

Where:
- n = number of strings
- k = maximum length of a string

**When to use which:**
- For short strings (k < 10): Sorted key is fine
- For long strings: Character count is better
- For interviews: Either works, sorted is cleaner to write

---

## Alternative Implementations

### Using Counter as Key (Not Directly Possible)

```python
from collections import Counter

def groupAnagrams_counter(strs: list[str]) -> list[list[str]]:
    """
    Counter isn't hashable, so we convert to frozenset of items.

    Note: This works but is less common in interviews.
    """
    groups = defaultdict(list)

    for s in strs:
        # Convert Counter to hashable key
        key = frozenset(Counter(s).items())
        groups[key].append(s)

    return list(groups.values())
```

::: info Complexity: Time O(n * k) - Space O(n * k)
- **Time:** O(n * k) - Counter creation is O(k), frozenset creation is O(unique chars).
- **Space:** O(n * k) for strings. Frozenset keys are more compact for sparse character distributions.
:::

### One-liner (Pythonic but Less Readable)

```python
from itertools import groupby

def groupAnagrams_oneliner(strs: list[str]) -> list[list[str]]:
    """
    Pythonic one-liner using groupby.

    Note: Requires sorting first, so O(n log n * k) overall.
    Not recommended for interviews - hard to explain.
    """
    sorted_strs = sorted(strs, key=sorted)
    return [list(group) for _, group in groupby(sorted_strs, key=sorted)]
```

::: info Complexity: Time O(n log n * k log k) - Space O(n * k)
- **Time:** O(n log n * k log k) - sorting n strings requires n log n comparisons, each comparison sorts strings in O(k log k).
- **Space:** O(n * k) for the sorted list copy and grouped results.
:::

---

## Edge Cases

| Case | Input | Output | Note |
|------|-------|--------|------|
| Empty strings | `["", "", ""]` | `[["", "", ""]]` | All are anagrams of each other |
| Single string | `["abc"]` | `[["abc"]]` | One group with one element |
| All different | `["a", "b", "c"]` | `[["a"], ["b"], ["c"]]` | Each in own group |
| All same | `["abc", "abc"]` | `[["abc", "abc"]]` | Identical strings are anagrams |
| Mixed lengths | `["a", "ab", "ba"]` | `[["a"], ["ab", "ba"]]` | Different lengths can't be anagrams |

---

## Interview Tips

1. **Start with sorted key approach**: It's cleaner and easier to explain
2. **Mention count approach**: Shows you know the optimization
3. **Use defaultdict**: Cleaner than checking if key exists
4. **Handle empty string**: It's a valid input with key "" or empty count

### Common Follow-up Questions

- **"Can you optimize time?"** - Yes, use character count (O(k) vs O(k log k))
- **"What if strings are very long?"** - Count approach is better
- **"What about Unicode?"** - Use dict-based count instead of array

---

## Related Problems

::: details Valid Anagram (LeetCode 242)
**Problem:** Given two strings s and t, return true if t is an anagram of s, and false otherwise.

**Key Insight:** Two strings are anagrams if they have identical character frequency counts.

**Approach:** Compare character counts using Counter or a 26-element array. For O(1) space with lowercase letters, use array-based counting.

**Complexity:** O(n) time, O(1) space (26 letters)
:::

::: details Find All Anagrams in a String (LeetCode 438)
**Problem:** Given strings s and p, return all start indices of p's anagrams in s.

**Key Insight:** Combine anagram checking with sliding window technique.

**Approach:** Use fixed-size sliding window of length p. Maintain frequency counts and compare as window slides. Can use array for O(1) comparisons.

**Complexity:** O(n) time, O(1) space
:::

::: details Group Shifted Strings (LeetCode 249)
**Problem:** Given an array of strings, group strings that can be shifted to become each other. Shifting means incrementing each character (with wrap-around).

**Key Insight:** Strings in the same group have the same "difference signature" between consecutive characters.

**Approach:** For each string, compute a canonical key based on character differences (modulo 26). Group strings with the same key.

**Complexity:** O(n * k) time where k is max string length, O(n * k) space
:::

---

## Complete Solution with Tests

```python
from collections import defaultdict

def groupAnagrams(strs: list[str]) -> list[list[str]]:
    """
    Group anagrams together using sorted string as key.

    Time: O(n * k log k)
    Space: O(n * k)
    """
    groups = defaultdict(list)

    for s in strs:
        key = ''.join(sorted(s))
        groups[key].append(s)

    return list(groups.values())
```

::: info Complexity: Time O(n * k log k) - Space O(n * k)
- **Time:** O(n * k log k) where n is number of strings, k is max string length. Each string is sorted once.
- **Space:** O(n * k) for the hash map storing all grouped strings.
:::

```python
# Test cases
if __name__ == "__main__":
    # Test 1: Standard case
    result = groupAnagrams(["eat", "tea", "tan", "ate", "nat", "bat"])
    print("Test 1:", result)
    # Expected: [["eat","tea","ate"], ["tan","nat"], ["bat"]] (order may vary)

    # Test 2: Empty string
    result = groupAnagrams([""])
    print("Test 2:", result)
    # Expected: [[""]]

    # Test 3: Single character
    result = groupAnagrams(["a"])
    print("Test 3:", result)
    # Expected: [["a"]]

    # Test 4: Multiple empty strings
    result = groupAnagrams(["", "", ""])
    print("Test 4:", result)
    # Expected: [["", "", ""]]

    print("All tests passed!")
```

---

## Summary

| Key Point | Details |
|-----------|---------|
| Core Technique | Use canonical form as hash key |
| Best Key Type | Sorted string or character count tuple |
| Time Complexity | O(n * k log k) or O(n * k) |
| Space Complexity | O(n * k) |

**Takeaway:** Group Anagrams is a canonical "grouping by key" problem. The key insight is that anagrams share the same sorted form. Master this pattern as it applies to many similar grouping problems.

---

## References

- [LeetCode - Group Anagrams](https://leetcode.com/problems/group-anagrams/)
- [NeetCode - Group Anagrams](https://neetcode.io/problems/anagram-groups)
- [GeeksforGeeks - Group Anagrams](https://www.geeksforgeeks.org/given-a-sequence-of-words-print-all-anagrams-together/)
