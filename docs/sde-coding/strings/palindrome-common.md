# Valid Palindrome & Most Common Words

> **String validation and word frequency problems**

---

## Valid Palindrome

### Problem Statement
Given a string s, return true if it is a palindrome after removing non-alphanumeric characters and ignoring cases.

A phrase is a palindrome if, after converting all uppercase letters into lowercase letters and removing all non-alphanumeric characters, it reads the same forward and backward. Alphanumeric characters include letters and numbers.

**LeetCode Problem:** [125. Valid Palindrome](https://leetcode.com/problems/valid-palindrome/)

### Example
```
Input: "A man, a plan, a canal: Panama"
Output: true
Explanation: "amanaplanacanalpanama" is a palindrome.

Input: "race a car"
Output: false
Explanation: "raceacar" is not a palindrome.

Input: " "
Output: true
Explanation: Empty string after removing non-alphanumeric characters is a palindrome.
```

### Approach
**Two Pointers from Both Ends**

1. Initialize two pointers: `left = 0` and `right = len(s) - 1`
2. Skip non-alphanumeric characters by moving pointers inward
3. Compare lowercase versions of characters at both pointers
4. If they differ, return false; if they match, continue inward
5. If loop completes without mismatches, return true

**Why This Works:**
- Avoids overhead of creating a new filtered string
- Checks character pairs directly within the original string
- Lowercasing standardizes comparison
- Skipping non-alphanumeric ensures compliance with constraints

### Solution (Python)
```python
def isPalindrome(s: str) -> bool:
    left, right = 0, len(s) - 1

    while left < right:
        # Skip non-alphanumeric
        while left < right and not s[left].isalnum():
            left += 1
        while left < right and not s[right].isalnum():
            right -= 1

        if s[left].lower() != s[right].lower():
            return False

        left += 1
        right -= 1

    return True

# One-liner (less efficient but clean)
def isPalindrome_v2(s: str) -> bool:
    cleaned = ''.join(c.lower() for c in s if c.isalnum())
    return cleaned == cleaned[::-1]
```

::: info Complexity: Time O(n) / O(n) - Space O(1) / O(n)
- **Time:** Both solutions iterate through the string once. Two-pointer visits each character at most once. One-liner also processes each character once for filtering.
- **Space:** Two-pointer uses only constant extra space (two index variables). One-liner creates a new cleaned string of length up to n, requiring O(n) space.
:::

### Complexity
- **Time:** O(n) - each character is visited at most once
- **Space:** O(1) for two-pointer approach, O(n) for one-liner (creates cleaned string)

### Google Follow-ups
1. **What if we can delete at most one character?** (LeetCode 680: Valid Palindrome II)
   - Use two pointers; when mismatch found, try skipping either left or right character
   - Check if remaining substring is palindrome

2. **Count palindromic substrings?** (LeetCode 647)
   - Expand around center technique
   - Consider both odd and even length palindromes

3. **Longest palindromic substring?**
   - Dynamic programming or expand around center approach

---

## Most Common Words

### Problem Statement
Given a paragraph and list of banned words, return the most frequent word that is not banned.

The words in paragraph are case-insensitive and the answer should be returned in lowercase. Words cannot contain punctuation symbols.

**LeetCode Problem:** [819. Most Common Word](https://leetcode.com/problems/most-common-word/)

### Example
```
Input: paragraph = "Bob hit a ball, the hit BALL flew far after it was hit."
       banned = ["hit"]
Output: "ball"

Explanation:
- "hit" occurs 3 times, but it is banned
- "ball" occurs twice (no other word does)
- Note: case-insensitive, punctuation ignored
```

### Approach
**Hash Map with Text Preprocessing**

1. **Normalize:** Convert paragraph to lowercase
2. **Extract Words:** Use regex to find all alphanumeric word sequences
3. **Filter:** Create a set of banned words for O(1) lookup
4. **Count:** Use hash map to count frequency of non-banned words
5. **Find Max:** Return the word with highest count

### Solution (Python)
```python
from collections import Counter
import re

def mostCommonWord(paragraph: str, banned: list[str]) -> str:
    # Normalize: lowercase, extract words
    words = re.findall(r'\w+', paragraph.lower())

    # Filter banned words
    banned_set = set(banned)
    valid_words = [w for w in words if w not in banned_set]

    # Count and return most common
    return Counter(valid_words).most_common(1)[0][0]

# Alternative without regex
def mostCommonWord_v2(paragraph: str, banned: list[str]) -> str:
    banned_set = set(banned)

    # Replace punctuation with spaces
    for c in "!?',;.":
        paragraph = paragraph.replace(c, ' ')

    words = paragraph.lower().split()

    # Count frequencies
    word_count = {}
    for word in words:
        if word not in banned_set:
            word_count[word] = word_count.get(word, 0) + 1

    # Find most common
    return max(word_count, key=word_count.get)
```

::: info Complexity: Time O(n + b) - Space O(n)
- **Time:** O(n) to process the paragraph using regex/split and count word frequencies, plus O(b) to create the banned set from the banned list. Finding the most common word is O(unique words).
- **Space:** O(n) for storing word counts and the list of valid words. Banned set uses O(b) space.
:::

### Complexity
- **Time:** O(n + b) where n is paragraph length and b is banned list size
  - O(n) to process paragraph and count words
  - O(b) to create banned set
- **Space:** O(n) for storing word counts

### Edge Cases
- Extra spaces from punctuation replacement can lead to empty strings
- Ensure splitting mechanism ignores empty strings
- Handle single word paragraphs
- All words except one are banned

### Google Follow-ups
1. **Return top K most common words** (LeetCode 692: Top K Frequent Words)
   - Use heap or bucket sort
   - Time: O(n log k) with min-heap

2. **Handle streaming text input**
   - Maintain running count with hash map
   - Use min-heap of size K for top K tracking
   - Consider approximate algorithms for very large streams

3. **Case where multiple words have same frequency**
   - Return lexicographically smallest
   - Use custom comparator in heap

---

## Past Google Interview Reference

These problems are commonly asked in Google interviews as they test fundamental string manipulation skills:

- **Valid Palindrome** tests understanding of two-pointer technique and character handling
- **Most Common Word** tests hash map usage, text preprocessing, and edge case handling

**Related Patterns:**
- Two pointers for in-place string processing
- Hash maps for frequency counting
- Regular expressions for text parsing

---

## Sources

- [Valid Palindrome - LeetCode](https://leetcode.com/problems/valid-palindrome/)
- [125. Valid Palindrome - In-Depth Explanation](https://algo.monster/liteproblems/125)
- [Valid Palindrome - Leetcode Solution](https://algomap.io/problems/valid-palindrome)
- [680. Valid Palindrome II - In-Depth Explanation](https://algo.monster/liteproblems/680)
- [Most Common Word - LeetCode](https://leetcode.com/problems/most-common-word/)
- [819. Most Common Word - In-Depth Explanation](https://algo.monster/liteproblems/819)
- [819. Most Common Word - Detailed Explanation](https://www.designgurus.io/answers/detail/819-most-common-word-8com9word)
- [Top K Frequent Words - LeetCode](https://leetcode.com/problems/top-k-frequent-words/)
