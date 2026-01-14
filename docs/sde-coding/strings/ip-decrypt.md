# Validate IP Address & Decrypt Message

> **String parsing and validation techniques**

---

## Validate IP Address

### Problem Statement
Given a string queryIP, return "IPv4" if valid IPv4, "IPv6" if valid IPv6, or "Neither".

This is [LeetCode Problem 468](https://leetcode.com/problems/validate-ip-address/).

### IPv4 Rules
- 4 decimal numbers (0-255) separated by dots
- No leading zeros (except "0" itself)
- Example valid: "172.16.254.1", "192.168.1.0"
- Example invalid: "192.168.01.1" (leading zero), "256.256.256.256" (out of range)

### IPv6 Rules
- 8 hexadecimal groups separated by colons
- Each group has 1-4 hex digits (0-9, a-f, A-F)
- Leading zeros allowed
- Example valid: "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
- Example invalid: "2001:0db8:85a3::8A2E:037j:7334" (invalid character 'j')

### Solution (Python)
```python
def validIPAddress(queryIP: str) -> str:
    def is_ipv4(ip):
        parts = ip.split('.')
        if len(parts) != 4:
            return False
        for part in parts:
            if not part or not part.isdigit():
                return False
            if len(part) > 1 and part[0] == '0':  # Leading zero
                return False
            if int(part) > 255:
                return False
        return True

    def is_ipv6(ip):
        parts = ip.split(':')
        if len(parts) != 8:
            return False
        hex_chars = set('0123456789abcdefABCDEF')
        for part in parts:
            if not part or len(part) > 4:
                return False
            if not all(c in hex_chars for c in part):
                return False
        return True

    if is_ipv4(queryIP):
        return "IPv4"
    if is_ipv6(queryIP):
        return "IPv6"
    return "Neither"
```

::: info Complexity: Time O(n) - Space O(n)
- **Time:** O(n) where n is the length of the string. Splitting by delimiter is O(n), and validating each part involves checking characters proportional to part length.
- **Space:** O(n) for storing the split parts array and the hex character set.
:::

### Edge Cases to Consider
```python
# Test cases
print(validIPAddress("172.16.254.1"))           # "IPv4"
print(validIPAddress("2001:0db8:85a3:0:0:8A2E:0370:7334"))  # "IPv6"
print(validIPAddress("256.256.256.256"))        # "Neither" - out of range
print(validIPAddress("192.168.01.1"))           # "Neither" - leading zero
print(validIPAddress("1.1.1.1."))               # "Neither" - trailing dot
print(validIPAddress("1e1.4.5.6"))              # "Neither" - invalid character
```

### Complexity
- Time: O(n) where n is the length of the string
- Space: O(n) for storing split parts

---

## Decrypt Message

### Problem Statement
Given an encrypted message where each letter is replaced by the first occurrence index, decrypt it.

Related problems:
- [Decrypt String from Alphabet to Integer Mapping](https://leetcode.com/problems/decrypt-string-from-alphabet-to-integer-mapping/)
- [Encode and Decode Strings](https://leetcode.com/problems/encrypt-and-decrypt-strings/)
- [Decode String](https://leetcode.com/problems/decode-string/)

### Example Approach
Character mapping based on first occurrence - the first unique character maps to 'a', second to 'b', etc.

### Solution (Python)
```python
def decrypt(word: str) -> str:
    # Example: decrypt based on character frequency or pattern
    # This varies by specific problem definition

    # Caesar cipher variant
    def caesar_decrypt(text, shift):
        result = []
        for char in text:
            if char.isalpha():
                base = ord('a') if char.islower() else ord('A')
                result.append(chr((ord(char) - base - shift) % 26 + base))
            else:
                result.append(char)
        return ''.join(result)

    return caesar_decrypt(word, 3)

# First occurrence mapping
def decrypt_by_position(word: str) -> str:
    char_map = {}
    result = []
    current_char = 'a'

    for char in word:
        if char == ' ':
            result.append(' ')
        elif char not in char_map:
            char_map[char] = current_char
            result.append(current_char)
            current_char = chr(ord(current_char) + 1)
        else:
            result.append(char_map[char])

    return ''.join(result)
```

::: info Complexity: Time O(n) - Space O(n)
- **Time:** O(n) for both Caesar cipher and position mapping variants, as each character is processed once.
- **Space:** O(n) for the result list and O(k) for the character mapping dictionary where k is unique characters.
:::

### Decrypt String from Alphabet to Integer Mapping
```python
def freqAlphabets(s: str) -> str:
    """
    '1' to '9' represent 'a' to 'i'
    '10#' to '26#' represent 'j' to 'z'
    """
    result = []
    i = len(s) - 1

    while i >= 0:
        if s[i] == '#':
            # Two-digit number followed by #
            num = int(s[i-2:i])
            result.append(chr(ord('a') + num - 1))
            i -= 3
        else:
            # Single digit
            num = int(s[i])
            result.append(chr(ord('a') + num - 1))
            i -= 1

    return ''.join(reversed(result))
```

::: info Complexity: Time O(n) - Space O(n)
- **Time:** O(n) where n is the length of the input string. We process each character once, moving backwards through the string.
- **Space:** O(n) for the result list that stores the decoded characters.
:::

### Decode String with Nested Brackets
```python
def decodeString(s: str) -> str:
    """
    Decode strings like "3[a2[c]]" -> "accaccacc"
    """
    stack = []
    current_num = 0
    current_str = ""

    for char in s:
        if char.isdigit():
            current_num = current_num * 10 + int(char)
        elif char == '[':
            stack.append((current_str, current_num))
            current_str = ""
            current_num = 0
        elif char == ']':
            prev_str, num = stack.pop()
            current_str = prev_str + current_str * num
        else:
            current_str += char

    return current_str
```

::: info Complexity: Time O(n * k) - Space O(n * k)
- **Time:** O(n * k) where n is the length of the encoded string and k is the maximum nesting depth times repetition count. Each character may be replicated multiple times.
- **Space:** O(n * k) for the stack which stores intermediate strings, and for the final decoded string which can be much larger than input.
:::

---

## String Encoding/Decoding Patterns

### Pattern 1: Character Substitution
Replace each character based on a mapping (Caesar cipher, character frequency, etc.)

```python
def substitute_cipher(text: str, mapping: dict) -> str:
    return ''.join(mapping.get(c, c) for c in text)
```

::: info Complexity: Time O(n) - Space O(n)
- **Time:** O(n) to iterate through each character once and perform O(1) dictionary lookup.
- **Space:** O(n) for the output string.
:::

### Pattern 2: Run-Length Encoding/Decoding
Compress repeated characters: "aaabbc" -> "3a2b1c"

```python
def run_length_decode(encoded: str) -> str:
    result = []
    i = 0
    while i < len(encoded):
        count = 0
        while i < len(encoded) and encoded[i].isdigit():
            count = count * 10 + int(encoded[i])
            i += 1
        if i < len(encoded):
            result.append(encoded[i] * count)
            i += 1
    return ''.join(result)
```

::: info Complexity: Time O(n + m) - Space O(m)
- **Time:** O(n) to parse the encoded string plus O(m) to build the decoded output, where m is the decoded length.
- **Space:** O(m) for the result list where m is the total length of the decoded string.
:::

### Pattern 3: Nested Structure Parsing
Handle nested brackets with stack-based approach.

```python
def parse_nested(s: str) -> list:
    stack = [[]]
    for char in s:
        if char == '[':
            stack.append([])
        elif char == ']':
            inner = stack.pop()
            stack[-1].append(inner)
        else:
            stack[-1].append(char)
    return stack[0]
```

::: info Complexity: Time O(n) - Space O(n)
- **Time:** O(n) as each character is processed once with O(1) stack operations.
- **Space:** O(n) for the stack which in worst case stores all characters in nested lists.
:::

### Pattern 4: Delimiter-Based Parsing
Split and validate based on specific delimiters (like IP addresses).

```python
def parse_by_delimiter(s: str, delimiter: str, validator) -> list:
    parts = s.split(delimiter)
    return [part for part in parts if validator(part)]
```

::: info Complexity: Time O(n * v) - Space O(n)
- **Time:** O(n) for splitting plus O(v) per part for validation, where v is the validator's complexity.
- **Space:** O(n) for storing the split parts.
:::

---

## Interview Applications

### Why These Problems Matter
1. **IP Validation**: Tests careful attention to edge cases and specification adherence
2. **Decryption Problems**: Evaluate pattern recognition and string manipulation skills
3. **String Parsing**: Core skill for processing logs, configurations, and user input

### Common Interview Variations
- Validate URLs or email addresses
- Parse log files with specific formats
- Implement custom serialization/deserialization
- Handle escaped characters in strings

### Tips for Success
1. **Clarify the requirements**: Ask about edge cases (empty strings, special characters)
2. **Use helper functions**: Break down validation into smaller, testable pieces
3. **Consider all edge cases**: Leading zeros, boundary values, invalid characters
4. **Test incrementally**: Verify each validation step before combining

### Related LeetCode Problems
| Problem | Difficulty | Key Concept |
|---------|------------|-------------|
| [468. Validate IP Address](https://leetcode.com/problems/validate-ip-address/) | Medium | String parsing, validation |
| [93. Restore IP Addresses](https://leetcode.com/problems/restore-ip-addresses/) | Medium | Backtracking, string partitioning |
| [394. Decode String](https://leetcode.com/problems/decode-string/) | Medium | Stack, nested parsing |
| [1309. Decrypt String](https://leetcode.com/problems/decrypt-string-from-alphabet-to-integer-mapping/) | Easy | Character mapping |
| [2227. Encrypt and Decrypt](https://leetcode.com/problems/encrypt-and-decrypt-strings/) | Hard | Hash map, reverse mapping |

---

## Sources
- [Validate IP Address - LeetCode](https://leetcode.com/problems/validate-ip-address/)
- [468. Validate IP Address - In-Depth Explanation](https://algo.monster/liteproblems/468)
- [Decrypt String from Alphabet to Integer Mapping - LeetCode](https://leetcode.com/problems/decrypt-string-from-alphabet-to-integer-mapping/)
- [Decode String - LeetCode](https://leetcode.com/problems/decode-string/)
- [Encrypt and Decrypt Strings - LeetCode](https://leetcode.com/problems/encrypt-and-decrypt-strings/)
- [Decrypt a String According to Given Rules - GeeksforGeeks](https://www.geeksforgeeks.org/dsa/decrypt-a-string-according-to-given-rules/)
- [Program to Validate an IP Address - GeeksforGeeks](https://www.geeksforgeeks.org/dsa/program-to-validate-an-ip-address/)
