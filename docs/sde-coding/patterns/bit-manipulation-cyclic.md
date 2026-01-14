# Bit Manipulation & Cyclic Sort Patterns

> **Low-level operations and in-place sorting for constrained problems**

These two patterns are essential for optimizing space and time complexity in coding interviews. Bit manipulation leverages binary representations for lightning-fast operations, while cyclic sort provides an elegant O(n) solution for problems involving numbers in a given range.

---

## Bit Manipulation Pattern

### Why Bit Manipulation Matters

Every piece of data in a computer is ultimately represented as bits (0s and 1s). Understanding bit manipulation provides:

- **Hardware-level speed**: Operations execute in a single CPU cycle
- **Space efficiency**: Pack multiple values into a single integer
- **Elegant solutions**: Replace complex logic with simple bit operations
- **Interview success**: Common in FAANG interviews for optimization questions

### Core Bitwise Operators

| Operator | Symbol | Description | Example |
|----------|--------|-------------|---------|
| AND | `&` | 1 if both bits are 1 | `5 & 3 = 1` (101 & 011 = 001) |
| OR | `\|` | 1 if either bit is 1 | `5 \| 3 = 7` (101 \| 011 = 111) |
| XOR | `^` | 1 if bits are different | `5 ^ 3 = 6` (101 ^ 011 = 110) |
| NOT | `~` | Inverts all bits | `~5 = -6` (inverts 101) |
| Left Shift | `<<` | Shift bits left, fill with 0 | `5 << 1 = 10` (101 -> 1010) |
| Right Shift | `>>` | Shift bits right | `5 >> 1 = 2` (101 -> 10) |

### Essential Bit Operations

| Operation | Code | Use Case |
|-----------|------|----------|
| Check if bit i is set | `n & (1 << i)` | Test if bit i is 1 |
| Set bit i | `n \| (1 << i)` | Turn on bit i |
| Clear bit i | `n & ~(1 << i)` | Turn off bit i |
| Toggle bit i | `n ^ (1 << i)` | Flip bit i |
| Check power of 2 | `n & (n-1) == 0` | True if single bit set |
| Get lowest set bit | `n & (-n)` | Isolate rightmost 1-bit |
| Clear lowest set bit | `n & (n-1)` | Turn off rightmost 1-bit |
| Check if even/odd | `n & 1` | 0 if even, 1 if odd |
| Multiply by 2^k | `n << k` | Left shift by k |
| Divide by 2^k | `n >> k` | Right shift by k |

### XOR Properties

XOR is particularly powerful for solving interview problems:

```
a ^ a = 0          // Any number XOR itself is 0
a ^ 0 = a          // Any number XOR 0 is itself
a ^ b ^ a = b      // XOR is self-inverse
a ^ b = b ^ a      // Commutative property
(a ^ b) ^ c = a ^ (b ^ c)  // Associative property
```

**Key Insight**: XOR cancels out pairs, making it perfect for finding unique elements.

### Template Code

```python
# ============================================
# SINGLE NUMBER - Find element appearing once
# ============================================
def single_number(nums: list[int]) -> int:
    """
    All elements appear twice except one.
    XOR cancels pairs: a ^ a = 0
    Time: O(n), Space: O(1)
    """
    result = 0
    for num in nums:
        result ^= num
    return result

# Example: [4, 1, 2, 1, 2] -> 4
# 4 ^ 1 ^ 2 ^ 1 ^ 2 = 4 ^ (1^1) ^ (2^2) = 4 ^ 0 ^ 0 = 4


# ============================================
# COUNT SET BITS (Brian Kernighan's Algorithm)
# ============================================
def count_bits(n: int) -> int:
    """
    Count number of 1-bits in binary representation.
    n & (n-1) clears the lowest set bit.
    Time: O(number of set bits)
    """
    count = 0
    while n:
        n &= (n - 1)  # Clear lowest set bit
        count += 1
    return count

# Alternative using built-in
def count_bits_builtin(n: int) -> int:
    return bin(n).count('1')


# ============================================
# CHECK IF POWER OF TWO
# ============================================
def is_power_of_two(n: int) -> bool:
    """
    Power of 2 has exactly one bit set.
    n & (n-1) == 0 checks for single set bit.
    """
    return n > 0 and (n & (n - 1)) == 0

# Example: 8 = 1000, 8-1 = 0111, 1000 & 0111 = 0000


# ============================================
# MISSING NUMBER (XOR approach)
# ============================================
def missing_number(nums: list[int]) -> int:
    """
    Array [0, n] with one number missing.
    XOR all numbers with all indices.
    """
    result = len(nums)
    for i, num in enumerate(nums):
        result ^= i ^ num
    return result


# ============================================
# SWAP WITHOUT TEMP VARIABLE
# ============================================
def swap_xor(a: int, b: int) -> tuple[int, int]:
    """
    Swap two numbers using XOR (interview trick).
    """
    a ^= b  # a now contains a ^ b
    b ^= a  # b = b ^ (a ^ b) = a
    a ^= b  # a = (a ^ b) ^ a = b
    return a, b


# ============================================
# REVERSE BITS
# ============================================
def reverse_bits(n: int) -> int:
    """
    Reverse 32-bit unsigned integer.
    """
    result = 0
    for _ in range(32):
        result = (result << 1) | (n & 1)
        n >>= 1
    return result


# ============================================
# SINGLE NUMBER II - Element appears once, others three times
# ============================================
def single_number_ii(nums: list[int]) -> int:
    """
    Count bits at each position mod 3.
    Time: O(32n), Space: O(1)
    """
    result = 0
    for i in range(32):
        bit_sum = 0
        for num in nums:
            bit_sum += (num >> i) & 1
        if bit_sum % 3:
            result |= (1 << i)
    # Handle negative numbers in Python
    if result >= 2**31:
        result -= 2**32
    return result


# ============================================
# SINGLE NUMBER III - Two elements appear once
# ============================================
def single_number_iii(nums: list[int]) -> list[int]:
    """
    Find two numbers that appear only once.
    1. XOR all to get a ^ b
    2. Find differing bit to split into two groups
    3. XOR each group separately
    """
    xor_all = 0
    for num in nums:
        xor_all ^= num

    # Find rightmost set bit (where a and b differ)
    diff_bit = xor_all & (-xor_all)

    a, b = 0, 0
    for num in nums:
        if num & diff_bit:
            a ^= num
        else:
            b ^= num

    return [a, b]


# ============================================
# GENERATE ALL SUBSETS USING BITMASK
# ============================================
def subsets(nums: list[int]) -> list[list[int]]:
    """
    Generate all 2^n subsets using bit manipulation.
    Each bit represents whether to include element.
    """
    n = len(nums)
    result = []

    for mask in range(1 << n):  # 0 to 2^n - 1
        subset = []
        for i in range(n):
            if mask & (1 << i):  # Check if bit i is set
                subset.append(nums[i])
        result.append(subset)

    return result
```

::: info Complexity Summary for Bit Manipulation Functions
- **single_number:** Time O(n), Space O(1) - XOR all elements once
- **count_bits:** Time O(k) where k is number of set bits, Space O(1)
- **is_power_of_two:** Time O(1), Space O(1) - Single bit operation
- **missing_number:** Time O(n), Space O(1) - XOR all indices and values
- **reverse_bits:** Time O(32) = O(1), Space O(1) - Fixed 32 iterations
- **single_number_ii:** Time O(32n) = O(n), Space O(1) - Check each of 32 bits
- **single_number_iii:** Time O(n), Space O(1) - Two passes through array
- **subsets:** Time O(n * 2^n), Space O(n * 2^n) - Generate all 2^n subsets
:::

### Classic Problems

| Problem | Difficulty | Key Technique |
|---------|------------|---------------|
| Single Number | Easy | XOR all elements |
| Single Number II | Medium | Bit counting mod 3 |
| Single Number III | Medium | XOR + split by diff bit |
| Missing Number | Easy | XOR with indices |
| Reverse Bits | Easy | Bit-by-bit reversal |
| Power of Two | Easy | `n & (n-1) == 0` |
| Counting Bits | Easy | Dynamic programming with bits |
| Hamming Distance | Easy | XOR + count set bits |
| Number of 1 Bits | Easy | Brian Kernighan's algorithm |
| Sum of Two Integers | Medium | XOR for sum, AND for carry |
| Subsets | Medium | Bitmask enumeration |

---

## Cyclic Sort Pattern

### Concept

Cyclic Sort is an elegant in-place sorting algorithm designed specifically for arrays containing numbers in a contiguous range (typically 1 to n or 0 to n-1).

**Key Insight**: If an array contains numbers from 1 to n, each number `x` should be at index `x-1`. We can place each number in its correct position by swapping.

### When to Use Cyclic Sort

Use this pattern when:
- Array contains numbers in range [1, n] or [0, n-1]
- You need to find missing or duplicate numbers
- O(1) space constraint is required
- The problem involves "should be at index" relationship

**Cannot use when**:
- Numbers are outside the expected range
- Floating point numbers are involved
- Array modification is not allowed

### Algorithm Characteristics

| Property | Value |
|----------|-------|
| Time Complexity | O(n) |
| Space Complexity | O(1) |
| In-place | Yes |
| Stable | No |

### Template Code

```python
# ============================================
# BASIC CYCLIC SORT
# ============================================
def cyclic_sort(nums: list[int]) -> list[int]:
    """
    Sort array containing numbers 1 to n.
    Place each number at index (number - 1).
    Time: O(n), Space: O(1)
    """
    i = 0
    while i < len(nums):
        correct_idx = nums[i] - 1  # Where nums[i] should be
        if nums[i] != nums[correct_idx]:
            # Swap to correct position
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1  # Already in place or duplicate
    return nums

# Example: [3, 1, 5, 4, 2]
# i=0: 3 should be at index 2, swap -> [5, 1, 3, 4, 2]
# i=0: 5 should be at index 4, swap -> [2, 1, 3, 4, 5]
# i=0: 2 should be at index 1, swap -> [1, 2, 3, 4, 5]
# i=0: 1 is at correct index, i++
# ... continue until sorted


# ============================================
# FIND MISSING NUMBER
# ============================================
def find_missing(nums: list[int]) -> int:
    """
    Array [1, n] with one number missing.
    Sort cyclically, then find misplaced element.
    """
    n = len(nums)
    i = 0

    while i < n:
        correct_idx = nums[i] - 1
        # Only swap if within range and not at correct position
        if 0 <= correct_idx < n and nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    # Find the missing number
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    return n + 1  # All in place, missing is n+1


# ============================================
# FIND ALL MISSING NUMBERS (LC 448)
# ============================================
def find_all_missing(nums: list[int]) -> list[int]:
    """
    Array of n integers where 1 <= nums[i] <= n.
    Some elements appear twice, some don't appear.
    Find all that don't appear.
    """
    n = len(nums)
    i = 0

    while i < n:
        correct_idx = nums[i] - 1
        if nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    missing = []
    for i in range(n):
        if nums[i] != i + 1:
            missing.append(i + 1)

    return missing


# ============================================
# FIND DUPLICATE NUMBER (LC 287)
# ============================================
def find_duplicate(nums: list[int]) -> int:
    """
    Array of n+1 integers where each is in [1, n].
    One number appears more than once.
    """
    i = 0
    while i < len(nums):
        if nums[i] != i + 1:
            correct_idx = nums[i] - 1
            if nums[i] == nums[correct_idx]:
                return nums[i]  # Found duplicate!
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1
    return -1


# ============================================
# FIND ALL DUPLICATES (LC 442)
# ============================================
def find_all_duplicates(nums: list[int]) -> list[int]:
    """
    Array of n integers where 1 <= nums[i] <= n.
    Some elements appear twice, others once.
    Find all that appear twice.
    """
    n = len(nums)
    i = 0

    while i < n:
        correct_idx = nums[i] - 1
        if nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    duplicates = []
    for i in range(n):
        if nums[i] != i + 1:
            duplicates.append(nums[i])

    return duplicates


# ============================================
# FIRST MISSING POSITIVE (LC 41) - Hard
# ============================================
def first_missing_positive(nums: list[int]) -> int:
    """
    Find smallest missing positive integer.
    Key: Answer must be in [1, n+1].
    Ignore non-positive and numbers > n.
    """
    n = len(nums)
    i = 0

    while i < n:
        correct_idx = nums[i] - 1
        # Only process if in valid range [1, n]
        if 0 < nums[i] <= n and nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    # First position where nums[i] != i+1 is answer
    for i in range(n):
        if nums[i] != i + 1:
            return i + 1

    return n + 1


# ============================================
# SET MISMATCH (LC 645)
# ============================================
def find_error_nums(nums: list[int]) -> list[int]:
    """
    Array [1, n] with one duplicate and one missing.
    Return [duplicate, missing].
    """
    n = len(nums)
    i = 0

    while i < n:
        correct_idx = nums[i] - 1
        if nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1

    # Find the number at wrong position
    for i in range(n):
        if nums[i] != i + 1:
            return [nums[i], i + 1]

    return []


# ============================================
# CYCLIC SORT FOR 0-INDEXED RANGE [0, n-1]
# ============================================
def cyclic_sort_zero_indexed(nums: list[int]) -> list[int]:
    """
    Sort array containing numbers 0 to n-1.
    Place each number at its own index.
    """
    i = 0
    while i < len(nums):
        correct_idx = nums[i]  # For 0-indexed, number IS the index
        if nums[i] < len(nums) and nums[i] != nums[correct_idx]:
            nums[i], nums[correct_idx] = nums[correct_idx], nums[i]
        else:
            i += 1
    return nums
```

::: info Complexity Summary for Cyclic Sort Functions
- **cyclic_sort:** Time O(n), Space O(1) - Each element swapped at most once to its correct position
- **find_missing:** Time O(n), Space O(1) - Cyclic sort phase + linear scan
- **find_all_missing:** Time O(n), Space O(1) extra - Only output array uses additional space
- **find_duplicate:** Time O(n), Space O(1) - Early termination when duplicate found
- **find_all_duplicates:** Time O(n), Space O(1) extra - Only output array uses additional space
- **first_missing_positive:** Time O(n), Space O(1) - Handles negative and out-of-range values
- **find_error_nums:** Time O(n), Space O(1) - Returns both duplicate and missing in single pass
:::

### Classic Problems

| Problem | Difficulty | Key Variation |
|---------|------------|---------------|
| Missing Number | Easy | Find one missing in [0,n] |
| Find All Missing Numbers | Easy | Find all missing |
| Find the Duplicate | Medium | Find one duplicate |
| Find All Duplicates | Medium | Find all duplicates |
| First Missing Positive | Hard | Handle negatives, > n |
| Set Mismatch | Easy | Find both duplicate and missing |

### Visual Example: Cyclic Sort Process

```
Input: [3, 1, 5, 4, 2]
Goal: Place each number at index (number - 1)

Step 1: i=0, nums[0]=3, should be at index 2
        Swap positions 0 and 2
        [5, 1, 3, 4, 2]

Step 2: i=0, nums[0]=5, should be at index 4
        Swap positions 0 and 4
        [2, 1, 3, 4, 5]

Step 3: i=0, nums[0]=2, should be at index 1
        Swap positions 0 and 1
        [1, 2, 3, 4, 5]

Step 4: i=0, nums[0]=1, correct position!
        Move to i=1

Step 5: i=1, nums[1]=2, correct position!
        Move to i=2
        ... (all remaining are correct)

Output: [1, 2, 3, 4, 5]
```

---

## Google Interview Applications

### Bit Manipulation at Google

Google frequently asks bit manipulation questions to assess:
- Understanding of computer fundamentals
- Ability to optimize space-constrained solutions
- Knowledge of binary representation tricks

**Common Google Interview Questions**:
1. **UTF-8 Validation**: Validate if array represents valid UTF-8 encoding
2. **Maximum XOR of Two Numbers**: Use trie + bit manipulation
3. **Counting Bits**: Dynamic programming with bit properties
4. **Bitwise AND of Numbers Range**: Find common prefix bits

### Cyclic Sort at Google

The cyclic sort pattern appears in Google interviews for:
- Array manipulation with space constraints
- Finding missing/duplicate elements in O(1) space
- Problems requiring in-place transformations

**Common Google Interview Questions**:
1. **First Missing Positive**: Classic Google question - find smallest missing positive in O(n) time, O(1) space
2. **Find All Duplicates**: Without extra space
3. **Set Mismatch**: Identify both duplicate and missing number

### Interview Tips

**For Bit Manipulation**:
1. Always visualize binary representations on paper
2. Remember common identities: `n & (n-1)` clears lowest bit, `n & -n` isolates lowest bit
3. XOR is your friend for finding unique elements
4. Think about how bits propagate through operations

**For Cyclic Sort**:
1. Recognize the pattern: "numbers 1 to n in array of size n"
2. The swap condition `nums[i] != nums[correct_idx]` handles duplicates
3. After sorting, scan for mismatches to find answers
4. Handle edge cases: numbers outside range, zero-indexed vs one-indexed

### Complexity Comparison

| Approach | Time | Space | When to Use |
|----------|------|-------|-------------|
| Hash Set | O(n) | O(n) | When modification not allowed |
| Sorting | O(n log n) | O(1) | When order doesn't matter |
| Cyclic Sort | O(n) | O(1) | When range is [1,n] or [0,n-1] |
| Bit Manipulation | O(n) | O(1) | When elements have XOR property |

---

## Practice Problems by Difficulty

### Easy

::: details Single Number (LeetCode 136)
**Problem:** Every element appears twice except one. Find the single element.

**Key Insight:** XOR of a number with itself is 0. XOR of all elements cancels pairs, leaving the single number.

**Approach:** Initialize result = 0, XOR all elements: result ^= num for each num in array.

**Complexity:** Time O(n), Space O(1)
:::

::: details Missing Number (LeetCode 268)
**Problem:** Array contains n distinct numbers from range [0, n]. Find the missing one.

**Key Insight:** XOR all indices 0 to n with all array elements. Pairs cancel, leaving missing number.

**Approach:** XOR all indices (0 to n) with all array values. Alternative: use Gauss formula n*(n+1)/2 - sum(array).

**Complexity:** Time O(n), Space O(1)
:::

::: details Number of 1 Bits (LeetCode 191)
**Problem:** Return the number of set bits (1s) in the binary representation of an integer.

**Key Insight:** n & (n-1) clears the lowest set bit. Count iterations until n becomes 0.

**Approach:** While n != 0: increment count and do n = n & (n-1). This is Brian Kernighan's algorithm.

**Complexity:** Time O(number of set bits), Space O(1)
:::

::: details Find All Numbers Disappeared in an Array (LeetCode 448)
**Problem:** Array of n integers in range [1, n], some appear twice, some don't. Find all missing numbers.

**Key Insight:** Use array indices as markers. Negate value at index corresponding to each number seen.

**Approach:** For each num, mark nums[abs(num)-1] as negative. After marking, indices with positive values are missing.

**Complexity:** Time O(n), Space O(1) extra
:::

::: details Power of Two (LeetCode 231)
**Problem:** Determine if n is a power of two.

**Key Insight:** Powers of two have exactly one set bit. n & (n-1) clears that bit, resulting in 0.

**Approach:** Return n > 0 and (n & (n-1)) == 0. Must check n > 0 since 0 would pass the bit check.

**Complexity:** Time O(1), Space O(1)
:::

### Medium

::: details Single Number II (LeetCode 137)
**Problem:** Every element appears three times except one which appears once. Find it.

**Key Insight:** Count bits at each position mod 3. Bits from the single number remain.

**Approach:** For each of 32 bit positions, count total set bits across all numbers. If count % 3 != 0, that bit is set in result.

**Complexity:** Time O(32n) = O(n), Space O(1)
:::

::: details Single Number III (LeetCode 260)
**Problem:** Exactly two elements appear once, all others appear twice. Find both single elements.

**Key Insight:** XOR all gives a ^ b. Find any set bit to split numbers into two groups.

**Approach:** Get xor_all = a ^ b. Find differing bit with xor_all & (-xor_all). Split array by this bit, XOR each group separately.

**Complexity:** Time O(n), Space O(1)
:::

::: details Find All Duplicates in an Array (LeetCode 442)
**Problem:** Array of n integers in range [1, n], each appears once or twice. Find all duplicates.

**Key Insight:** Use sign flipping - negate value at index when number is seen. If already negative, it's a duplicate.

**Approach:** For each num, check if nums[abs(num)-1] is negative. If yes, add to result. Otherwise negate it.

**Complexity:** Time O(n), Space O(1) extra
:::

::: details Find the Duplicate Number (LeetCode 287)
**Problem:** Array of n+1 integers in [1, n] with one duplicate. Find it without modifying array in O(1) space.

**Key Insight:** Treat as linked list where index i points to nums[i]. Duplicate creates cycle, use Floyd's algorithm.

**Approach:** Use slow/fast pointers to find cycle, then find cycle entrance which is the duplicate.

**Complexity:** Time O(n), Space O(1)
:::

::: details Subsets (LeetCode 78)
**Problem:** Given array of unique elements, return all possible subsets (power set).

**Key Insight:** Each subset corresponds to a binary number from 0 to 2^n - 1. Bit i indicates whether to include element i.

**Approach:** For mask from 0 to 2^n - 1: check each bit position, include element if bit is set.

**Complexity:** Time O(n * 2^n), Space O(n * 2^n) for output
:::

::: details Sum of Two Integers (LeetCode 371)
**Problem:** Calculate sum of two integers without using + or - operators.

**Key Insight:** XOR gives sum without carry. AND shifted left gives carry. Repeat until no carry.

**Approach:** While b != 0: carry = a & b, a = a ^ b, b = carry << 1. Handle negative numbers carefully.

**Complexity:** Time O(1), Space O(1)
:::

::: details Maximum XOR of Two Numbers in Array (LeetCode 421)
**Problem:** Find maximum XOR of two elements in an array.

**Key Insight:** Build answer bit by bit from MSB. Use Trie or set to check if higher bit can be set.

**Approach:** For each bit position from high to low, try to set it in result. Use prefix set to verify achievability.

**Complexity:** Time O(32n), Space O(n)
:::

### Hard

::: details First Missing Positive (LeetCode 41)
**Problem:** Find the smallest missing positive integer. Must run in O(n) time and O(1) space.

**Key Insight:** Answer must be in [1, n+1]. Use cyclic sort to place each number at its correct index.

**Approach:** Place num at index num-1 using swaps. Then scan for first index where nums[i] != i+1. That's the answer.

**Complexity:** Time O(n), Space O(1)
:::

::: details Maximum AND Sum of Array (LeetCode 2172)
**Problem:** Distribute array elements into numSlots slots (each slot holds at most 2 elements). Maximize sum of (slot_number AND element).

**Key Insight:** Use bitmask DP where state encodes slot assignments.

**Approach:** DP with bitmask representing which slot positions are filled. Transition by assigning current element to each valid slot.

**Complexity:** Time O(n * 3^numSlots), Space O(3^numSlots)
:::

---

## Quick Reference Card

### Bit Manipulation Cheat Sheet
```python
# Check bit i
(n >> i) & 1        # or: n & (1 << i)

# Set bit i
n | (1 << i)

# Clear bit i
n & ~(1 << i)

# Toggle bit i
n ^ (1 << i)

# Clear lowest set bit
n & (n - 1)

# Isolate lowest set bit
n & (-n)

# Check power of 2
n > 0 and n & (n - 1) == 0

# Count set bits
bin(n).count('1')   # Python built-in
```

### Cyclic Sort Template
```python
i = 0
while i < len(nums):
    correct = nums[i] - 1  # Adjust for 0-indexed: correct = nums[i]
    if nums[i] != nums[correct]:
        nums[i], nums[correct] = nums[correct], nums[i]
    else:
        i += 1
```

---

## References

- [GeeksforGeeks - Bit Manipulation Problems](https://www.geeksforgeeks.org/dsa/top-problems-on-bit-manipulation-for-interviews/)
- [AlgoCademy - Bit Manipulation Guide](https://algocademy.com/blog/approaching-bit-manipulation-problems-a-comprehensive-guide-for-coding-interviews/)
- [DevInterview - 40 Bit Manipulation Questions](https://devinterview.io/blog/bit-manipulation-interview-questions/)
- [LeetCode - Cyclic Sort Pattern Discussion](https://leetcode.com/discuss/study-guide/2958275/cyclic-sort-important-pattern/)
- [Coding Patterns: Cyclic Sort](https://emre.me/coding-patterns/cyclic-sort/)
- [LeetCode The Hard Way - Cyclic Sort](https://leetcodethehardway.com/tutorials/basic-topics/sorting/cyclic-sort)
- [Educative - Grokking Bit Manipulation](https://www.educative.io/courses/bit-manipulation)
