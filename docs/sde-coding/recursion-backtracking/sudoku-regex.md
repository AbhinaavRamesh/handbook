# Sudoku Solver & Regex Parser

> **Complex backtracking applications** - Two classic problems that demonstrate the power of recursive backtracking and dynamic programming optimization techniques.

---

## Table of Contents
1. [Sudoku Solver](#sudoku-solver)
2. [Basic Regex Parser](#basic-regex-parser)
3. [Interview Applications](#interview-applications)
4. [Complexity Comparison](#complexity-comparison)

---

## Sudoku Solver

### Problem Statement
Solve a 9x9 Sudoku puzzle by filling empty cells (marked as '.').

**LeetCode:** [Problem 37 - Sudoku Solver](https://leetcode.com/problems/sudoku-solver/)

**Rules:**
- Each row must contain digits 1-9 with no repetition
- Each column must contain digits 1-9 with no repetition
- Each 3x3 sub-box must contain digits 1-9 with no repetition
- The input puzzle has exactly one solution

### Core Intuition

The backtracking approach treats the Sudoku board like a maze: at each empty cell (intersection), we try each possible digit (path). If we hit an invalid state (wall), we backtrack and try the next digit. This systematic exploration guarantees finding the solution without wasting time on impossible paths.

### Visualization: Backtracking Process

```
INITIAL BOARD STATE:
+-------+-------+-------+
| 5 3 . | . 7 . | . . . |
| 6 . . | 1 9 5 | . . . |
| . 9 8 | . . . | . 6 . |
+-------+-------+-------+
| 8 . . | . 6 . | . . 3 |
| 4 . . | 8 . 3 | . . 1 |
| 7 . . | . 2 . | . . 6 |
+-------+-------+-------+
| . 6 . | . . . | 2 8 . |
| . . . | 4 1 9 | . . 5 |
| . . . | . 8 . | . 7 9 |
+-------+-------+-------+

STEP-BY-STEP BACKTRACKING:

Step 1: Find first empty cell (0,2)
        Available numbers: {1,2,4}  (eliminating 3,5,6,7,8,9 due to constraints)
        Try 4... Valid!

+-------+-------+-------+
| 5 3 4 | . 7 . | . . . |   <-- Placed 4
| 6 . . | 1 9 5 | . . . |
| . 9 8 | . . . | . 6 . |
+-------+-------+-------+
        ...

Step 2: Move to next empty cell (0,3)
        Try 2... Valid!
        Try 6... Valid! Continue...

...LATER IN THE PROCESS...

BACKTRACKING TRIGGERED:
+-------+-------+-------+
| 5 3 4 | 6 7 8 | 9 1 2 |
| 6 7 2 | 1 9 5 | 3 4 8 |
| 1 9 8 | 3 4 2 | 5 6 7 |
+-------+-------+-------+
| 8 5 9 | 7 6 1 | 4 2 3 |
| 4 2 6 | 8 5 3 | 7 X . |   <-- Can't place valid number at X!
        ...

        Dead end detected!

        ACTION: Backtrack to previous cell
                Remove last placed number
                Try next candidate

        Undo: board[4][7] = '.'
        Try next number at (4,6)...

FINAL SOLVED STATE:
+-------+-------+-------+
| 5 3 4 | 6 7 8 | 9 1 2 |
| 6 7 2 | 1 9 5 | 3 4 8 |
| 1 9 8 | 3 4 2 | 5 6 7 |
+-------+-------+-------+
| 8 5 9 | 7 6 1 | 4 2 3 |
| 4 2 6 | 8 5 3 | 7 9 1 |
| 7 1 3 | 9 2 4 | 8 5 6 |
+-------+-------+-------+
| 9 6 1 | 5 3 7 | 2 8 4 |
| 2 8 7 | 4 1 9 | 6 3 5 |
| 3 4 5 | 2 8 6 | 1 7 9 |
+-------+-------+-------+
```

### Visualization: Constraint Checking

```
Checking if '5' can be placed at position (4,4):

ROW CHECK (row 4):          COLUMN CHECK (col 4):       BOX CHECK (box 4):
+---+---+---+---+---+       +---+                       +-------+
| 4 | 2 | 6 | 8 |[?]| ...   | 7 |                       | . 6 . |
+---+---+---+---+---+       | 9 |                       | 8 . 3 |
                            |[?]|                       | . 2 . |
5 in row? NO                | 6 |                       +-------+
                            | 2 |                       5 in 3x3 box? NO
                            | . |
                            | 1 |                       ALL CLEAR!
                            | 8 |                       Place 5 at (4,4)
                            +---+
                            5 in col? NO

Box index formula: box_idx = (row // 3) * 3 + (col // 3)
For (4,4): box_idx = (4//3)*3 + (4//3) = 1*3 + 1 = 4 (center box)
```

### Solution (Python) - Basic Backtracking

```python
def solveSudoku(board: list[list[str]]) -> None:
    """
    Solve Sudoku puzzle in-place using backtracking.
    Time: O(9^m) where m is number of empty cells (worst case)
    Space: O(m) for recursion stack
    """
    def is_valid(row: int, col: int, num: str) -> bool:
        # Check row constraint
        if num in board[row]:
            return False

        # Check column constraint
        if num in [board[r][col] for r in range(9)]:
            return False

        # Check 3x3 box constraint
        box_row, box_col = 3 * (row // 3), 3 * (col // 3)
        for r in range(box_row, box_row + 3):
            for c in range(box_col, box_col + 3):
                if board[r][c] == num:
                    return False

        return True

    def solve() -> bool:
        # Find next empty cell
        for row in range(9):
            for col in range(9):
                if board[row][col] == '.':
                    # Try each digit 1-9
                    for num in '123456789':
                        if is_valid(row, col, num):
                            board[row][col] = num  # Place number

                            if solve():  # Recurse
                                return True

                            board[row][col] = '.'  # Backtrack

                    return False  # No valid number found - trigger backtrack

        return True  # All cells filled successfully

    solve()


# Example usage
board = [
    ["5","3",".",".","7",".",".",".","."],
    ["6",".",".","1","9","5",".",".","."],
    [".","9","8",".",".",".",".","6","."],
    ["8",".",".",".","6",".",".",".","3"],
    ["4",".",".","8",".","3",".",".","1"],
    ["7",".",".",".","2",".",".",".","6"],
    [".","6",".",".",".",".","2","8","."],
    [".",".",".","4","1","9",".",".","5"],
    [".",".",".",".","8",".",".","7","9"]
]
solveSudoku(board)
# board is now solved in-place
```

::: info Complexity: Time O(9^m) · Space O(m)
- **Time:** O(9^m) worst case where m is the number of empty cells - trying up to 9 digits per empty cell
- **Space:** O(m) for recursion stack depth equal to the number of empty cells
:::

### Optimized Solution with Sets (O(1) Validation)

```python
def solveSudoku_optimized(board: list[list[str]]) -> None:
    """
    Optimized Sudoku solver using sets for O(1) constraint checking.
    Time: O(9^m) worst case, but faster in practice
    Space: O(81) for sets + O(m) for recursion
    """
    # Pre-compute constraints using sets
    rows = [set() for _ in range(9)]
    cols = [set() for _ in range(9)]
    boxes = [set() for _ in range(9)]
    empty = []  # List of empty cell coordinates

    # Initialize constraint sets and find empty cells
    for r in range(9):
        for c in range(9):
            if board[r][c] != '.':
                num = board[r][c]
                rows[r].add(num)
                cols[c].add(num)
                boxes[(r // 3) * 3 + c // 3].add(num)
            else:
                empty.append((r, c))

    def backtrack(idx: int) -> bool:
        # Base case: all empty cells filled
        if idx == len(empty):
            return True

        r, c = empty[idx]
        box_idx = (r // 3) * 3 + c // 3

        for num in '123456789':
            # O(1) constraint checking with sets
            if num not in rows[r] and num not in cols[c] and num not in boxes[box_idx]:
                # Place number
                board[r][c] = num
                rows[r].add(num)
                cols[c].add(num)
                boxes[box_idx].add(num)

                if backtrack(idx + 1):
                    return True

                # Backtrack - remove number
                board[r][c] = '.'
                rows[r].remove(num)
                cols[c].remove(num)
                boxes[box_idx].remove(num)

        return False

    backtrack(0)
```

::: info Complexity: Time O(9^m) · Space O(81 + m)
- **Time:** O(9^m) worst case but faster in practice due to O(1) constraint checking using sets
- **Space:** O(81) for the three sets (rows, cols, boxes) plus O(m) for recursion stack
:::

### Advanced: Bitmask Optimization

```python
def solveSudoku_bitmask(board: list[list[str]]) -> None:
    """
    Ultra-optimized using bitmasks for constraint tracking.
    Each bit represents availability of digits 1-9.
    Time: O(9^m), Space: O(1) for masks + O(m) for recursion
    """
    rows = [0] * 9  # Bitmask for each row
    cols = [0] * 9  # Bitmask for each column
    boxes = [0] * 9  # Bitmask for each 3x3 box
    empty = []

    def get_box(r: int, c: int) -> int:
        return (r // 3) * 3 + c // 3

    # Initialize bitmasks
    for r in range(9):
        for c in range(9):
            if board[r][c] != '.':
                bit = 1 << (int(board[r][c]) - 1)
                rows[r] |= bit
                cols[c] |= bit
                boxes[get_box(r, c)] |= bit
            else:
                empty.append((r, c))

    def backtrack(idx: int) -> bool:
        if idx == len(empty):
            return True

        r, c = empty[idx]
        box = get_box(r, c)

        # Available numbers = NOT (row | col | box)
        available = ~(rows[r] | cols[c] | boxes[box]) & 0x1FF

        while available:
            # Get lowest set bit (next available number)
            bit = available & (-available)
            available &= available - 1

            num = str(bit.bit_length())
            board[r][c] = num
            rows[r] |= bit
            cols[c] |= bit
            boxes[box] |= bit

            if backtrack(idx + 1):
                return True

            board[r][c] = '.'
            rows[r] &= ~bit
            cols[c] &= ~bit
            boxes[box] &= ~bit

        return False

    backtrack(0)
```

::: info Complexity: Time O(9^m) · Space O(1 + m)
- **Time:** O(9^m) worst case with fastest constant factor due to bitwise operations for constraint checking
- **Space:** O(1) for fixed-size bitmask arrays (27 integers) plus O(m) for recursion stack
:::

### Visual Guides

![Sudoku Backtracking](/sde-coding/recursion/sudoku_backtracking.png)

---

## Basic Regex Parser

### Problem Statement
Implement regular expression matching with support for `.` and `*`:
- `.` matches any single character
- `*` matches zero or more of the preceding element

The matching should cover the **entire** input string (not partial).

**LeetCode:** [Problem 10 - Regular Expression Matching](https://leetcode.com/problems/regular-expression-matching/)

### Pattern Matching Visualization

```
EXAMPLE 1: Pattern "a*b" matches String "aab"

Decision Tree Visualization:

                        isMatch("aab", "a*b")
                       /                      \
              [skip a*]                      [use a* to match 'a']
                   |                               |
        isMatch("aab", "b")              isMatch("ab", "a*b")
               |                        /                    \
         'a' != 'b'            [skip a*]               [use a* to match 'a']
               |                   |                          |
            FALSE        isMatch("ab", "b")          isMatch("b", "a*b")
                               |                    /                 \
                         'a' != 'b'          [skip a*]          [use a* - can't match]
                               |                 |                      |
                            FALSE        isMatch("b", "b")         'a' != 'b'
                                                |                       |
                                         MATCH! TRUE                 FALSE


EXAMPLE 2: Pattern ".*" matches ANY string

".*" with "abc":
    '.' matches any char, '*' means zero or more
    So ".*" = match any character, any number of times

    isMatch("abc", ".*")
    -> Use .* to match 'a': isMatch("bc", ".*")
    -> Use .* to match 'b': isMatch("c", ".*")
    -> Use .* to match 'c': isMatch("", ".*")
    -> Skip .*: isMatch("", "") = TRUE!


EXAMPLE 3: Pattern "a.b" with String "acb"

Step-by-step matching:
    Pattern: a . b
    String:  a c b
             |  |  |
    Match:   a  c  b
             ^  ^  ^
             |  |  |
           'a'='a' '.'=any 'b'='b'
             OK!    OK!    OK!

    Result: TRUE


EXAMPLE 4: Pattern "ab*c" with String "ac" (zero occurrences)

    Pattern: a b* c
    String:  a    c

    'a' matches 'a' -> continue
    'b*' can match ZERO 'b's -> skip to 'c'
    'c' matches 'c' -> TRUE!
```

### State Machine Visualization

```
Pattern: "a*b" as a state machine:

     +-----+   'a'    +-----+   'b'    +-----+
---->|  0  |--------->|  0  |--------->|  1  |
     +-----+          +-----+          +-----+
        |                                 |
        |            'b'                  | (accepting)
        +-------------------->------------+

State 0: Haven't matched 'b' yet
         - On 'a': stay in state 0 (a* consumes it)
         - On 'b': move to state 1

State 1: Matched 'b' (accepting state)


Pattern: "a.b*c" as transitions:

    +---+  'a'  +---+  any  +---+  'b'  +---+  'c'  +---+
--->| 0 |------>| 1 |------>| 2 |<----->| 2 |------>| 3 |
    +---+       +---+       +---+       +---+       +---+
                                  \_____/     (accepting)
                                  stay on 'b'
                                  (b* loop)
```

### DP Table Visualization

```
String s = "aab", Pattern p = "c*a*b"

Build DP table where dp[i][j] = s[0:i] matches p[0:j]

           ""    c    *    a    *    b
      j:    0    1    2    3    4    5
    +-----+----+----+----+----+----+----+
"" i: 0  |  T |  F |  T |  F |  T |  F |
    +-----+----+----+----+----+----+----+
 a i: 1  |  F |  F |  F |  T |  T |  F |
    +-----+----+----+----+----+----+----+
 a i: 2  |  F |  F |  F |  F |  T |  F |
    +-----+----+----+----+----+----+----+
 b i: 3  |  F |  F |  F |  F |  F |  T |
    +-----+----+----+----+----+----+----+

Key observations:
- dp[0][0] = True (empty matches empty)
- dp[0][2] = True (c* can match empty by taking 0 c's)
- dp[0][4] = True (c*a* can match empty)
- dp[3][5] = True = ANSWER (full match!)

Transitions:
- If p[j-1] == '*':
    dp[i][j] = dp[i][j-2]  OR  (s[i-1] matches p[j-2] AND dp[i-1][j])
- Else if p[j-1] == '.' or p[j-1] == s[i-1]:
    dp[i][j] = dp[i-1][j-1]
```

### Solution (Python) - Recursive with Memoization

```python
def isMatch(s: str, p: str) -> bool:
    """
    Regular expression matching with '.' and '*'.

    Time: O(m * n) with memoization
    Space: O(m * n) for memo + O(m + n) for recursion stack

    Args:
        s: Input string to match
        p: Pattern with '.' (any char) and '*' (zero or more of preceding)

    Returns:
        True if s matches p completely
    """
    memo = {}

    def dp(i: int, j: int) -> bool:
        """Check if s[i:] matches p[j:]"""
        # Return cached result if available
        if (i, j) in memo:
            return memo[(i, j)]

        # Base case: pattern exhausted
        if j == len(p):
            result = (i == len(s))  # True only if string also exhausted
            memo[(i, j)] = result
            return result

        # Check if current characters match
        first_match = (i < len(s)) and (p[j] in {s[i], '.'})

        # Handle '*' - two choices
        if j + 1 < len(p) and p[j + 1] == '*':
            # Choice 1: Skip the 'x*' pattern (match zero occurrences)
            # Choice 2: Use 'x*' to match current char (if it matches)
            result = (
                dp(i, j + 2) or                    # Skip pattern
                (first_match and dp(i + 1, j))     # Match one char, stay on pattern
            )
        else:
            # No '*', must match current char and move forward
            result = first_match and dp(i + 1, j + 1)

        memo[(i, j)] = result
        return result

    return dp(0, 0)


# Test cases
assert isMatch("aa", "a") == False      # Single 'a' doesn't match "aa"
assert isMatch("aa", "a*") == True      # 'a*' matches any number of 'a's
assert isMatch("ab", ".*") == True      # '.*' matches any string
assert isMatch("aab", "c*a*b") == True  # c* matches empty, a* matches "aa"
assert isMatch("mississippi", "mis*is*p*.") == False
```

::: info Complexity: Time O(m * n) · Space O(m * n)
- **Time:** O(m * n) with memoization where m is string length and n is pattern length - each state computed once
- **Space:** O(m * n) for memoization cache plus O(m + n) for recursion stack
:::

### Solution (Python) - Bottom-Up Dynamic Programming

```python
def isMatch_dp(s: str, p: str) -> bool:
    """
    Regular expression matching using bottom-up DP.

    Time: O(m * n)
    Space: O(m * n) for DP table
    """
    m, n = len(s), len(p)

    # dp[i][j] = True if s[:i] matches p[:j]
    dp = [[False] * (n + 1) for _ in range(m + 1)]

    # Base case: empty string matches empty pattern
    dp[0][0] = True

    # Handle patterns like "a*", "a*b*", ".*" matching empty string
    # Pattern "x*" can match empty string by taking zero x's
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]  # Skip the 'x*' pattern

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # '*' case: either skip pattern or match if chars align
                # Option 1: Skip 'x*' (zero occurrences)
                dp[i][j] = dp[i][j - 2]

                # Option 2: Match current char with 'x*'
                if p[j - 2] in {s[i - 1], '.'}:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]

            elif p[j - 1] in {s[i - 1], '.'}:
                # Current chars match (or pattern has '.')
                dp[i][j] = dp[i - 1][j - 1]

    return dp[m][n]


# Visualization helper
def print_dp_table(s: str, p: str):
    """Print the DP table for debugging."""
    m, n = len(s), len(p)
    dp = [[False] * (n + 1) for _ in range(m + 1)]

    dp[0][0] = True
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            dp[0][j] = dp[0][j - 2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j - 1] == '*':
                dp[i][j] = dp[i][j - 2]
                if p[j - 2] in {s[i - 1], '.'}:
                    dp[i][j] = dp[i][j] or dp[i - 1][j]
            elif p[j - 1] in {s[i - 1], '.'}:
                dp[i][j] = dp[i - 1][j - 1]

    # Print header
    print("    ", end="")
    print("  \"\"", end="")
    for c in p:
        print(f"   {c}", end="")
    print()

    # Print rows
    for i in range(m + 1):
        char = f"\"{s[i-1]}\"" if i > 0 else "\"\""
        print(f"{char:4}", end="")
        for j in range(n + 1):
            val = "T" if dp[i][j] else "F"
            print(f"   {val}", end="")
        print()

    print(f"\nResult: {dp[m][n]}")


# Example usage
print_dp_table("aab", "c*a*b")
```

::: info Complexity: Time O(m * n) · Space O(m * n)
- **Time:** O(m * n) where m is string length and n is pattern length - filling the DP table
- **Space:** O(m * n) for the DP table storing all subproblem results
:::

### Space-Optimized DP (O(n) Space)

```python
def isMatch_optimized(s: str, p: str) -> bool:
    """
    Space-optimized regex matching.

    Time: O(m * n)
    Space: O(n) - only keep current and previous row
    """
    m, n = len(s), len(p)

    # Only need previous row
    prev = [False] * (n + 1)
    prev[0] = True

    # Initialize first row (empty string)
    for j in range(2, n + 1):
        if p[j - 1] == '*':
            prev[j] = prev[j - 2]

    for i in range(1, m + 1):
        curr = [False] * (n + 1)

        for j in range(1, n + 1):
            if p[j - 1] == '*':
                curr[j] = curr[j - 2]  # Skip pattern
                if p[j - 2] in {s[i - 1], '.'}:
                    curr[j] = curr[j] or prev[j]  # Match and continue
            elif p[j - 1] in {s[i - 1], '.'}:
                curr[j] = prev[j - 1]

        prev = curr

    return prev[n]
```

::: info Complexity: Time O(m * n) · Space O(n)
- **Time:** O(m * n) where m is string length and n is pattern length - same computation as full DP
- **Space:** O(n) for storing only the previous and current row of the DP table
:::

### Related Problem: Wildcard Matching (LeetCode 44)

```python
def isMatch_wildcard(s: str, p: str) -> bool:
    """
    Wildcard matching where:
    - '?' matches any single character
    - '*' matches any sequence (including empty)

    Note: Different from regex - here '*' is independent,
    not modifying the preceding character.

    Time: O(m * n)
    Space: O(n)
    """
    m, n = len(s), len(p)

    prev = [False] * (n + 1)
    prev[0] = True

    # Handle leading *'s
    for j in range(1, n + 1):
        if p[j - 1] == '*':
            prev[j] = prev[j - 1]
        else:
            break

    for i in range(1, m + 1):
        curr = [False] * (n + 1)

        for j in range(1, n + 1):
            if p[j - 1] == '*':
                # '*' matches empty OR matches s[i-1] and continues
                curr[j] = curr[j - 1] or prev[j]
            elif p[j - 1] == '?' or p[j - 1] == s[i - 1]:
                curr[j] = prev[j - 1]

        prev = curr

    return prev[n]
```

::: info Complexity: Time O(m * n) · Space O(n)
- **Time:** O(m * n) where m is string length and n is pattern length
- **Space:** O(n) for storing only the previous and current row - wildcard '*' is independent (not modifying preceding char)
:::

---

## Interview Applications

### Common Interview Patterns

These problems demonstrate critical skills sought by Google:

1. **Constraint Propagation** (Sudoku)
   - Used in: Google's constraint satisfaction systems
   - Example: Resource allocation, scheduling problems

2. **Pattern Matching** (Regex)
   - Used in: Search query processing, log analysis
   - Example: Google Search pattern matching

3. **Backtracking with Pruning**
   - Used in: Route optimization, game AI
   - Example: Google Maps route finding

### Interview Tips

```
SUDOKU SOLVER - Key Points:
+------------------------------------------+
| 1. Start with brute-force explanation    |
| 2. Discuss constraint checking (O(n) vs O(1)) |
| 3. Mention bitmask optimization          |
| 4. Discuss pruning strategies            |
| 5. Handle edge cases (invalid input)     |
+------------------------------------------+

REGEX PARSER - Key Points:
+------------------------------------------+
| 1. Clarify the pattern symbols           |
| 2. Draw the recursion tree               |
| 3. Identify overlapping subproblems      |
| 4. Discuss memo vs bottom-up trade-offs  |
| 5. Handle edge cases ("", ".*", "a**")   |
+------------------------------------------+
```

### Follow-up Questions to Expect

**Sudoku:**
- "How would you generate a valid Sudoku puzzle?"
- "How would you count all possible solutions?"
- "How would you optimize for 16x16 or 25x25 boards?"

**Regex:**
- "How would you add support for `+` (one or more)?"
- "How would you handle character classes like `[a-z]`?"
- "How would you implement backreferences?"

---

## Complexity Comparison

| Problem | Approach | Time | Space |
|---------|----------|------|-------|
| **Sudoku** | Basic Backtracking | O(9^m) | O(m) |
| **Sudoku** | Set Optimization | O(9^m) | O(81 + m) |
| **Sudoku** | Bitmask | O(9^m) | O(1 + m) |
| **Regex** | Recursive (no memo) | O(2^(m+n)) | O(m+n) |
| **Regex** | Memoization | O(m*n) | O(m*n) |
| **Regex** | Bottom-up DP | O(m*n) | O(m*n) |
| **Regex** | Space-optimized DP | O(m*n) | O(n) |

Where:
- m = number of empty cells (Sudoku) or string length (Regex)
- n = pattern length (Regex)

---

## Practice Problems

1. **Sudoku Solver** - [LeetCode 37](https://leetcode.com/problems/sudoku-solver/)
2. **Valid Sudoku** - [LeetCode 36](https://leetcode.com/problems/valid-sudoku/)
3. **Regular Expression Matching** - [LeetCode 10](https://leetcode.com/problems/regular-expression-matching/)
4. **Wildcard Matching** - [LeetCode 44](https://leetcode.com/problems/wildcard-matching/)
5. **N-Queens** - [LeetCode 51](https://leetcode.com/problems/n-queens/) (Similar backtracking pattern)

---

## References

- [Sudoku Solver - GeeksforGeeks](https://www.geeksforgeeks.org/dsa/sudoku-backtracking-7/)
- [37. Sudoku Solver - In-Depth Explanation](https://algo.monster/liteproblems/37)
- [10. Regular Expression Matching - In-Depth Explanation](https://algo.monster/liteproblems/10)
- [Regular Expression Matching - GeeksforGeeks](https://www.geeksforgeeks.org/dsa/implementing-regular-expression-matching/)
- [LeetCode 10. Regular Expression Matching - Dynamic Programming](https://www.deepdevblog.com/regular-expression-matching-dp/)
