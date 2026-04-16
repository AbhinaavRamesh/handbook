# Tier 6: String / Pattern Matching

> **3 problems** | Priority: **Lower** | Time estimate: **1-1.5 hours**

---

## Why Strings?

At their core, tech companies are text-processing companies. Search indexing, autocomplete, spell-check, regex engines, natural language processing -- strings are baked into the DNA of nearly every product. This is not an accident: interviewers reach for string problems because they reflect real engineering work.

In SDE interviews, string problems serve a specific purpose:

- **Trie problems** test whether you can design a data structure from scratch. They often appear as a "warm-up" design question before a harder algorithmic follow-up.
- **Two-pointer string comparison** tests careful implementation. These problems are deceptively simple to describe but require precise handling of edge cases.
- **Backtracking on grids** tests whether you can write clean recursive code without an IDE to catch your bugs.

::: info REAL INTERVIEW DATA
Expressive Words (LC 809) is a **confirmed SDE interview question**. Multiple candidates have reported receiving this exact problem or close variants. If you only have time for one problem in this tier, make it this one.
:::

---

## Problem Table

| # | Problem | LC # | Pattern | Why |
|---|---------|------|---------|-----|
| 30 | [Implement Trie](https://leetcode.com/problems/implement-trie-prefix-tree) | 208 | Trie | IP pattern matching frequently asked |
| 31 | [Expressive Words](https://leetcode.com/problems/expressive-words) | 809 | Two pointer | Frequently asked in SDE interviews |
| 32 | [Word Search](https://leetcode.com/problems/word-search) | 79 | Backtracking + DFS | Classic |

---

## Key Patterns

### Trie: When and How to Use

A Trie (prefix tree) is the go-to data structure when you need:

- **Prefix matching** -- "find all words that start with `pre`"
- **Autocomplete** -- "given a prefix typed so far, suggest completions"
- **Word dictionary with efficient lookup** -- faster than hashing for prefix queries
- **IP routing / longest prefix match** -- a real systems problem

**When to reach for a Trie instead of a HashMap:**

| Use Case | HashMap | Trie |
|----------|---------|------|
| Exact word lookup | O(k) per lookup, simpler | O(k) per lookup, more code |
| Prefix search ("starts with") | Requires iterating all keys | O(k) -- built-in |
| Autocomplete / suggestions | Awkward | Natural -- DFS from prefix node |
| Shared prefix storage | Stores full keys | Shares common prefixes, saves memory |

Rule of thumb: if the problem mentions "prefix", "starts with", or "autocomplete", use a Trie.

### Two-Pointer String Comparison

The pattern: walk through two strings simultaneously, character by character, with rules governing when you advance each pointer.

:::code-group

```python [Python]
def compare_strings(s, t):
    """General two-pointer string comparison template."""
    i, j = 0, 0
    while i < len(s) and j < len(t):
        if s[i] == t[j]:
            # Characters match -- advance both or apply rules
            i += 1
            j += 1
        else:
            # Mismatch -- apply problem-specific logic
            # Maybe advance only one pointer, or return False
            pass
    # Check remaining characters in either string
    return i == len(s) and j == len(t)
```

```java [Java]
boolean compareStrings(String s, String t) {
    int i = 0, j = 0;
    while (i < s.length() && j < t.length()) {
        if (s.charAt(i) == t.charAt(j)) {
            // Characters match -- advance both or apply rules
            i++;
            j++;
        } else {
            // Mismatch -- apply problem-specific logic
            // Maybe advance only one pointer, or return false
        }
    }
    // Check remaining characters in either string
    return i == s.length() && j == t.length();
}
```

:::

This pattern shows up in: Expressive Words, Backspace String Compare (LC 844), Is Subsequence (LC 392), and many custom string problems.

### Backtracking on Grids

DFS + a visited set (or in-place marking) to explore all paths through a grid. The key is **undoing your choice** after returning from the recursive call.

:::code-group

```python [Python]
def backtrack_grid(board, word):
    """General grid backtracking template."""
    rows, cols = len(board), len(board[0])

    def dfs(r, c, idx):
        if idx == len(word):
            return True  # found the complete word
        if (r < 0 or r >= rows or c < 0 or c >= cols
                or board[r][c] != word[idx]):
            return False

        # Mark as visited (in-place to save space)
        temp = board[r][c]
        board[r][c] = '#'

        # Explore all 4 directions
        found = (dfs(r + 1, c, idx + 1) or
                 dfs(r - 1, c, idx + 1) or
                 dfs(r, c + 1, idx + 1) or
                 dfs(r, c - 1, idx + 1))

        # Undo the choice (backtrack)
        board[r][c] = temp
        return found

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    return False
```

```java [Java]
boolean backtrackGrid(char[][] board, String word) {
    int rows = board.length, cols = board[0].length;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (dfsGrid(board, word, r, c, 0)) return true;
        }
    }
    return false;
}

private boolean dfsGrid(char[][] board, String word, int r, int c, int idx) {
    if (idx == word.length()) return true;  // found the complete word
    if (r < 0 || r >= board.length || c < 0 || c >= board[0].length
            || board[r][c] != word.charAt(idx)) return false;

    // Mark as visited (in-place to save space)
    char temp = board[r][c];
    board[r][c] = '#';

    // Explore all 4 directions
    boolean found = dfsGrid(board, word, r + 1, c, idx + 1) ||
                    dfsGrid(board, word, r - 1, c, idx + 1) ||
                    dfsGrid(board, word, r, c + 1, idx + 1) ||
                    dfsGrid(board, word, r, c - 1, idx + 1);

    // Undo the choice (backtrack)
    board[r][c] = temp;
    return found;
}
```

:::

::: warning BACKTRACKING vs. REGULAR DFS
In regular DFS (e.g., Number of Islands), once you visit a cell, it stays visited forever. In backtracking (e.g., Word Search), you **unmark** visited cells after returning, because different paths may need to reuse the same cell in different orderings.
:::

![Backtracking Decision Tree — Subset Generation](/sde-coding/sprint/backtracking_tree.png)

---

## Trie Implementation Template

This is the full Trie implementation you should be able to write from memory in a plain text editor. Practice until you can write it in under 5 minutes.

:::code-group

```python [Python]
class TrieNode:
    def __init__(self):
        self.children = {}       # char -> TrieNode
        self.is_end = False      # marks end of a complete word


class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """Insert a word into the trie. O(k) time."""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word: str) -> bool:
        """Return True if the word is in the trie. O(k) time."""
        node = self._find(word)
        return node is not None and node.is_end

    def startsWith(self, prefix: str) -> bool:
        """Return True if any word starts with the given prefix. O(k) time."""
        return self._find(prefix) is not None

    def _find(self, prefix: str):
        """Traverse the trie following the given prefix.
        Returns the node at the end of the prefix, or None."""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node
```

```java [Java]
class TrieNode {
    Map<Character, TrieNode> children = new HashMap<>();
    boolean isEnd = false;
}

class Trie {
    private final TrieNode root = new TrieNode();

    void insert(String word) {
        TrieNode node = root;
        for (char ch : word.toCharArray()) {
            node.children.putIfAbsent(ch, new TrieNode());
            node = node.children.get(ch);
        }
        node.isEnd = true;
    }

    boolean search(String word) {
        TrieNode node = find(word);
        return node != null && node.isEnd;
    }

    boolean startsWith(String prefix) {
        return find(prefix) != null;
    }

    private TrieNode find(String prefix) {
        TrieNode node = root;
        for (char ch : prefix.toCharArray()) {
            if (!node.children.containsKey(ch)) return null;
            node = node.children.get(ch);
        }
        return node;
    }
}
```

:::

::: tip DESIGN CHOICES TO DISCUSS
In an interview, mention these trade-offs:
- **`children` as dict vs. array of size 26:** Dict is more flexible (handles Unicode, sparse alphabets). Array is faster for lowercase-English-only problems. Start with dict unless told otherwise.
- **`is_end` flag:** Some variants store a count (for frequency) or store the full word (for easy retrieval during DFS).
- **Memory:** Tries can use more memory than a hash set for small dictionaries, but less memory for large dictionaries with shared prefixes.
:::

![Trie — Prefix Tree Structure](/sde-coding/sprint/trie_structure.png)

---

## Problem Walkthroughs

::: details Problem 30: Implement Trie (LC 208) -- Trie

**Key Insight:** Each node stores a map from character to child node. Insertion walks down the tree, creating nodes as needed. Search walks down and checks `is_end`. `startsWith` walks down and just checks existence.

:::code-group

```python [Python]
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

    def search(self, word: str) -> bool:
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def startsWith(self, prefix: str) -> bool:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True
```

```java [Java]
class TrieNode {
    Map<Character, TrieNode> children = new HashMap<>();
    boolean isEnd = false;
}

class Trie {
    private final TrieNode root = new TrieNode();

    public void insert(String word) {
        TrieNode node = root;
        for (char ch : word.toCharArray()) {
            node.children.putIfAbsent(ch, new TrieNode());
            node = node.children.get(ch);
        }
        node.isEnd = true;
    }

    public boolean search(String word) {
        TrieNode node = root;
        for (char ch : word.toCharArray()) {
            if (!node.children.containsKey(ch)) return false;
            node = node.children.get(ch);
        }
        return node.isEnd;
    }

    public boolean startsWith(String prefix) {
        TrieNode node = root;
        for (char ch : prefix.toCharArray()) {
            if (!node.children.containsKey(ch)) return false;
            node = node.children.get(ch);
        }
        return true;
    }
}
```

:::

**Complexity:**
- Time: O(k) for insert, search, and startsWith, where k = length of the word/prefix.
- Space: O(T) total, where T = total number of characters across all inserted words (worst case, no shared prefixes).

**Common Follow-ups:**
- "Add a `delete` method" -- decrement a count or use reference counting on nodes.
- "Add a `countWordsWithPrefix` method" -- store a `prefix_count` at each node, increment during insert.
- "Make it support wildcard `.` matching" -- use DFS with branching at wildcard characters (this becomes LC 211: Design Add and Search Words Data Structure).
:::

::: details Problem 31: Expressive Words (LC 809) -- Two Pointer

**Key Insight:** Compare `s` and each word character group by character group. A "group" is a run of the same character. For each group, the characters must match, and the count in `s` must be either equal to the count in `word`, or the count in `s` must be >= 3 (stretched).

:::code-group

```python [Python]
def expressiveWords(s: str, words: list) -> int:
    def get_groups(word):
        """Convert a word into a list of (char, count) groups."""
        groups = []
        i = 0
        while i < len(word):
            char = word[i]
            count = 0
            while i < len(word) and word[i] == char:
                count += 1
                i += 1
            groups.append((char, count))
        return groups

    def is_stretchy(word):
        """Check if word can be stretched to match s."""
        sg = get_groups(s)
        wg = get_groups(word)

        if len(sg) != len(wg):
            return False

        for (sc, sn), (wc, wn) in zip(sg, wg):
            if sc != wc:
                return False
            if sn < wn:
                return False          # s has fewer -- can't shrink
            if sn != wn and sn < 3:
                return False          # stretched but group too small
        return True

    return sum(1 for word in words if is_stretchy(word))
```

```java [Java]
int expressiveWords(String s, String[] words) {
    int count = 0;
    for (String word : words) {
        if (isStretchy(s, word)) count++;
    }
    return count;
}

private boolean isStretchy(String s, String word) {
    List<int[]> sg = getGroups(s);  // [char, count] pairs
    List<int[]> wg = getGroups(word);

    if (sg.size() != wg.size()) return false;

    for (int i = 0; i < sg.size(); i++) {
        int sc = sg.get(i)[0], sn = sg.get(i)[1];
        int wc = wg.get(i)[0], wn = wg.get(i)[1];
        if (sc != wc) return false;
        if (sn < wn) return false;           // s has fewer -- can't shrink
        if (sn != wn && sn < 3) return false; // stretched but group too small
    }
    return true;
}

private List<int[]> getGroups(String word) {
    List<int[]> groups = new ArrayList<>();
    int i = 0;
    while (i < word.length()) {
        char ch = word.charAt(i);
        int count = 0;
        while (i < word.length() && word.charAt(i) == ch) { count++; i++; }
        groups.add(new int[]{ch, count});
    }
    return groups;
}
```

:::

**Complexity:**
- Time: O(W * max(|s|, |word|)) where W = number of words.
- Space: O(|s| + |word|) for the groups.

**Key Rules:**
1. Character groups must appear in the same order with the same characters.
2. For each group, the count in `s` must be >= count in `word`.
3. If the count in `s` is greater than the count in `word`, the count in `s` must be >= 3.

**Why rule 3?** A group of 2 identical characters looks intentional (like "ll" in "hello"). You can only call it "stretched" if the group has 3+ characters (like "helllo" with 3 l's -- that looks like someone held the key too long).

**Alternative -- inline two-pointer (no group extraction):**

:::code-group

```python [Python]
def is_stretchy_inline(s, word):
    """Direct two-pointer approach without building group lists."""
    i, j = 0, 0
    while i < len(s) and j < len(word):
        if s[i] != word[j]:
            return False
        # Count the length of the current group in both strings
        si = i
        while i < len(s) and s[i] == s[si]:
            i += 1
        sj = j
        while j < len(word) and word[j] == word[sj]:
            j += 1

        s_count = i - si
        w_count = j - sj

        if s_count < w_count:
            return False
        if s_count != w_count and s_count < 3:
            return False

    return i == len(s) and j == len(word)
```

```java [Java]
boolean isStretchyInline(String s, String word) {
    int i = 0, j = 0;
    while (i < s.length() && j < word.length()) {
        if (s.charAt(i) != word.charAt(j)) return false;
        // Count the length of the current group in both strings
        int si = i, sj = j;
        while (i < s.length() && s.charAt(i) == s.charAt(si)) i++;
        while (j < word.length() && word.charAt(j) == word.charAt(sj)) j++;

        int sCount = i - si;
        int wCount = j - sj;

        if (sCount < wCount) return false;
        if (sCount != wCount && sCount < 3) return false;
    }
    return i == s.length() && j == word.length();
}
```

:::
:::

::: details Problem 32: Word Search (LC 79) -- Backtracking + DFS

**Key Insight:** For each cell in the grid that matches the first character of the word, start a DFS. At each step, mark the current cell as visited (to avoid reusing it), explore all 4 neighbors, and **unmark** the cell when backtracking.

:::code-group

```python [Python]
def exist(board, word):
    rows, cols = len(board), len(board[0])

    def dfs(r, c, idx):
        # Base case: found all characters
        if idx == len(word):
            return True

        # Bounds check + character match + visited check
        if (r < 0 or r >= rows or c < 0 or c >= cols
                or board[r][c] != word[idx]):
            return False

        # Mark as visited by temporarily replacing the character
        temp = board[r][c]
        board[r][c] = '#'

        # Explore all 4 directions
        found = (dfs(r + 1, c, idx + 1) or
                 dfs(r - 1, c, idx + 1) or
                 dfs(r, c + 1, idx + 1) or
                 dfs(r, c - 1, idx + 1))

        # Backtrack: restore the original character
        board[r][c] = temp
        return found

    for r in range(rows):
        for c in range(cols):
            if dfs(r, c, 0):
                return True
    return False
```

```java [Java]
boolean exist(char[][] board, String word) {
    int rows = board.length, cols = board[0].length;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            if (existDFS(board, word, r, c, 0)) return true;
        }
    }
    return false;
}

private boolean existDFS(char[][] board, String word, int r, int c, int idx) {
    if (idx == word.length()) return true;
    if (r < 0 || r >= board.length || c < 0 || c >= board[0].length
            || board[r][c] != word.charAt(idx)) return false;

    // Mark as visited by temporarily replacing the character
    char temp = board[r][c];
    board[r][c] = '#';

    // Explore all 4 directions
    boolean found = existDFS(board, word, r + 1, c, idx + 1) ||
                    existDFS(board, word, r - 1, c, idx + 1) ||
                    existDFS(board, word, r, c + 1, idx + 1) ||
                    existDFS(board, word, r, c - 1, idx + 1);

    // Backtrack: restore the original character
    board[r][c] = temp;
    return found;
}
```

:::

**Complexity:**
- Time: O(m * n * 4^L) where m x n is the grid size and L is the word length. Each cell can branch into 4 directions up to L levels deep.
- Space: O(L) for the recursion stack (we modify the board in-place instead of using a separate visited set).

**Optimization -- early termination with character counting:**

:::code-group

```python [Python]
from collections import Counter

def exist_optimized(board, word):
    # Prune: check if the board even contains enough of each character
    board_count = Counter(c for row in board for c in row)
    word_count = Counter(word)
    for char, count in word_count.items():
        if board_count[char] < count:
            return False

    # Optimization: if the last char is rarer than the first,
    # search the word reversed (finds dead ends faster)
    if board_count[word[0]] > board_count[word[-1]]:
        word = word[::-1]

    # ... then run the standard DFS from above
```

```java [Java]
boolean existOptimized(char[][] board, String word) {
    // Prune: check if the board even contains enough of each character
    int[] boardCount = new int[26];
    for (char[] row : board) for (char c : row) boardCount[c - 'a']++;
    int[] wordCount = new int[26];
    for (char c : word.toCharArray()) wordCount[c - 'a']++;
    for (int i = 0; i < 26; i++) {
        if (wordCount[i] > boardCount[i]) return false;
    }

    // Optimization: if the last char is rarer than the first,
    // search the word reversed (finds dead ends faster)
    if (boardCount[word.charAt(0) - 'a'] > boardCount[word.charAt(word.length() - 1) - 'a']) {
        word = new StringBuilder(word).reverse().toString();
    }

    // ... then run the standard DFS from above
    return exist(board, word);
}
```

:::

**Why in-place marking instead of a visited set?**
- A `visited` set of `(r, c)` tuples costs O(L) space and has dict overhead.
- Replacing `board[r][c]` with `'#'` uses O(1) extra space and is faster.
- Just remember to restore it during backtracking. This is the standard interview approach.
:::

---

## Interview Tips

::: tip THINGS CANDIDATES WISH THEY KNEW

**1. Expressive Words is a REAL interview question -- know it cold.**

Multiple candidates (2024-2025) have reported receiving this problem or close variants during their interview. The two-pointer group comparison is the core technique. Practice both the "extract groups then compare" and the "inline two-pointer" approaches until you can write either from memory. The inline version is shorter and impresses interviewers.

**2. Trie problems often come as "design" warm-ups before a harder follow-up.**

A common SDE interview flow:
1. "Implement a Trie with insert, search, and startsWith." (10-15 minutes)
2. "Now, using your Trie, solve [harder problem]." (remaining time)

The harder problem might be: word search II (LC 212), search suggestions system (LC 1268), or a custom IP routing / prefix matching problem. If you spend 20 minutes struggling with the basic Trie, you will not have time for the follow-up. **Practice until the basic Trie takes under 5 minutes.**

**3. Word Search tests clean backtracking code -- practice writing it without IDE help.**

In a plain text editor, you cannot:
- Run your code to check for bugs
- Use autocomplete for method names
- Rely on syntax highlighting to spot typos

The backtracking template for Word Search has several places where bugs hide:
- Forgetting to restore `board[r][c]` after the recursive call
- Getting the bounds check order wrong (check bounds BEFORE accessing `board[r][c]`)
- Using `and` instead of `or` when combining the four directional results
- Off-by-one on the `idx == len(word)` base case

Write this solution 3 times on paper. By the third time, it should be muscle memory.

**4. String problems often have "hidden" O(n) solutions.**

Many candidates jump to O(n^2) approaches for string problems. Before coding, ask yourself:
- Can I solve this with a single pass using two pointers? (Often yes.)
- Can I precompute something with a hash map to avoid nested loops? (Often yes.)
- Is there a sliding window formulation? (Sometimes.)

Interviewers notice when you jump to the optimal approach immediately -- it signals pattern recognition.
:::

---

## Practice Checklist

Use this to track your progress. Each problem should be solvable in under 15 minutes without looking at the solution.

- [ ] **Implement Trie** (LC 208) -- Can you write TrieNode + Trie with insert/search/startsWith in under 5 minutes on paper?
- [ ] **Expressive Words** (LC 809) -- Can you explain the three rules for "stretchy" groups? Can you write the inline two-pointer version?
- [ ] **Word Search** (LC 79) -- Can you write the backtracking DFS without forgetting to restore the cell?

::: tip FINAL STRETCH
This is the last tier. If you have made it through all 32 problems, you are more prepared than the vast majority of candidates. Spend your remaining time doing **timed mock rounds** -- pick a random problem, set a 45-minute timer, write your solution in a plain text editor, and practice explaining your thought process out loud.
:::
