# Design File System

## Problem Statement

You are asked to design a file system that allows you to create new paths and associate them with different values.

The format of a path is one or more concatenated strings of the form: `/` followed by one or more lowercase English letters. For example, `/leetcode` and `/leetcode/problems` are valid paths while an empty string `""` and `/` are not.

Implement the `FileSystem` class:

- `bool createPath(string path, int value)` Creates a new path and associates a value to it if possible and returns `true`. Returns `false` if the path already exists or its parent path doesn't exist.
- `int get(string path)` Returns the value associated with path or returns `-1` if the path doesn't exist.

**LeetCode:** [1166. Design File System](https://leetcode.com/problems/design-file-system/)

## Examples

**Example 1:**
```
Input:
["FileSystem","createPath","get"]
[[],["/a",1],["/a"]]

Output:
[null,true,1]

Explanation:
FileSystem fileSystem = new FileSystem();
fileSystem.createPath("/a", 1);  // return true
fileSystem.get("/a");            // return 1
```

**Example 2:**
```
Input:
["FileSystem","createPath","createPath","get","createPath","get"]
[[],["/leet",1],["/leet/code",2],["/leet/code"],["/c/d",3],["/c"]]

Output:
[null,true,true,2,false,-1]

Explanation:
FileSystem fileSystem = new FileSystem();
fileSystem.createPath("/leet", 1);      // return true
fileSystem.createPath("/leet/code", 2); // return true
fileSystem.get("/leet/code");           // return 2
fileSystem.createPath("/c/d", 3);       // return false (parent "/c" doesn't exist)
fileSystem.get("/c");                   // return -1 (path doesn't exist)
```

## Constraints

- `2 <= path.length <= 100`
- `1 <= value <= 10^9`
- Each path is valid and consists of lowercase English letters and '/'.
- At most `10^4` calls in total will be made to `createPath` and `get`.

## Hash Map Approach

![Design File System Visualization](./assets/design-file-system-viz.png)

### Key Insight

Store full paths as keys in a hash map. Before creating a path, validate that the parent path exists.

### Algorithm

**createPath:**
1. Check if path already exists -> return false
2. Find parent path (everything before last '/')
3. Check if parent exists (or is root) -> return false if not
4. Add path to hash map with value
5. Return true

**get:**
1. Return value if path exists, else -1

## Solution

```python
class FileSystem:
    """
    File System using HashMap.

    createPath: O(path_length) - string operations
    get: O(1) - hash lookup
    Space: O(total_path_characters)
    """

    def __init__(self):
        self.paths = {}

    def createPath(self, path: str, value: int) -> bool:
        # Path already exists
        if path in self.paths:
            return False

        # Find parent path
        last_slash = path.rfind('/')
        parent = path[:last_slash]

        # Parent must exist (empty parent means direct child of root)
        if parent and parent not in self.paths:
            return False

        # Create the path
        self.paths[path] = value
        return True

    def get(self, path: str) -> int:
        return self.paths.get(path, -1)
```

::: info Complexity: Time O(path_length) · Space O(total_path_chars)
- **Time:** O(path_length) for createPath due to string operations (rfind, slicing); O(1) average for get with hash lookup
- **Space:** O(total_path_chars) for storing all paths as keys in the hash map
:::

## Trie Approach

For more complex file system operations, a Trie (prefix tree) is more suitable:

```python
class TrieNode:
    def __init__(self):
        self.children = {}
        self.value = -1
        self.is_path = False


class FileSystemTrie:
    """
    File System using Trie.
    Better for operations like listing directories.
    """

    def __init__(self):
        self.root = TrieNode()

    def createPath(self, path: str, value: int) -> bool:
        parts = path.split('/')[1:]  # Skip empty string before first '/'

        node = self.root
        for i, part in enumerate(parts):
            if part not in node.children:
                # Last part: create it
                if i == len(parts) - 1:
                    node.children[part] = TrieNode()
                else:
                    # Parent doesn't exist
                    return False

            node = node.children[part]

        # Path already exists
        if node.is_path:
            return False

        node.is_path = True
        node.value = value
        return True

    def get(self, path: str) -> int:
        parts = path.split('/')[1:]

        node = self.root
        for part in parts:
            if part not in node.children:
                return -1
            node = node.children[part]

        return node.value if node.is_path else -1
```

::: info Complexity: Time O(path_length) · Space O(total_path_chars)
- **Time:** O(path_length) for both createPath and get, as we traverse the path components
- **Space:** O(total_path_chars) for the Trie nodes storing all path components
:::

## Extended File System

```python
from typing import List

class ExtendedFileSystem:
    """
    Extended file system with more operations.
    """

    def __init__(self):
        self.root = TrieNode()

    def createPath(self, path: str, value: int) -> bool:
        if not path or path == '/':
            return False

        parts = path.split('/')[1:]
        node = self.root

        for i, part in enumerate(parts):
            if i == len(parts) - 1:
                # Last part
                if part in node.children and node.children[part].is_path:
                    return False  # Already exists
                if part not in node.children:
                    node.children[part] = TrieNode()
                node.children[part].is_path = True
                node.children[part].value = value
                return True
            else:
                # Intermediate parts must exist
                if part not in node.children or not node.children[part].is_path:
                    return False
                node = node.children[part]

        return False

    def get(self, path: str) -> int:
        node = self._get_node(path)
        return node.value if node and node.is_path else -1

    def _get_node(self, path: str) -> TrieNode:
        if not path or path == '/':
            return self.root

        parts = path.split('/')[1:]
        node = self.root

        for part in parts:
            if part not in node.children:
                return None
            node = node.children[part]

        return node

    def delete(self, path: str) -> bool:
        """Delete a path (only if it has no children)."""
        if not path or path == '/':
            return False

        parts = path.split('/')[1:]
        node = self.root
        parent = None
        last_part = None

        for part in parts:
            if part not in node.children:
                return False
            parent = node
            last_part = part
            node = node.children[part]

        if not node.is_path:
            return False

        # Can only delete if no children
        if node.children:
            return False

        del parent.children[last_part]
        return True

    def listDir(self, path: str) -> List[str]:
        """List immediate children of a directory."""
        node = self._get_node(path)
        if not node:
            return []

        return [name for name, child in node.children.items()
                if child.is_path]

    def exists(self, path: str) -> bool:
        """Check if path exists."""
        node = self._get_node(path)
        return node is not None and node.is_path

    def update(self, path: str, value: int) -> bool:
        """Update value at path."""
        node = self._get_node(path)
        if not node or not node.is_path:
            return False
        node.value = value
        return True
```

## Complexity Analysis

| Operation | HashMap | Trie |
|-----------|---------|------|
| createPath | O(path_len) | O(path_len) |
| get | O(1)* | O(path_len) |
| Space | O(total_chars) | O(total_chars) |

*O(path_len) for hashing in worst case

## Edge Cases

1. **Root path:** "/" - Not a valid path per problem
2. **Path already exists:** Return false
3. **Parent doesn't exist:** Return false
4. **Single level path:** "/a" - Parent is root, always valid
5. **Deep nesting:** "/a/b/c/d" - All parents must exist

## Variations

### File System with File Content

```python
class FileSystemWithContent:
    """File system that stores file content."""

    def __init__(self):
        self.paths = {}  # path -> (is_dir, content/None)

    def mkdir(self, path: str) -> None:
        """Create directory (and all parents)."""
        parts = path.split('/')[1:]
        current = ""
        for part in parts:
            current += "/" + part
            if current not in self.paths:
                self.paths[current] = (True, None)

    def addContentToFile(self, path: str, content: str) -> None:
        """Add/append content to file."""
        # Ensure parent directories exist
        last_slash = path.rfind('/')
        if last_slash > 0:
            self.mkdir(path[:last_slash])

        if path in self.paths:
            _, existing = self.paths[path]
            self.paths[path] = (False, (existing or "") + content)
        else:
            self.paths[path] = (False, content)

    def readContentFromFile(self, path: str) -> str:
        """Read file content."""
        if path in self.paths:
            is_dir, content = self.paths[path]
            if not is_dir:
                return content
        return ""

    def ls(self, path: str) -> List[str]:
        """List directory contents or file name."""
        if path not in self.paths:
            return []

        is_dir, _ = self.paths[path]
        if not is_dir:
            # Return file name
            return [path.split('/')[-1]]

        # List directory contents
        prefix = path if path != "/" else ""
        result = set()
        for p in self.paths:
            if p.startswith(prefix + "/"):
                rest = p[len(prefix)+1:]
                first_part = rest.split('/')[0]
                result.add(first_part)

        return sorted(result)
```

## Interview Tips

1. **Clarify requirements:** What operations needed? File content?
2. **Discuss trade-offs:** HashMap vs Trie
3. **Consider extensions:** Delete, rename, move operations
4. **Thread safety:** Mention if asked about concurrent access

## Related Problems

::: details Design In-Memory File System (LeetCode 588)
**Problem:** Implement file system with ls, mkdir, addContentToFile, readContentFromFile.

**Key Insight:** Trie-like structure where nodes can be files or directories.

**Approach:** Each node has children map and optional file content. Parse path to navigate.

**Complexity:** Time O(path_length) for operations, Space O(total_path_length)
:::

::: details Implement Trie (Prefix Tree) (LeetCode 208)
**Problem:** Implement Trie with insert, search, startsWith operations.

**Key Insight:** Tree structure where each node represents a character. Mark word endings.

**Approach:** Each node has children map and isEnd flag. Insert/search by traversing characters.

**Complexity:** Time O(word_length) for all operations, Space O(total_chars)
:::

::: details Design Search Autocomplete System (LeetCode 642)
**Problem:** Design autocomplete returning top 3 sentences by frequency.

**Key Insight:** Combine Trie for prefix matching with frequency tracking.

**Approach:** Trie stores sentences with counts. For each input char, filter and rank by frequency.

**Complexity:** Time O(n log k) for suggestions where n is matches, Space O(total_chars)
:::
