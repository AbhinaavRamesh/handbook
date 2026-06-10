# LCA & BST Operations

> **Ancestor finding and BST key operations**

---

## Lowest Common Ancestor (Binary Tree)

### Problem Statement
Given a binary tree and two nodes p and q, find their lowest common ancestor.

The lowest common ancestor (LCA) is defined as the lowest node in the tree that has both p and q as descendants (where a node can be a descendant of itself).

**LeetCode Reference:** [236. Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)

### Examples
```
Example 1:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 1
Output: 3
Explanation: The LCA of nodes 5 and 1 is 3.

Example 2:
Input: root = [3,5,1,6,2,0,8,null,null,7,4], p = 5, q = 4
Output: 5
Explanation: Node 5 is an ancestor of 4, and a node can be a descendant of itself.
```

### Solution

::: code-group
```python [Python]
def lowestCommonAncestor(root, p, q):
    if not root or root == p or root == q:
        return root

    left = lowestCommonAncestor(root.left, p, q)
    right = lowestCommonAncestor(root.right, p, q)

    # If both sides return a node, current root is LCA
    if left and right:
        return root

    return left if left else right
```
```java [Java]
public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    if (root == null || root == p || root == q) return root;

    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);

    if (left != null && right != null) return root;
    return left != null ? left : right;
}
```
:::

::: info Complexity: Time O(n) · Space O(h)
- **Time:** O(n) because we may need to search the entire tree in worst case
- **Space:** O(h) for recursion stack where h is tree height
:::

### Algorithm Explanation
1. **Base case:** If root is None, or root equals p or q, return root
2. **Recursive search:** Search both left and right subtrees
3. **Decision logic:**
   - If both left and right return non-null, p and q are on opposite sides - current node is LCA
   - If only one side returns non-null, both nodes are in that subtree
   - If both return null, neither p nor q found in this subtree

### LCA in BST (More efficient)

**LeetCode Reference:** [235. Lowest Common Ancestor of a Binary Search Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)

```python
def lowestCommonAncestorBST(root, p, q):
    while root:
        if p.val < root.val and q.val < root.val:
            root = root.left
        elif p.val > root.val and q.val > root.val:
            root = root.right
        else:
            return root
    return None
```

::: info Complexity: Time O(h) · Space O(1)
- **Time:** O(h) because we follow one path using BST property to find the split point
- **Space:** O(1) because iterative approach uses only a pointer variable
:::

### Why BST Version is More Efficient
- Uses BST property: left subtree < root < right subtree
- No need to search both subtrees
- The split point (where one node goes left and another goes right) is the LCA

### Complexity
| Version | Time | Space |
|---------|------|-------|
| Binary Tree | O(n) | O(h) - recursion stack |
| BST | O(h) | O(1) - iterative |

Where h = tree height (O(log n) for balanced, O(n) for skewed)

---

## Find Largest Smaller BST Key

### Problem Statement
Given a BST and a target value, find the largest key in the tree that is smaller than the target.

This is also known as finding the **floor** of a value in a BST (largest value <= target, but strictly less than in this case).

### Examples
```
Example 1:
BST:      20
         /  \
        9    25
       / \
      5   12
         /  \
        11   14

Target: 17
Output: 14 (largest value smaller than 17)

Target: 4
Output: None (no value smaller than 4)
```

### Solution

::: code-group
```python [Python]
def find_largest_smaller(root, target):
    result = None

    while root:
        if root.val < target:
            result = root.val  # Potential answer
            root = root.right  # Look for larger
        else:
            root = root.left  # Too big, go left

    return result
```
```java [Java]
public Integer findLargestSmaller(TreeNode root, int target) {
    Integer result = null;

    while (root != null) {
        if (root.val < target) {
            result = root.val; // Potential answer
            root = root.right; // Look for larger
        } else {
            root = root.left;  // Too big, go left
        }
    }

    return result;
}
```
:::

::: info Complexity: Time O(h) · Space O(1)
- **Time:** O(h) because we traverse one path from root tracking candidates
- **Space:** O(1) because we only store the result variable
:::

### Algorithm Walkthrough
1. Start at root
2. If current value < target:
   - This is a candidate answer (save it)
   - Move right to find potentially larger valid values
3. If current value >= target:
   - Too big, move left to find smaller values
4. Continue until we've searched the entire relevant path

### Variations

**Find Smallest Greater (Ceiling)**
```python
def find_smallest_greater(root, target):
    result = None

    while root:
        if root.val > target:
            result = root.val  # Potential answer
            root = root.left   # Look for smaller
        else:
            root = root.right  # Too small, go right

    return result
```

::: info Complexity: Time O(h) · Space O(1)
- **Time:** O(h) because we follow one path looking for ceiling value
- **Space:** O(1) for single result variable
:::

**Find Floor (Largest <= target)**
```python
def find_floor(root, target):
    result = None

    while root:
        if root.val == target:
            return root.val
        elif root.val < target:
            result = root.val
            root = root.right
        else:
            root = root.left

    return result
```

::: info Complexity: Time O(h) · Space O(1)
- **Time:** O(h) because we traverse one path; includes exact match check
- **Space:** O(1) for single result variable
:::

### Complexity
- **Time:** O(h) - traverse one path from root to leaf
- **Space:** O(1) - iterative solution

---

## BST Successor Search

### Problem Statement
Find the in-order successor of a given node in a BST. The successor is the node with the smallest key greater than the given node's value.

**LeetCode References:**
- [285. Inorder Successor in BST](https://leetcode.com/problems/inorder-successor-in-bst/)
- [510. Inorder Successor in BST II](https://leetcode.com/problems/inorder-successor-in-bst-ii/)

### Examples
```
Example 1:
BST:    2
       / \
      1   3

p = 1
Output: 2 (The in-order successor of 1 is 2)

Example 2:
BST:       5
          / \
         3   6
        / \
       2   4
      /
     1

p = 6
Output: null (No successor exists)
```

### Case Analysis
1. **Node has right subtree:** Successor is the leftmost node in the right subtree
2. **Node has no right subtree:** Successor is the nearest ancestor where the node is in the left subtree

### Solution - With Root Access

::: code-group
```python [Python]
def inorderSuccessor(root, p):
    successor = None

    while root:
        if p.val < root.val:
            successor = root  # Potential successor
            root = root.left
        else:
            root = root.right

    return successor
```
```java [Java]
public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
    TreeNode successor = null;

    while (root != null) {
        if (p.val < root.val) {
            successor = root; // Potential successor
            root = root.left;
        } else {
            root = root.right;
        }
    }

    return successor;
}
```
:::

::: info Complexity: Time O(h) · Space O(1)
- **Time:** O(h) because we follow one path tracking potential successors
- **Space:** O(1) for single successor variable
:::

### Solution (Python) - With Parent Pointer

When you have access to parent pointers but not the root:

```python
def inorderSuccessor_with_parent(node):
    # Case 1: Has right subtree - find leftmost in right
    if node.right:
        node = node.right
        while node.left:
            node = node.left
        return node

    # Case 2: No right subtree - go up until we're a left child
    while node.parent and node == node.parent.right:
        node = node.parent

    return node.parent
```

::: info Complexity: Time O(h) · Space O(1)
- **Time:** O(h) either traversing right subtree or going up to ancestors
- **Space:** O(1) as we only use pointer manipulation
:::

### Predecessor Search (Opposite Direction)

```python
def inorderPredecessor(root, p):
    predecessor = None

    while root:
        if p.val > root.val:
            predecessor = root  # Potential predecessor
            root = root.right
        else:
            root = root.left

    return predecessor

def inorderPredecessor_with_parent(node):
    # Case 1: Has left subtree - find rightmost in left
    if node.left:
        node = node.left
        while node.right:
            node = node.right
        return node

    # Case 2: No left subtree - go up until we're a right child
    while node.parent and node == node.parent.left:
        node = node.parent

    return node.parent
```

::: info Complexity: Time O(h) · Space O(1)
- **Time:** O(h) either traversing left subtree or going up to ancestors
- **Space:** O(1) as we only use pointer manipulation
:::

### Complexity
- **Time:** O(h)
- **Space:** O(1)

---

## BST Operations Summary

| Operation | Time | Space | Key Insight |
|-----------|------|-------|-------------|
| Search | O(h) | O(1) | Compare and go left/right |
| Insert | O(h) | O(1) | Find position, add leaf |
| Delete | O(h) | O(1) | Replace with successor |
| LCA (BST) | O(h) | O(1) | Split point where paths diverge |
| LCA (Binary Tree) | O(n) | O(h) | DFS both subtrees |
| Successor | O(h) | O(1) | Right subtree min or nearest right ancestor |
| Largest Smaller | O(h) | O(1) | Track candidates while traversing |

**Note:** h = O(log n) for balanced BST, O(n) for skewed BST

---

## Common Patterns and Techniques

### Pattern 1: Two-Pointer Tracking
Used in Successor and Largest Smaller problems:
```python
candidate = None
while root:
    if condition:
        candidate = root  # Update candidate
        # Move in direction to find better
    else:
        # Move in other direction
```

### Pattern 2: BST Property Exploitation
Always ask: Can I use left < root < right to eliminate half the tree?

### Pattern 3: Case Analysis by Subtree Existence
For successor/predecessor:
- Has relevant subtree? -> Find extreme in that subtree
- No relevant subtree? -> Traverse up to ancestors

---

## Interview Applications

These BST operations frequently appear in technical interviews in various forms:

1. **Database Range Queries:** Finding keys within a range uses floor/ceiling operations
2. **Calendar Systems:** Finding next available slot (successor-like operations)
3. **Autocomplete:** Finding the smallest string greater than prefix
4. **Version Control:** Finding common ancestor of two commits (LCA)
5. **File Systems:** Finding common parent directory (LCA variant)

### Common Follow-up Questions
- What if we need the k-th successor?
- How would you handle duplicates in the BST?
- Can you modify the solution to work with a balanced tree (AVL/Red-Black)?
- What if the tree is too large to fit in memory?

---

## Practice Problems

::: details Lowest Common Ancestor of a Binary Tree (LeetCode 236)
**Problem:** Find the lowest common ancestor of two given nodes in a binary tree.

**Key Insight:** If both subtrees return a node, current node is LCA. If only one returns, propagate it upward.

**Approach:** Recursive DFS. Base case: if node is null or equals p or q, return it. Recurse on both children. If both return non-null, current is LCA.

**Complexity:** Time O(n), Space O(h)
:::

::: details Lowest Common Ancestor of a BST (LeetCode 235)
**Problem:** Find the LCA of two nodes in a Binary Search Tree.

**Key Insight:** Use BST property. LCA is the split point where p and q go to different subtrees.

**Approach:** If both p and q are smaller, go left. If both are larger, go right. Otherwise, current node is LCA. Can be done iteratively in O(1) space.

**Complexity:** Time O(h), Space O(1)
:::

::: details Inorder Successor in BST (LeetCode 285)
**Problem:** Find the in-order successor (smallest key greater than given node) in a BST.

**Key Insight:** The successor is either in right subtree (leftmost node) or is an ancestor where we last went left.

**Approach:** If p.val < root.val, record root as potential successor and go left. Otherwise go right. Return the last recorded successor.

**Complexity:** Time O(h), Space O(1)
:::

::: details Inorder Successor in BST II (LeetCode 510)
**Problem:** Find in-order successor when you have parent pointers but not root access.

**Key Insight:** Two cases: if right child exists, go right then all the way left. Otherwise, go up until you find an ancestor where you came from the left.

**Approach:** If node has right subtree, return leftmost in right subtree. Otherwise, traverse up while current is parent's right child.

**Complexity:** Time O(h), Space O(1)
:::

::: details Lowest Common Ancestor of Deepest Leaves (LeetCode 1123)
**Problem:** Return the LCA of all the deepest leaves in a binary tree.

**Key Insight:** If left and right subtrees have equal depth, current node is LCA. Otherwise, LCA is in the deeper subtree.

**Approach:** DFS returning (node, depth) pair. At each node, compare left and right depths. Return deeper subtree's LCA, or current node if depths are equal.

**Complexity:** Time O(n), Space O(h)
:::

---

## References

- [LeetCode 236 - Lowest Common Ancestor of a Binary Tree](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/)
- [LeetCode 235 - Lowest Common Ancestor of a BST](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/)
- [LeetCode 285 - Inorder Successor in BST](https://leetcode.com/problems/inorder-successor-in-bst/)
- [LeetCode 510 - Inorder Successor in BST II](https://leetcode.com/problems/inorder-successor-in-bst-ii/)
- [Algo.monster - LCA Explanation](https://algo.monster/liteproblems/236)
- [Algo.monster - BST Successor Explanation](https://algo.monster/liteproblems/285)
- [GeeksforGeeks - Inorder Successor in BST](https://www.geeksforgeeks.org/dsa/inorder-successor-in-binary-search-tree/)
- [GeeksforGeeks - LCA in Binary Tree](https://www.geeksforgeeks.org/dsa/lowest-common-ancestor-binary-tree-set-1/)
