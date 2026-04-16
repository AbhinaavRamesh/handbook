# Tier 2: Trees

> **Second highest frequency in SDE interviews.** Tree problems appeared consistently in onsite coding rounds throughout 2024-2025. Expect DFS/BFS traversal, tree DP, LCA, and serialization.

---

## Why Trees?

Trees are a go-to for testing **recursive thinking** and **clean code under pressure**. Here is what makes them important:

- **Recursive structure maps to recursive solutions** -- if you can think recursively, trees become mechanical. If you cannot, they are a wall.
- **Clean code matters more here than anywhere else** -- tree solutions are short (10-20 lines), so every line is scrutinized. Sloppy variable names or forgotten base cases stand out.
- **Follow-ups are common** -- you solve "Diameter of Binary Tree" and the interviewer asks "now do Max Path Sum." The patterns compound.
- **Plain text editor friendly** -- tree solutions are concise enough to write correctly without an IDE.

::: tip TREES ARE HIGH ROI
6 problems, 4 core patterns. If you internalize the post-order DFS pattern and the "global variable trick," you can solve most tree problems in under 15 minutes.
:::

---

## Problem Set

| # | Problem | LC # | Pattern | Why |
|---|---------|------|---------|-----|
| 9 | [Binary Tree Max Path Sum](https://leetcode.com/problems/binary-tree-maximum-path-sum/) | 124 | Tree DP | Frequently asked in SDE interviews |
| 10 | [Lowest Common Ancestor of a BST](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-search-tree/) | 235 | BST property | Quick win |
| 11 | [Lowest Common Ancestor of a BT](https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/) | 236 | Recursion | Frequently asked |
| 12 | [Serialize and Deserialize BT](https://leetcode.com/problems/serialize-and-deserialize-binary-tree/) | 297 | BFS/DFS | Tests implementation |
| 13 | [Validate BST](https://leetcode.com/problems/validate-binary-search-tree/) | 98 | Inorder / bounds | Classic |
| 14 | [Diameter of Binary Tree](https://leetcode.com/problems/diameter-of-binary-tree/) | 543 | Post-order DFS | Common follow-up |

---

## Key Patterns

### When to use DFS vs BFS on trees

| Use DFS when... | Use BFS when... |
|-----------------|-----------------|
| You need to process children before the parent (post-order) | You need level-by-level traversal |
| The answer depends on subtree results (max path, diameter) | You need the shortest path to a node |
| You need to validate properties top-down (BST bounds) | You need to serialize level by level |
| Memory is a concern on wide trees | Memory is a concern on deep trees |

::: info RULE OF THUMB
**90% of tree questions are DFS.** Default to recursive DFS unless the problem explicitly asks for level-order or shortest path.
:::

### Post-order pattern (process children before parent)

This is the single most important tree pattern. You solve the left subtree, solve the right subtree, then combine results at the current node.

:::code-group

```python [Python]
def solve(node):
    if not node:
        return BASE_CASE

    left_result = solve(node.left)     # solve left subtree
    right_result = solve(node.right)   # solve right subtree

    # combine and return result for current subtree
    return combine(left_result, right_result, node.val)
```

```java [Java]
int solve(TreeNode node) {
    if (node == null) return BASE_CASE;

    int leftResult = solve(node.left);    // solve left subtree
    int rightResult = solve(node.right);  // solve right subtree

    // combine and return result for current subtree
    return combine(leftResult, rightResult, node.val);
}
```

:::

Used in: **Diameter** (#14), **Max Path Sum** (#9), **LCA of BT** (#11), **Validate BST** (#13).

### The "global variable" trick for path problems

Some problems need you to track a value across multiple recursive calls -- but each call only returns one thing. The trick: use an instance variable or a list to track the global answer, and use the return value for the recursive subproblem.

:::code-group

```python [Python]
def solve(root):
    self.ans = 0  # global tracker  # [!code highlight]

    def dfs(node):
        if not node:
            return 0
        left = dfs(node.left)
        right = dfs(node.right)
        self.ans = max(self.ans, left + right + node.val)  # update global  # [!code highlight]
        return max(left, right) + node.val  # return to parent  # [!code highlight]

    dfs(root)
    return self.ans
```

```java [Java]
int ans = 0; // global tracker (instance variable)

int solve(TreeNode root) {
    ans = 0;
    dfs(root);
    return ans;
}

private int dfs(TreeNode node) {
    if (node == null) return 0;
    int left = dfs(node.left);
    int right = dfs(node.right);
    ans = Math.max(ans, left + right + node.val);  // update global
    return Math.max(left, right) + node.val;        // return to parent
}
```

:::

Used in: **Max Path Sum** (#9), **Diameter** (#14).

### BST property exploitation (left < root < right)

In a BST, for every node: all values in the left subtree are less than the node, and all values in the right subtree are greater. This means:

- **LCA in a BST** is the first node where `p` and `q` split to different subtrees -- no need to search the whole tree.
- **Validation** can be done by passing `(low, high)` bounds down the tree.
- **Search** is O(h) -- go left if target is smaller, right if larger.

:::code-group

```python [Python]
# BST search -- O(h) instead of O(n)
def search_bst(node, target):
    while node:
        if target < node.val:
            node = node.left
        elif target > node.val:
            node = node.right
        else:
            return node
    return None
```

```java [Java]
// BST search -- O(h) instead of O(n)
TreeNode searchBST(TreeNode node, int target) {
    while (node != null) {
        if (target < node.val) node = node.left;
        else if (target > node.val) node = node.right;
        else return node;
    }
    return null;
}
```

:::

---

## Core Templates

### TreeNode class definition

Write this at the top of your solution. Interviewers expect you to know this by heart.

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
```

### Recursive DFS (preorder, inorder, postorder)

:::code-group

```python [Python]
# Preorder: root -> left -> right
# Use for: top-down problems, serialization
def preorder(node):
    if not node:
        return
    process(node)          # visit root first
    preorder(node.left)
    preorder(node.right)

# Inorder: left -> root -> right
# Use for: BST problems (gives sorted order)
def inorder(node):
    if not node:
        return
    inorder(node.left)
    process(node)          # visit root in the middle
    inorder(node.right)

# Postorder: left -> right -> root
# Use for: bottom-up problems (diameter, max path sum)
def postorder(node):
    if not node:
        return
    postorder(node.left)
    postorder(node.right)
    process(node)          # visit root last
```

```java [Java]
// Preorder: root -> left -> right
void preorder(TreeNode node, List<Integer> result) {
    if (node == null) return;
    result.add(node.val);        // visit root first
    preorder(node.left, result);
    preorder(node.right, result);
}

// Inorder: left -> root -> right (BST gives sorted order)
void inorder(TreeNode node, List<Integer> result) {
    if (node == null) return;
    inorder(node.left, result);
    result.add(node.val);        // visit root in the middle
    inorder(node.right, result);
}

// Postorder: left -> right -> root
void postorder(TreeNode node, List<Integer> result) {
    if (node == null) return;
    postorder(node.left, result);
    postorder(node.right, result);
    result.add(node.val);        // visit root last
}
```

:::

![Tree Traversals — Preorder, Inorder, Postorder](/sde-coding/sprint/tree_traversals.png)

### Iterative DFS with stack

:::code-group

```python [Python]
# Iterative preorder -- useful when recursion depth is a concern
def iterative_dfs(root):
    if not root:
        return []
    stack = [root]
    result = []
    while stack:
        node = stack.pop()
        result.append(node.val)
        # push right first so left is processed first (LIFO)
        if node.right:
            stack.append(node.right)
        if node.left:
            stack.append(node.left)
    return result
```

```java [Java]
// Iterative preorder -- useful when recursion depth is a concern
List<Integer> iterativeDFS(TreeNode root) {
    List<Integer> result = new ArrayList<>();
    if (root == null) return result;
    Deque<TreeNode> stack = new ArrayDeque<>();
    stack.push(root);
    while (!stack.isEmpty()) {
        TreeNode node = stack.pop();
        result.add(node.val);
        // push right first so left is processed first (LIFO)
        if (node.right != null) stack.push(node.right);
        if (node.left != null) stack.push(node.left);
    }
    return result;
}
```

:::

### BFS level-order with deque

:::code-group

```python [Python]
from collections import deque

def level_order(root):
    if not root:
        return []
    queue = deque([root])
    result = []
    while queue:
        level_size = len(queue)
        level = []
        for _ in range(level_size):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)
    return result
```

```java [Java]
List<List<Integer>> levelOrder(TreeNode root) {
    List<List<Integer>> result = new ArrayList<>();
    if (root == null) return result;
    Deque<TreeNode> queue = new ArrayDeque<>();
    queue.offer(root);
    while (!queue.isEmpty()) {
        int levelSize = queue.size();
        List<Integer> level = new ArrayList<>();
        for (int i = 0; i < levelSize; i++) {
            TreeNode node = queue.poll();
            level.add(node.val);
            if (node.left != null) queue.offer(node.left);
            if (node.right != null) queue.offer(node.right);
        }
        result.add(level);
    }
    return result;
}
```

:::

---

## Problem Walkthroughs

### #9 -- Binary Tree Max Path Sum (LC 124)

::: details Full walkthrough -- Binary Tree Max Path Sum

**Pattern:** Post-order DFS + global variable trick

**Approach:** At each node, compute the maximum "gain" from its left and right subtrees. A subtree gain is clamped to 0 (we can choose not to include a negative path). The maximum path **through** this node is `left_gain + right_gain + node.val`. But the value we **return** to the parent is `max(left_gain, right_gain) + node.val` because a path cannot fork.

:::code-group

```python [Python]
class Solution:
    def maxPathSum(self, root: TreeNode) -> int:
        self.max_sum = float('-inf')  # global tracker

        def max_gain(node):
            if not node:
                return 0

            # only take positive gains from subtrees
            left_gain = max(max_gain(node.left), 0)   # [!code highlight]
            right_gain = max(max_gain(node.right), 0)  # [!code highlight]

            # path through this node (potentially the answer)
            path_sum = node.val + left_gain + right_gain  # [!code highlight]
            self.max_sum = max(self.max_sum, path_sum)

            # return max gain to parent -- can only go one direction
            return node.val + max(left_gain, right_gain)  # [!code highlight]

        max_gain(root)
        return self.max_sum
```

```java [Java]
int maxPathSum(TreeNode root) {
    int[] maxSum = {Integer.MIN_VALUE}; // single-element array for mutable closure
    maxGain(root, maxSum);
    return maxSum[0];
}

private int maxGain(TreeNode node, int[] maxSum) {
    if (node == null) return 0;

    // only take positive gains from subtrees
    int leftGain = Math.max(maxGain(node.left, maxSum), 0);
    int rightGain = Math.max(maxGain(node.right, maxSum), 0);

    // path through this node (potentially the answer)
    int pathSum = node.val + leftGain + rightGain;
    maxSum[0] = Math.max(maxSum[0], pathSum);

    // return max gain to parent -- can only go one direction
    return node.val + Math.max(leftGain, rightGain);
}
```

:::

**Time:** O(n) -- visit every node once.

**Space:** O(h) -- recursion stack, where h is the height of the tree.

**Key insight:** The return value and the global update serve **different purposes**. The global update considers the full path (left + node + right). The return value only sends the best single-direction path upward, because a path cannot split at two nodes.

:::

---

### #10 -- Lowest Common Ancestor of a BST (LC 235)

::: details Full walkthrough -- LCA of BST

**Pattern:** BST property exploitation

**Approach:** Start at the root. If both `p` and `q` are smaller, go left. If both are larger, go right. The moment they split (or one equals the current node), you have found the LCA. No need to search the entire tree.

:::code-group

```python [Python]
class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        node = root
        while node:
            if p.val < node.val and q.val < node.val:   # [!code highlight]
                node = node.left       # both in left subtree
            elif p.val > node.val and q.val > node.val:  # [!code highlight]
                node = node.right      # both in right subtree
            else:
                return node            # split point = LCA  # [!code highlight]
```

```java [Java]
TreeNode lowestCommonAncestorBST(TreeNode root, TreeNode p, TreeNode q) {
    TreeNode node = root;
    while (node != null) {
        if (p.val < node.val && q.val < node.val)
            node = node.left;       // both in left subtree
        else if (p.val > node.val && q.val > node.val)
            node = node.right;      // both in right subtree
        else
            return node;            // split point = LCA
    }
    return null;
}
```

:::

**Time:** O(h) -- where h is the height of the BST. O(log n) if balanced.

**Space:** O(1) -- iterative, no extra space.

**Key insight:** The BST ordering property guarantees that the first node where `p` and `q` diverge to different subtrees is the LCA. This is O(h) instead of O(n).

:::

---

### #11 -- Lowest Common Ancestor of a Binary Tree (LC 236)

::: details Full walkthrough -- LCA of Binary Tree

**Pattern:** Post-order DFS / recursion

**Approach:** Recursively search left and right subtrees for `p` and `q`. If a node is `p` or `q`, return it. If both left and right recursive calls return non-null, the current node is the LCA. Otherwise, propagate whichever non-null result upward.

:::code-group

```python [Python]
class Solution:
    def lowestCommonAncestor(self, root: TreeNode, p: TreeNode, q: TreeNode) -> TreeNode:
        # base case: reached null or found p or q
        if not root or root == p or root == q:  # [!code highlight]
            return root

        left = self.lowestCommonAncestor(root.left, p, q)
        right = self.lowestCommonAncestor(root.right, p, q)

        if left and right:     # p and q are in different subtrees  # [!code highlight]
            return root        # current node is the LCA
        return left or right   # propagate the non-null result up  # [!code highlight]
```

```java [Java]
TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
    // base case: reached null or found p or q
    if (root == null || root == p || root == q) return root;

    TreeNode left = lowestCommonAncestor(root.left, p, q);
    TreeNode right = lowestCommonAncestor(root.right, p, q);

    if (left != null && right != null) return root;  // p and q in different subtrees
    return left != null ? left : right;              // propagate the non-null result up
}
```

:::

**Time:** O(n) -- worst case visits every node.

**Space:** O(h) -- recursion stack depth.

**Key insight:** This works because each node asks its subtrees "did you find p or q?" If both sides say yes, the current node must be where they meet. If only one side says yes, that side contains the LCA (or one of the targets), so propagate it up.

:::

![Lowest Common Ancestor Visualization](/sde-coding/sprint/lca_visualization.png)

---

### #12 -- Serialize and Deserialize Binary Tree (LC 297)

::: details Full walkthrough -- Serialize and Deserialize BT

**Pattern:** BFS (level-order) or preorder DFS

**Approach (preorder DFS):** Serialize using preorder traversal, encoding `None` nodes as a sentinel (e.g., `"#"`). Deserialize by reading values in the same preorder sequence and reconstructing the tree recursively.

:::code-group

```python [Python]
class Codec:
    def serialize(self, root: TreeNode) -> str:
        """Encodes a tree to a single string."""
        result = []

        def dfs(node):
            if not node:
                result.append('#')   # sentinel for null  # [!code highlight]
                return
            result.append(str(node.val))
            dfs(node.left)
            dfs(node.right)

        dfs(root)
        return ','.join(result)

    def deserialize(self, data: str) -> TreeNode:
        """Decodes your encoded data to tree."""
        values = iter(data.split(','))  # [!code highlight]

        def dfs():
            val = next(values)
            if val == '#':             # null sentinel  # [!code highlight]
                return None
            node = TreeNode(int(val))
            node.left = dfs()
            node.right = dfs()
            return node

        return dfs()
```

```java [Java]
// Serialize: preorder DFS, '#' for null
String serialize(TreeNode root) {
    StringBuilder sb = new StringBuilder();
    serializeDFS(root, sb);
    return sb.toString();
}

private void serializeDFS(TreeNode node, StringBuilder sb) {
    if (node == null) { sb.append("#,"); return; }  // sentinel for null
    sb.append(node.val).append(",");
    serializeDFS(node.left, sb);
    serializeDFS(node.right, sb);
}

// Deserialize: reconstruct tree from preorder string
TreeNode deserialize(String data) {
    Deque<String> values = new ArrayDeque<>(Arrays.asList(data.split(",")));
    return deserializeDFS(values);
}

private TreeNode deserializeDFS(Deque<String> values) {
    String val = values.poll();
    if ("#".equals(val)) return null;   // null sentinel
    TreeNode node = new TreeNode(Integer.parseInt(val));
    node.left = deserializeDFS(values);
    node.right = deserializeDFS(values);
    return node;
}
```

:::

**Time:** O(n) for both serialize and deserialize.

**Space:** O(n) for the serialized string and recursion stack.

**Key insight:** Preorder DFS naturally records the structure of the tree because we always process left before right and use a sentinel for null. The iterator in `deserialize` advances exactly once per node (including nulls), so the tree is reconstructed unambiguously without needing any index management.

:::

---

### #13 -- Validate Binary Search Tree (LC 98)

::: details Full walkthrough -- Validate BST

**Pattern:** Inorder traversal / recursive bounds checking

**Approach (bounds):** Pass a valid range `(low, high)` down the tree. At each node, check that its value falls strictly within the range. Narrow the range as you go: left children get `high = node.val`, right children get `low = node.val`.

:::code-group

```python [Python]
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        def validate(node, low, high):
            if not node:
                return True

            if node.val <= low or node.val >= high:  # [!code highlight]
                return False                          # out of valid range

            # left subtree: all values must be < node.val
            # right subtree: all values must be > node.val
            return (validate(node.left, low, node.val) and   # [!code highlight]
                    validate(node.right, node.val, high))     # [!code highlight]

        return validate(root, float('-inf'), float('inf'))
```

```java [Java]
boolean isValidBST(TreeNode root) {
    return validate(root, Long.MIN_VALUE, Long.MAX_VALUE);
}

private boolean validate(TreeNode node, long low, long high) {
    if (node == null) return true;

    if (node.val <= low || node.val >= high) return false;  // out of valid range

    // left subtree: all values must be < node.val
    // right subtree: all values must be > node.val
    return validate(node.left, low, node.val) &&
           validate(node.right, node.val, high);
}
```

:::

**Alternative -- Inorder traversal (values must be strictly increasing):**

:::code-group

```python [Python]
class Solution:
    def isValidBST(self, root: TreeNode) -> bool:
        self.prev = float('-inf')

        def inorder(node):
            if not node:
                return True
            if not inorder(node.left):
                return False
            if node.val <= self.prev:   # must be strictly increasing
                return False
            self.prev = node.val
            return inorder(node.right)

        return inorder(root)
```

```java [Java]
long prev = Long.MIN_VALUE;

boolean isValidBSTInorder(TreeNode root) {
    prev = Long.MIN_VALUE;
    return inorder(root);
}

private boolean inorder(TreeNode node) {
    if (node == null) return true;
    if (!inorder(node.left)) return false;
    if (node.val <= prev) return false;   // must be strictly increasing
    prev = node.val;
    return inorder(node.right);
}
```

:::

**Time:** O(n) -- visit every node once.

**Space:** O(h) -- recursion stack.

**Key insight:** The bounds approach is cleaner for a plain text editor because it has no mutable state. The common mistake is checking only `node.left.val < node.val` -- this misses cases where a deeper node violates the BST property against an ancestor.

:::

---

### #14 -- Diameter of Binary Tree (LC 543)

::: details Full walkthrough -- Diameter of Binary Tree

**Pattern:** Post-order DFS + global variable trick

**Approach:** The diameter is the longest path between any two nodes, measured in edges. At each node, compute the height of the left and right subtrees. The diameter **through** this node is `left_height + right_height`. Track the global maximum. Return `max(left_height, right_height) + 1` to the parent (the height of this subtree).

:::code-group

```python [Python]
class Solution:
    def diameterOfBinaryTree(self, root: TreeNode) -> int:
        self.diameter = 0

        def height(node):
            if not node:
                return 0

            left_h = height(node.left)
            right_h = height(node.right)

            # diameter through this node = left height + right height
            self.diameter = max(self.diameter, left_h + right_h)  # [!code highlight]

            # return height of this subtree to parent
            return max(left_h, right_h) + 1  # [!code highlight]

        height(root)
        return self.diameter
```

```java [Java]
int diameterOfBinaryTree(TreeNode root) {
    int[] diameter = {0};
    height(root, diameter);
    return diameter[0];
}

private int height(TreeNode node, int[] diameter) {
    if (node == null) return 0;

    int leftH = height(node.left, diameter);
    int rightH = height(node.right, diameter);

    // diameter through this node = left height + right height
    diameter[0] = Math.max(diameter[0], leftH + rightH);

    // return height of this subtree to parent
    return Math.max(leftH, rightH) + 1;
}
```

:::

**Time:** O(n) -- visit every node once.

**Space:** O(h) -- recursion stack.

**Key insight:** This is structurally identical to Max Path Sum (#9) but simpler -- no negative values to worry about, and the "gain" is just the height. If you understand this problem, Max Path Sum is a direct extension. The interviewer knows this, which is why Diameter is a common warm-up before Max Path Sum.

:::

---

## Common Mistakes

::: danger MISTAKES THAT COST OFFERS

**Forgetting to handle `None` nodes**

Every recursive tree function needs a base case for `None`. If you skip it, you get an `AttributeError` on `node.left` or `node.val`. In a plain text editor, you cannot run your code to catch this -- write the base case first, every time.

```python
# WRONG -- crashes on empty subtrees
def height(node):
    return max(height(node.left), height(node.right)) + 1

# CORRECT -- handle None first
def height(node):
    if not node:        # always check this first
        return 0
    return max(height(node.left), height(node.right)) + 1
```

**Confusing BST vs BT algorithms**

LCA of a BST (#10) is O(h) using the split-point technique. LCA of a general BT (#11) is O(n) using post-order recursion. If you use the BT algorithm on a BST problem, it works but is suboptimal -- the interviewer will notice and ask you to do better. If you use the BST algorithm on a general BT, it is **wrong**.

**Not tracking global state for path sum problems**

In Max Path Sum (#9) and Diameter (#14), the answer is **not** the return value of the recursion. The return value is the best single-direction path for the parent to use. The actual answer is tracked in a separate global variable. If you try to return the answer directly, your solution will be incorrect for any tree where the optimal path does not pass through the root.

```python
# WRONG -- only considers paths through the root
return left + right + root.val

# CORRECT -- update global, return single direction to parent
self.ans = max(self.ans, left + right + root.val)  # global update
return max(left, right) + root.val                  # return to parent
```

:::

---

## Practice Checklist

Use this to track your progress. Solve each problem in a plain text editor (no IDE) to simulate interview conditions.

| Done | # | Problem | Time Target | Notes |
|------|---|---------|-------------|-------|
| [ ] | 9 | Binary Tree Max Path Sum | 20 min | Global variable trick. Clamp negatives to 0. |
| [ ] | 10 | LCA of BST | 8 min | Iterative. Follow the split. Quick win. |
| [ ] | 11 | LCA of BT | 12 min | Post-order. Both sides non-null = LCA. |
| [ ] | 12 | Serialize / Deserialize BT | 20 min | Preorder + sentinel. Use `iter()` for deserialize. |
| [ ] | 13 | Validate BST | 12 min | Bounds approach. Use `-inf` / `inf` as initial range. |
| [ ] | 14 | Diameter of Binary Tree | 10 min | Same structure as Max Path Sum but simpler. |

::: tip SOLVE ORDER
Start with **#14 (Diameter)** -- it is the simplest and teaches the post-order + global variable pattern. Then do **#9 (Max Path Sum)** -- it is the same pattern with negative values. Then **#10 (LCA BST)** and **#11 (LCA BT)** as a pair. Then **#13 (Validate BST)**. Save **#12 (Serialize/Deserialize)** for last -- it is the most implementation-heavy.
:::

---

<div style="text-align: center; margin-top: 2rem;">

**6 problems. 4 patterns. You know what to do.**

[Back to Battle Plan](./) | [Next: Sliding Window](./sliding-window) | [Templates](./templates)

</div>
