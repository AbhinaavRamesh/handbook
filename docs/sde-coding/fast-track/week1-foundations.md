# Week 1: Foundations - Day-by-Day Breakdown

> **Build structural intuition and master fundamental data structures**

---

## Overview

Week 1 focuses on establishing a solid foundation in the core data structures that appear most frequently in technical interviews. By the end of this week, you should be comfortable with arrays, hash tables, linked lists, stacks, queues, and binary trees.

**Weekly Goals:**
- Solve 30-35 problems across fundamental data structures
- Complete 1 full mock interview (recorded if possible)
- Build pattern recognition for basic problem types
- Establish consistent daily practice routine

**Daily Time Commitment:** 3-4 hours

---

## Day 1: Arrays & Strings

### Learning Objectives
- Master two-pointer techniques
- Understand sliding window patterns
- Implement prefix sum for range queries

### Study Plan (3-4 hours)

| Time Block | Activity | Details |
|------------|----------|---------|
| 0:00-0:30 | **Concept Review** | Two pointers, sliding window theory |
| 0:30-2:00 | **Problem Solving** | Solve 5-6 problems (see list below) |
| 2:00-2:30 | **Pattern Review** | Document patterns you discovered |
| 2:30-3:00 | **Edge Cases** | Practice empty array, single element |
| 3:00-3:30 | **Verbal Practice** | Explain one solution out loud |

### Must-Solve Problems

| # | Problem | Difficulty | Pattern | Time Limit |
|---|---------|------------|---------|------------|
| 1 | Two Sum | Easy | Hash Map | 15 min |
| 2 | Best Time to Buy and Sell Stock | Easy | Single Pass | 15 min |
| 3 | Contains Duplicate | Easy | Hash Set | 10 min |
| 4 | Maximum Subarray | Medium | Kadane's Algorithm | 20 min |
| 5 | Product of Array Except Self | Medium | Prefix/Suffix | 25 min |
| 6 | 3Sum | Medium | Two Pointers | 30 min |

### Code Templates

```python
# Two Pointers - Opposite Direction
def two_sum_sorted(arr, target):
    left, right = 0, len(arr) - 1
    while left < right:
        current = arr[left] + arr[right]
        if current == target:
            return [left, right]
        elif current < target:
            left += 1
        else:
            right -= 1
    return []

# Kadane's Algorithm - Maximum Subarray
def max_subarray(nums):
    max_sum = current_sum = nums[0]
    for num in nums[1:]:
        current_sum = max(num, current_sum + num)
        max_sum = max(max_sum, current_sum)
    return max_sum

# Prefix Sum
def prefix_sum(arr):
    prefix = [0]
    for num in arr:
        prefix.append(prefix[-1] + num)
    return prefix
    # Range sum [i, j] = prefix[j+1] - prefix[i]
```

::: info Complexity: Time O(n) · Space O(1) to O(n)
- **Time:** Two pointers and Kadane's are O(n) single pass
- **Space:** Prefix sum uses O(n); others use O(1)
:::

### Day 1 Checklist

```
[ ] Reviewed two-pointer theory (15 min)
[ ] Solved Two Sum using hash map approach
[ ] Solved Best Time to Buy and Sell Stock
[ ] Solved Contains Duplicate
[ ] Solved Maximum Subarray using Kadane's
[ ] Solved Product of Array Except Self
[ ] Attempted 3Sum with proper duplicate handling
[ ] Documented 2-3 patterns learned today
[ ] Practiced explaining one solution verbally
```

---

## Day 2: Hash Tables

### Learning Objectives
- Master frequency counting techniques
- Understand two-sum pattern variations
- Implement anagram detection efficiently

### Study Plan (3-4 hours)

| Time Block | Activity | Details |
|------------|----------|---------|
| 0:00-0:30 | **Concept Review** | Hash table operations, collision handling |
| 0:30-2:00 | **Problem Solving** | Solve 5-6 problems |
| 2:00-2:30 | **Pattern Review** | When hash map vs hash set |
| 2:30-3:00 | **Optimization** | Space-time tradeoffs |
| 3:00-3:30 | **Verbal Practice** | Explain one solution out loud |

### Must-Solve Problems

| # | Problem | Difficulty | Pattern | Time Limit |
|---|---------|------------|---------|------------|
| 1 | Valid Anagram | Easy | Frequency Count | 10 min |
| 2 | Two Sum | Easy | Hash Map Lookup | 15 min |
| 3 | Group Anagrams | Medium | Sorted Key | 25 min |
| 4 | Top K Frequent Elements | Medium | Frequency + Heap | 25 min |
| 5 | Longest Consecutive Sequence | Medium | Hash Set | 25 min |
| 6 | Subarray Sum Equals K | Medium | Prefix Sum + Hash | 30 min |

### Code Templates

```python
# Frequency Counter
from collections import Counter, defaultdict

def is_anagram(s, t):
    return Counter(s) == Counter(t)

# Hash Map for Two Sum Pattern
def two_sum(nums, target):
    seen = {}
    for i, num in enumerate(nums):
        complement = target - num
        if complement in seen:
            return [seen[complement], i]
        seen[num] = i
    return []

# Group by Key Pattern
def group_anagrams(strs):
    groups = defaultdict(list)
    for s in strs:
        key = tuple(sorted(s))  # or character count tuple
        groups[key].append(s)
    return list(groups.values())

# Prefix Sum + Hash Map
def subarray_sum(nums, k):
    count = 0
    prefix_sum = 0
    prefix_count = {0: 1}  # Empty prefix

    for num in nums:
        prefix_sum += num
        if prefix_sum - k in prefix_count:
            count += prefix_count[prefix_sum - k]
        prefix_count[prefix_sum] = prefix_count.get(prefix_sum, 0) + 1

    return count
```

::: info Complexity: Time O(n) · Space O(n)
- **Time:** Single pass through elements with O(1) hash operations
- **Space:** Hash maps store up to n unique keys
:::

### Day 2 Checklist

```
[ ] Reviewed hash table fundamentals (15 min)
[ ] Solved Valid Anagram
[ ] Solved Two Sum (review from Day 1)
[ ] Solved Group Anagrams
[ ] Solved Top K Frequent Elements
[ ] Solved Longest Consecutive Sequence
[ ] Attempted Subarray Sum Equals K
[ ] Understood prefix sum + hash map pattern
[ ] Practiced explaining frequency counting verbally
```

---

## Day 3: Linked Lists

### Learning Objectives
- Master list reversal techniques
- Implement cycle detection (Floyd's algorithm)
- Perform merge operations efficiently

### Study Plan (3-4 hours)

| Time Block | Activity | Details |
|------------|----------|---------|
| 0:00-0:30 | **Concept Review** | List operations, pointer manipulation |
| 0:30-2:00 | **Problem Solving** | Solve 4-5 problems |
| 2:00-2:30 | **Pattern Review** | Fast/slow pointer applications |
| 2:30-3:00 | **Edge Cases** | Empty list, single node, cycles |
| 3:00-3:30 | **Verbal Practice** | Draw and explain pointer movements |

### Must-Solve Problems

| # | Problem | Difficulty | Pattern | Time Limit |
|---|---------|------------|---------|------------|
| 1 | Reverse Linked List | Easy | Iterative/Recursive | 15 min |
| 2 | Merge Two Sorted Lists | Easy | Two Pointers | 15 min |
| 3 | Linked List Cycle | Easy | Fast/Slow Pointers | 15 min |
| 4 | Remove Nth Node From End | Medium | Two Pointers | 20 min |
| 5 | Reorder List | Medium | Multiple Techniques | 30 min |

### Code Templates

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# Iterative Reversal
def reverse_list(head):
    prev = None
    curr = head
    while curr:
        next_temp = curr.next
        curr.next = prev
        prev = curr
        curr = next_temp
    return prev

# Cycle Detection (Floyd's Algorithm)
def has_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# Find Middle Node
def find_middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

# Merge Two Sorted Lists
def merge_lists(l1, l2):
    dummy = ListNode(0)
    curr = dummy

    while l1 and l2:
        if l1.val <= l2.val:
            curr.next = l1
            l1 = l1.next
        else:
            curr.next = l2
            l2 = l2.next
        curr = curr.next

    curr.next = l1 or l2
    return dummy.next
```

::: info Complexity: Time O(n) · Space O(1)
- **Time:** Reversal and merge are single pass O(n)
- **Space:** All operations use constant extra space
:::

### Day 3 Checklist

```
[ ] Reviewed linked list operations (15 min)
[ ] Drew pointer diagrams on paper
[ ] Solved Reverse Linked List (both iterative and recursive)
[ ] Solved Merge Two Sorted Lists
[ ] Solved Linked List Cycle
[ ] Solved Remove Nth Node From End
[ ] Attempted Reorder List
[ ] Practiced dry-running pointer movements
```

---

## Day 4: Stacks & Queues

### Learning Objectives
- Master monotonic stack patterns
- Implement queue using stacks
- Solve parentheses matching problems

### Study Plan (3-4 hours)

| Time Block | Activity | Details |
|------------|----------|---------|
| 0:00-0:30 | **Concept Review** | LIFO vs FIFO, stack applications |
| 0:30-2:00 | **Problem Solving** | Solve 4-5 problems |
| 2:00-2:30 | **Pattern Review** | Monotonic stack use cases |
| 2:30-3:00 | **Implementation** | Queue with two stacks |
| 3:00-3:30 | **Verbal Practice** | Explain stack state changes |

### Must-Solve Problems

| # | Problem | Difficulty | Pattern | Time Limit |
|---|---------|------------|---------|------------|
| 1 | Valid Parentheses | Easy | Stack Matching | 15 min |
| 2 | Min Stack | Medium | Auxiliary Stack | 20 min |
| 3 | Daily Temperatures | Medium | Monotonic Stack | 25 min |
| 4 | Evaluate RPN | Medium | Stack Operations | 20 min |
| 5 | Implement Queue using Stacks | Easy | Two Stacks | 20 min |

### Code Templates

```python
# Valid Parentheses
def is_valid(s):
    stack = []
    mapping = {')': '(', '}': '{', ']': '['}

    for char in s:
        if char in mapping:
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            stack.append(char)

    return len(stack) == 0

# Monotonic Stack - Next Greater Element
def next_greater(nums):
    result = [-1] * len(nums)
    stack = []  # Store indices

    for i in range(len(nums)):
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)

    return result

# Min Stack
class MinStack:
    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        min_val = min(val, self.min_stack[-1] if self.min_stack else val)
        self.min_stack.append(min_val)

    def pop(self):
        self.stack.pop()
        self.min_stack.pop()

    def top(self):
        return self.stack[-1]

    def getMin(self):
        return self.min_stack[-1]

# Queue Using Two Stacks
class MyQueue:
    def __init__(self):
        self.in_stack = []
        self.out_stack = []

    def push(self, x):
        self.in_stack.append(x)

    def pop(self):
        self._transfer()
        return self.out_stack.pop()

    def peek(self):
        self._transfer()
        return self.out_stack[-1]

    def _transfer(self):
        if not self.out_stack:
            while self.in_stack:
                self.out_stack.append(self.in_stack.pop())
```

::: info Complexity: Time O(n) · Space O(n)
- **Time:** Stack operations are O(1); monotonic stack is amortized O(n)
- **Space:** Stack stores up to n elements
:::

### Day 4 Checklist

```
[ ] Reviewed stack/queue fundamentals (15 min)
[ ] Solved Valid Parentheses
[ ] Implemented Min Stack
[ ] Solved Daily Temperatures with monotonic stack
[ ] Solved Evaluate Reverse Polish Notation
[ ] Implemented Queue using Two Stacks
[ ] Understood when to use monotonic stack
[ ] Practiced explaining stack state verbally
```

---

## Day 5: Binary Trees

### Learning Objectives
- Master all tree traversals (in/pre/post/level)
- Calculate tree properties (height, diameter)
- Validate tree structure (BST validation)

### Study Plan (3-4 hours)

| Time Block | Activity | Details |
|------------|----------|---------|
| 0:00-0:30 | **Concept Review** | Tree terminology, traversal orders |
| 0:30-2:00 | **Problem Solving** | Solve 5-6 problems |
| 2:00-2:30 | **Pattern Review** | Recursive vs iterative traversals |
| 2:30-3:00 | **Edge Cases** | Empty tree, single node, skewed trees |
| 3:00-3:30 | **Verbal Practice** | Explain recursion base cases |

### Must-Solve Problems

| # | Problem | Difficulty | Pattern | Time Limit |
|---|---------|------------|---------|------------|
| 1 | Invert Binary Tree | Easy | Recursive DFS | 10 min |
| 2 | Maximum Depth of Binary Tree | Easy | DFS/BFS | 10 min |
| 3 | Same Tree | Easy | Parallel Recursion | 10 min |
| 4 | Binary Tree Level Order Traversal | Medium | BFS | 20 min |
| 5 | Validate Binary Search Tree | Medium | Inorder/Range | 25 min |
| 6 | Lowest Common Ancestor | Medium | Recursive DFS | 25 min |

### Code Templates

```python
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

# DFS Traversals (Recursive)
def inorder(root):
    if not root:
        return []
    return inorder(root.left) + [root.val] + inorder(root.right)

def preorder(root):
    if not root:
        return []
    return [root.val] + preorder(root.left) + preorder(root.right)

def postorder(root):
    if not root:
        return []
    return postorder(root.left) + postorder(root.right) + [root.val]

# BFS Level Order
from collections import deque

def level_order(root):
    if not root:
        return []

    result = []
    queue = deque([root])

    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(node.val)
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        result.append(level)

    return result

# Maximum Depth
def max_depth(root):
    if not root:
        return 0
    return 1 + max(max_depth(root.left), max_depth(root.right))

# Validate BST
def is_valid_bst(root, min_val=float('-inf'), max_val=float('inf')):
    if not root:
        return True
    if root.val <= min_val or root.val >= max_val:
        return False
    return (is_valid_bst(root.left, min_val, root.val) and
            is_valid_bst(root.right, root.val, max_val))

# Lowest Common Ancestor
def lowest_common_ancestor(root, p, q):
    if not root or root == p or root == q:
        return root

    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left and right:
        return root
    return left or right
```

::: info Complexity: Time O(n) · Space O(h)
- **Time:** Visit each node at most once
- **Space:** Recursion depth is O(h); BFS uses O(w) queue space
:::

### Day 5 Checklist

```
[ ] Reviewed tree traversal orders (15 min)
[ ] Drew tree diagrams and traced traversals
[ ] Solved Invert Binary Tree
[ ] Solved Maximum Depth of Binary Tree
[ ] Solved Same Tree
[ ] Solved Binary Tree Level Order Traversal
[ ] Solved Validate Binary Search Tree
[ ] Solved Lowest Common Ancestor
[ ] Practiced explaining recursion verbally
```

---

## Day 6: Binary Search Trees

### Learning Objectives
- Perform BST operations (search, insert, delete)
- Utilize BST properties for optimization
- Handle BST-specific problems

### Study Plan (3-4 hours)

| Time Block | Activity | Details |
|------------|----------|---------|
| 0:00-0:30 | **Concept Review** | BST properties, AVL basics |
| 0:30-2:00 | **Problem Solving** | Solve 4-5 problems |
| 2:00-2:30 | **Pattern Review** | Inorder = sorted, range queries |
| 2:30-3:00 | **Edge Cases** | Duplicates, unbalanced trees |
| 3:00-3:30 | **Verbal Practice** | Explain BST properties |

### Must-Solve Problems

| # | Problem | Difficulty | Pattern | Time Limit |
|---|---------|------------|---------|------------|
| 1 | Search in BST | Easy | BST Search | 10 min |
| 2 | Insert into BST | Medium | BST Insert | 15 min |
| 3 | Delete Node in BST | Medium | BST Delete | 25 min |
| 4 | Kth Smallest in BST | Medium | Inorder Traversal | 20 min |
| 5 | Convert Sorted Array to BST | Easy | Divide and Conquer | 20 min |

### Code Templates

```python
# BST Search
def search_bst(root, val):
    if not root or root.val == val:
        return root
    if val < root.val:
        return search_bst(root.left, val)
    return search_bst(root.right, val)

# BST Insert
def insert_bst(root, val):
    if not root:
        return TreeNode(val)
    if val < root.val:
        root.left = insert_bst(root.left, val)
    else:
        root.right = insert_bst(root.right, val)
    return root

# BST Delete
def delete_node(root, key):
    if not root:
        return None

    if key < root.val:
        root.left = delete_node(root.left, key)
    elif key > root.val:
        root.right = delete_node(root.right, key)
    else:
        # Node to delete found
        if not root.left:
            return root.right
        if not root.right:
            return root.left

        # Two children: find inorder successor
        successor = root.right
        while successor.left:
            successor = successor.left
        root.val = successor.val
        root.right = delete_node(root.right, successor.val)

    return root

# Kth Smallest (Inorder)
def kth_smallest(root, k):
    stack = []
    curr = root
    count = 0

    while stack or curr:
        while curr:
            stack.append(curr)
            curr = curr.left

        curr = stack.pop()
        count += 1
        if count == k:
            return curr.val

        curr = curr.right

    return -1

# Sorted Array to BST
def sorted_array_to_bst(nums):
    if not nums:
        return None

    mid = len(nums) // 2
    root = TreeNode(nums[mid])
    root.left = sorted_array_to_bst(nums[:mid])
    root.right = sorted_array_to_bst(nums[mid+1:])

    return root
```

::: info Complexity: Time O(h) to O(n) · Space O(h)
- **Time:** BST operations are O(h); kth smallest and array-to-BST are O(n)
- **Space:** Recursion depth is O(h) where h = log n for balanced trees
:::

### Day 6 Checklist

```
[ ] Reviewed BST properties (15 min)
[ ] Understood BST invariant: left < root < right
[ ] Solved Search in BST
[ ] Solved Insert into BST
[ ] Solved Delete Node in BST
[ ] Solved Kth Smallest in BST
[ ] Solved Convert Sorted Array to BST
[ ] Practiced explaining BST operations verbally
```

---

## Day 7: Review + Mock Interview #1

### Learning Objectives
- Consolidate Week 1 learning
- Identify weak areas
- Complete first mock interview

### Study Plan (4 hours)

| Time Block | Activity | Details |
|------------|----------|---------|
| 0:00-0:30 | **Pattern Review** | Review all patterns from Days 1-6 |
| 0:30-1:00 | **Weak Area Focus** | Re-solve problems you struggled with |
| 1:00-1:30 | **Mock Prep** | Review mock interview format |
| 1:30-2:30 | **Mock Interview** | 45-60 min timed interview |
| 2:30-3:00 | **Self-Review** | Analyze mock performance |
| 3:00-3:30 | **Week 1 Summary** | Document learnings and areas to improve |

### Mock Interview Format

**Duration:** 45-60 minutes

**Structure:**
1. **Introduction (2-3 min)**: Brief background
2. **Problem 1 (20-25 min)**: Medium difficulty
3. **Problem 2 (20-25 min)**: Medium difficulty
4. **Questions (5 min)**: Ask interviewer questions

**Recommended Mock Problems:**

| Problem | Topics Covered | Expected Time |
|---------|----------------|---------------|
| Container With Most Water | Two Pointers | 20 min |
| LRU Cache | Hash Map + Linked List | 25 min |

### Mock Interview Evaluation Criteria

Rate yourself 1-5 on each:

```
[ ] Problem Understanding - Did I clarify requirements?
[ ] Approach Communication - Did I explain my strategy before coding?
[ ] Code Quality - Was my code clean and readable?
[ ] Testing - Did I test with examples and edge cases?
[ ] Time Management - Did I pace myself appropriately?
[ ] Debugging - Did I handle bugs calmly?
```

### Week 1 Summary Template

```markdown
## Week 1 Summary

### Problems Solved: ___/35

### Strongest Topics:
1.
2.
3.

### Areas Needing Work:
1.
2.
3.

### Patterns Mastered:
- [ ] Two Pointers
- [ ] Sliding Window
- [ ] Hash Map Lookup
- [ ] Fast/Slow Pointers
- [ ] Stack Matching
- [ ] Tree Traversals
- [ ] BST Operations

### Mock Interview Score: ___/30

### Goals for Week 2:
1.
2.
3.
```

---

## Week 1 Problem Checklist

Track your progress with this comprehensive list:

### Day 1: Arrays and Strings
- [ ] Two Sum
- [ ] Best Time to Buy and Sell Stock
- [ ] Contains Duplicate
- [ ] Maximum Subarray
- [ ] Product of Array Except Self
- [ ] 3Sum

### Day 2: Hash Tables
- [ ] Valid Anagram
- [ ] Two Sum (review)
- [ ] Group Anagrams
- [ ] Top K Frequent Elements
- [ ] Longest Consecutive Sequence
- [ ] Subarray Sum Equals K

### Day 3: Linked Lists
- [ ] Reverse Linked List
- [ ] Merge Two Sorted Lists
- [ ] Linked List Cycle
- [ ] Remove Nth Node From End
- [ ] Reorder List

### Day 4: Stacks and Queues
- [ ] Valid Parentheses
- [ ] Min Stack
- [ ] Daily Temperatures
- [ ] Evaluate RPN
- [ ] Implement Queue using Stacks

### Day 5: Binary Trees
- [ ] Invert Binary Tree
- [ ] Maximum Depth of Binary Tree
- [ ] Same Tree
- [ ] Binary Tree Level Order Traversal
- [ ] Validate Binary Search Tree
- [ ] Lowest Common Ancestor

### Day 6: Binary Search Trees
- [ ] Search in BST
- [ ] Insert into BST
- [ ] Delete Node in BST
- [ ] Kth Smallest in BST
- [ ] Convert Sorted Array to BST

### Day 7: Mock Interview
- [ ] Completed 45-min mock interview
- [ ] Reviewed mock performance
- [ ] Identified Week 2 focus areas

---

## Tips for Week 1 Success

### Time Management
- Start with the easy problem to build confidence
- If stuck for 10 minutes, look at the hint (not solution)
- If stuck for 20 minutes, review the solution and implement it yourself

### Pattern Recognition
- After each problem, ask: "What pattern did this use?"
- Group problems by pattern, not by difficulty
- Review patterns at end of each day

### Communication Practice
- Even when solving alone, speak your thoughts aloud
- Explain why you are choosing a particular approach
- Practice saying "Let me think about edge cases..."

### Energy Management
- Take 5-minute breaks every 45 minutes
- Stay hydrated
- Sleep 7-8 hours - memory consolidation happens during sleep

---

*Proceed to [Week 2: Advanced + Practice](./week2-advanced) after completing this week.*

---

*Last updated: January 2026*
