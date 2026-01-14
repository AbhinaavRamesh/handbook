# Calculator, Flatten Dictionary & More

> **Practical recursion applications**

---

## Build a Calculator

### Problem Statement
Implement a basic calculator to evaluate a simple expression string with +, -, *, /, (, ).

This is a classic interview problem that appears in multiple variations on LeetCode:
- **Basic Calculator (224)** - Handles +, -, (, ) with spaces
- **Basic Calculator II** - Handles +, -, *, / without parentheses
- **Basic Calculator III** - Combines all operators with parentheses

### Approach
There are two main approaches:
1. **Stack-based**: Process characters sequentially, using stack to handle operator precedence and parentheses
2. **Recursive Descent Parser**: Build a proper expression parser following grammar rules

The stack-based approach has O(n) time complexity and O(n) space complexity as the stack size is proportional to the input length.

### Solution (Stack-based)
```python
def calculate(s: str) -> int:
    def helper(s, i):
        stack = []
        num = 0
        sign = '+'

        while i < len(s):
            char = s[i]

            if char.isdigit():
                num = num * 10 + int(char)

            if char == '(':
                num, i = helper(s, i + 1)

            if char in '+-*/)' or i == len(s) - 1:
                if sign == '+':
                    stack.append(num)
                elif sign == '-':
                    stack.append(-num)
                elif sign == '*':
                    stack.append(stack.pop() * num)
                elif sign == '/':
                    stack.append(int(stack.pop() / num))

                num = 0
                sign = char

                if char == ')':
                    return sum(stack), i

            i += 1

        return sum(stack), i

    return helper(s.replace(' ', ''), 0)[0]
```

::: info Complexity: Time O(n) · Space O(n)
- **Time:** O(n) where n is the length of the expression string - each character is processed exactly once
- **Space:** O(n) for the stack storing intermediate values, plus O(d) recursion depth where d is the maximum nesting depth of parentheses
:::

### How the Stack Solution Works

1. **Process digits**: Build multi-digit numbers character by character
2. **Handle parentheses**: Recursively process subexpressions
3. **Apply operators**:
   - For +/-, push number with sign to stack
   - For *///, pop top, compute, push result back
4. **Final result**: Sum all values in stack

```
Example: "2+3*4"
Step 1: num=2, sign='+'
Step 2: Push 2, num=0, sign='+'
Step 3: num=3, sign='+'
Step 4: Push 3, num=0, sign='*'
Step 5: num=4
Step 6: Pop 3, compute 3*4=12, push 12
Result: sum([2, 12]) = 14
```

### Recursive Descent Parser
```python
class Calculator:
    def __init__(self, s):
        self.s = s.replace(' ', '')
        self.i = 0

    def parse_expr(self):
        return self.parse_add_sub()

    def parse_add_sub(self):
        left = self.parse_mul_div()

        while self.i < len(self.s) and self.s[self.i] in '+-':
            op = self.s[self.i]
            self.i += 1
            right = self.parse_mul_div()
            left = left + right if op == '+' else left - right

        return left

    def parse_mul_div(self):
        left = self.parse_factor()

        while self.i < len(self.s) and self.s[self.i] in '*/':
            op = self.s[self.i]
            self.i += 1
            right = self.parse_factor()
            left = left * right if op == '*' else int(left / right)

        return left

    def parse_factor(self):
        if self.s[self.i] == '(':
            self.i += 1  # Skip '('
            result = self.parse_expr()
            self.i += 1  # Skip ')'
            return result

        # Parse number
        start = self.i
        while self.i < len(self.s) and self.s[self.i].isdigit():
            self.i += 1
        return int(self.s[start:self.i])

# Usage
calc = Calculator("(2+3)*4")
result = calc.parse_expr()  # Returns 20
```

::: info Complexity: Time O(n) · Space O(d)
- **Time:** O(n) where n is the length of the expression - each character is visited once during parsing
- **Space:** O(d) where d is the maximum nesting depth of parentheses (recursion stack for nested expressions)
:::

### Grammar Rules (Recursive Descent)
```
expr      -> add_sub
add_sub   -> mul_div (('+' | '-') mul_div)*
mul_div   -> factor (('*' | '/') factor)*
factor    -> '(' expr ')' | number
number    -> digit+
```

This grammar naturally enforces operator precedence: multiplication/division binds tighter than addition/subtraction.

---

## Sales Path (Tree DFS)

### Problem Statement
Given an n-ary tree where each node has a cost, find the path from root to any leaf with the minimum total cost sum.

This is a common interview question that tests understanding of tree traversal and recursion.

### Solution (Python)
```python
class Node:
    def __init__(self, cost):
        self.cost = cost
        self.children = []

def minSalesPath(root):
    if not root:
        return 0

    if not root.children:  # Leaf node
        return root.cost

    min_child_cost = float('inf')
    for child in root.children:
        min_child_cost = min(min_child_cost, minSalesPath(child))

    return root.cost + min_child_cost
```

::: info Complexity: Time O(n) · Space O(h)
- **Time:** O(n) where n is the number of nodes - each node is visited exactly once
- **Space:** O(h) where h is the height of the tree for the recursion call stack
:::

### Iterative Solution with Stack
```python
def minSalesPath_iterative(root):
    if not root:
        return 0

    min_cost = float('inf')
    stack = [(root, root.cost)]  # (node, path_cost_so_far)

    while stack:
        node, path_cost = stack.pop()

        if not node.children:  # Leaf node
            min_cost = min(min_cost, path_cost)
        else:
            for child in node.children:
                stack.append((child, path_cost + child.cost))

    return min_cost
```

::: info Complexity: Time O(n) · Space O(w)
- **Time:** O(n) where n is the number of nodes - each node is processed exactly once
- **Space:** O(w) where w is the maximum width of the tree (maximum nodes at any level stored in stack)
:::

### Visual Example
```
        0
      / | \
     5  3  6
    /  / \   \
   4  2   0   1
      |   |
      1   10

Paths:
0 -> 5 -> 4 = 9
0 -> 3 -> 2 -> 1 = 6
0 -> 3 -> 0 -> 10 = 13
0 -> 6 -> 1 = 7

Minimum: 6
```

---

## Flatten a Dictionary

### Problem Statement
Flatten a nested dictionary by joining nested keys with a delimiter (typically '.').

### Example
```
Input:  {"a": {"b": 1, "c": {"d": 2}}}
Output: {"a.b": 1, "a.c.d": 2}
```

### Solution (Python) - Recursive
```python
def flatten_dict(d, parent_key='', sep='.'):
    items = []

    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            items.extend(flatten_dict(value, new_key, sep).items())
        else:
            items.append((new_key, value))

    return dict(items)
```

::: info Complexity: Time O(n) · Space O(d)
- **Time:** O(n) where n is the total number of key-value pairs across all nesting levels
- **Space:** O(d) where d is the maximum nesting depth of the dictionary (recursion stack)
:::

### Solution (Python) - Iterative with Stack
```python
def flatten_dict_iterative(d, sep='.'):
    result = {}
    stack = [('', d)]

    while stack:
        prefix, current = stack.pop()

        for key, value in current.items():
            new_key = f"{prefix}{sep}{key}" if prefix else key

            if isinstance(value, dict):
                stack.append((new_key, value))
            else:
                result[new_key] = value

    return result
```

::: info Complexity: Time O(n) · Space O(d)
- **Time:** O(n) where n is the total number of key-value pairs including nested ones
- **Space:** O(d) where d is the maximum nesting depth (stack stores path to current level)
:::

### Handle Empty Dictionaries and Lists
```python
def flatten_dict_complete(d, parent_key='', sep='.'):
    """
    Complete implementation handling edge cases:
    - Empty nested dictionaries
    - Lists as values
    - None values
    """
    items = []

    for key, value in d.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            if value:  # Non-empty dict
                items.extend(flatten_dict_complete(value, new_key, sep).items())
            else:  # Empty dict - keep as empty dict or special value
                items.append((new_key, {}))
        elif isinstance(value, list):
            # Option 1: Keep list as-is
            items.append((new_key, value))
            # Option 2: Flatten list indices
            # for i, item in enumerate(value):
            #     items.append((f"{new_key}[{i}]", item))
        else:
            items.append((new_key, value))

    return dict(items)
```

::: info Complexity: Time O(n) · Space O(d)
- **Time:** O(n) where n is the total number of key-value pairs at all levels
- **Space:** O(d) where d is the maximum nesting depth for recursion stack
:::

### Reverse: Unflatten Dictionary
```python
def unflatten_dict(d, sep='.'):
    """Convert flat dictionary back to nested structure."""
    result = {}

    for key, value in d.items():
        parts = key.split(sep)
        current = result

        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        current[parts[-1]] = value

    return result

# Example
flat = {"a.b": 1, "a.c.d": 2}
nested = unflatten_dict(flat)
# Result: {"a": {"b": 1, "c": {"d": 2}}}
```

::: info Complexity: Time O(n * k) · Space O(n * k)
- **Time:** O(n * k) where n is the number of flat keys and k is the average key depth (splitting and traversing each key path)
- **Space:** O(n * k) for storing the nested dictionary structure
:::

---

## Implement Promise.all()

### Problem Statement
Implement Promise.all that takes an array of promises and resolves when all complete, or rejects if any fails.

### Solution (JavaScript-style in Python)
```python
import asyncio

async def promise_all(promises):
    """
    Execute all promises concurrently and return results in order.
    Raises first exception if any promise fails.
    """
    return await asyncio.gather(*promises)

# With error handling
async def promise_all_settled(promises):
    """
    Execute all promises and return all results (including errors).
    Similar to JavaScript's Promise.allSettled()
    """
    results = []

    async def safe_await(promise, index):
        try:
            result = await promise
            return {'status': 'fulfilled', 'value': result, 'index': index}
        except Exception as e:
            return {'status': 'rejected', 'reason': str(e), 'index': index}

    tasks = [safe_await(p, i) for i, p in enumerate(promises)]
    results = await asyncio.gather(*tasks)

    # Sort by original index to maintain order
    return sorted(results, key=lambda x: x['index'])

# Sequential version (for comparison)
async def promise_sequence(promises):
    results = []
    for promise in promises:
        results.append(await promise)
    return results
```

::: info Complexity: Time O(max(t_i)) · Space O(n)
- **Time:** O(max(t_i)) for promise_all where t_i is the time for each promise (parallel execution limited by slowest); O(sum(t_i)) for sequential version
- **Space:** O(n) for storing results array where n is the number of promises
:::

### JavaScript Implementation
```javascript
function promiseAll(promises) {
    return new Promise((resolve, reject) => {
        if (!Array.isArray(promises)) {
            return reject(new TypeError('Argument must be an array'));
        }

        const results = [];
        let completed = 0;

        if (promises.length === 0) {
            return resolve(results);
        }

        promises.forEach((promise, index) => {
            Promise.resolve(promise)
                .then(value => {
                    results[index] = value;
                    completed++;

                    if (completed === promises.length) {
                        resolve(results);
                    }
                })
                .catch(reject);  // Reject immediately on first error
        });
    });
}
```

::: info Complexity: Time O(max(t_i)) · Space O(n)
- **Time:** O(max(t_i)) where t_i is execution time of each promise - all promises run concurrently
- **Space:** O(n) for storing n results in the results array
:::

### Concurrent Execution Visualization
```
Promise.all([p1, p2, p3]):

Time -->
p1: |========|          (2s)
p2: |====|              (1s)
p3: |============|      (3s)
    ---------------------
Total: 3s (parallel, limited by slowest)

Sequential:
p1: |========|
p2:          |====|
p3:               |============|
    -------------------------------
Total: 6s (sum of all)
```

### Related Promise Methods
```python
import asyncio

async def promise_race(promises):
    """Return result of first promise to complete (success or failure)."""
    done, pending = await asyncio.wait(
        promises,
        return_when=asyncio.FIRST_COMPLETED
    )

    # Cancel pending tasks
    for task in pending:
        task.cancel()

    return done.pop().result()

async def promise_any(promises):
    """Return first successful result, fail only if all fail."""
    errors = []

    async def try_promise(p):
        try:
            return await p
        except Exception as e:
            errors.append(e)
            raise

    tasks = [asyncio.create_task(try_promise(p)) for p in promises]

    while tasks:
        done, tasks = await asyncio.wait(
            tasks,
            return_when=asyncio.FIRST_COMPLETED
        )

        for task in done:
            if not task.exception():
                # Cancel remaining
                for t in tasks:
                    t.cancel()
                return task.result()

    raise AggregateError(errors)
```

::: info Complexity: Time O(max(t_i)) · Space O(n)
- **Time:** O(max(t_i)) for promise_race (returns first completed); O(max(t_i)) for promise_any (returns first successful)
- **Space:** O(n) for tracking task states and potentially storing all errors
:::

---

## Interview Applications

### Calculator Problems
- **Expression parsing**: Common in system design and language implementation questions
- **Operator precedence**: Understanding grammar and parsing techniques
- **Stack manipulation**: Demonstrates understanding of data structures

### Sales Path / Tree DFS
- **Cost optimization**: Finding minimum/maximum paths in trees
- **N-ary tree traversal**: Not just binary trees
- **Real-world modeling**: Organization hierarchies, file systems

### Flatten Dictionary
- **Data transformation**: Common in data processing pipelines
- **JSON/API handling**: Flattening nested API responses
- **Configuration management**: Handling nested config files

### Promise.all Implementation
- **Concurrency patterns**: Understanding parallel vs sequential execution
- **Error handling**: Aggregating results and failures
- **System design**: Building robust async systems

### Key Interview Tips

1. **Ask clarifying questions**:
   - Calculator: What operators? Negative numbers? Invalid input?
   - Flatten: Handle empty dicts? Lists? Circular references?

2. **Start with examples**:
   - Trace through your algorithm with the interviewer
   - Handle edge cases explicitly

3. **Discuss trade-offs**:
   - Recursive vs iterative approaches
   - Time vs space complexity

4. **Consider production concerns**:
   - Input validation
   - Error handling
   - Performance at scale

---

## References

- [Basic Calculator - LeetCode](https://leetcode.com/problems/basic-calculator/)
- [Solving Basic Calculator I, II, III on LeetCode](https://medium.com/@CalvinChankf/solving-basic-calculator-i-ii-iii-on-leetcode-74d926732437)
- [224. Basic Calculator - In-Depth Explanation](https://algo.monster/liteproblems/224)
- [How to Flatten a Dictionary in Python - freeCodeCamp](https://www.freecodecamp.org/news/how-to-flatten-a-dictionary-in-python-in-4-different-ways/)
- [Python - Convert Nested Dictionary into Flattened Dictionary - GeeksforGeeks](https://www.geeksforgeeks.org/python/python-convert-nested-dictionary-into-flattened-dictionary/)
- [Flatten Nested Dictionary using Recursive Function](https://medium.com/@huasa0115/python-how-to-flatten-a-nested-dictionary-with-recursive-function-069ed277adce)
- [flatten-dict PyPI Package](https://pypi.org/project/flatten-dict/)
