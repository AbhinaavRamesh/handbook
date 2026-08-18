# SQL for Data Science

> **The lingua franca of data** - joins, aggregations, window functions, and interview patterns

---

## Overview

SQL is tested in almost every data science interview. You'll be given a schema and asked to write queries under time pressure. Mastering window functions and self-joins separates strong candidates from average ones.

---

## Core Operations

### JOIN Types

| Join | Returns | Use When |
|------|---------|----------|
| **INNER JOIN** | Matching rows in both tables | Only want matched records |
| **LEFT JOIN** | All left rows + matching right | Keep all from primary table |
| **RIGHT JOIN** | All right rows + matching left | Rarely used (rewrite as LEFT) |
| **FULL OUTER** | All rows from both tables | Need complete picture |
| **CROSS JOIN** | Cartesian product | Generate combinations |
| **SELF JOIN** | Table joined with itself | Hierarchical or sequential data |

```sql
-- Users who made at least one purchase
SELECT DISTINCT u.user_id, u.name
FROM users u
INNER JOIN orders o ON u.user_id = o.user_id;

-- All users with their order count (including 0)
SELECT u.user_id, u.name, COUNT(o.order_id) AS order_count
FROM users u
LEFT JOIN orders o ON u.user_id = o.user_id
GROUP BY u.user_id, u.name;
```

---

## Aggregation Patterns

### GROUP BY + HAVING

```sql
-- Customers with more than 5 orders in the last 30 days
SELECT customer_id, COUNT(*) AS order_count, SUM(amount) AS total_spent
FROM orders
WHERE order_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY customer_id
HAVING COUNT(*) > 5
ORDER BY total_spent DESC;
```

### Common Aggregation Functions

| Function | Purpose |
|----------|--------|
| `COUNT(*)` | Total rows |
| `COUNT(DISTINCT col)` | Unique values |
| `SUM(col)` | Total |
| `AVG(col)` | Average |
| `MIN(col)` / `MAX(col)` | Extremes |
| `PERCENTILE_CONT(0.5)` | Median |

---

## Window Functions

Window functions perform calculations across rows related to the current row without collapsing the result set.

### Syntax

```sql
FUNCTION() OVER (
    PARTITION BY col1
    ORDER BY col2
    ROWS BETWEEN ... AND ...
)
```

### Common Window Functions

| Function | Purpose | Example |
|----------|---------|--------|
| `ROW_NUMBER()` | Unique sequential rank | Deduplicate, top-N per group |
| `RANK()` | Rank with gaps for ties | Competition ranking |
| `DENSE_RANK()` | Rank without gaps | Sequential ranking |
| `LAG(col, n)` | Value from $n$ rows before | Compare to previous period |
| `LEAD(col, n)` | Value from $n$ rows after | Lookahead |
| `SUM() OVER()` | Running total | Cumulative revenue |
| `AVG() OVER()` | Moving average | 7-day rolling average |
| `NTILE(n)` | Divide into $n$ buckets | Percentile groups |

### Top-N Per Group Pattern

```sql
-- Top 3 products by revenue per category
WITH ranked AS (
    SELECT
        category,
        product_name,
        revenue,
        ROW_NUMBER() OVER (
            PARTITION BY category
            ORDER BY revenue DESC
        ) AS rn
    FROM products
)
SELECT category, product_name, revenue
FROM ranked
WHERE rn <= 3;
```

### Running Total

```sql
SELECT
    order_date,
    daily_revenue,
    SUM(daily_revenue) OVER (ORDER BY order_date) AS cumulative_revenue
FROM daily_sales;
```

### Period-over-Period Comparison

```sql
SELECT
    month,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY month) AS prev_month,
    revenue - LAG(revenue, 1) OVER (ORDER BY month) AS mom_change,
    ROUND(100.0 * (revenue - LAG(revenue, 1) OVER (ORDER BY month))
        / LAG(revenue, 1) OVER (ORDER BY month), 2) AS mom_pct_change
FROM monthly_revenue;
```

---

## Common Interview Patterns

### 1. Retention / Cohort Analysis

```sql
-- Day-1 retention by signup cohort
WITH first_day AS (
    SELECT user_id, MIN(DATE(event_time)) AS signup_date
    FROM events
    GROUP BY user_id
),
next_day AS (
    SELECT DISTINCT e.user_id, f.signup_date
    FROM events e
    JOIN first_day f ON e.user_id = f.user_id
    WHERE DATE(e.event_time) = f.signup_date + INTERVAL '1 day'
)
SELECT
    f.signup_date,
    COUNT(DISTINCT f.user_id) AS cohort_size,
    COUNT(DISTINCT n.user_id) AS retained,
    ROUND(100.0 * COUNT(DISTINCT n.user_id) / COUNT(DISTINCT f.user_id), 2) AS retention_pct
FROM first_day f
LEFT JOIN next_day n ON f.user_id = n.user_id AND f.signup_date = n.signup_date
GROUP BY f.signup_date
ORDER BY f.signup_date;
```

### 2. Consecutive Events

```sql
-- Users with 3+ consecutive days of activity
WITH daily_activity AS (
    SELECT DISTINCT user_id, DATE(event_time) AS activity_date
    FROM events
),
with_groups AS (
    SELECT
        user_id,
        activity_date,
        activity_date - ROW_NUMBER() OVER (
            PARTITION BY user_id ORDER BY activity_date
        ) * INTERVAL '1 day' AS grp
    FROM daily_activity
)
SELECT user_id, MIN(activity_date) AS streak_start, COUNT(*) AS streak_length
FROM with_groups
GROUP BY user_id, grp
HAVING COUNT(*) >= 3;
```

### 3. Funnel Analysis

```sql
-- Conversion funnel: view -> cart -> purchase
SELECT
    COUNT(DISTINCT CASE WHEN event = 'view' THEN user_id END) AS viewed,
    COUNT(DISTINCT CASE WHEN event = 'add_to_cart' THEN user_id END) AS carted,
    COUNT(DISTINCT CASE WHEN event = 'purchase' THEN user_id END) AS purchased,
    ROUND(100.0 *
        COUNT(DISTINCT CASE WHEN event = 'purchase' THEN user_id END) /
        COUNT(DISTINCT CASE WHEN event = 'view' THEN user_id END), 2
    ) AS overall_conversion
FROM events
WHERE event_date >= '2025-01-01';
```

---

## Performance Tips

| Technique | Why |
|-----------|-----|
| Filter early with `WHERE` | Reduce rows before joins/aggregations |
| Use `EXISTS` over `IN` for large subqueries | Short-circuits on first match |
| Avoid `SELECT *` | Only fetch needed columns |
| Index join and filter columns | Speed up lookups |
| Use CTEs for readability | Optimizer usually inlines them |

---

## Interview Questions

1. **"What's the difference between WHERE and HAVING?"**
   - `WHERE` filters rows before aggregation. `HAVING` filters groups after aggregation.

2. **"What's the difference between RANK, DENSE_RANK, and ROW_NUMBER?"**
   - Given values [100, 100, 90]: `RANK` = [1, 1, 3], `DENSE_RANK` = [1, 1, 2], `ROW_NUMBER` = [1, 2, 3].

3. **"How do you find the median in SQL?"**
   - Use `PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY col)`, or manually with `ROW_NUMBER` and filtering to the middle row(s).

4. **"How do you deduplicate rows?"**
   - `ROW_NUMBER()` partitioned by the dedup key, ordered by recency, then filter to `rn = 1`.
