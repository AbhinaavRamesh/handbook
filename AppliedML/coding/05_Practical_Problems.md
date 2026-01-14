# Practical ML Coding Problems

> **Real-world interview problems**: User deletion, toxic text, data splitting

---

## Problem 1: Predict User App Deletion

### Problem Statement

Build a model to predict whether a user will delete/uninstall an app within the next 30 days. You have access to user activity logs, app usage metrics, and user profile data.

### Clarifying Questions

- What data is available? (Usage logs, demographics, device info?)
- What's the time horizon? (Predict 7-day, 30-day deletion?)
- Class balance? (What % of users delete?)
- Latency requirements? (Real-time prediction or batch?)
- Action if predicted to delete? (Re-engagement campaign?)

### Solution Approach

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_user_deletion_features(user_logs, user_profiles, lookback_days=30):
    """
    Feature engineering for user deletion prediction.

    Args:
        user_logs: DataFrame with columns [user_id, timestamp, event_type, ...]
        user_profiles: DataFrame with columns [user_id, signup_date, device, ...]
        lookback_days: Days of history to use for features

    Returns:
        features: DataFrame with engineered features
    """
    features = {}

    # === Engagement Features ===
    # How much they use the app recently
    features['sessions_last_7d'] = count_sessions(user_logs, days=7)
    features['sessions_last_30d'] = count_sessions(user_logs, days=30)
    features['session_trend'] = (
        features['sessions_last_7d'] / (features['sessions_last_30d'] / 4 + 1)
    )  # Compare recent vs historical

    # Time spent
    features['total_time_last_7d'] = sum_session_duration(user_logs, days=7)
    features['avg_session_duration'] = (
        features['total_time_last_7d'] / (features['sessions_last_7d'] + 1)
    )

    # === Recency Features ===
    # When did they last engage?
    features['days_since_last_session'] = compute_recency(user_logs)
    features['days_since_last_purchase'] = compute_purchase_recency(user_logs)

    # === Behavioral Features ===
    # What do they do in the app?
    features['unique_features_used'] = count_unique_features(user_logs, days=30)
    features['core_action_rate'] = compute_core_action_rate(user_logs, days=30)

    # === User Profile Features ===
    features['days_since_signup'] = compute_tenure(user_profiles)
    features['is_premium'] = user_profiles['subscription_type'].apply(
        lambda x: 1 if x == 'premium' else 0
    )

    # === Device/Context Features ===
    features['device_type'] = encode_device(user_profiles['device'])
    features['os_version'] = encode_os(user_profiles['os_version'])

    return pd.DataFrame(features)


def count_sessions(logs, days):
    """Count user sessions in last N days."""
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    recent = logs[logs['timestamp'] >= cutoff]
    return recent.groupby('user_id').size()


def compute_core_action_rate(logs, days):
    """
    Ratio of 'core' actions (purchases, shares, content creation) to total actions.
    Higher ratio = more engaged, less likely to delete.
    """
    cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
    recent = logs[logs['timestamp'] >= cutoff]

    core_actions = ['purchase', 'share', 'create', 'subscribe']
    recent['is_core'] = recent['event_type'].isin(core_actions).astype(int)

    return recent.groupby('user_id')['is_core'].mean()
```

### Key Features for Deletion Prediction

| Feature Category | Examples | Signal |
|-----------------|----------|--------|
| **Engagement Drop** | Session trend < 0.5 | Strong predictor |
| **Recency** | Days since last session > 7 | High risk |
| **Core Actions** | No purchases/shares | Less invested |
| **Tenure** | New users (< 7 days) | Higher churn |
| **Device** | Low storage devices | May delete for space |

### Model Selection

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import precision_recall_curve, average_precision_score

def train_deletion_model(X, y):
    """
    Train deletion prediction model.

    Using Gradient Boosting because:
    - Handles mixed feature types well
    - Captures non-linear relationships
    - Built-in feature importance
    - Works well with imbalanced data (with class_weight)
    """
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Handle class imbalance
    pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

    model = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        min_samples_leaf=20,  # Prevent overfitting
        random_state=42
    )

    model.fit(X_train, y_train)

    # Evaluate
    y_proba = model.predict_proba(X_val)[:, 1]
    ap_score = average_precision_score(y_val, y_proba)
    print(f"Average Precision: {ap_score:.4f}")

    return model, X_val, y_val, y_proba


def select_threshold_for_action(y_true, y_proba, target_precision=0.7):
    """
    Select probability threshold for taking action (re-engagement campaign).

    We want high precision: when we predict deletion, we want to be right.
    Low precision = wasting re-engagement budget on non-churners.
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)

    # Find threshold that achieves target precision
    for i, prec in enumerate(precisions):
        if prec >= target_precision:
            return thresholds[min(i, len(thresholds)-1)], recalls[i]

    return thresholds[-1], recalls[-1]
```

---

## Problem 2: Predict Harmful/Toxic Text

### Problem Statement

Build a text classification model to detect harmful, toxic, or abusive content in user-generated text (comments, messages, posts).

### Clarifying Questions

- What types of harm? (Hate speech, harassment, threats, spam?)
- Multi-label or multi-class? (Text can be toxic AND threat?)
- False positive tolerance? (Over-moderation vs under-moderation)
- Latency requirements? (Real-time for comments?)
- Available data? (Labeled examples, volume?)

### Solution Approach

```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re

def preprocess_text(text):
    """
    Clean and preprocess text for toxicity detection.
    """
    if not isinstance(text, str):
        return ""

    # Lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)

    # Remove special characters but keep some punctuation (can signal tone)
    text = re.sub(r'[^a-zA-Z0-9\s!?.]', '', text)

    # Remove extra whitespace
    text = ' '.join(text.split())

    return text


def extract_text_features(texts):
    """
    Extract features beyond bag-of-words for toxicity detection.
    """
    features = {}

    for i, text in enumerate(texts):
        # === Lexical Features ===
        features[i] = {
            'char_count': len(text),
            'word_count': len(text.split()),
            'avg_word_length': np.mean([len(w) for w in text.split()]) if text else 0,

            # Caps ratio (shouting)
            'caps_ratio': sum(1 for c in text if c.isupper()) / (len(text) + 1),

            # Punctuation ratio (emphasis)
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),

            # Profanity indicators (without explicit list)
            'asterisk_count': text.count('*'),  # Often used to censor

            # Repetition (emphasis/frustration)
            'repeated_chars': count_repeated_chars(text),
        }

    return pd.DataFrame(features).T


def count_repeated_chars(text):
    """Count instances of 3+ repeated characters (e.g., 'soooo')."""
    import re
    pattern = r'(.)\1{2,}'
    matches = re.findall(pattern, text.lower())
    return len(matches)


def build_toxicity_classifier(X_texts, y_labels):
    """
    Build toxicity classifier with TF-IDF and additional features.
    """
    # TF-IDF Pipeline
    tfidf_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=5,
            max_df=0.9,
            sublinear_tf=True  # Log scaling
        )),
        ('clf', LogisticRegression(
            C=1.0,
            class_weight='balanced',  # Handle imbalance
            max_iter=1000
        ))
    ])

    # Preprocess
    X_clean = [preprocess_text(text) for text in X_texts]

    # Fit
    tfidf_pipeline.fit(X_clean, y_labels)

    return tfidf_pipeline


def build_ensemble_toxicity_classifier(X_texts, y_labels):
    """
    Ensemble approach combining TF-IDF with engineered features.
    """
    from sklearn.ensemble import VotingClassifier

    # Model 1: TF-IDF Logistic Regression
    tfidf_lr = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    # Model 2: Character n-grams (catches misspellings, character patterns)
    char_ngram = Pipeline([
        ('char_tfidf', TfidfVectorizer(
            max_features=5000,
            analyzer='char',
            ngram_range=(3, 5)  # Character 3-5 grams
        )),
        ('lr', LogisticRegression(class_weight='balanced', max_iter=1000))
    ])

    # Soft voting ensemble
    ensemble = VotingClassifier(
        estimators=[
            ('tfidf_word', tfidf_lr),
            ('tfidf_char', char_ngram)
        ],
        voting='soft'
    )

    X_clean = [preprocess_text(text) for text in X_texts]
    ensemble.fit(X_clean, y_labels)

    return ensemble
```

### Production Considerations

```python
def predict_toxicity_with_threshold(model, text, threshold=0.5):
    """
    Predict with configurable threshold.

    Higher threshold = more permissive (fewer false positives)
    Lower threshold = more aggressive (fewer false negatives)
    """
    clean_text = preprocess_text(text)
    proba = model.predict_proba([clean_text])[0, 1]

    return {
        'is_toxic': proba >= threshold,
        'toxicity_score': proba,
        'action': get_action(proba)
    }


def get_action(score):
    """
    Determine moderation action based on confidence.
    """
    if score >= 0.9:
        return 'auto_remove'      # High confidence toxic
    elif score >= 0.7:
        return 'human_review'     # Medium confidence, needs review
    elif score >= 0.5:
        return 'flag_for_review'  # Low confidence, queue for review
    else:
        return 'allow'            # Likely safe
```

---

## Problem 3: Split Dataset for Training, Evaluation, Testing

### Problem Statement

Implement functions to properly split a dataset for ML training. Handle considerations like stratification, time-based splits, and group-based splits.

### Basic Random Split

```python
import numpy as np

def train_test_split_basic(X, y, test_size=0.2, random_state=None):
    """
    Basic train/test split.

    Args:
        X: Features, shape (n_samples, n_features)
        y: Labels, shape (n_samples,)
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    # Shuffle indices
    indices = np.random.permutation(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def train_val_test_split(X, y, val_size=0.1, test_size=0.1, random_state=None):
    """
    Three-way split: train, validation, test.
    """
    # First split off test
    X_temp, X_test, y_temp, y_test = train_test_split_basic(
        X, y, test_size=test_size, random_state=random_state
    )

    # Then split temp into train and val
    # Adjust val_size to account for already removed test
    adjusted_val_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split_basic(
        X_temp, y_temp, test_size=adjusted_val_size,
        random_state=random_state + 1 if random_state else None
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
```

### Stratified Split (Preserves Class Distribution)

```python
def stratified_split(X, y, test_size=0.2, random_state=None):
    """
    Stratified split that preserves class distribution.

    Critical for imbalanced datasets!
    """
    if random_state is not None:
        np.random.seed(random_state)

    # Get unique classes and their indices
    classes = np.unique(y)

    train_indices = []
    test_indices = []

    for cls in classes:
        # Get indices of this class
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)

        # Split this class proportionally
        n_test = int(len(cls_indices) * test_size)

        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])

    # Shuffle final splits
    train_indices = np.array(train_indices)
    test_indices = np.array(test_indices)
    np.random.shuffle(train_indices)
    np.random.shuffle(test_indices)

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def verify_stratification(y_train, y_test):
    """Verify class distributions match."""
    train_dist = np.bincount(y_train) / len(y_train)
    test_dist = np.bincount(y_test) / len(y_test)

    print("Train distribution:", train_dist)
    print("Test distribution:", test_dist)
    print("Max difference:", np.max(np.abs(train_dist - test_dist)))
```

### Time-Based Split (For Temporal Data)

```python
def time_based_split(X, y, timestamps, test_size=0.2):
    """
    Split based on time - train on past, test on future.

    Critical for time series and any prediction task where
    you're predicting the future.
    """
    # Sort by timestamp
    sort_idx = np.argsort(timestamps)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]
    timestamps_sorted = timestamps[sort_idx]

    # Split at time boundary
    n_samples = len(X)
    split_point = int(n_samples * (1 - test_size))

    X_train = X_sorted[:split_point]
    X_test = X_sorted[split_point:]
    y_train = y_sorted[:split_point]
    y_test = y_sorted[split_point:]

    # Report split timestamp
    split_time = timestamps_sorted[split_point]
    print(f"Split at timestamp: {split_time}")
    print(f"Train: {timestamps_sorted[0]} to {timestamps_sorted[split_point-1]}")
    print(f"Test: {timestamps_sorted[split_point]} to {timestamps_sorted[-1]}")

    return X_train, X_test, y_train, y_test


def time_series_cv_split(X, y, timestamps, n_splits=5):
    """
    Time series cross-validation with expanding window.

    Each fold uses all previous data for training.
    """
    sort_idx = np.argsort(timestamps)
    X_sorted = X[sort_idx]
    y_sorted = y[sort_idx]

    n_samples = len(X)
    fold_size = n_samples // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        # Train on everything up to fold i+1
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n_samples)

        train_idx = np.arange(train_end)
        test_idx = np.arange(test_start, test_end)

        splits.append((train_idx, test_idx))

    return splits
```

### Group-Based Split (No Data Leakage)

```python
def group_split(X, y, groups, test_size=0.2, random_state=None):
    """
    Split ensuring same group doesn't appear in both train and test.

    Use when:
    - Multiple samples per user (don't leak user info)
    - Multiple samples per session (don't leak session info)
    - Multiple samples per entity (patients, stores, etc.)
    """
    if random_state is not None:
        np.random.seed(random_state)

    unique_groups = np.unique(groups)
    np.random.shuffle(unique_groups)

    n_test_groups = int(len(unique_groups) * test_size)

    test_groups = set(unique_groups[:n_test_groups])
    train_groups = set(unique_groups[n_test_groups:])

    train_mask = np.array([g in train_groups for g in groups])
    test_mask = np.array([g in test_groups for g in groups])

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]


# Example usage
def demo_group_split():
    """Demo: user-based split."""
    # Data: multiple samples per user
    X = np.random.rand(1000, 10)
    y = np.random.randint(0, 2, 1000)
    user_ids = np.random.randint(0, 100, 1000)  # 100 users, ~10 samples each

    X_train, X_test, y_train, y_test = group_split(X, y, user_ids, test_size=0.2)

    # Verify no user appears in both
    train_users = set(user_ids[np.isin(np.arange(len(X)), np.where(y_train)[0])])
    test_users = set(user_ids[np.isin(np.arange(len(X)), np.where(y_test)[0])])

    overlap = train_users & test_users
    print(f"User overlap: {len(overlap)} (should be 0)")
```

### K-Fold Cross-Validation Implementation

```python
def k_fold_split(X, y, n_folds=5, shuffle=True, random_state=None):
    """
    Generate K-Fold cross-validation splits.

    Returns generator of (train_indices, val_indices) tuples.
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.arange(n_samples)

    if shuffle:
        np.random.shuffle(indices)

    fold_sizes = np.full(n_folds, n_samples // n_folds)
    fold_sizes[:n_samples % n_folds] += 1  # Distribute remainder

    current = 0
    for fold_size in fold_sizes:
        val_indices = indices[current:current + fold_size]
        train_indices = np.concatenate([indices[:current], indices[current + fold_size:]])
        yield train_indices, val_indices
        current += fold_size


def cross_validate(model_class, X, y, n_folds=5, **model_params):
    """
    Perform cross-validation and return scores.
    """
    scores = []

    for fold, (train_idx, val_idx) in enumerate(k_fold_split(X, y, n_folds)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = model_class(**model_params)
        model.fit(X_train, y_train)
        score = model.score(X_val, y_val)
        scores.append(score)

        print(f"Fold {fold + 1}: {score:.4f}")

    print(f"\nMean: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
    return scores
```

---

## Summary: When to Use Each Split

| Split Type | Use When |
|------------|----------|
| **Random** | IID data, no temporal/group structure |
| **Stratified** | Imbalanced classes |
| **Time-based** | Temporal data, predicting future |
| **Group-based** | Multiple samples per entity |
| **K-Fold** | Limited data, need robust estimate |

---

**Previous**: [← 04_CNN_Filter_Implementation](./04_CNN_Filter_Implementation.md) | **Next**: [06_Coding_Reference →](./06_Coding_Reference.md)
