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

::: code-group

```python [Python]
from typing import Dict, List, Any
from datetime import datetime, timedelta
from collections import defaultdict

def prepare_user_deletion_features(user_logs: List[Dict], user_profiles: List[Dict],
                                    lookback_days: int = 30) -> Dict[str, Dict[str, float]]:
    """
    Feature engineering for user deletion prediction.

    Args:
        user_logs: List of log dicts with keys [user_id, timestamp, event_type, ...]
        user_profiles: List of profile dicts with keys [user_id, signup_date, device, ...]
        lookback_days: Days of history to use for features

    Returns:
        features: Dict mapping user_id to feature dict
    """
    features: Dict[str, Dict[str, float]] = {}
    now = datetime.now()

    # Group logs by user
    user_log_groups: Dict[str, List[Dict]] = defaultdict(list)
    for log in user_logs:
        user_log_groups[log['user_id']].append(log)

    # Build profile lookup
    profile_lookup = {p['user_id']: p for p in user_profiles}

    for user_id, logs in user_log_groups.items():
        profile = profile_lookup.get(user_id, {})

        # === Engagement Features ===
        sessions_7d = count_sessions(logs, now, days=7)
        sessions_30d = count_sessions(logs, now, days=30)

        features[user_id] = {
            'sessions_last_7d': sessions_7d,
            'sessions_last_30d': sessions_30d,
            'session_trend': sessions_7d / (sessions_30d / 4 + 1),

            # === Recency Features ===
            'days_since_last_session': compute_recency(logs, now),

            # === Behavioral Features ===
            'unique_features_used': count_unique_features(logs, now, days=30),
            'core_action_rate': compute_core_action_rate(logs, now, days=30),

            # === User Profile Features ===
            'days_since_signup': compute_tenure(profile, now),
            'is_premium': 1 if profile.get('subscription_type') == 'premium' else 0,
        }

    return features


def count_sessions(logs: List[Dict], now: datetime, days: int) -> int:
    """Count user sessions in last N days."""
    cutoff = now - timedelta(days=days)
    return sum(1 for log in logs
               if log.get('event_type') == 'session_start'
               and log['timestamp'] >= cutoff)


def compute_recency(logs: List[Dict], now: datetime) -> float:
    """Days since last session."""
    session_logs = [log for log in logs if log.get('event_type') == 'session_start']
    if not session_logs:
        return 999.0  # Very long time
    last_session = max(log['timestamp'] for log in session_logs)
    return (now - last_session).days


def count_unique_features(logs: List[Dict], now: datetime, days: int) -> int:
    """Count unique features used in last N days."""
    cutoff = now - timedelta(days=days)
    features_used = set()
    for log in logs:
        if log['timestamp'] >= cutoff and 'feature_name' in log:
            features_used.add(log['feature_name'])
    return len(features_used)


def compute_core_action_rate(logs: List[Dict], now: datetime, days: int) -> float:
    """Ratio of 'core' actions to total actions."""
    cutoff = now - timedelta(days=days)
    recent = [log for log in logs if log['timestamp'] >= cutoff]
    if not recent:
        return 0.0

    core_actions = ['purchase', 'share', 'create', 'subscribe']
    core_count = sum(1 for log in recent if log.get('event_type') in core_actions)
    return core_count / len(recent)


def compute_tenure(profile: Dict, now: datetime) -> float:
    """Days since signup."""
    signup_date = profile.get('signup_date')
    if signup_date is None:
        return 0.0
    return (now - signup_date).days
```

```python [NumPy]
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

```python [PyTorch]
import torch
import torch.nn as nn
from typing import Dict, List

class UserDeletionFeatureExtractor:
    """
    Feature extractor for user deletion prediction using PyTorch tensors.
    """
    def __init__(self, lookback_days: int = 30):
        self.lookback_days = lookback_days

    def extract_features(self, user_logs: torch.Tensor,
                         user_profiles: torch.Tensor) -> torch.Tensor:
        """
        Extract features from user data.

        Args:
            user_logs: Tensor of shape (n_logs, n_log_features)
                       Features: [user_id, timestamp, event_type_encoded, ...]
            user_profiles: Tensor of shape (n_users, n_profile_features)
                          Features: [user_id, signup_timestamp, is_premium, ...]

        Returns:
            features: Tensor of shape (n_users, n_features)
        """
        n_users = user_profiles.shape[0]
        feature_list = []

        for user_idx in range(n_users):
            user_id = user_profiles[user_idx, 0]

            # Filter logs for this user
            user_mask = user_logs[:, 0] == user_id
            user_log_data = user_logs[user_mask]

            # Compute features
            sessions_7d = self._count_recent_sessions(user_log_data, days=7)
            sessions_30d = self._count_recent_sessions(user_log_data, days=30)
            session_trend = sessions_7d / (sessions_30d / 4 + 1)

            recency = self._compute_recency(user_log_data)
            core_action_rate = self._compute_core_action_rate(user_log_data)

            # Profile features
            tenure = user_profiles[user_idx, 1]  # days since signup
            is_premium = user_profiles[user_idx, 2]

            user_features = torch.tensor([
                sessions_7d, sessions_30d, session_trend,
                recency, core_action_rate, tenure, is_premium
            ])
            feature_list.append(user_features)

        return torch.stack(feature_list)

    def _count_recent_sessions(self, logs: torch.Tensor, days: int) -> float:
        """Count sessions in last N days."""
        if logs.shape[0] == 0:
            return 0.0
        # Assuming column 1 is timestamp (days ago), column 2 is event type
        recent_mask = logs[:, 1] <= days
        session_mask = logs[:, 2] == 0  # 0 = session_start
        return float(torch.sum(recent_mask & session_mask).item())

    def _compute_recency(self, logs: torch.Tensor) -> float:
        """Days since last session."""
        if logs.shape[0] == 0:
            return 999.0
        session_mask = logs[:, 2] == 0
        session_logs = logs[session_mask]
        if session_logs.shape[0] == 0:
            return 999.0
        return float(torch.min(session_logs[:, 1]).item())

    def _compute_core_action_rate(self, logs: torch.Tensor) -> float:
        """Ratio of core actions to total."""
        if logs.shape[0] == 0:
            return 0.0
        # Core actions encoded as 1, 2, 3, 4
        core_mask = (logs[:, 2] >= 1) & (logs[:, 2] <= 4)
        return float(torch.mean(core_mask.float()).item())
```

:::

### Key Features for Deletion Prediction

| Feature Category | Examples | Signal |
|-----------------|----------|--------|
| **Engagement Drop** | Session trend < 0.5 | Strong predictor |
| **Recency** | Days since last session > 7 | High risk |
| **Core Actions** | No purchases/shares | Less invested |
| **Tenure** | New users (< 7 days) | Higher churn |
| **Device** | Low storage devices | May delete for space |

### Model Selection

::: code-group

```python [Python]
from typing import List, Tuple, Dict
import math

def train_deletion_model_simple(X: List[List[float]], y: List[int],
                                test_size: float = 0.2) -> Dict:
    """
    Simple logistic regression for deletion prediction.
    """
    # Manual train/test split
    n = len(X)
    n_test = int(n * test_size)
    indices = list(range(n))

    # Shuffle (simple implementation)
    import random
    random.shuffle(indices)

    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    # Train simple logistic regression
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0
    lr = 0.01

    for _ in range(1000):
        for xi, yi in zip(X_train, y_train):
            # Forward pass
            z = sum(w * x for w, x in zip(weights, xi)) + bias
            pred = 1 / (1 + math.exp(-z))

            # Gradient update
            error = pred - yi
            for j in range(n_features):
                weights[j] -= lr * error * xi[j]
            bias -= lr * error

    # Evaluate
    correct = 0
    for xi, yi in zip(X_test, y_test):
        z = sum(w * x for w, x in zip(weights, xi)) + bias
        pred = 1 if z > 0 else 0
        if pred == yi:
            correct += 1

    return {
        'weights': weights,
        'bias': bias,
        'accuracy': correct / len(X_test)
    }
```

```python [NumPy]
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
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

```python [PyTorch]
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class DeletionPredictor(nn.Module):
    """Neural network for user deletion prediction."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = [64, 32]):
        super().__init__()
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def train_deletion_model_pytorch(X: torch.Tensor, y: torch.Tensor,
                                  epochs: int = 100, batch_size: int = 32):
    """
    Train deletion prediction model using PyTorch.
    """
    # Split data
    n = len(X)
    n_val = int(n * 0.2)
    perm = torch.randperm(n)

    X_train, y_train = X[perm[n_val:]], y[perm[n_val:]]
    X_val, y_val = X[perm[:n_val]], y[perm[:n_val]]

    # Handle class imbalance
    pos_count = y_train.sum().item()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / (pos_count + 1e-8)])

    # Model
    model = DeletionPredictor(X.shape[1])
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_x).squeeze()
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_preds = model(X_val).squeeze()
                val_loss = criterion(val_preds, y_val)
                accuracy = ((val_preds > 0.5) == y_val).float().mean()
                print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, Acc={accuracy:.4f}")

    return model
```

:::

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

::: code-group

```python [Python]
from typing import List, Dict
import re
from collections import Counter
import math

def preprocess_text(text: str) -> str:
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


def extract_text_features(texts: List[str]) -> List[Dict[str, float]]:
    """
    Extract features beyond bag-of-words for toxicity detection.
    """
    features = []

    for text in texts:
        words = text.split() if text else []
        char_count = len(text)

        feature_dict = {
            'char_count': char_count,
            'word_count': len(words),
            'avg_word_length': sum(len(w) for w in words) / (len(words) + 1),

            # Caps ratio (shouting)
            'caps_ratio': sum(1 for c in text if c.isupper()) / (char_count + 1),

            # Punctuation ratio (emphasis)
            'exclamation_count': text.count('!'),
            'question_count': text.count('?'),

            # Profanity indicators (without explicit list)
            'asterisk_count': text.count('*'),  # Often used to censor

            # Repetition (emphasis/frustration)
            'repeated_chars': count_repeated_chars(text),
        }
        features.append(feature_dict)

    return features


def count_repeated_chars(text: str) -> int:
    """Count instances of 3+ repeated characters (e.g., 'soooo')."""
    pattern = r'(.)\1{2,}'
    matches = re.findall(pattern, text.lower())
    return len(matches)


def build_vocabulary(texts: List[str], max_vocab: int = 10000) -> Dict[str, int]:
    """Build vocabulary from texts."""
    word_counts: Counter = Counter()
    for text in texts:
        words = preprocess_text(text).split()
        word_counts.update(words)

    vocab = {'<UNK>': 0, '<PAD>': 1}
    for word, _ in word_counts.most_common(max_vocab - 2):
        vocab[word] = len(vocab)

    return vocab


def text_to_bow(text: str, vocab: Dict[str, int]) -> List[float]:
    """Convert text to bag-of-words vector."""
    words = preprocess_text(text).split()
    bow = [0.0] * len(vocab)

    for word in words:
        idx = vocab.get(word, vocab['<UNK>'])
        bow[idx] += 1

    # Normalize
    total = sum(bow) + 1e-8
    return [x / total for x in bow]


def train_toxicity_classifier_simple(texts: List[str], labels: List[int],
                                      max_vocab: int = 5000) -> Dict:
    """
    Simple toxicity classifier using bag-of-words and logistic regression.
    """
    # Build vocabulary
    vocab = build_vocabulary(texts, max_vocab)

    # Convert to features
    X = [text_to_bow(text, vocab) for text in texts]

    # Train logistic regression (simplified)
    n_features = len(vocab)
    weights = [0.0] * n_features
    bias = 0.0
    lr = 0.1

    for _ in range(100):
        for xi, yi in zip(X, labels):
            z = sum(w * x for w, x in zip(weights, xi)) + bias
            pred = 1 / (1 + math.exp(-max(-500, min(500, z))))
            error = pred - yi
            for j in range(n_features):
                weights[j] -= lr * error * xi[j]
            bias -= lr * error

    return {'weights': weights, 'bias': bias, 'vocab': vocab}
```

```python [NumPy]
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import re
import pandas as pd

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

```python [PyTorch]
import torch
import torch.nn as nn
from typing import List, Dict
import re

class ToxicityClassifier(nn.Module):
    """LSTM-based toxicity classifier."""

    def __init__(self, vocab_size: int, embedding_dim: int = 128,
                 hidden_dim: int = 256, num_layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers,
                           batch_first=True, bidirectional=True, dropout=0.3)
        self.fc = nn.Linear(hidden_dim * 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, (hidden, _) = self.lstm(embedded)

        # Use last hidden states from both directions
        hidden_cat = torch.cat((hidden[-2], hidden[-1]), dim=1)
        output = self.fc(hidden_cat)
        return self.sigmoid(output).squeeze()


def preprocess_text(text: str) -> str:
    """Clean and preprocess text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-zA-Z0-9\s!?.]', '', text)
    return ' '.join(text.split())


def build_vocabulary(texts: List[str], max_vocab: int = 10000) -> Dict[str, int]:
    """Build vocabulary from texts."""
    from collections import Counter
    word_counts: Counter = Counter()
    for text in texts:
        words = preprocess_text(text).split()
        word_counts.update(words)

    vocab = {'<PAD>': 0, '<UNK>': 1}
    for word, _ in word_counts.most_common(max_vocab - 2):
        vocab[word] = len(vocab)

    return vocab


def texts_to_tensor(texts: List[str], vocab: Dict[str, int],
                    max_len: int = 100) -> torch.Tensor:
    """Convert texts to padded tensor."""
    sequences = []
    for text in texts:
        words = preprocess_text(text).split()[:max_len]
        seq = [vocab.get(w, vocab['<UNK>']) for w in words]
        # Pad
        seq = seq + [vocab['<PAD>']] * (max_len - len(seq))
        sequences.append(seq)

    return torch.tensor(sequences)


def train_toxicity_model_pytorch(texts: List[str], labels: List[int],
                                  epochs: int = 10, batch_size: int = 32):
    """Train toxicity classifier using PyTorch."""
    # Build vocabulary
    vocab = build_vocabulary(texts)

    # Convert to tensors
    X = texts_to_tensor(texts, vocab)
    y = torch.tensor(labels, dtype=torch.float)

    # Split
    n = len(X)
    n_val = int(n * 0.2)
    perm = torch.randperm(n)

    X_train, y_train = X[perm[n_val:]], y[perm[n_val:]]
    X_val, y_val = X[perm[:n_val]], y[perm[:n_val]]

    # Model
    model = ToxicityClassifier(len(vocab))
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    for epoch in range(epochs):
        model.train()
        for i in range(0, len(X_train), batch_size):
            batch_x = X_train[i:i+batch_size]
            batch_y = y_train[i:i+batch_size]

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val)
            accuracy = ((val_preds > 0.5) == y_val).float().mean()
            print(f"Epoch {epoch+1}: Val Loss={val_loss:.4f}, Acc={accuracy:.4f}")

    return model, vocab
```

:::

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

::: code-group

```python [Python]
from typing import List, Tuple, TypeVar
import random

T = TypeVar('T')

def train_test_split_basic(X: List[T], y: List[T], test_size: float = 0.2,
                           random_state: int = None) -> Tuple[List[T], List[T], List[T], List[T]]:
    """
    Basic train/test split.

    Args:
        X: Features
        y: Labels
        test_size: Fraction for test set
        random_state: Random seed

    Returns:
        X_train, X_test, y_train, y_test
    """
    if random_state is not None:
        random.seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    # Shuffle indices
    indices = list(range(n_samples))
    random.shuffle(indices)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test


def train_val_test_split(X: List[T], y: List[T], val_size: float = 0.1,
                         test_size: float = 0.1, random_state: int = None):
    """
    Three-way split: train, validation, test.
    """
    # First split off test
    X_temp, X_test, y_temp, y_test = train_test_split_basic(
        X, y, test_size=test_size, random_state=random_state
    )

    # Then split temp into train and val
    adjusted_val_size = val_size / (1 - test_size)
    seed2 = random_state + 1 if random_state else None
    X_train, X_val, y_train, y_val = train_test_split_basic(
        X_temp, y_temp, test_size=adjusted_val_size, random_state=seed2
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
```

```python [NumPy]
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

```python [PyTorch]
import torch
from typing import Tuple

def train_test_split_basic(X: torch.Tensor, y: torch.Tensor, test_size: float = 0.2,
                           random_state: int = None) -> Tuple[torch.Tensor, torch.Tensor,
                                                               torch.Tensor, torch.Tensor]:
    """
    Basic train/test split.
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    n_samples = len(X)
    n_test = int(n_samples * test_size)

    # Shuffle indices
    indices = torch.randperm(n_samples)

    test_indices = indices[:n_test]
    train_indices = indices[n_test:]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]


def train_val_test_split(X: torch.Tensor, y: torch.Tensor, val_size: float = 0.1,
                         test_size: float = 0.1, random_state: int = None):
    """
    Three-way split: train, validation, test.
    """
    # First split off test
    X_temp, X_test, y_temp, y_test = train_test_split_basic(
        X, y, test_size=test_size, random_state=random_state
    )

    # Then split temp into train and val
    adjusted_val_size = val_size / (1 - test_size)
    seed2 = random_state + 1 if random_state else None
    X_train, X_val, y_train, y_val = train_test_split_basic(
        X_temp, y_temp, test_size=adjusted_val_size, random_state=seed2
    )

    return X_train, X_val, X_test, y_train, y_val, y_test
```

:::

### Stratified Split (Preserves Class Distribution)

::: code-group

```python [Python]
from typing import List, Tuple, TypeVar
import random
from collections import defaultdict

T = TypeVar('T')

def stratified_split(X: List[T], y: List[int], test_size: float = 0.2,
                     random_state: int = None) -> Tuple[List[T], List[T], List[int], List[int]]:
    """
    Stratified split that preserves class distribution.

    Critical for imbalanced datasets!
    """
    if random_state is not None:
        random.seed(random_state)

    # Group indices by class
    class_indices: dict = defaultdict(list)
    for i, label in enumerate(y):
        class_indices[label].append(i)

    train_indices = []
    test_indices = []

    for cls, indices in class_indices.items():
        random.shuffle(indices)

        # Split this class proportionally
        n_test = int(len(indices) * test_size)

        test_indices.extend(indices[:n_test])
        train_indices.extend(indices[n_test:])

    # Shuffle final splits
    random.shuffle(train_indices)
    random.shuffle(test_indices)

    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]

    return X_train, X_test, y_train, y_test
```

```python [NumPy]
import numpy as np

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

```python [PyTorch]
import torch
from typing import Tuple

def stratified_split(X: torch.Tensor, y: torch.Tensor, test_size: float = 0.2,
                     random_state: int = None) -> Tuple[torch.Tensor, torch.Tensor,
                                                         torch.Tensor, torch.Tensor]:
    """
    Stratified split that preserves class distribution.
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    # Get unique classes
    classes = torch.unique(y)

    train_indices = []
    test_indices = []

    for cls in classes:
        # Get indices of this class
        cls_indices = torch.where(y == cls)[0]
        # Shuffle
        perm = torch.randperm(len(cls_indices))
        cls_indices = cls_indices[perm]

        # Split this class proportionally
        n_test = int(len(cls_indices) * test_size)

        test_indices.append(cls_indices[:n_test])
        train_indices.append(cls_indices[n_test:])

    # Concatenate and shuffle
    train_indices = torch.cat(train_indices)
    test_indices = torch.cat(test_indices)
    train_indices = train_indices[torch.randperm(len(train_indices))]
    test_indices = test_indices[torch.randperm(len(test_indices))]

    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]
```

:::

### Time-Based Split (For Temporal Data)

::: code-group

```python [Python]
from typing import List, Tuple, TypeVar

T = TypeVar('T')

def time_based_split(X: List[T], y: List[T], timestamps: List[float],
                     test_size: float = 0.2) -> Tuple[List[T], List[T], List[T], List[T]]:
    """
    Split based on time - train on past, test on future.

    Critical for time series and any prediction task where
    you're predicting the future.
    """
    # Sort by timestamp
    sorted_data = sorted(zip(timestamps, X, y))
    timestamps_sorted = [t for t, _, _ in sorted_data]
    X_sorted = [x for _, x, _ in sorted_data]
    y_sorted = [y for _, _, y in sorted_data]

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

    return X_train, X_test, y_train, y_test
```

```python [NumPy]
import numpy as np

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

```python [PyTorch]
import torch
from typing import Tuple, List

def time_based_split(X: torch.Tensor, y: torch.Tensor, timestamps: torch.Tensor,
                     test_size: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor,
                                                       torch.Tensor, torch.Tensor]:
    """
    Split based on time - train on past, test on future.
    """
    # Sort by timestamp
    sort_idx = torch.argsort(timestamps)
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
    split_time = timestamps_sorted[split_point].item()
    print(f"Split at timestamp: {split_time}")

    return X_train, X_test, y_train, y_test


def time_series_cv_split(X: torch.Tensor, y: torch.Tensor, timestamps: torch.Tensor,
                         n_splits: int = 5) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Time series cross-validation with expanding window.
    """
    sort_idx = torch.argsort(timestamps)

    n_samples = len(X)
    fold_size = n_samples // (n_splits + 1)

    splits = []
    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n_samples)

        train_idx = sort_idx[:train_end]
        test_idx = sort_idx[test_start:test_end]

        splits.append((train_idx, test_idx))

    return splits
```

:::

### Group-Based Split (No Data Leakage)

::: code-group

```python [Python]
from typing import List, Tuple, TypeVar
import random

T = TypeVar('T')

def group_split(X: List[T], y: List[T], groups: List[int], test_size: float = 0.2,
                random_state: int = None) -> Tuple[List[T], List[T], List[T], List[T]]:
    """
    Split ensuring same group doesn't appear in both train and test.

    Use when:
    - Multiple samples per user (don't leak user info)
    - Multiple samples per session (don't leak session info)
    - Multiple samples per entity (patients, stores, etc.)
    """
    if random_state is not None:
        random.seed(random_state)

    unique_groups = list(set(groups))
    random.shuffle(unique_groups)

    n_test_groups = int(len(unique_groups) * test_size)

    test_groups = set(unique_groups[:n_test_groups])
    train_groups = set(unique_groups[n_test_groups:])

    X_train, X_test, y_train, y_test = [], [], [], []
    for i in range(len(X)):
        if groups[i] in train_groups:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])

    return X_train, X_test, y_train, y_test
```

```python [NumPy]
import numpy as np

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

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
```

```python [PyTorch]
import torch
from typing import Tuple

def group_split(X: torch.Tensor, y: torch.Tensor, groups: torch.Tensor,
                test_size: float = 0.2, random_state: int = None) -> Tuple[torch.Tensor,
                                                                           torch.Tensor,
                                                                           torch.Tensor,
                                                                           torch.Tensor]:
    """
    Split ensuring same group doesn't appear in both train and test.
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    unique_groups = torch.unique(groups)
    perm = torch.randperm(len(unique_groups))
    unique_groups = unique_groups[perm]

    n_test_groups = int(len(unique_groups) * test_size)

    test_groups = set(unique_groups[:n_test_groups].tolist())
    train_groups = set(unique_groups[n_test_groups:].tolist())

    train_mask = torch.tensor([g.item() in train_groups for g in groups])
    test_mask = torch.tensor([g.item() in test_groups for g in groups])

    return X[train_mask], X[test_mask], y[train_mask], y[test_mask]
```

:::

### K-Fold Cross-Validation Implementation

::: code-group

```python [Python]
from typing import List, Tuple, Generator, TypeVar
import random

T = TypeVar('T')

def k_fold_split(X: List[T], y: List[T], n_folds: int = 5, shuffle: bool = True,
                 random_state: int = None) -> Generator[Tuple[List[int], List[int]], None, None]:
    """
    Generate K-Fold cross-validation splits.

    Returns generator of (train_indices, val_indices) tuples.
    """
    if random_state is not None:
        random.seed(random_state)

    n_samples = len(X)
    indices = list(range(n_samples))

    if shuffle:
        random.shuffle(indices)

    fold_sizes = [n_samples // n_folds] * n_folds
    for i in range(n_samples % n_folds):
        fold_sizes[i] += 1

    current = 0
    for fold_size in fold_sizes:
        val_indices = indices[current:current + fold_size]
        train_indices = indices[:current] + indices[current + fold_size:]
        yield train_indices, val_indices
        current += fold_size
```

```python [NumPy]
import numpy as np

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

```python [PyTorch]
import torch
from typing import Generator, Tuple

def k_fold_split(X: torch.Tensor, y: torch.Tensor, n_folds: int = 5,
                 shuffle: bool = True, random_state: int = None) -> Generator:
    """
    Generate K-Fold cross-validation splits.
    """
    if random_state is not None:
        torch.manual_seed(random_state)

    n_samples = len(X)
    indices = torch.arange(n_samples)

    if shuffle:
        indices = indices[torch.randperm(n_samples)]

    fold_size = n_samples // n_folds
    remainder = n_samples % n_folds

    current = 0
    for fold in range(n_folds):
        this_fold_size = fold_size + (1 if fold < remainder else 0)
        val_indices = indices[current:current + this_fold_size]
        train_indices = torch.cat([indices[:current], indices[current + this_fold_size:]])
        yield train_indices, val_indices
        current += this_fold_size
```

:::

---

## Summary: When to Use Each Split

| Split Type | Use When |
|------------|----------|
| **Random** | IID data, no temporal/group structure |
| **Stratified** | Imbalanced classes |
| **Time-based** | Temporal data, predicting future |
| **Group-based** | Multiple samples per entity |
| **K-Fold** | Limited data, need robust estimate |
