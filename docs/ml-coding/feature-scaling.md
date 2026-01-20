---
title: Feature Scaling Implementation
description: Implement StandardScaler, MinMaxScaler, and other normalization techniques from scratch.
---

# Feature Scaling Implementation

> **Essential preprocessing for ML algorithms**

Feature scaling transforms features to similar ranges, critical for distance-based algorithms (KNN, SVM, K-Means) and gradient descent optimization. This guide implements scalers from scratch with the fit/transform pattern.

## When to Use Each Scaler

| Scaler | Best For | Handles Outliers | Range | Preserves Zero |
|--------|----------|------------------|-------|----------------|
| StandardScaler | Normally distributed data | No | Unbounded | No |
| MinMaxScaler | Bounded ranges needed | No | [0, 1] | No |
| RobustScaler | Data with outliers | Yes | Unbounded | No |
| MaxAbsScaler | Sparse data | Somewhat | [-1, 1] | Yes |
| L1 Normalizer | Text/frequency data | No | Unit sum | No |
| L2 Normalizer | Cosine similarity | No | Unit norm | No |

## Base Scaler Class

```python
import numpy as np
from typing import Optional, Union, List
from abc import ABC, abstractmethod

class BaseScaler(ABC):
    """
    Abstract base class for all scalers.
    Implements the fit/transform pattern used in scikit-learn.
    """

    def __init__(self):
        self.is_fitted = False

    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseScaler':
        """Compute statistics from training data."""
        pass

    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using computed statistics."""
        pass

    @abstractmethod
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Reverse the transformation."""
        pass

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)

    def _validate_input(self, X: np.ndarray) -> np.ndarray:
        """Convert input to 2D numpy array."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        return X

    def _check_is_fitted(self):
        """Verify the scaler has been fitted."""
        if not self.is_fitted:
            raise RuntimeError("Scaler must be fitted before transform")
```

## StandardScaler (Z-Score Normalization)

Transforms features to have zero mean and unit variance: `z = (x - mean) / std`

```python
class StandardScaler(BaseScaler):
    """
    Standardize features by removing mean and scaling to unit variance.

    Formula: z = (x - μ) / σ

    Best for:
    - Normally distributed features
    - Algorithms assuming zero-centered data (PCA, SVM)
    - Gradient descent optimization

    Limitations:
    - Sensitive to outliers (they affect mean and std)
    - Does not bound the output range

    Time Complexity: O(n * d) for fit, O(n * d) for transform
    Space Complexity: O(d) for storing statistics
    """

    def __init__(self, with_mean: bool = True, with_std: bool = True):
        """
        Args:
            with_mean: If True, center data by removing mean
            with_std: If True, scale data to unit variance
        """
        super().__init__()
        self.with_mean = with_mean
        self.with_std = with_std
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self.var_: Optional[np.ndarray] = None
        self.n_samples_seen_: int = 0

    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute mean and standard deviation from training data.

        Args:
            X: Training data of shape (n_samples, n_features)

        Returns:
            self: Fitted scaler
        """
        X = self._validate_input(X)

        self.n_samples_seen_ = X.shape[0]

        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])

        if self.with_std:
            self.var_ = np.var(X, axis=0)
            # Avoid division by zero for constant features
            self.std_ = np.sqrt(self.var_)
            self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        else:
            self.std_ = np.ones(X.shape[1])
            self.var_ = np.ones(X.shape[1])

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize data using fitted statistics.

        Args:
            X: Data to transform of shape (n_samples, n_features)

        Returns:
            X_scaled: Standardized data
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        return (X - self.mean_) / self.std_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse standardization to original scale.

        Args:
            X: Standardized data

        Returns:
            X_original: Data in original scale
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        return X * self.std_ + self.mean_

    def partial_fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Incrementally update statistics with new data (for streaming).
        Uses Welford's online algorithm for numerical stability.

        Args:
            X: New batch of data

        Returns:
            self: Updated scaler
        """
        X = self._validate_input(X)

        if not self.is_fitted:
            return self.fit(X)

        # Welford's online algorithm for mean and variance
        n_new = X.shape[0]
        new_mean = np.mean(X, axis=0)
        new_var = np.var(X, axis=0)

        n_total = self.n_samples_seen_ + n_new

        # Combined mean
        delta = new_mean - self.mean_
        combined_mean = self.mean_ + delta * n_new / n_total

        # Combined variance (parallel algorithm)
        combined_var = (
            (self.n_samples_seen_ * self.var_ + n_new * new_var) / n_total +
            (self.n_samples_seen_ * n_new * delta ** 2) / (n_total ** 2)
        )

        self.mean_ = combined_mean
        self.var_ = combined_var
        self.std_ = np.sqrt(self.var_)
        self.std_ = np.where(self.std_ == 0, 1.0, self.std_)
        self.n_samples_seen_ = n_total

        return self


# Example usage
print("=== StandardScaler Demo ===")
X_train = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]])
X_test = np.array([[2, 3, 4], [8, 9, 10]])

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"Training mean: {scaler.mean_}")
print(f"Training std: {scaler.std_}")
print(f"Scaled train:\n{X_train_scaled}")
print(f"Scaled test:\n{X_test_scaled}")
print(f"Inverse:\n{scaler.inverse_transform(X_train_scaled)}")
```

## MinMaxScaler

Scales features to a fixed range, typically [0, 1].

```python
class MinMaxScaler(BaseScaler):
    """
    Scale features to a given range [min, max].

    Formula: X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min

    Best for:
    - Neural networks with bounded activation functions
    - Image pixel values
    - Features that need bounded range
    - Algorithms sensitive to magnitude (KNN)

    Limitations:
    - Very sensitive to outliers
    - New data may fall outside [0, 1] range

    Time Complexity: O(n * d) for fit, O(n * d) for transform
    Space Complexity: O(d) for storing min/max
    """

    def __init__(self, feature_range: tuple = (0, 1)):
        """
        Args:
            feature_range: Desired range of transformed data (min, max)
        """
        super().__init__()
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.max_: Optional[np.ndarray] = None
        self.data_range_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Compute min and max from training data.
        """
        X = self._validate_input(X)

        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        self.data_range_ = self.max_ - self.min_

        # Handle constant features (range = 0)
        self.data_range_ = np.where(self.data_range_ == 0, 1.0, self.data_range_)

        # Precompute scale factor
        range_min, range_max = self.feature_range
        self.scale_ = (range_max - range_min) / self.data_range_

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features to the configured range.
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        range_min, range_max = self.feature_range

        # Scale to [0, 1] first, then to desired range
        X_std = (X - self.min_) / self.data_range_
        X_scaled = X_std * (range_max - range_min) + range_min

        return X_scaled

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse scaling to original range.
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        range_min, range_max = self.feature_range

        X_std = (X - range_min) / (range_max - range_min)
        X_original = X_std * self.data_range_ + self.min_

        return X_original

    def clip(self, X: np.ndarray) -> np.ndarray:
        """
        Clip transformed values to feature_range.
        Useful for new data that falls outside training range.
        """
        self._check_is_fitted()
        X_scaled = self.transform(X)
        return np.clip(X_scaled, self.feature_range[0], self.feature_range[1])


# Example usage
print("\n=== MinMaxScaler Demo ===")
X = np.array([[1, 10], [2, 20], [3, 30], [4, 40]])

scaler = MinMaxScaler(feature_range=(0, 1))
X_scaled = scaler.fit_transform(X)

print(f"Original:\n{X}")
print(f"Scaled [0,1]:\n{X_scaled}")
print(f"Data range: {scaler.data_range_}")

# Custom range example
scaler_custom = MinMaxScaler(feature_range=(-1, 1))
X_scaled_custom = scaler_custom.fit_transform(X)
print(f"Scaled [-1,1]:\n{X_scaled_custom}")
```

## RobustScaler

Uses statistics robust to outliers (median and IQR).

```python
class RobustScaler(BaseScaler):
    """
    Scale features using statistics robust to outliers.

    Formula: X_scaled = (X - median) / IQR

    Uses median instead of mean and interquartile range (IQR)
    instead of standard deviation.

    Best for:
    - Data with significant outliers
    - Skewed distributions
    - When you can't remove outliers

    Properties:
    - Outliers don't affect scaling as much
    - Output is unbounded
    - Median is at 0, middle 50% of data in [-0.5, 0.5]

    Time Complexity: O(n * d * log(n)) for fit (due to median)
    Space Complexity: O(d) for storing statistics
    """

    def __init__(self, with_centering: bool = True, with_scaling: bool = True,
                 quantile_range: tuple = (25.0, 75.0)):
        """
        Args:
            with_centering: If True, center data at median
            with_scaling: If True, scale by IQR
            quantile_range: Quantile range for IQR (default 25th to 75th)
        """
        super().__init__()
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        self.center_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'RobustScaler':
        """
        Compute median and IQR from training data.
        """
        X = self._validate_input(X)

        q_min, q_max = self.quantile_range

        if self.with_centering:
            self.center_ = np.median(X, axis=0)
        else:
            self.center_ = np.zeros(X.shape[1])

        if self.with_scaling:
            q_lower = np.percentile(X, q_min, axis=0)
            q_upper = np.percentile(X, q_max, axis=0)
            self.scale_ = q_upper - q_lower
            # Handle zero IQR
            self.scale_ = np.where(self.scale_ == 0, 1.0, self.scale_)
        else:
            self.scale_ = np.ones(X.shape[1])

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale features using median and IQR.
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        return (X - self.center_) / self.scale_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse robust scaling.
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        return X * self.scale_ + self.center_


# Example with outliers
print("\n=== RobustScaler Demo ===")
X_with_outliers = np.array([[1], [2], [3], [4], [5], [100]])  # 100 is outlier

standard = StandardScaler()
robust = RobustScaler()

X_standard = standard.fit_transform(X_with_outliers)
X_robust = robust.fit_transform(X_with_outliers)

print(f"Original: {X_with_outliers.flatten()}")
print(f"StandardScaler: {X_standard.flatten().round(2)}")
print(f"RobustScaler: {X_robust.flatten().round(2)}")
print(f"Standard mean: {standard.mean_[0]:.2f}, std: {standard.std_[0]:.2f}")
print(f"Robust median: {robust.center_[0]:.2f}, IQR: {robust.scale_[0]:.2f}")
```

## MaxAbsScaler

Scales by maximum absolute value, preserving sparsity.

```python
class MaxAbsScaler(BaseScaler):
    """
    Scale features by their maximum absolute value.

    Formula: X_scaled = X / max(|X|)

    Output range: [-1, 1]

    Best for:
    - Sparse data (preserves zeros)
    - Data already centered at zero
    - When you need bounded output without shifting

    Properties:
    - Preserves sparsity (zero stays zero)
    - Does not shift/center data
    - Output bounded to [-1, 1]

    Time Complexity: O(n * d)
    Space Complexity: O(d)
    """

    def __init__(self):
        super().__init__()
        self.max_abs_: Optional[np.ndarray] = None

    def fit(self, X: np.ndarray) -> 'MaxAbsScaler':
        """
        Compute maximum absolute value per feature.
        """
        X = self._validate_input(X)

        self.max_abs_ = np.max(np.abs(X), axis=0)
        # Handle all-zero features
        self.max_abs_ = np.where(self.max_abs_ == 0, 1.0, self.max_abs_)

        self.is_fitted = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale by maximum absolute value.
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        return X / self.max_abs_

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Reverse max-abs scaling.
        """
        self._check_is_fitted()
        X = self._validate_input(X)

        return X * self.max_abs_


# Example with sparse-like data
print("\n=== MaxAbsScaler Demo ===")
X_sparse = np.array([[0, 5, 0], [3, 0, -2], [0, 0, 4], [-1, 2, 0]])

scaler = MaxAbsScaler()
X_scaled = scaler.fit_transform(X_sparse)

print(f"Original:\n{X_sparse}")
print(f"MaxAbs scaled:\n{X_scaled}")
print(f"Max abs values: {scaler.max_abs_}")
print(f"Zeros preserved: {np.sum(X_sparse == 0) == np.sum(X_scaled == 0)}")
```

## L1 and L2 Normalizers

Normalize samples individually to unit norm.

```python
class Normalizer:
    """
    Normalize samples individually to unit norm.

    Unlike scalers that work on features (columns), normalizers work on
    samples (rows), making each sample have unit norm.

    L1 norm: sum(|x_i|) = 1 (values sum to 1 in absolute terms)
    L2 norm: sqrt(sum(x_i^2)) = 1 (Euclidean length of 1)
    Max norm: max(|x_i|) = 1

    Best for:
    - Text data (TF-IDF, word counts)
    - When magnitude doesn't matter, only direction
    - Cosine similarity calculations (L2)
    - Probability distributions (L1)

    Note: This is stateless - no fit required, but we implement
    fit for API consistency.

    Time Complexity: O(n * d)
    Space Complexity: O(1) - stateless
    """

    def __init__(self, norm: str = 'l2'):
        """
        Args:
            norm: Type of norm ('l1', 'l2', or 'max')
        """
        if norm not in ('l1', 'l2', 'max'):
            raise ValueError("norm must be 'l1', 'l2', or 'max'")
        self.norm = norm

    def fit(self, X: np.ndarray) -> 'Normalizer':
        """
        No-op for API consistency. Normalizer is stateless.
        """
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize each sample to unit norm.

        Args:
            X: Data of shape (n_samples, n_features)

        Returns:
            X_normalized: Normalized data where each row has unit norm
        """
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        if self.norm == 'l1':
            norms = np.sum(np.abs(X), axis=1, keepdims=True)
        elif self.norm == 'l2':
            norms = np.sqrt(np.sum(X ** 2, axis=1, keepdims=True))
        else:  # max
            norms = np.max(np.abs(X), axis=1, keepdims=True)

        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)

        return X / norms

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform (fit is no-op)."""
        return self.fit(X).transform(X)


def compute_norm(X: np.ndarray, norm: str = 'l2') -> np.ndarray:
    """
    Compute norm of each sample.

    Args:
        X: Data array
        norm: 'l1', 'l2', or 'max'

    Returns:
        norms: Norm of each row
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)

    if norm == 'l1':
        return np.sum(np.abs(X), axis=1)
    elif norm == 'l2':
        return np.sqrt(np.sum(X ** 2, axis=1))
    elif norm == 'max':
        return np.max(np.abs(X), axis=1)
    else:
        raise ValueError("norm must be 'l1', 'l2', or 'max'")


# Example usage
print("\n=== Normalizer Demo ===")
X = np.array([[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]])

l1_norm = Normalizer(norm='l1')
l2_norm = Normalizer(norm='l2')
max_norm = Normalizer(norm='max')

X_l1 = l1_norm.fit_transform(X)
X_l2 = l2_norm.fit_transform(X)
X_max = max_norm.fit_transform(X)

print(f"Original:\n{X}")
print(f"\nL1 normalized (rows sum to 1):\n{X_l1.round(3)}")
print(f"L1 row sums: {np.sum(np.abs(X_l1), axis=1).round(3)}")

print(f"\nL2 normalized (Euclidean norm = 1):\n{X_l2.round(3)}")
print(f"L2 row norms: {compute_norm(X_l2, 'l2').round(3)}")

print(f"\nMax normalized (max abs = 1):\n{X_max.round(3)}")
print(f"Max values: {np.max(np.abs(X_max), axis=1).round(3)}")
```

## Unified Scaler Interface

A factory pattern for easy scaler selection.

```python
class FeatureScaler:
    """
    Unified interface for all scaling methods.

    Provides a consistent API and handles the common pattern of:
    1. Fit on training data
    2. Transform training data
    3. Transform test/new data using training statistics
    """

    SCALERS = {
        'standard': StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler,
        'maxabs': MaxAbsScaler,
    }

    NORMALIZERS = {
        'l1': lambda: Normalizer(norm='l1'),
        'l2': lambda: Normalizer(norm='l2'),
        'max': lambda: Normalizer(norm='max'),
    }

    def __init__(self, method: str = 'standard', **kwargs):
        """
        Args:
            method: Scaling method name
            **kwargs: Additional arguments for the scaler
        """
        method = method.lower()

        if method in self.SCALERS:
            self.scaler = self.SCALERS[method](**kwargs)
            self.is_normalizer = False
        elif method in self.NORMALIZERS:
            self.scaler = self.NORMALIZERS[method]()
            self.is_normalizer = True
        else:
            raise ValueError(f"Unknown method: {method}. "
                           f"Available: {list(self.SCALERS.keys()) + list(self.NORMALIZERS.keys())}")

        self.method = method

    def fit(self, X: np.ndarray) -> 'FeatureScaler':
        self.scaler.fit(X)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.scaler.fit_transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        if self.is_normalizer:
            raise NotImplementedError("Normalizers don't support inverse_transform")
        return self.scaler.inverse_transform(X)

    @staticmethod
    def recommend(X: np.ndarray, check_outliers: bool = True) -> str:
        """
        Recommend a scaling method based on data characteristics.

        Args:
            X: Input data
            check_outliers: Whether to check for outliers

        Returns:
            Recommended scaler name
        """
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        # Check for sparsity
        sparsity = np.mean(X == 0)
        if sparsity > 0.5:
            return 'maxabs'  # Preserves zeros

        # Check for outliers using IQR method
        if check_outliers:
            q1 = np.percentile(X, 25, axis=0)
            q3 = np.percentile(X, 75, axis=0)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            outlier_mask = (X < lower) | (X > upper)
            outlier_ratio = np.mean(outlier_mask)

            if outlier_ratio > 0.05:  # More than 5% outliers
                return 'robust'

        # Check normality (simplified check using skewness)
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        std = np.where(std == 0, 1, std)
        skewness = np.mean(((X - mean) / std) ** 3, axis=0)

        if np.all(np.abs(skewness) < 0.5):
            return 'standard'  # Approximately normal

        return 'minmax'  # Default for bounded range


# Demo of unified interface
print("\n=== Unified FeatureScaler Demo ===")
X = np.random.randn(100, 3) * 10 + 5

for method in ['standard', 'minmax', 'robust', 'maxabs', 'l2']:
    scaler = FeatureScaler(method=method)
    X_scaled = scaler.fit_transform(X)
    print(f"{method:10s}: mean={X_scaled.mean():.3f}, std={X_scaled.std():.3f}, "
          f"min={X_scaled.min():.3f}, max={X_scaled.max():.3f}")

# Recommendation
print(f"\nRecommended scaler: {FeatureScaler.recommend(X)}")
```

## Handling New Data Correctly

Critical pattern for ML pipelines.

```python
class ScalingPipeline:
    """
    Demonstrates correct handling of train/test data scaling.

    CRITICAL: Always fit on training data only, then transform both
    train and test using the same statistics.

    Common mistake: Fitting on all data causes data leakage.
    """

    def __init__(self, scaler: BaseScaler):
        self.scaler = scaler
        self.train_stats = {}

    def fit_on_train(self, X_train: np.ndarray) -> np.ndarray:
        """
        Fit scaler on training data and transform it.

        Args:
            X_train: Training features

        Returns:
            X_train_scaled: Scaled training features
        """
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Store statistics for reference
        if hasattr(self.scaler, 'mean_'):
            self.train_stats['mean'] = self.scaler.mean_.copy()
        if hasattr(self.scaler, 'std_'):
            self.train_stats['std'] = self.scaler.std_.copy()
        if hasattr(self.scaler, 'min_'):
            self.train_stats['min'] = self.scaler.min_.copy()
        if hasattr(self.scaler, 'max_'):
            self.train_stats['max'] = self.scaler.max_.copy()

        return X_train_scaled

    def transform_test(self, X_test: np.ndarray) -> np.ndarray:
        """
        Transform test data using training statistics.

        Args:
            X_test: Test features

        Returns:
            X_test_scaled: Scaled test features (using train stats)
        """
        return self.scaler.transform(X_test)

    def transform_new_sample(self, x: np.ndarray) -> np.ndarray:
        """
        Transform a single new sample for prediction.

        Args:
            x: Single sample (1D or 2D array)

        Returns:
            x_scaled: Scaled sample
        """
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
        return self.scaler.transform(x)


# Demonstrate correct vs incorrect scaling
print("\n=== Correct Train/Test Scaling ===")
np.random.seed(42)

# Simulate train/test split
X_train = np.random.randn(80, 2) * 10 + 50
X_test = np.random.randn(20, 2) * 10 + 50

print("Train set - mean:", X_train.mean(axis=0).round(2))
print("Test set - mean:", X_test.mean(axis=0).round(2))

# Correct way
pipeline = ScalingPipeline(StandardScaler())
X_train_scaled = pipeline.fit_on_train(X_train)
X_test_scaled = pipeline.transform_test(X_test)

print("\nCorrect scaling (fit on train only):")
print(f"  Train scaled mean: {X_train_scaled.mean(axis=0).round(4)}")
print(f"  Test scaled mean: {X_test_scaled.mean(axis=0).round(4)}")

# Wrong way (for comparison)
wrong_scaler = StandardScaler()
X_all = np.vstack([X_train, X_test])
X_all_scaled = wrong_scaler.fit_transform(X_all)  # DATA LEAKAGE!
X_train_wrong = X_all_scaled[:80]
X_test_wrong = X_all_scaled[80:]

print("\nWrong scaling (fit on all data - DATA LEAKAGE!):")
print(f"  Train scaled mean: {X_train_wrong.mean(axis=0).round(4)}")
print(f"  Test scaled mean: {X_test_wrong.mean(axis=0).round(4)}")
```

## Comparison and Visualization

```python
def compare_scalers(X: np.ndarray, feature_names: list = None):
    """
    Compare all scalers on the same dataset.

    Args:
        X: Input data
        feature_names: Optional feature names
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    n_features = X.shape[1]
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(n_features)]

    scalers = {
        'Original': None,
        'StandardScaler': StandardScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'RobustScaler': RobustScaler(),
        'MaxAbsScaler': MaxAbsScaler(),
    }

    print("=" * 70)
    print("SCALER COMPARISON")
    print("=" * 70)

    results = {}
    for name, scaler in scalers.items():
        if scaler is None:
            X_scaled = X
        else:
            X_scaled = scaler.fit_transform(X)

        results[name] = X_scaled

        print(f"\n{name}:")
        print(f"  Mean:  {X_scaled.mean(axis=0).round(3)}")
        print(f"  Std:   {X_scaled.std(axis=0).round(3)}")
        print(f"  Min:   {X_scaled.min(axis=0).round(3)}")
        print(f"  Max:   {X_scaled.max(axis=0).round(3)}")

    return results


# Example comparison
print("\n=== Scaler Comparison ===")
# Create data with different characteristics
np.random.seed(42)
X_comparison = np.column_stack([
    np.random.randn(100) * 10 + 100,  # Normal, large values
    np.random.exponential(5, 100),      # Skewed
    np.concatenate([np.random.randn(95), [100, 150, -50, -80, 200]])  # With outliers
])

compare_scalers(X_comparison, ['Normal', 'Skewed', 'Outliers'])
```

## Interview Tips

```python
"""
INTERVIEW DISCUSSION POINTS

1. Why is feature scaling important?
   - Distance-based algorithms (KNN, K-Means, SVM) are sensitive to scale
   - Gradient descent converges faster with scaled features
   - Regularization penalizes features equally after scaling
   - Neural networks train better with normalized inputs

2. When NOT to scale?
   - Tree-based models (Random Forest, XGBoost) - split decisions are scale-invariant
   - When feature magnitude has meaning (e.g., counts)
   - Naive Bayes (works with probabilities)

3. Fit/Transform Pattern - Why?
   - Prevents data leakage from test set
   - Simulates real-world deployment where future data stats are unknown
   - Must use training statistics for all future transformations

4. StandardScaler vs MinMaxScaler?
   - StandardScaler: unbounded output, better for normally distributed data
   - MinMaxScaler: bounded [0,1], better for neural networks, sensitive to outliers

5. How to handle outliers?
   - Use RobustScaler (median/IQR instead of mean/std)
   - Or clip outliers before scaling
   - Or use quantile transformation

6. Sparse data considerations?
   - Use MaxAbsScaler to preserve zeros
   - StandardScaler would destroy sparsity (subtracting mean)

7. Streaming/Online learning?
   - Use partial_fit() to update statistics incrementally
   - Welford's algorithm for numerical stability

8. Common mistakes:
   - Fitting on entire dataset (train + test)
   - Scaling target variable for regression (usually not needed)
   - Forgetting to save scaler for production
   - Not handling new categories in categorical data
"""


# Quick reference implementation for interviews
def quick_standard_scale(X_train, X_test):
    """Minimal StandardScaler for interviews."""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1  # Handle constant features

    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std  # Use TRAIN statistics!

    return X_train_scaled, X_test_scaled, mean, std


def quick_minmax_scale(X_train, X_test, feature_range=(0, 1)):
    """Minimal MinMaxScaler for interviews."""
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = X_max - X_min
    X_range[X_range == 0] = 1

    scale = feature_range[1] - feature_range[0]

    X_train_scaled = (X_train - X_min) / X_range * scale + feature_range[0]
    X_test_scaled = (X_test - X_min) / X_range * scale + feature_range[0]

    return X_train_scaled, X_test_scaled


# Time complexity summary
print("""
TIME COMPLEXITY SUMMARY:
-----------------------
StandardScaler:  O(n*d) fit, O(n*d) transform
MinMaxScaler:    O(n*d) fit, O(n*d) transform
RobustScaler:    O(n*d*log(n)) fit (due to percentile), O(n*d) transform
MaxAbsScaler:    O(n*d) fit, O(n*d) transform
Normalizer:      O(n*d) transform (stateless)

Where n = samples, d = features
""")
```

## Complete Test Suite

```python
def run_all_tests():
    """Comprehensive tests for all scalers."""
    print("\n" + "=" * 70)
    print("RUNNING ALL TESTS")
    print("=" * 70)

    np.random.seed(42)

    # Test data
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    X_new = np.array([[2, 3], [6, 7]])

    # Test 1: StandardScaler
    print("\n[TEST] StandardScaler")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_inverse = scaler.inverse_transform(X_scaled)
    assert np.allclose(X_scaled.mean(axis=0), 0), "Mean should be 0"
    assert np.allclose(X_scaled.std(axis=0), 1), "Std should be 1"
    assert np.allclose(X_inverse, X), "Inverse should recover original"
    print("  PASSED: mean=0, std=1, inverse works")

    # Test 2: MinMaxScaler
    print("\n[TEST] MinMaxScaler")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    assert np.allclose(X_scaled.min(axis=0), 0), "Min should be 0"
    assert np.allclose(X_scaled.max(axis=0), 1), "Max should be 1"
    X_inverse = scaler.inverse_transform(X_scaled)
    assert np.allclose(X_inverse, X), "Inverse should recover original"
    print("  PASSED: range [0,1], inverse works")

    # Test 3: RobustScaler
    print("\n[TEST] RobustScaler")
    X_outliers = np.array([[1], [2], [3], [4], [100]])
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X_outliers)
    assert scaler.center_[0] == 3, "Median should be 3"
    X_inverse = scaler.inverse_transform(X_scaled)
    assert np.allclose(X_inverse, X_outliers), "Inverse should recover original"
    print("  PASSED: uses median, inverse works")

    # Test 4: MaxAbsScaler preserves zeros
    print("\n[TEST] MaxAbsScaler")
    X_sparse = np.array([[0, 5], [3, 0], [0, -2]])
    scaler = MaxAbsScaler()
    X_scaled = scaler.fit_transform(X_sparse)
    assert np.sum(X_sparse == 0) == np.sum(X_scaled == 0), "Zeros preserved"
    assert X_scaled.max() <= 1 and X_scaled.min() >= -1, "Range [-1,1]"
    print("  PASSED: zeros preserved, range [-1,1]")

    # Test 5: Normalizer L2
    print("\n[TEST] Normalizer L2")
    normalizer = Normalizer(norm='l2')
    X_norm = normalizer.fit_transform(X)
    norms = np.sqrt(np.sum(X_norm ** 2, axis=1))
    assert np.allclose(norms, 1), "L2 norms should be 1"
    print("  PASSED: L2 norm = 1 for all rows")

    # Test 6: Normalizer L1
    print("\n[TEST] Normalizer L1")
    normalizer = Normalizer(norm='l1')
    X_norm = normalizer.fit_transform(X)
    sums = np.sum(np.abs(X_norm), axis=1)
    assert np.allclose(sums, 1), "L1 norms should be 1"
    print("  PASSED: L1 norm = 1 for all rows")

    # Test 7: Train/test consistency
    print("\n[TEST] Train/test consistency")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X)
    X_new_scaled = scaler.transform(X_new)
    # Transform new data point that equals mean of training
    X_mean = X.mean(axis=0).reshape(1, -1)
    X_mean_scaled = scaler.transform(X_mean)
    assert np.allclose(X_mean_scaled, 0), "Mean of training should map to 0"
    print("  PASSED: training mean maps to 0")

    # Test 8: Constant feature handling
    print("\n[TEST] Constant feature handling")
    X_const = np.array([[1, 5], [1, 6], [1, 7]])
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_const)
    assert not np.any(np.isnan(X_scaled)), "No NaN for constant features"
    assert not np.any(np.isinf(X_scaled)), "No Inf for constant features"
    print("  PASSED: constant features handled")

    print("\n" + "=" * 70)
    print("ALL TESTS PASSED!")
    print("=" * 70)


# Run tests
run_all_tests()
```

## Summary

Feature scaling is essential for many ML algorithms. Key takeaways:

1. **StandardScaler**: Use for normally distributed data, gradient descent
2. **MinMaxScaler**: Use for bounded ranges, neural networks
3. **RobustScaler**: Use when outliers are present
4. **MaxAbsScaler**: Use for sparse data
5. **Normalizer**: Use for text data, cosine similarity

Always fit on training data only and use those statistics for all transformations!
