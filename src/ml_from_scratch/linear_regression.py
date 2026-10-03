"""Linear regression and the gradient-descent machinery it is trained with.

Everything here is written against NumPy alone. The exploratory notebook this
module came from
(``notebooks/05-algorithms-from-scratch/linear-models/02-batch-gd-early-stopping-ridge.ipynb``)
had three defects that made it unrunnable, all corrected here:

* ``normalize`` mutated its input arrays in place, so calling ``predict`` after
  ``fit`` silently corrupted the caller's data. The scaler below never mutates
  its arguments.
* ``un_norm`` inverted the x-axis with ``(max + min) + min`` instead of
  ``(max - min) + min``, which does not undo the forward transform.
* The early-stopping loop referenced an undefined name ``sel.patience`` and
  compared a per-sample error array against a scalar tolerance.

The notebook was also titled "mini batch" while computing a full-batch update;
the batch size is now a real parameter.

Example:
    >>> import numpy as np
    >>> from ml_from_scratch.linear_regression import LinearRegression
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(500, 1))
    >>> y = 3.0 * X[:, 0] - 2.0 + rng.normal(scale=0.1, size=500)
    >>> model = LinearRegression(n_epochs=2000, learning_rate=0.2,
    ...                          random_state=0)
    >>> _ = model.fit(X, y)
    >>> bool(abs(model.coef_[0] - 3.0) < 0.1)
    True
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["MinMaxScaler", "LinearRegression"]


class MinMaxScaler:
    """Scale features onto ``[0, 1]`` using observed extrema.

    Kept deliberately minimal so that the forward and inverse transforms are
    visible in one place. Unlike ``sklearn.preprocessing.MinMaxScaler`` this
    class never writes to its input: every method returns a new array. That
    matters here, because the original notebook's in-place mutation meant a
    second call to :meth:`transform` operated on already-rescaled data.

    Attributes:
        data_min_ (numpy.ndarray): Per-feature minimum seen during :meth:`fit`,
            shape ``(n_features,)``.
        data_max_ (numpy.ndarray): Per-feature maximum seen during :meth:`fit`,
            shape ``(n_features,)``.
    """

    def __init__(self, feature_range: tuple[float, float] = (0.0, 1.0)) -> None:
        """Initialise the scaler.

        Args:
            feature_range: Inclusive ``(low, high)`` target interval. Defaults to
                ``(0.0, 1.0)``.

        Raises:
            ValueError: If ``feature_range`` does not have exactly two entries
                or if ``low >= high``.
        """
        if len(feature_range) != 2:
            raise ValueError(
                f"feature_range must be a (low, high) pair, got {feature_range}."
            )
        low, high = feature_range
        if low >= high:
            raise ValueError(
                f"feature_range low must be below high, got {low} >= {high}."
            )
        self.feature_range = (float(low), float(high))
        self.data_min_ = None
        self.data_max_ = None

    def fit(self, X: np.ndarray) -> "MinMaxScaler":
        """Record the per-feature extrema.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            self: The fitted scaler, to allow chaining.

        Raises:
            ValueError: If ``X`` is not two-dimensional or is empty.
        """
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit a scaler on an empty array.")
        self.data_min_ = X.min(axis=0)
        self.data_max_ = X.max(axis=0)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Scale a matrix onto the fitted target interval.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Newly allocated scaled copy of ``X``, with the same
            shape. The input is not modified.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``X`` has the wrong number of columns, or if a
                feature is constant in the training data (zero range) and so
                cannot be rescaled.
        """
        if self.data_min_ is None:
            raise RuntimeError("MinMaxScaler has not been fitted yet; call fit first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.data_min_.size:
            raise ValueError(
                f"X must be 2-D with {self.data_min_.size} features, got shape {X.shape}."
            )
        span = self.data_max_ - self.data_min_
        if np.any(span == 0):
            const = np.flatnonzero(span == 0).tolist()
            raise ValueError(
                f"Feature(s) {const} are constant in the training data and cannot "
                "be min-max scaled."
            )

        low, high = self.feature_range
        scaled = (X - self.data_min_) / span
        return low + scaled * (high - low)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit on ``X`` and return the scaled result in one call.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Newly allocated scaled copy of ``X``.

        Raises:
            ValueError: If ``X`` is not two-dimensional or is empty.
        """
        return self.fit(X).transform(X)

    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Map a scaled matrix back to the original data range.

        Args:
            X: Scaled matrix of shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Newly allocated copy of ``X`` on the original scale.
            The input is not modified.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``X`` has the wrong number of columns.

        Note:
            This is the exact inverse of :meth:`transform`:
            ``x = (scaled - low) * span / (high - low) + data_min_``.
        """
        if self.data_min_ is None:
            raise RuntimeError("MinMaxScaler has not been fitted yet; call fit first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.data_min_.size:
            raise ValueError(
                f"X must be 2-D with {self.data_min_.size} features, got shape {X.shape}."
            )
        low, high = self.feature_range
        span = self.data_max_ - self.data_min_
        return (X - low) * span / (high - low) + self.data_min_


@dataclass
class _Split:
    """Preprocessed training/validation arrays for one fit.

    Attributes:
        X_train: Feature matrix for gradient updates.
        y_train: Targets aligned with ``X_train``, shape ``(n_train, 1)``.
        X_val: Held-out feature matrix, or ``None`` when no split was made.
        y_val: Held-out targets, or ``None``.
    """

    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray | None
    y_val: np.ndarray | None


class LinearRegression:
    """Ordinary least squares fitted with gradient descent.

    The objective is mean squared error plus an optional regularisation term:

    .. math::

        J(\\mathbf{w}) = \\frac{1}{n} \\sum_{i} (\\hat{y}_i - y_i)^2
            + \\alpha \\left( r \\lVert \\mathbf{w} \\rVert_2^2
            + \\frac{1 - r}{2} \\lVert \\mathbf{w} \\rVert_1 \\right)

    where :math:`r` interpolates between L2 (ridge, ``r=1``) and L1 (lasso,
    ``r=0``) penalties. Setting ``alpha=0`` recovers plain OLS.

    Features are min-max scaled before optimisation and predictions are mapped
    back to the original scale. Scaling matters here: the gradient step size is
    a single scalar, so badly scaled features make convergence slow or unstable.

    Attributes:
        coef_ (numpy.ndarray): Learned coefficients, shape ``(n_features,)``.
        intercept_ (float): Learned intercept on the original target scale.
        n_iter_ (int): Epochs executed.
        loss_curve_ (list[float]): Training loss recorded at each epoch.
        val_loss_curve_ (list[float]): Validation loss per epoch; empty when
            ``val_size`` is zero.
        stopped_early_ (bool): Whether early stopping halted training before
            ``n_epochs`` was exhausted.

    Example:
        >>> import numpy as np
        >>> from ml_from_scratch.linear_regression import LinearRegression
        >>> rng = np.random.default_rng(1)
        >>> X = rng.uniform(size=(200, 3))
        >>> y = X @ np.array([1.0, -2.0, 0.5]) + 0.3
        >>> model = LinearRegression(n_epochs=500, learning_rate=0.1, alpha=0.01,
        ...                          random_state=1)
        >>> _ = model.fit(X, y)
        >>> preds = model.predict(X)
        >>> bool(np.max(np.abs(preds - y)) < 0.5)
        True
    """

    def __init__(
        self,
        n_epochs: int = 1000,
        learning_rate: float = 0.05,
        alpha: float = 0.0,
        l1_ratio: float = 0.5,
        penalty: str = "l2",
        batch_size: int | None = None,
        val_size: float = 0.2,
        early_stopping: bool = True,
        patience: int = 20,
        tol: float = 1e-6,
        random_state: int | None = None,
    ) -> None:
        """Configure the regressor.

        Args:
            n_epochs: Maximum passes over the training data.
            learning_rate: Gradient-descent step size.
            alpha: Total regularisation strength. ``0`` disables it.
            l1_ratio: Blend between L1 and L2 in ``[0, 1]``. ``1`` is pure L1,
                ``0`` is pure L2. Ignored when ``penalty`` is ``"none"``.
            penalty: One of ``"l1"``, ``"l2"`` or ``"none"``. Selects the
                penalty form directly, overriding ``l1_ratio``.
            batch_size: Samples per update. ``None`` selects full-batch
                gradient descent, which is deterministic. An integer enables
                mini-batch updates.
            val_size: Fraction of data held out for early stopping, in
                ``[0, 1)``. ``0`` disables the split and trains on everything.
            early_stopping: Whether to stop when validation loss stops
                improving.
            patience: Number of epochs without improvement tolerated before
                stopping.
            tol: Minimum relative decrease in validation loss that counts as
                improvement.
            random_state: Seed controlling shuffling and the train/validation
                split.

        Raises:
            ValueError: If ``n_epochs`` is negative, ``learning_rate`` is not
                positive, ``alpha`` is negative, ``l1_ratio`` or ``val_size``
                falls outside ``[0, 1)``, ``patience`` is negative, ``tol`` is
                negative, ``batch_size`` is not positive, or ``penalty`` is
                unrecognised.
        """
        if n_epochs < 0:
            raise ValueError(f"n_epochs must be non-negative, got {n_epochs}.")
        if learning_rate <= 0:
            raise ValueError(f"learning_rate must be positive, got {learning_rate}.")
        if alpha < 0:
            raise ValueError(f"alpha must be non-negative, got {alpha}.")
        if not 0.0 <= l1_ratio <= 1.0:
            raise ValueError(f"l1_ratio must be in [0, 1], got {l1_ratio}.")
        if not 0.0 <= val_size < 1.0:
            raise ValueError(f"val_size must be in [0, 1), got {val_size}.")
        if patience < 0:
            raise ValueError(f"patience must be non-negative, got {patience}.")
        if tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol}.")
        if batch_size is not None and batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}.")
        if penalty not in {"l1", "l2", "none"}:
            raise ValueError(
                f"penalty must be 'l1', 'l2' or 'none', got {penalty!r}."
            )

        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.penalty = penalty
        self.batch_size = batch_size
        self.val_size = val_size
        self.early_stopping = early_stopping
        self.patience = patience
        self.tol = tol
        self.random_state = random_state

        self.coef_ = None
        self.intercept_ = 0.0
        self.n_iter_ = 0
        self.loss_curve_ = []
        self.val_loss_curve_ = []
        self.stopped_early_ = False
        self._x_scaler = None
        self._y_scaler = None

    def _regularisation(self, weights: np.ndarray) -> float:
        """Compute the penalty value for a weight vector.

        Args:
            weights: Coefficient vector, shape ``(n_features,)``. The intercept
                is excluded deliberately: shrinking it would bias predictions
                toward zero.

        Returns:
            float: Penalty contribution to the total loss, or ``0.0`` when
            regularisation is disabled.
        """
        if self.alpha == 0.0 or self.penalty == "none":
            return 0.0
        if self.penalty == "l1":
            return self.alpha * np.abs(weights).sum()
        if self.penalty == "l2":
            return self.alpha * float(np.dot(weights, weights))
        l2 = float(np.dot(weights, weights))
        l1 = float(np.abs(weights).sum())
        return self.alpha * ((1 - self.l1_ratio) * l2 + self.l1_ratio * l1)

    @staticmethod
    def _loss(
        y_true: np.ndarray, y_pred: np.ndarray, penalty_value: float = 0.0
    ) -> float:
        """Return mean squared error plus an optional penalty.

        Args:
            y_true: Targets of shape ``(n_samples, 1)``.
            y_pred: Predictions of the same shape.
            penalty_value: Precomputed regularisation term.

        Returns:
            float: The scalar objective minimised during training.
        """
        return float(np.mean((y_true - y_pred) ** 2) + penalty_value)

    def _prepare(self, X: np.ndarray, y: np.ndarray) -> _Split:
        """Scale data and split it into training and validation partitions.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Targets of shape ``(n_samples,)`` or ``(n_samples, 1)``.

        Returns:
            _Split: Scaled training and validation arrays.

        Raises:
            ValueError: If shapes are inconsistent.
        """
        rng = np.random.default_rng(self.random_state)
        X_scaled = self._x_scaler.fit_transform(X)
        y_col = y.reshape(-1, 1)
        y_scaled = self._y_scaler.fit_transform(y_col)

        n = X_scaled.shape[0]
        order = rng.permutation(n)

        if self.val_size > 0:
            cut = int(n * self.val_size)
            if cut == 0:
                cut = 1
            val_idx, train_idx = order[:cut], order[cut:]
            if train_idx.size == 0:
                # Tiny dataset with a large val_size: fall back to full training.
                return _Split(X_scaled, y_scaled, None, None)
            return _Split(
                X_scaled[train_idx],
                y_scaled[train_idx],
                X_scaled[val_idx],
                y_scaled[val_idx],
            )
        return _Split(X_scaled, y_scaled, None, None)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        """Fit the model with (mini-batch) gradient descent.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Targets of shape ``(n_samples,)`` or ``(n_samples, 1)``.

        Returns:
            self: The fitted estimator, to allow chaining.

        Raises:
            ValueError: If ``X`` is not two-dimensional, if sample counts
                disagree, or if ``X`` contains a constant feature that cannot be
                scaled.

        Note:
            Fitted attributes are ``coef_``, ``intercept_``, ``n_iter_``,
            ``loss_curve_``, ``val_loss_curve_`` and ``stopped_early_``.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1, 1)

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]}."
            )
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on an empty dataset.")

        self._x_scaler = MinMaxScaler()
        self._y_scaler = MinMaxScaler()
        split = self._prepare(X, y)

        X_train, y_train = split.X_train, split.y_train
        rng = np.random.default_rng(self.random_state)

        weights = np.zeros(X_train.shape[1], dtype=float)
        bias = 0.0
        self.loss_curve_ = []
        self.val_loss_curve_ = []
        self.stopped_early_ = False

        n_train = X_train.shape[0]
        batch = self.batch_size or n_train
        best_val = np.inf
        epochs_without_gain = 0

        for epoch in range(self.n_epochs):
            order = rng.permutation(n_train)
            epoch_loss = 0.0
            n_batches = 0

            for start in range(0, n_train, batch):
                idx = order[start : start + batch]
                xb, yb = X_train[idx], y_train[idx]

                # weights is 1-D, so xb @ weights is (batch,). Reshape to a
                # column so it lines up with yb's (batch, 1); without this,
                # `pred - yb` silently broadcasts to a (batch, batch) matrix.
                pred = (xb @ weights + bias).reshape(-1, 1)
                error = pred - yb
                # Average the gradient over the batch to keep the step size
                # comparable across batch sizes. ravel() collapses the
                # (n_features, 1) result of xb.T @ error to a 1-D vector.
                grad_w = (xb.T @ error).ravel() / xb.shape[0]
                grad_b = float(error.mean())

                penalty_grad = self._penalty_gradient(weights)
                weights -= self.learning_rate * (grad_w + penalty_grad)
                bias -= self.learning_rate * grad_b

                epoch_loss += self._loss(yb, pred, self._regularisation(weights))
                n_batches += 1

            self.n_iter_ = epoch + 1
            self.loss_curve_.append(epoch_loss / max(n_batches, 1))

            if split.X_val is not None:
                val_pred = (split.X_val @ weights + bias).reshape(-1, 1)
                val_loss = self._loss(split.y_val, val_pred)
                self.val_loss_curve_.append(val_loss)

                if self.early_stopping and val_loss < best_val - self.tol:
                    best_val = val_loss
                    epochs_without_gain = 0
                else:
                    # `tol` sets the smallest drop that still counts as real
                    # progress. Without it, floating-point noise in a converged
                    # model would count as improvement forever and early
                    # stopping would never fire.
                    epochs_without_gain += 1
                    if (
                        self.early_stopping
                        and epochs_without_gain >= self.patience
                        and self.patience > 0
                    ):
                        self.stopped_early_ = True
                        break

        # Map the fitted parameters back onto the original feature and target
        # scales so that callers see interpretable coefficients.
        #
        #   y = y_min + y_span * (X_scaled @ weights + bias),  X_scaled = (X - x_min) / x_span
        #
        # expanding gives coef_j = weights_j * y_span / x_span_j and
        # intercept = y_min - sum_j (x_min_j * y_span/x_span_j * weights_j) + y_span * bias.
        y_min = float(self._y_scaler.data_min_[0])
        y_span = float(self._y_scaler.data_max_[0] - y_min)
        x_min = self._x_scaler.data_min_
        x_span = self._x_scaler.data_max_ - x_min
        scale = y_span / x_span
        self.coef_ = (weights * scale).ravel()
        self.intercept_ = float(
            y_min - np.dot(x_min * scale, weights) + y_span * bias
        )
        return self

    def _penalty_gradient(self, weights: np.ndarray) -> np.ndarray:
        """Return the gradient of the regularisation term.

        Args:
            weights: Coefficient vector, shape ``(n_features,)``.

        Returns:
            numpy.ndarray: Penalty gradient shaped like ``weights``, or zeros
            when regularisation is disabled.
        """
        if self.alpha == 0.0 or self.penalty == "none":
            return np.zeros_like(weights)
        if self.penalty == "l1":
            return self.alpha * self.l1_ratio * np.sign(weights)
        if self.penalty == "l2":
            return 2.0 * self.alpha * (1.0 - self.l1_ratio) * weights
        return self.alpha * (
            (1.0 - self.l1_ratio) * 2.0 * weights
            + self.l1_ratio * np.sign(weights)
        )

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for new samples.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Predictions of shape ``(n_samples,)`` on the
            original target scale. The input is not modified.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``X`` has the wrong number of features.
        """
        if self.coef_ is None:
            raise RuntimeError("LinearRegression has not been fitted yet; call fit first.")
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.coef_.size:
            raise ValueError(
                f"X must be 2-D with {self.coef_.size} features, got shape {X.shape}."
            )
        # coef_ and intercept_ already live on the original scales, so predict
        # consumes raw X directly and never needs to touch the scalers.
        return X @ self.coef_ + self.intercept_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Return the coefficient of determination :math:`R^2`.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: True targets of shape ``(n_samples,)``.

        Returns:
            float: :math:`R^2` score. ``1.0`` is a perfect fit, ``0.0`` matches
            predicting the mean, and negative values are worse than that.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``X`` has the wrong number of features.
        """
        y = np.asarray(y, dtype=float).ravel()
        preds = self.predict(X)
        ss_res = float(np.sum((y - preds) ** 2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        if ss_tot == 0.0:
            return 0.0
        return 1.0 - ss_res / ss_tot
