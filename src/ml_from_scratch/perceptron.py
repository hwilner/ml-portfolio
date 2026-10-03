"""Linear classifiers implemented from first principles with NumPy only.

This module provides a dependency-free reimplementation of the perceptron
learning rule. It exists because relying on ``sklearn.linear_model`` hides the
update rule that the algorithm actually executes, and understanding that rule is
the point of writing it by hand.

The implementations here were extracted from exploratory notebooks
(``notebooks/05-algorithms-from-scratch/linear-models/``) and corrected: the
original notebooks had swapped argument orders, an aliasing bug, and two
identical update branches where the sign should have differed.

References:
    Rosenblatt, F. (1958). The perceptron: A probabilistic model for
        information storage and organization in the brain.
    Rosenblatt, F. (1962). Principles of Neural Science.

Example:
    >>> import numpy as np
    >>> from ml_from_scratch.perceptron import Perceptron
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(200, 2))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> clf = Perceptron(n_epochs=25, learning_rate=0.1, random_state=0)
    >>> _ = clf.fit(X, y)
    >>> clf.score(X, y) > 0.95
    True
"""

from __future__ import annotations

import numpy as np

__all__ = ["Perceptron"]


class Perceptron:
    """Binary linear classifier trained with the perceptron learning rule.

    The perceptron predicts ``1`` when the weighted sum of a sample's features
    is non-negative and ``0`` otherwise. Training consists of repeatedly
    scanning the dataset and nudging the weight vector towards samples that were
    classified incorrectly:

    .. math::

        \\mathbf{w} \\leftarrow \\mathbf{w} + \\eta (y_i - \\hat{y}_i) \\mathbf{x}_i

    where :math:`\\eta` is the learning rate. Because the rule only fires on
    mistakes, a separable dataset converges in a finite number of updates;
    a non-separable one may oscillate, so training is always bounded by
    ``n_epochs``.

    Attributes:
        weights_ (numpy.ndarray): Learned weight vector, shape ``(n_features + 1,)``.
            The final entry is the bias term.
        n_iter_ (int): Number of training epochs actually executed.
        errors_ (list[int]): Misclassification count recorded at the end of each
            epoch. Useful for diagnosing convergence.

    Example:
        >>> import numpy as np
        >>> from ml_from_scratch.perceptron import Perceptron
        >>> X = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0], [-2.0, -1.0]])
        >>> y = np.array([1, 1, 0, 0])
        >>> clf = Perceptron(n_epochs=10, learning_rate=0.5, random_state=0)
        >>> _ = clf.fit(X, y)
        >>> clf.predict(X).tolist()
        [1, 1, 0, 0]
    """

    def __init__(
        self,
        n_epochs: int = 50,
        learning_rate: float = 0.1,
        tol: float | None = None,
        random_state: int | None = None,
    ) -> None:
        """Initialise the perceptron.

        Args:
            n_epochs: Maximum number of passes over the training data. Bounds
                training so that non-separable data cannot loop forever.
            learning_rate: Step size ``eta`` applied to each weight update.
            tol: Optional early-stopping tolerance on the epoch miscount.
                Training halts early when the absolute change in errors between
                consecutive epochs is at or below this value. ``None`` disables
                early stopping.
            random_state: Seed for shuffling the data each epoch. Shuffling
                escapes the pathological ordering in which the perceptron can
                repeatedly correct the same sample. ``None`` uses fresh
                randomness.

        Raises:
            ValueError: If ``n_epochs`` is negative, ``learning_rate`` is not
                positive, or ``tol`` is negative.
        """
        if n_epochs < 0:
            raise ValueError(f"n_epochs must be non-negative, got {n_epochs}.")
        if learning_rate <= 0:
            raise ValueError(
                f"learning_rate must be positive, got {learning_rate}."
            )
        if tol is not None and tol < 0:
            raise ValueError(f"tol must be non-negative, got {tol}.")

        self.n_epochs = n_epochs
        self.learning_rate = learning_rate
        self.tol = tol
        self.random_state = random_state

        self.weights_ = None
        self.n_iter_ = 0
        self.errors_ = []

    def fit(self, X: np.ndarray, y: np.ndarray) -> "Perceptron":
        """Fit the perceptron to a training set.

        Samples are augmented with a constant column so the bias term is learned
        by the same update rule as the feature weights, rather than being
        special-cased.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Binary targets of shape ``(n_samples,)`` with values in ``{0, 1}``.

        Returns:
            self: The fitted estimator, to allow chaining.

        Raises:
            ValueError: If ``X`` is not two-dimensional, if the sample counts of
                ``X`` and ``y`` disagree, or if ``y`` contains labels outside
                ``{0, 1}``.

        Note:
            Fitted attributes are ``weights_``, ``n_iter_`` and ``errors_``.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X has {X.shape[0]} samples but y has {y.shape[0]}."
            )
        if not np.isin(y, [0, 1]).all():
            raise ValueError("y must contain only 0 and 1 for a binary perceptron.")

        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]

        # Augment with a bias column so one uniform rule updates everything.
        X_aug = np.hstack([X, np.ones((n_samples, 1))])
        self.weights_ = np.zeros(X_aug.shape[1], dtype=float)
        self.errors_ = []

        previous_errors = None
        for epoch in range(self.n_epochs):
            order = rng.permutation(n_samples)
            epoch_errors = 0

            for i in order:
                prediction = int(X_aug[i] @ self.weights_ > 0)
                # Update fires only on mistakes, and the sign of the correction
                # is carried by (y - y_hat), not by the branch taken.
                update = y[i] - prediction
                if update != 0:
                    self.weights_ += self.learning_rate * update * X_aug[i]
                    epoch_errors += 1

            self.n_iter_ = epoch + 1
            self.errors_.append(epoch_errors)

            if self.tol is not None and previous_errors is not None:
                if abs(previous_errors - epoch_errors) <= self.tol:
                    break
            previous_errors = epoch_errors

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Return the signed distance to the separating hyperplane.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Raw scores of shape ``(n_samples,)``. Positive values
            fall on the positive side of the boundary, negative values on the
            negative side.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If the feature count does not match the fitted model.
        """
        self._check_fitted(X)
        X = np.asarray(X, dtype=float)
        X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
        return X_aug @ self.weights_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary class labels.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Predicted labels of shape ``(n_samples,)`` with dtype
            ``int`` and values in ``{0, 1}``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If the feature count does not match the fitted model.
        """
        return (self.decision_function(X) > 0).astype(int)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the mean classification accuracy on a labelled set.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: True labels of shape ``(n_samples,)``.

        Returns:
            float: Fraction of correctly classified samples, in ``[0, 1]``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If the feature count does not match the fitted model.
        """
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))

    def _check_fitted(self, X: np.ndarray) -> None:
        """Validate that the model is fitted and that features align.

        Args:
            X: Feature matrix whose column count should match the fitted model.

        Raises:
            RuntimeError: If the model has not been fitted.
            ValueError: If ``X`` has the wrong number of columns.
        """
        if self.weights_ is None:
            raise RuntimeError("Perceptron has not been fitted yet; call fit first.")
        if np.asarray(X).shape[1] != self.weights_.size - 1:
            raise ValueError(
                f"X has {np.asarray(X).shape[1]} features but the model was "
                f"fitted on {self.weights_.size - 1}."
            )
