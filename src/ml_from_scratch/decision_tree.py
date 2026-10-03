"""A CART-style binary decision tree classifier built on NumPy alone.

Splitting is greedy: at every node the tree evaluates every feature and every
midpoint between adjacent sorted values, and commits to the split with the
largest information gain. Leaves predict the majority class.

Three defects in the original notebook
(``notebooks/05-algorithms-from-scratch/tree-models/01-decision-tree-information-gain.ipynb``)
are corrected here:

* ``choose_best_feature`` referenced an undefined name ``data_n`` instead of
  its own ``data`` argument, so the function raised ``NameError`` on every call.
* The best-split guard read ``if ent < min_avg_ent and col:``. Because column
  ``0`` is falsy in Python, feature 0 could never win a split. The tree was
  silently crippled; the ``and col`` clause is gone.
* ``predict`` looked up ``self.labels``, which was never populated.

Unlike ``sklearn.tree.DecisionTreeClassifier``, this implementation can stop
splitting on minimum leaf size, which makes it useful for studying how leaf
size controls the bias/variance trade-off.

Example:
    >>> import numpy as np
    >>> from ml_from_scratch.decision_tree import DecisionTreeClassifier
    >>> rng = np.random.default_rng(0)
    >>> X = rng.normal(size=(300, 2))
    >>> y = (X[:, 0] > 0).astype(int)
    >>> tree = DecisionTreeClassifier(min_samples_leaf=1, random_state=0)
    >>> _ = tree.fit(X, y)
    >>> tree.score(X, y)
    1.0
"""

from __future__ import annotations

import numpy as np

__all__ = ["DecisionTreeClassifier", "shannon_entropy", "information_gain"]


def shannon_entropy(labels: np.ndarray) -> float:
    """Compute Shannon entropy of a label distribution.

    Entropy is maximal when classes are evenly split and zero when the node is
    pure. The implementation uses base-2 logarithms so the result is measured
    in bits.

    Args:
        labels: Array of class labels of any shape. Only the value counts
            matter, so labels need not be numeric or contiguous.

    Returns:
        float: Entropy in bits, in ``[0, log2(k)]`` where ``k`` is the number of
        distinct classes. Returns ``0.0`` for an empty input.

    Example:
        >>> import numpy as np
        >>> from ml_from_scratch.decision_tree import shannon_entropy
        >>> shannon_entropy(np.array([0, 0, 1, 1]))
        1.0
    """
    labels = np.asarray(labels)
    if labels.size == 0:
        return 0.0
    _, counts = np.unique(labels, return_counts=True)
    probabilities = counts / counts.sum()
    # Guard against log2(0) for empty classes.
    probabilities = probabilities[probabilities > 0]
    return float(-np.sum(probabilities * np.log2(probabilities)))


def information_gain(parent: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    """Compute the information gain of a binary split.

    Args:
        parent: All labels at the parent node.
        left: Labels routed to the left child.
        right: Labels routed to the right child.

    Returns:
        float: Reduction in entropy, in bits. Non-negative for a valid split;
            values near zero mean the split separates nothing useful.

    Example:
        >>> import numpy as np
        >>> from ml_from_scratch.decision_tree import information_gain
        >>> parent = np.array([0, 0, 1, 1])
        >>> round(information_gain(parent, parent[:2], parent[2:]), 6)
        1.0
    """
    n_parent = np.asarray(parent).size
    if n_parent == 0:
        return 0.0
    n_left = np.asarray(left).size
    n_right = np.asarray(right).size
    remainder = (n_left * shannon_entropy(left) + n_right * shannon_entropy(right)) / n_parent
    return shannon_entropy(parent) - remainder


class _Node:
    """Internal node of the fitted tree.

    Attributes:
        feature: Index of the splitting feature, or ``None`` at a leaf.
        threshold: Value of the splitting feature, or ``None`` at a leaf.
        left: Left child node, or ``None`` at a leaf.
        right: Right child node, or ``None`` at a leaf.
        prediction: Majority class at a leaf, otherwise ``None``.
    """

    __slots__ = ("feature", "threshold", "left", "right", "prediction", "counts")

    def __init__(self) -> None:
        self.feature = None
        self.threshold = None
        self.left = None
        self.right = None
        self.prediction = None
        self.counts = None

    @property
    def is_leaf(self) -> bool:
        """bool: Whether this node makes a prediction instead of splitting."""
        return self.feature is None


class DecisionTreeClassifier:
    """Binary decision tree classifier using entropy-based information gain.

    The tree grows greedily and stops when any of these conditions hold: the
    node is pure, the node has fewer than ``min_samples_split`` samples, the
    node has fewer than ``2 * min_samples_leaf`` samples (so at least one leaf
    could be formed), the depth limit is reached, or no split yields positive
    information gain.

    Attributes:
        tree_ (_Node): Root of the fitted tree.
        n_features_ (int): Number of features seen during :meth:`fit`.
        classes_ (numpy.ndarray): Sorted distinct labels observed in training.
        max_depth_ (int): Depth of the fitted tree, where a leaf-only tree has
            depth ``0``.
        n_nodes_ (int): Total number of nodes, counting leaves.

    Example:
        >>> import numpy as np
        >>> from ml_from_scratch.decision_tree import DecisionTreeClassifier
        >>> X = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
        >>> y = np.array([0, 0, 0, 1, 1, 1])
        >>> tree = DecisionTreeClassifier(min_samples_leaf=1)
        >>> _ = tree.fit(X, y)
        >>> tree.predict(np.array([[0.5], [4.5]])).tolist()
        [0, 1]
    """

    def __init__(
        self,
        min_samples_leaf: int = 1,
        min_samples_split: int = 2,
        max_depth: int | None = None,
        random_state: int | None = None,
    ) -> None:
        """Configure the tree.

        Args:
            min_samples_leaf: Smallest number of samples allowed in a leaf.
                Larger values regularise the tree by preventing it from
                memorising individual points.
            min_samples_split: Smallest number of samples a node needs before it
                is allowed to split.
            max_depth: Maximum number of splits from root to leaf. ``None``
                grows the tree until other stopping rules trigger.
            random_state: Seed reserved for tie-breaking in future extensions.
                Accepted for API symmetry with scikit-learn.

        Raises:
            ValueError: If ``min_samples_leaf`` or ``min_samples_split`` is less
                than ``1``, or if ``max_depth`` is less than ``1``.
        """
        if min_samples_leaf < 1:
            raise ValueError(
                f"min_samples_leaf must be at least 1, got {min_samples_leaf}."
            )
        if min_samples_split < 2:
            raise ValueError(
                f"min_samples_split must be at least 2, got {min_samples_split}."
            )
        if max_depth is not None and max_depth < 1:
            raise ValueError(f"max_depth must be at least 1 or None, got {max_depth}.")

        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.random_state = random_state

        self.tree_ = None
        self.n_features_ = 0
        self.classes_ = None
        self.max_depth_ = 0
        self.n_nodes_ = 0

    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Search every feature and midpoint for the highest-gain split.

        Args:
            X: Feature matrix at the current node, shape ``(n_samples, n_features)``.
            y: Labels at the current node, shape ``(n_samples,)``.

        Returns:
            tuple | None: ``(feature_index, threshold, gain)`` for the best split,
            or ``None`` when no split yields positive gain.
        """
        parent_entropy = shannon_entropy(y)
        if parent_entropy == 0.0:
            return None

        best_feature, best_threshold, best_gain = None, None, 0.0

        for feature in range(X.shape[1]):
            column = X[:, feature]
            unique_values = np.unique(column)
            if unique_values.size < 2:
                continue

            # Candidate thresholds sit midway between adjacent distinct values,
            # which is the only place a binary split can actually change.
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
            for threshold in thresholds:
                left_mask = column <= threshold
                # Respect the minimum leaf size on both sides.
                if left_mask.sum() < self.min_samples_leaf:
                    continue
                if (~left_mask).sum() < self.min_samples_leaf:
                    continue

                left_y = y[left_mask]
                right_y = y[~left_mask]
                gain = parent_entropy - (
                    left_y.size * shannon_entropy(left_y)
                    + right_y.size * shannon_entropy(right_y)
                ) / y.size

                # Strictly greater, so ties keep the first feature found.
                # Every feature is eligible, including index 0.
                if gain > best_gain + 1e-12:
                    best_feature, best_threshold, best_gain = (
                        feature,
                        float(threshold),
                        float(gain),
                    )

        if best_feature is None or best_gain <= 0.0:
            return None
        return best_feature, best_threshold, best_gain

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        """Recursively build the tree for one node.

        Args:
            X: Feature matrix at this node, shape ``(n_samples, n_features)``.
            y: Labels at this node, shape ``(n_samples,)``.
            depth: Current depth, where the root is ``0``.

        Returns:
            _Node: A leaf if a stopping rule fires, otherwise an internal node
            with both children attached.
        """
        node = _Node()
        self.n_nodes_ += 1
        self.max_depth_ = max(self.max_depth_, depth)

        values, counts = np.unique(y, return_counts=True)
        node.prediction = values[np.argmax(counts)]
        # Keep the full class histogram so predict_proba can report the leaf's
        # empirical distribution without re-scanning the training data.
        node.counts = dict(zip(values.tolist(), counts.tolist()))

        n_samples = X.shape[0]
        if (
            shannon_entropy(y) == 0.0
            or n_samples < self.min_samples_split
            or n_samples < 2 * self.min_samples_leaf
            or (self.max_depth is not None and depth >= self.max_depth)
        ):
            return node

        split = self._best_split(X, y)
        if split is None:
            return node

        feature, threshold, _ = split
        left_mask = X[:, feature] <= threshold
        node.feature = feature
        node.threshold = threshold
        node.left = self._build(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        """Fit the decision tree.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Labels of shape ``(n_samples,)``. May be any hashable type.

        Returns:
            self: The fitted estimator, to allow chaining.

        Raises:
            ValueError: If ``X`` is not two-dimensional, if sample counts
                disagree, if the dataset is empty, or if fewer than two distinct
                classes are present.

        Note:
            Fitted attributes are ``tree_``, ``n_features_``, ``classes_``,
            ``max_depth_`` and ``n_nodes_``.
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()

        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}.")
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} samples but y has {y.shape[0]}.")
        if X.shape[0] == 0:
            raise ValueError("Cannot fit on an empty dataset.")
        if np.unique(y).size < 2:
            raise ValueError(
                "DecisionTreeClassifier needs at least 2 distinct classes."
            )

        self.n_features_ = X.shape[1]
        self.classes_ = np.unique(y)
        self.n_nodes_ = 0
        self.max_depth_ = 0
        self.tree_ = self._build(X, y, depth=0)
        return self

    def _predict_one(self, x: np.ndarray) -> np.generic:
        """Route a single sample through the tree.

        Args:
            x: Feature vector of shape ``(n_features,)``.

        Returns:
            numpy.generic: The predicted class label from the reached leaf.

        Raises:
            RuntimeError: If the tree has not been fitted.
        """
        if self.tree_ is None:
            raise RuntimeError(
                "DecisionTreeClassifier has not been fitted yet; call fit first."
            )
        node = self.tree_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.prediction

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels for a batch of samples.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Predicted labels of shape ``(n_samples,)``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``X`` has the wrong number of features.
        """
        if self.tree_ is None:
            raise RuntimeError(
                "DecisionTreeClassifier has not been fitted yet; call fit first."
            )
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(
                f"X must be 2-D with {self.n_features_} features, got shape {X.shape}."
            )
        return np.array([self._predict_one(row) for row in X])

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities as the leaf class distribution.

        Probabilities come from the proportion of training labels in the
        reached leaf, so they are quantised to multiples of ``1/leaf_size``.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.

        Returns:
            numpy.ndarray: Array of shape ``(n_samples, n_classes)`` whose rows
            sum to one. Columns follow the order of ``classes_``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``X`` has the wrong number of features.
        """
        if self.tree_ is None:
            raise RuntimeError(
                "DecisionTreeClassifier has not been fitted yet; call fit first."
            )
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.n_features_:
            raise ValueError(
                f"X must be 2-D with {self.n_features_} features, got shape {X.shape}."
            )
        return np.array([self._predict_proba_one(row) for row in X])

    def _predict_proba_one(self, x: np.ndarray) -> np.ndarray:
        """Return the leaf class distribution for one sample.

        Args:
            x: Feature vector of shape ``(n_features,)``.

        Returns:
            numpy.ndarray: Probabilities of length ``n_classes``.
        """
        node = self.tree_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return self._leaf_distribution_[node]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute the mean classification accuracy on a labelled set.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: True labels of shape ``(n_samples,)``.

        Returns:
            float: Fraction of correctly classified samples, in ``[0, 1]``.

        Raises:
            RuntimeError: If called before :meth:`fit`.
            ValueError: If ``X`` has the wrong number of features.
        """
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))

    @property
    def _leaf_distribution_(self) -> dict:
        """dict: Mapping from leaf node to its class-probability vector.

        Built lazily on first access and cached, since it is only needed by
        :meth:`predict_proba`.

        Returns:
            dict: Keyed by identity of each leaf ``_Node``.
        """
        cached = getattr(self, "_leaf_dist_cache", None)
        if cached is None:
            cached = {}
            self._collect_distributions(self.tree_, cached)
            self._leaf_dist_cache = cached
        return cached

    def _collect_distributions(self, node: _Node, out: dict) -> None:
        """Walk the tree recording a probability vector for every leaf.

        Args:
            node: Node to visit.
            out: Dictionary populated in place with ``node -> probabilities``.

        Raises:
            RuntimeError: If a node is not an internal node or a leaf.
        """
        if node is None:
            raise RuntimeError("Cannot build distributions from an unfitted tree.")
        if node.is_leaf:
            counts = np.array(
                [node.counts.get(cls, 0) for cls in self.classes_], dtype=float
            )
            total = counts.sum()
            out[node] = counts / total if total else np.zeros(len(self.classes_))
            return
        self._collect_distributions(node.left, out)
        self._collect_distributions(node.right, out)
