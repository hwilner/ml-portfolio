"""Behavioural tests for the from-scratch algorithm implementations.

These go beyond the doctests in the modules themselves. Where a doctest checks
that a function returns something, these tests check that the algorithm *learns
the right thing* and that it degrades the way theory says it should.

Run with::

    python -m pytest tests/ -v

or, without pytest installed::

    python tests/test_algorithms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ml_from_scratch.decision_tree import (  # noqa: E402
    DecisionTreeClassifier,
    information_gain,
    shannon_entropy,
)
from ml_from_scratch.linear_regression import (  # noqa: E402
    LinearRegression,
    MinMaxScaler,
)
from ml_from_scratch.perceptron import Perceptron  # noqa: E402


class TestShannonEntropy:
    """Entropy behaves as the textbook formula predicts."""

    def test_perfectly_balanced_split_is_one_bit(self):
        assert shannon_entropy(np.array([0, 0, 1, 1])) == 1.0

    def test_pure_node_has_zero_entropy(self):
        assert shannon_entropy(np.array([1, 1, 1, 1])) == 0.0

    def test_empty_input_is_zero(self):
        assert shannon_entropy(np.array([])) == 0.0

    def test_four_equal_classes_is_two_bits(self):
        assert abs(shannon_entropy(np.array([0, 1, 2, 3])) - 2.0) < 1e-12

    def test_unequal_split_matches_closed_form(self):
        # 3/4 vs 1/4: -(0.75*log2(0.75) + 0.25*log2(0.25)) = 0.811278...
        expected = -(0.75 * np.log2(0.75) + 0.25 * np.log2(0.25))
        assert abs(shannon_entropy(np.array([0, 0, 0, 1])) - expected) < 1e-12

    def test_label_type_does_not_matter(self):
        strings = np.array(["cat", "cat", "dog", "dog"])
        assert shannon_entropy(strings) == 1.0


class TestInformationGain:
    """Information gain measures the entropy a split removes."""

    def test_perfect_separation_gains_one_bit(self):
        parent = np.array([0, 0, 1, 1])
        assert abs(information_gain(parent, parent[:2], parent[2:]) - 1.0) < 1e-12

    def test_useless_split_gains_nothing(self):
        # Each class appears in both children, so the split removes no
        # information even though the children are disjoint subsets.
        parent = np.array([0, 1, 0, 1])
        assert abs(information_gain(parent, np.array([0, 1]), np.array([0, 1]))) < 1e-12

    def test_empty_parent_gains_nothing(self):
        assert information_gain(np.array([]), np.array([]), np.array([])) == 0.0


class TestMinMaxScaler:
    """The scaler is exact, reversible, and non-destructive."""

    def test_maps_onto_target_range(self):
        X = np.array([[1.0, 10.0], [3.0, 30.0], [5.0, 50.0]])
        scaled = MinMaxScaler().fit_transform(X)
        assert np.allclose(scaled.min(axis=0), 0.0)
        assert np.allclose(scaled.max(axis=0), 1.0)

    def test_round_trip_is_exact(self):
        rng = np.random.default_rng(0)
        X = rng.normal(loc=50, scale=10, size=(40, 3))
        scaler = MinMaxScaler().fit(X)
        assert np.allclose(scaler.inverse_transform(scaler.transform(X)), X)

    def test_supports_arbitrary_ranges(self):
        X = np.array([[0.0], [10.0]])
        scaled = MinMaxScaler(feature_range=(-1.0, 1.0)).fit_transform(X)
        assert np.allclose(scaled.ravel(), [-1.0, 1.0])

    def test_does_not_mutate_input(self):
        """Regression test: the original notebook rescaled in place."""
        X = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        original = X.copy()
        MinMaxScaler().fit_transform(X)
        assert np.array_equal(X, original), "fit_transform must not modify its input"

    def test_repeated_transform_is_stable(self):
        """A second transform must give the same answer as the first."""
        rng = np.random.default_rng(1)
        X = rng.normal(size=(20, 3))
        scaler = MinMaxScaler().fit(X)
        assert np.allclose(scaler.transform(X), scaler.transform(X))

    def test_constant_feature_is_rejected(self):
        X = np.array([[1.0, 5.0], [2.0, 5.0]])
        try:
            MinMaxScaler().fit_transform(X)
        except ValueError:
            return
        raise AssertionError("expected ValueError for a constant feature")

    def test_inverse_range_raises(self):
        try:
            MinMaxScaler(feature_range=(1.0, 0.0))
        except ValueError:
            return
        raise AssertionError("expected ValueError for an inverted range")


class TestPerceptron:
    """The perceptron converges on separable data and respects its bounds."""

    def test_learns_a_separable_problem(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(400, 2))
        y = (X[:, 0] > 0).astype(int)
        clf = Perceptron(n_epochs=30, learning_rate=0.1, random_state=0)
        clf.fit(X, y)
        assert clf.score(X, y) > 0.97

    def test_error_count_decreases(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(300, 2))
        y = (X[:, 0] > 0).astype(int)
        clf = Perceptron(n_epochs=25, learning_rate=0.1, random_state=0).fit(X, y)
        assert clf.errors_[0] > clf.errors_[-1]
        assert clf.errors_[-1] == 0

    def test_respects_epoch_budget_on_cyclic_data(self):
        """Non-separable data must not loop forever."""
        rng = np.random.default_rng(3)
        X = rng.normal(size=(200, 2))
        y = (X[:, 0] > 0).astype(int)
        # Make it non-separable by corrupting a slice of the labels.
        y[:40] = 1 - y[:40]
        clf = Perceptron(n_epochs=7, learning_rate=0.1, random_state=0).fit(X, y)
        assert clf.n_iter_ == 7
        assert len(clf.errors_) == 7

    def test_early_stopping_halts_early(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(200, 2))
        y = (X[:, 0] > 0).astype(int)
        clf = Perceptron(
            n_epochs=500, learning_rate=0.5, tol=0, random_state=0
        ).fit(X, y)
        assert clf.n_iter_ < 500

    def test_decision_function_separates_the_classes(self):
        X = np.array([[2.0], [-2.0]])
        y = np.array([1, 0])
        clf = Perceptron(n_epochs=10, learning_rate=0.5, random_state=0).fit(X, y)
        scores = clf.decision_function(X)
        assert scores[0] > 0 > scores[1]

    def test_rejects_non_binary_labels(self):
        X = np.zeros((4, 2))
        try:
            Perceptron().fit(X, np.array([0, 1, 2, 3]))
        except ValueError:
            return
        raise AssertionError("expected ValueError for non-binary y")

    def test_predict_before_fit_raises(self):
        try:
            Perceptron().predict(np.zeros((2, 2)))
        except RuntimeError:
            return
        raise AssertionError("expected RuntimeError before fit")

    def test_rejects_bad_learning_rate(self):
        try:
            Perceptron(learning_rate=0.0)
        except ValueError:
            return
        raise AssertionError("expected ValueError for learning_rate=0")


class TestLinearRegression:
    """Regression recovers known coefficients and honours regularisation."""

    def test_recovers_known_coefficients(self):
        rng = np.random.default_rng(0)
        X = rng.uniform(size=(600, 3))
        y = X @ np.array([2.0, -1.0, 0.5]) + 4.0
        model = LinearRegression(
            n_epochs=3000, learning_rate=0.2, early_stopping=False, val_size=0.0
        ).fit(X, y)
        assert np.allclose(model.coef_, [2.0, -1.0, 0.5], atol=0.05)
        assert abs(model.intercept_ - 4.0) < 0.2

    def test_score_is_near_one_for_linear_data(self):
        rng = np.random.default_rng(1)
        X = rng.uniform(size=(400, 2))
        y = X @ np.array([1.5, -0.5]) + 1.0
        model = LinearRegression(
            n_epochs=3000, learning_rate=0.2, early_stopping=False, val_size=0.0
        ).fit(X, y)
        assert model.score(X, y) > 0.99

    def test_predictions_match_manual_dot_product(self):
        rng = np.random.default_rng(2)
        X = rng.uniform(size=(200, 2))
        y = rng.normal(size=200)
        model = LinearRegression(
            n_epochs=2000, learning_rate=0.2, early_stopping=False, val_size=0.0
        ).fit(X, y)
        expected = X @ model.coef_ + model.intercept_
        assert np.allclose(model.predict(X), expected)

    def test_standard_scaling_does_not_change_fit_quality(self):
        """The reason the module scales features: scale must not matter."""
        rng = np.random.default_rng(3)
        X = rng.normal(size=(500, 2)) * np.array([1.0, 1000.0])
        y = X @ np.array([1.0, 0.001]) + 0.5
        model = LinearRegression(
            n_epochs=3000, learning_rate=0.2, early_stopping=False, val_size=0.0
        ).fit(X, y)
        assert model.score(X, y) > 0.95

    def test_fit_does_not_mutate_inputs(self):
        """Regression test for the notebook's in-place normalisation bug."""
        rng = np.random.default_rng(4)
        X = rng.normal(size=(200, 2))
        y = rng.normal(size=200)
        X_copy, y_copy = X.copy(), y.copy()
        LinearRegression(n_epochs=100, random_state=0).fit(X, y)
        assert np.array_equal(X, X_copy), "fit must not modify X"
        assert np.array_equal(y, y_copy), "fit must not modify y"

    def test_lasso_shrinks_more_than_ols(self):
        """Heavy L1 on a noise feature should drive it toward zero."""
        rng = np.random.default_rng(5)
        X = rng.normal(size=(300, 2))
        y = X[:, 0] * 1.0 + rng.normal(scale=0.1, size=300)
        lasso = LinearRegression(
            n_epochs=3000,
            learning_rate=0.2,
            alpha=0.5,
            penalty="l1",
            early_stopping=False,
            val_size=0.0,
        ).fit(X, y)
        # The second feature is pure noise; L1 should penalise it hard.
        assert abs(lasso.coef_[1]) < 0.5

    def test_minibatch_training_runs(self):
        rng = np.random.default_rng(6)
        X = rng.uniform(size=(300, 2))
        y = X @ np.array([1.0, 2.0])
        model = LinearRegression(
            n_epochs=500,
            learning_rate=0.1,
            batch_size=32,
            early_stopping=False,
            val_size=0.0,
            random_state=6,
        ).fit(X, y)
        assert model.coef_.shape == (2,)
        assert model.n_iter_ == 500

    def test_batch_size_changes_convergence_rate(self):
        """Mini-batching should still reach a good fit, just noisily."""
        rng = np.random.default_rng(9)
        X = rng.uniform(size=(400, 2))
        y = X @ np.array([1.0, 2.0]) + 1.0
        model = LinearRegression(
            n_epochs=2000,
            learning_rate=0.1,
            batch_size=64,
            early_stopping=False,
            val_size=0.0,
            random_state=9,
        ).fit(X, y)
        assert model.score(X, y) > 0.9

    def test_early_stopping_triggers(self):
        rng = np.random.default_rng(7)
        X = rng.normal(size=(200, 1))
        y = X[:, 0] + rng.normal(scale=0.01, size=200)
        model = LinearRegression(
            n_epochs=5000, learning_rate=0.2, patience=5, val_size=0.3
        ).fit(X, y)
        assert model.stopped_early_
        assert model.n_iter_ < 5000
        assert len(model.val_loss_curve_) == model.n_iter_

    def test_loss_decreases(self):
        rng = np.random.default_rng(8)
        X = rng.normal(size=(300, 2))
        y = rng.normal(size=300)
        model = LinearRegression(
            n_epochs=500, learning_rate=0.1, early_stopping=False, val_size=0.0
        ).fit(X, y)
        assert model.loss_curve_[-1] < model.loss_curve_[0]

    def test_predict_before_fit_raises(self):
        try:
            LinearRegression().predict(np.zeros((2, 2)))
        except RuntimeError:
            return
        raise AssertionError("expected RuntimeError before fit")

    def test_rejects_bad_penalty(self):
        try:
            LinearRegression(penalty="elastic")
        except ValueError:
            return
        raise AssertionError("expected ValueError for an unknown penalty")


class TestDecisionTree:
    """The tree splits sensibly, respects leaf size, and can use feature 0."""

    def test_learns_a_separable_problem(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(300, 2))
        y = (X[:, 0] > 0).astype(int)
        tree = DecisionTreeClassifier(min_samples_leaf=1).fit(X, y)
        assert tree.score(X, y) == 1.0

    def test_feature_zero_is_usable(self):
        """Regression test: the notebook's `and col` disabled feature 0.

        Here feature 0 is the only informative column, so the tree must split
        on it. A tree that cannot select column 0 scores at chance.
        """
        X = np.array([[0.0], [0.1], [0.2], [0.8], [0.9], [1.0]] * 20)
        y = np.array([0, 0, 0, 1, 1, 1] * 20)
        tree = DecisionTreeClassifier(min_samples_leaf=1).fit(X, y)
        assert tree.tree_.feature == 0
        assert tree.score(X, y) == 1.0

    def test_respects_min_samples_leaf(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(200, 2))
        y = (X[:, 0] > 0).astype(int)
        tree = DecisionTreeClassifier(min_samples_leaf=40).fit(X, y)
        leaf_sizes = _leaf_sizes(tree.tree_)
        assert min(leaf_sizes) >= 40, f"leaf smaller than min_samples_leaf: {leaf_sizes}"

    def test_min_leaf_reduces_node_count(self):
        rng = np.random.default_rng(2)
        X = rng.normal(size=(200, 2))
        y = (X[:, 0] + 0.3 * rng.normal(size=200) > 0).astype(int)
        small = DecisionTreeClassifier(min_samples_leaf=1).fit(X, y)
        large = DecisionTreeClassifier(min_samples_leaf=25).fit(X, y)
        assert large.n_nodes_ < small.n_nodes_

    def test_respects_max_depth(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(300, 3))
        y = ((X[:, 0] + X[:, 1] + X[:, 2]) > 0).astype(int)
        tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=1).fit(X, y)
        assert tree.max_depth_ <= 2

    def test_pure_child_becomes_a_leaf(self):
        """A node whose labels are all one class must stop splitting."""
        X = np.array([[0.0], [1.0], [2.0], [8.0], [9.0], [10.0]])
        y = np.array([0, 0, 0, 1, 1, 1])
        tree = DecisionTreeClassifier(min_samples_leaf=1).fit(X, y)
        # The root splits, and both children are pure, so both are leaves.
        assert tree.tree_.is_leaf is False
        assert tree.tree_.left.is_leaf and tree.tree_.right.is_leaf
        assert tree.tree_.left.prediction == 0
        assert tree.tree_.right.prediction == 1
        assert tree.max_depth_ == 1

    def test_impure_node_splits(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0, 0, 1, 1])
        tree = DecisionTreeClassifier(min_samples_leaf=1).fit(X, y)
        assert tree.tree_.is_leaf is False

    def test_predict_proba_rows_sum_to_one(self):
        rng = np.random.default_rng(4)
        X = rng.normal(size=(120, 2))
        y = (X[:, 0] > 0).astype(int)
        tree = DecisionTreeClassifier().fit(X, y)
        proba = tree.predict_proba(X)
        assert proba.shape == (120, 2)
        assert np.allclose(proba.sum(axis=1), 1.0)

    def test_predict_proba_argmax_matches_predict(self):
        rng = np.random.default_rng(5)
        X = rng.normal(size=(120, 2))
        y = (X[:, 0] > 0).astype(int)
        tree = DecisionTreeClassifier().fit(X, y)
        proba_argmax = tree.classes_[tree.predict_proba(X).argmax(axis=1)]
        assert np.array_equal(proba_argmax, tree.predict(X))

    def test_single_class_is_rejected(self):
        try:
            DecisionTreeClassifier().fit(np.zeros((5, 2)), np.zeros(5))
        except ValueError:
            return
        raise AssertionError("expected ValueError for a single-class target")

    def test_predict_before_fit_raises(self):
        try:
            DecisionTreeClassifier().predict(np.zeros((2, 2)))
        except RuntimeError:
            return
        raise AssertionError("expected RuntimeError before fit")

    def test_string_labels_work(self):
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array(["a", "a", "b", "b"])
        tree = DecisionTreeClassifier().fit(X, y)
        assert tree.predict(np.array([[0.1], [2.9]])).tolist() == ["a", "b"]


def _leaf_sizes(node) -> list[int]:
    """Collect the sample count stored at every leaf of a fitted tree.

    Args:
        node: Root ``_Node`` of a fitted tree.

    Returns:
        list[int]: Leaf sizes in traversal order.
    """
    if node.is_leaf:
        return [sum(node.counts.values())]
    return _leaf_sizes(node.left) + _leaf_sizes(node.right)


def _main() -> int:
    """Run every ``test_*`` method in this module without pytest.

    Returns:
        int: ``0`` when all tests pass, ``1`` otherwise.
    """
    failures = []
    total = 0
    for name, obj in sorted(globals().items()):
        if not name.startswith("Test") or not isinstance(obj, type):
            continue
        instance = obj()
        for method_name in sorted(dir(instance)):
            if not method_name.startswith("test_"):
                continue
            total += 1
            try:
                getattr(instance, method_name)()
                print(f"  PASS  {name}.{method_name}")
            except Exception as exc:  # noqa: BLE001
                failures.append((name, method_name, exc))
                print(f"  FAIL  {name}.{method_name}: {exc}")
    print(f"\n{total - len(failures)}/{total} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
