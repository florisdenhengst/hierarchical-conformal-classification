from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.base import BaseEstimator, ClassifierMixin

from mapie.classification import MapieClassifier
from mapie.hierarchical_classification import HierarchicalMapieClassifier
from mapie.multi_label_classification import MapieMultiLabelClassifier
from mapie.multiple_testing import WestfallYoungMaxT
from mapie.utils import compute_quantiles

N_CALIB = 200
N_CLASSES = 4
ALPHA = 0.1


class LookupClassifier(BaseEstimator, ClassifierMixin):
    """Prefit classifier returning stored probabilities for the ids in X."""

    def __init__(self, proba):
        self.proba = proba

    def fit(self, X, y=None):
        self.classes_ = np.arange(self.proba.shape[1])
        self.cv = "prefit"
        return self

    def predict_proba(self, X):
        return self.proba[np.asarray(X).ravel().astype(int)]

    def predict(self, X, *args, **kwargs):
        return self.predict_proba(X)


def _proba(seed, n=N_CALIB, k=N_CLASSES, temperature=1.0):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, k)) / temperature
    proba = np.exp(logits)
    return proba / proba.sum(axis=1, keepdims=True)


def _labels(seed, n=N_CALIB, k=N_CLASSES):
    return np.random.default_rng(seed).integers(0, k, size=n)


def _mapie_classifier(seed):
    proba = _proba(seed)
    X = np.arange(len(proba)).reshape(-1, 1)
    y = _labels(seed + 100)
    model = LookupClassifier(proba).fit(X, y)
    return MapieClassifier(estimator=model, method="lac", cv="prefit").fit(
        X, y
    )


def _hierarchical_classifier(seed):
    proba = _proba(seed)
    X = np.arange(len(proba)).reshape(-1, 1)
    rng = np.random.default_rng(seed + 200)
    y = rng.random(size=proba.shape) < 0.4
    y[np.arange(len(y)), rng.integers(0, N_CLASSES, size=len(y))] = True
    model = LookupClassifier(proba).fit(X, y)
    return HierarchicalMapieClassifier(
        estimator=model,
        method="first",
        classes=np.arange(N_CLASSES),
        cv="prefit",
        costs_per_node=np.ones(N_CLASSES),
        clip=(0.0, 0.999999),
        cp_method="lac",
    ).fit(X, y)


def _multi_label_classifier(seed):
    proba = _proba(seed)
    X = np.arange(len(proba)).reshape(-1, 1)
    rng = np.random.default_rng(seed + 300)
    y = rng.random(size=proba.shape) < 0.4
    y[np.arange(len(y)), rng.integers(0, N_CLASSES, size=len(y))] = True
    model = LookupClassifier(proba).fit(X, y)
    return MapieMultiLabelClassifier(
        estimator=model, method="crc", metric_control="recall"
    ).fit(X, y)


def _X():
    return np.arange(N_CALIB).reshape(-1, 1)


def test_max_t_scores_are_the_rowwise_maximum() -> None:
    family = [_mapie_classifier(seed) for seed in range(3)]
    family.append(_hierarchical_classifier(3))
    wy = WestfallYoungMaxT().fit(family)
    expected = np.column_stack(
        [np.ravel(e.conformity_scores_) for e in family]
    ).max(axis=1)
    np.testing.assert_allclose(wy.max_t_scores_, expected)
    assert wy.n_samples_ == N_CALIB
    assert wy.conformity_scores_.shape == (N_CALIB, len(family))


def test_quantile_is_the_split_conformal_quantile_of_the_maxima() -> None:
    family = [_mapie_classifier(seed) for seed in range(3)]
    wy = WestfallYoungMaxT().fit(family)
    np.testing.assert_allclose(
        wy.get_quantiles([0.05, ALPHA]),
        compute_quantiles(wy.max_t_scores_, np.array([0.05, ALPHA])),
    )


@pytest.mark.parametrize(
    "make", [_mapie_classifier, _hierarchical_classifier]
)
def test_corrected_copy_uses_the_corrected_quantile(make) -> None:
    family = [make(seed) for seed in range(3)]
    wy = WestfallYoungMaxT().fit(family)
    q_hat = wy.get_quantiles(ALPHA)
    for estimator in family:
        scores_before = np.copy(estimator.conformity_scores_)
        corrected = wy.correct(estimator)
        _, sets_corrected = corrected.predict(_X(), alpha=ALPHA)
        np.testing.assert_allclose(corrected.quantiles_, q_hat)
        _, sets_original = estimator.predict(_X(), alpha=ALPHA)
        # the original predictor is left untouched
        np.testing.assert_array_equal(
            estimator.conformity_scores_, scores_before
        )
        np.testing.assert_allclose(
            estimator.quantiles_,
            compute_quantiles(scores_before, np.array([ALPHA])),
        )
        # q_hat dominates each predictor's own quantile: sets only grow
        assert np.all(sets_corrected >= sets_original)


def test_single_predictor_family_is_not_corrected() -> None:
    estimator = _mapie_classifier(0)
    wy = WestfallYoungMaxT().fit([estimator])
    _, sets_original = estimator.predict(_X(), alpha=ALPHA)
    _, sets_corrected = wy.correct(estimator).predict(_X(), alpha=ALPHA)
    np.testing.assert_array_equal(sets_original, sets_corrected)
    np.testing.assert_allclose(wy.get_corrected_alpha(ALPHA)[0, 0],
                               np.mean(np.ravel(estimator.conformity_scores_)
                                       > wy.get_quantiles(ALPHA)[0]))


def test_perfectly_dependent_family_needs_no_correction() -> None:
    scores = np.random.default_rng(0).random(N_CALIB)
    family = [SimpleNamespace(conformity_scores_=scores) for _ in range(5)]
    wy = WestfallYoungMaxT().fit(family)
    np.testing.assert_allclose(
        wy.get_quantiles(ALPHA),
        compute_quantiles(scores, np.array([ALPHA])),
    )


def test_corrected_alpha_is_below_nominal_alpha() -> None:
    family = [_mapie_classifier(seed) for seed in range(5)]
    wy = WestfallYoungMaxT().fit(family)
    corrected_alpha = wy.get_corrected_alpha(ALPHA)
    assert corrected_alpha.shape == (1, len(family))
    assert np.all(corrected_alpha <= ALPHA)
    # the family-wise error on the calibration set is at most alpha
    assert np.mean(wy.max_t_scores_ > wy.get_quantiles(ALPHA)[0]) <= ALPHA


@pytest.mark.parametrize("rho", [0.0, 0.5, 0.9])
def test_simultaneous_coverage_simulation(rho) -> None:
    """
    Simultaneous coverage of all predictors (and hence of any selected
    predictor) is at least 1 - alpha with the correction, while the
    uncorrected family under-covers when scores are not fully dependent.
    """
    rng = np.random.default_rng(42)
    n_calib, n_test, m, n_repeats = 300, 1000, 8, 100
    covered_corrected, covered_uncorrected = [], []
    for _ in range(n_repeats):
        shared = rng.normal(size=(n_calib + n_test, 1))
        own = rng.normal(size=(n_calib + n_test, m))
        scores = np.sqrt(rho) * shared + np.sqrt(1 - rho) * own
        calib, test = scores[:n_calib], scores[n_calib:]
        family = [SimpleNamespace(conformity_scores_=calib[:, j])
                  for j in range(m)]
        q_hat = WestfallYoungMaxT().fit(family).get_quantiles(ALPHA)[0]
        covered_corrected.append(np.all(test <= q_hat, axis=1).mean())
        q_own = np.array([
            compute_quantiles(calib[:, j], np.array([ALPHA]))[0]
            for j in range(m)
        ])
        covered_uncorrected.append(np.all(test <= q_own, axis=1).mean())
    standard_error = np.sqrt(ALPHA * (1 - ALPHA) / n_calib / n_repeats)
    assert np.mean(covered_corrected) >= 1 - ALPHA - 3 * standard_error
    # the guarantee is tight: the correction is not overly conservative
    assert np.mean(covered_corrected) <= 1 - ALPHA + 0.02
    if rho < 0.9:
        assert np.mean(covered_uncorrected) < 1 - ALPHA - 0.05


def test_risk_based_members_use_the_maximum_loss() -> None:
    score_based = _mapie_classifier(0)
    risk_based = [_multi_label_classifier(1), _multi_label_classifier(2)]
    wy = WestfallYoungMaxT().fit([score_based] + risk_based)
    lambdas = risk_based[0].lambdas
    miscoverage = np.ravel(score_based.conformity_scores_)[:, None] >= (
        1 - lambdas[None, :]
    )
    expected = np.maximum.reduce(
        [risk_based[0].risks, risk_based[1].risks, miscoverage]
    )
    np.testing.assert_allclose(wy.max_risks_, expected)
    for estimator in risk_based:
        risks_before = np.copy(estimator.risks)
        corrected = wy.correct(estimator)
        _, sets_corrected = corrected.predict(_X(), alpha=ALPHA)
        _, sets_original = estimator.predict(_X(), alpha=ALPHA)
        np.testing.assert_array_equal(estimator.risks, risks_before)
        assert corrected.lambdas_star[0] <= estimator.lambdas_star[0]
        assert np.all(sets_corrected >= sets_original)
    # effective alpha: own score/risk at the corrected threshold
    corrected_alpha = wy.get_corrected_alpha(ALPHA)
    assert corrected_alpha.shape == (1, 3)
    np.testing.assert_allclose(
        corrected_alpha[0, 0],
        np.mean(np.ravel(score_based.conformity_scores_)
                > wy.get_quantiles(ALPHA)[0]),
    )
    corrected = wy.correct(risk_based[0])
    corrected.predict(_X(), alpha=ALPHA)
    lambda_star = corrected.lambdas_star[0]
    for i, estimator in enumerate(risk_based, start=1):
        np.testing.assert_allclose(
            corrected_alpha[0, i],
            estimator.risks[:, np.searchsorted(lambdas, lambda_star)].mean(),
        )
        assert corrected_alpha[0, i] <= ALPHA


def test_errors() -> None:
    with pytest.raises(ValueError, match="empty"):
        WestfallYoungMaxT().fit([])
    with pytest.raises(ValueError, match="same samples"):
        WestfallYoungMaxT().fit([
            SimpleNamespace(conformity_scores_=np.zeros(3)),
            SimpleNamespace(conformity_scores_=np.zeros(4)),
        ])
    with pytest.raises(ValueError, match="conformity_scores_"):
        WestfallYoungMaxT().fit([SimpleNamespace()])
    with pytest.raises(ValueError, match="not fitted"):
        WestfallYoungMaxT().correct(_mapie_classifier(0))
    wy = WestfallYoungMaxT().fit(
        [SimpleNamespace(conformity_scores_=np.zeros(3))]
    )
    with pytest.raises(ValueError, match="calibration samples"):
        wy.correct(_mapie_classifier(0))
    with pytest.raises(ValueError, match="risk-based"):
        wy.correct(_multi_label_classifier(0))
    # too few calibration samples for the requested alpha, as in MAPIE
    with pytest.raises(ValueError, match="too low"):
        wy.get_quantiles(0.1)
