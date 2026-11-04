"""
Multiple-testing corrections for families of split conformal predictors.

When several conformal predictors are evaluated on the same test input and
one of their prediction sets is selected afterwards (for instance the one
that minimizes some cost), the marginal coverage guarantee of each individual
predictor no longer transfers to the selected set. This module implements a
correction analogous to the single-step Westfall-Young maxT procedure [1],
which restores the guarantee for any selection rule by calibrating all
predictors of the family on the distribution of the *maximum*
non-conformity score across the family.
"""
from __future__ import annotations

import copy
from typing import Iterable, List, Optional, Sequence, Union, cast

import numpy as np

from mapie._typing import NDArray
from mapie.control_risk.crc_rcps import find_lambda_star, get_r_hat_plus
from mapie.utils import (check_alpha, check_alpha_and_n_samples,
                         compute_quantiles)


def _is_score_based(estimator: object) -> bool:
    """Whether ``estimator`` is calibrated on non-conformity scores."""
    return hasattr(estimator, "conformity_scores_")


def _is_risk_based(estimator: object) -> bool:
    """Whether ``estimator`` is calibrated with (conformal) risk control."""
    return (
        hasattr(estimator, "risks")
        and hasattr(estimator, "lambdas")
        and not _is_score_based(estimator)
    )


class WestfallYoungMaxT:
    """
    Single-step Westfall-Young maxT correction for a family of split
    conformal predictors that are calibrated on the same calibration set.

    For each calibration sample ``i`` the non-conformity scores of all ``m``
    predictors ``S_i = (s_{i,1}, ..., s_{i,m})`` are reduced to their maximum
    ``t_i = max(S_i)``. The corrected threshold is the usual split conformal
    quantile of these maxima::

        q_hat = Quantile({t_i}, (n + 1)(1 - alpha) / n)

    computed with :func:`mapie.utils.compute_quantiles`, i.e. exactly like the
    quantile of a single MAPIE predictor. Every corrected predictor uses
    ``q_hat`` as its threshold. By exchangeability of the calibration samples
    and the test sample, ``P(s_{t,j} <= q_hat for all j) = P(t_t <= q_hat)
    >= 1 - alpha``: coverage holds *simultaneously* for all predictors in the
    family and hence for the prediction set of any predictor selected after
    looking at the test input. In contrast to Bonferroni, the correction
    uses the dependence between the scores: a family of perfectly correlated
    predictors needs no correction at all, and no a priori ordering of the
    predictors is needed (single-step procedure).

    The correction is applied by :meth:`correct`, which returns a copy of a
    fitted predictor that is calibrated on the maxT statistics ``t_i``
    instead of on its own scores. The copy is used exactly like the original,
    e.g. ``predict(X, alpha=alpha)`` with the *nominal* ``alpha``; the
    original predictor is not modified.

    Score-based predictors (with a ``conformity_scores_`` attribute, e.g.
    ``MapieClassifier`` and ``HierarchicalMapieClassifier``) must produce
    prediction sets of the form ``{y: s(x, y) <= q_hat}``. For the maximum to
    be informative, their scores should be on a common scale, which is the
    case for LAC scores ``1 - p(y | x)``. Validity does not depend on this.

    Predictors based on risk control (``MapieMultiLabelClassifier``) have no
    non-conformity score. For those, the analogue of the maxT statistic in
    the loss domain is used: the corrected loss of calibration sample ``i``
    at threshold ``lambda`` is the maximum over the family of

    - the losses of all risk-based predictors at ``lambda``, and
    - the miscoverage ``1[t_i >= 1 - lambda]`` of the score-based predictors,
      which assumes LAC scores (a label is included by risk-based predictors
      when its probability exceeds ``lambda``).

    All risk-based predictors therefore share the same corrected loss curve,
    which bounds the loss of every member of the family.

    Attributes
    ----------
    estimators_: List
        The fitted predictors of the family, in the order given to ``fit``.

    n_samples_: int
        Number of calibration samples.

    conformity_scores_: NDArray of shape (n_samples, n_score_estimators)
        Non-conformity scores of the score-based predictors.

    max_t_scores_: Optional[NDArray] of shape (n_samples,)
        maxT statistics ``t_i``, ``None`` if the family has no score-based
        predictor.

    lambdas_: Optional[NDArray] of shape (n_lambdas,)
        Threshold grid shared by the risk-based predictors, ``None`` if the
        family has no risk-based predictor.

    max_risks_: Optional[NDArray] of shape (n_samples, n_lambdas)
        Corrected loss curves of the risk-based predictors.

    References
    ----------
    [1] Peter H. Westfall and S. Stanley Young.
    "Resampling-Based Multiple Testing: Examples and Methods for p-Value
    Adjustment." Wiley, 1993.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from mapie.classification import MapieClassifier
    >>> from mapie.multiple_testing import WestfallYoungMaxT
    >>> X = np.arange(30).reshape(-1, 1)
    >>> y = np.repeat([0, 1, 2], 10)
    >>> clf = GaussianNB().fit(X, y)
    >>> family = [
    ...     MapieClassifier(estimator=clf, cv="prefit").fit(X, y)
    ...     for _ in range(3)
    ... ]
    >>> wy = WestfallYoungMaxT().fit(family)
    >>> corrected = [wy.correct(predictor) for predictor in family]
    >>> _, y_ps = corrected[0].predict(X, alpha=0.2)
    """

    def fit(self, estimators: Sequence) -> WestfallYoungMaxT:
        """
        Build the maxT distribution of a family of fitted predictors.

        Parameters
        ----------
        estimators: Sequence
            Fitted MAPIE classifiers calibrated on the same calibration
            samples, in the same order.

        Returns
        -------
        WestfallYoungMaxT
            The fitted correction.

        Raises
        ------
        ValueError
            If the family is empty, contains a predictor that is neither
            score-based nor risk-based, if the predictors are calibrated on a
            different number of samples, or if the risk-based predictors do
            not share the same threshold grid.
        """
        estimators = list(estimators)
        if len(estimators) == 0:
            raise ValueError("The family of predictors is empty.")
        score_based = [e for e in estimators if _is_score_based(e)]
        risk_based = [e for e in estimators if _is_risk_based(e)]
        if len(score_based) + len(risk_based) != len(estimators):
            raise ValueError(
                "Every predictor must be fitted and expose either "
                "conformity_scores_ (score-based) or risks and lambdas "
                "(risk-based)."
            )

        n_samples = {
            len(np.asarray(e.conformity_scores_).reshape(-1))
            for e in score_based
        } | {len(np.asarray(e.risks)) for e in risk_based}
        if len(n_samples) != 1:
            raise ValueError(
                "All predictors must be calibrated on the same samples, got "
                "different numbers of calibration samples: "
                f"{sorted(n_samples)}."
            )
        self.n_samples_ = n_samples.pop()

        if score_based:
            self.conformity_scores_ = np.column_stack([
                np.asarray(e.conformity_scores_, dtype=float).reshape(-1)
                for e in score_based
            ])
            self.max_t_scores_: Optional[NDArray] = (
                self.conformity_scores_.max(axis=1)
            )
        else:
            self.conformity_scores_ = np.empty((self.n_samples_, 0))
            self.max_t_scores_ = None

        if risk_based:
            lambdas = np.asarray(risk_based[0].lambdas, dtype=float)
            for e in risk_based[1:]:
                if not np.array_equal(np.asarray(e.lambdas), lambdas):
                    raise ValueError(
                        "All risk-based predictors must share the same "
                        "threshold grid (lambdas)."
                    )
            max_risks = np.max(
                np.stack([np.asarray(e.risks, dtype=float)
                          for e in risk_based]),
                axis=0,
            )
            if self.max_t_scores_ is not None:
                miscoverage = np.greater_equal(
                    self.max_t_scores_[:, np.newaxis],
                    1 - lambdas[np.newaxis, :],
                ).astype(float)
                max_risks = np.maximum(max_risks, miscoverage)
            self.lambdas_: Optional[NDArray] = lambdas
            self.max_risks_: Optional[NDArray] = max_risks
        else:
            self.lambdas_ = None
            self.max_risks_ = None

        self.estimators_: List = estimators
        return self

    def _check_is_fitted(self) -> None:
        if not hasattr(self, "estimators_"):
            raise ValueError(
                "This WestfallYoungMaxT instance is not fitted yet. "
                "Call 'fit' with the family of predictors first."
            )

    def get_quantiles(
        self, alpha: Union[float, Iterable[float]]
    ) -> NDArray:
        """
        Corrected thresholds ``q_hat`` of the score-based predictors.

        Parameters
        ----------
        alpha: Union[float, Iterable[float]]
            Nominal family-wise error levels, between 0 and 1.

        Returns
        -------
        NDArray of shape (n_alpha,)
            The corrected threshold for each value of ``alpha``.
        """
        self._check_is_fitted()
        if self.max_t_scores_ is None:
            raise ValueError("The family has no score-based predictor.")
        alpha_np = cast(NDArray, check_alpha(alpha))
        check_alpha_and_n_samples(alpha_np, self.n_samples_)
        return compute_quantiles(self.max_t_scores_, alpha_np)

    def get_corrected_alpha(
        self, alpha: Union[float, Iterable[float]]
    ) -> NDArray:
        """
        Effective per-predictor error level ``alpha'`` after correction.

        For a score-based predictor, this is the fraction of calibration
        samples whose own score exceeds the corrected threshold ``q_hat``.
        For a risk-based predictor with method ``"crc"``, it is its own mean
        calibration loss at the corrected threshold ``lambda``. It is the
        (empirical) error level at which each predictor effectively operates
        after correction, and plays the role of ``alpha / m`` in a Bonferroni
        correction. It is reported for information only: the corrected
        predictors are used with the nominal ``alpha``.

        Parameters
        ----------
        alpha: Union[float, Iterable[float]]
            Nominal family-wise error levels, between 0 and 1.

        Returns
        -------
        NDArray of shape (n_alpha, n_estimators)
            Effective error level per alpha and predictor, in the order of
            ``estimators_``. ``nan`` for risk-based predictors whose method
            is not ``"crc"``.
        """
        self._check_is_fitted()
        alpha_np = cast(NDArray, check_alpha(alpha))
        corrected_alpha = np.full((len(alpha_np), len(self.estimators_)),
                                  np.nan)
        if self.max_t_scores_ is not None:
            quantiles = self.get_quantiles(alpha_np)
            score_idx = [i for i, e in enumerate(self.estimators_)
                         if _is_score_based(e)]
            for a, q in enumerate(quantiles):
                corrected_alpha[a, score_idx] = np.greater(
                    self.conformity_scores_, q
                ).mean(axis=0)
        if self.max_risks_ is not None:
            lambdas = cast(NDArray, self.lambdas_)
            # conformal risk control bound of the corrected (max) loss
            _, r_hat_plus = get_r_hat_plus(
                self.max_risks_, lambdas, "crc", None, None, None
            )
            lambdas_star = find_lambda_star(lambdas, r_hat_plus, alpha_np)
            for i, e in enumerate(self.estimators_):
                is_crc = getattr(e, "method", None) == "crc"
                if not (_is_risk_based(e) and is_crc):
                    continue
                risks = np.asarray(e.risks, dtype=float)
                for a, lambda_star in enumerate(lambdas_star):
                    corrected_alpha[a, i] = risks[
                        :, np.searchsorted(lambdas, lambda_star)
                    ].mean()
        return corrected_alpha

    def correct(self, estimator):
        """
        Return a copy of ``estimator`` calibrated on the maxT distribution.

        The copy shares the underlying (prefit) model with ``estimator``;
        only its calibration is replaced. ``estimator`` itself is left
        unchanged, so both can be used side by side.

        Parameters
        ----------
        estimator: fitted MAPIE classifier
            A member of the family given to :meth:`fit`, or any predictor
            calibrated on the same samples.

        Returns
        -------
        Corrected copy of ``estimator``.

        Raises
        ------
        ValueError
            If the correction is not fitted, or ``estimator`` is calibrated
            on a different number of samples or is of an unsupported type.
        """
        self._check_is_fitted()
        corrected = copy.copy(estimator)
        if _is_score_based(estimator):
            if self.max_t_scores_ is None:
                raise ValueError("The family has no score-based predictor.")
            scores = np.asarray(estimator.conformity_scores_)
            if scores.size != self.n_samples_:
                raise ValueError(
                    f"The estimator has {scores.size} calibration samples, "
                    f"the family has {self.n_samples_}."
                )
            corrected.conformity_scores_ = self.max_t_scores_.reshape(
                scores.shape
            ).astype(scores.dtype, copy=True)
        elif _is_risk_based(estimator):
            if self.max_risks_ is None:
                raise ValueError("The family has no risk-based predictor.")
            risks = np.asarray(estimator.risks)
            if risks.shape != self.max_risks_.shape:
                raise ValueError(
                    f"The estimator has risks of shape {risks.shape}, the "
                    f"family has {self.max_risks_.shape}."
                )
            corrected.risks = self.max_risks_.copy()
        else:
            raise ValueError(
                "The estimator must be fitted and expose either "
                "conformity_scores_ or risks and lambdas."
            )
        # Score functions store state (e.g. quantiles_) during predict:
        # give the copy its own instance so it does not overwrite the one of
        # the original predictor.
        for attribute in (
            "conformity_score_function_", "conformity_score_function"
        ):
            if hasattr(estimator, attribute):
                setattr(
                    corrected,
                    attribute,
                    copy.copy(getattr(estimator, attribute)),
                )
        return corrected
