from typing import Optional, Union, Dict
import numpy as np

from mapie._typing import NDArray
from mapie._machine_precision import EPSILON
import functools

@functools.cache
def cost_dict_to_array(
    costs: Dict[int, float],
):
    return np.array([costs[i] for i in range(len(costs))])
    

def minimize_costs(
    costs: NDArray,
    y_pred_proba: NDArray,
    y: NDArray,
):
    broadcasted = np.broadcast_to(costs, y.shape)
    # Create a masked array, removing all non-true labels from cost matrix
    masked = np.ma.masked_where(~y, broadcasted)

    # select the ground truth with the lowest cost
    y_single = masked.argmin(axis=1).reshape(-1,1)

    return y_single
    

def get_first_true_label_position(
    y_pred_proba: NDArray,
    y: NDArray
) -> NDArray:
    """
    Return the position of the first true label according to the sorted prediction.

    Parameters
    ----------
    y_pred_proba: NDArray of shape (n_samples, n_classes)
        Model prediction probabilities.

    y: NDArray of shape (n_samples, n_classes)
        Binary matrix of true labels.

    Returns
    -------
    NDArray of shape (n_samples, 1)
        Position of the first true label in the sorted prediction.
    """
    # Create a masked array, removing all non-true labels
    masked = np.ma.masked_where(~y, y_pred_proba)

    # select the ground truth with the highest predicted score
    y_single = masked.argmax(axis=1)

    return y_single

def get_last_true_label_position(
    y_pred_proba: NDArray,
    y: NDArray
) -> NDArray:
    """
    Return the position of the last true label according to the sorted prediction.

    Parameters
    ----------
    y_pred_proba: NDArray of shape (n_samples, n_classes)
        Model prediction probabilities.

    y: NDArray of shape (n_samples, n_classes)
        Binary matrix of true labels.

    Returns
    -------
    NDArray of shape (n_samples, 1)
        Position of the last true label in the sorted prediction.
    """
        # Create a masked array, removing all non-true labels
    masked = np.ma.masked_where(~y, y_pred_proba)

    # select the ground truth with the lowest predicted score
    y_single = masked.argmin(axis=1)

    return y_single

def get_true_label_position(
    y_pred_proba: NDArray,
    y: NDArray
) -> NDArray:
    """
    Return the sorted position of the true label in the prediction

    Parameters
    ----------
    y_pred_proba: NDArray of shape (n_samples, n_classes)
        Model prediction.

    y: NDArray of shape (n_samples)
        Labels.

    Returns
    -------
    NDArray of shape (n_samples, 1)
        Position of the true label in the prediction.
    """
    index = np.argsort(np.fliplr(np.argsort(y_pred_proba, axis=1)))
    position = np.take_along_axis(index, y.reshape(-1, 1), axis=1)

    return position


def check_include_last_label(
    include_last_label: Optional[Union[bool, str]]
) -> Optional[Union[bool, str]]:
    """
    Check if ``include_last_label`` is a boolean or a string.
    Else raise error.

    Parameters
    ----------
    include_last_label: Optional[Union[bool, str]]
        Whether or not to include last label in
        prediction sets for the ``"aps"`` method. Choose among:

        - ``False``, does not include label whose cumulated score is just
            over the quantile.

        - ``True``, includes label whose cumulated score is just over the
            quantile, unless there is only one label in the prediction set.

        - ``"randomized"``, randomly includes label whose cumulated score
            is just over the quantile based on the comparison of a uniform
            number and the difference between the cumulated score of the last
            label and the quantile.

    Returns
    -------
    Optional[Union[bool, str]]

    Raises
    ------
    ValueError
        "Invalid include_last_label argument. "
        "Should be a boolean or 'randomized'."
    """
    if (
        (not isinstance(include_last_label, bool)) and
        (not include_last_label == "randomized")
    ):
        raise ValueError(
            "Invalid include_last_label argument. "
            "Should be a boolean or 'randomized'."
        )
    else:
        return include_last_label


def check_proba_normalized(
    y_pred_proba: NDArray,
    axis: int = 1
) -> NDArray:
    """
    Check if for all the samples the sum of the probabilities is equal to one.

    Parameters
    ----------
    y_pred_proba: NDArray of shape (n_samples, n_classes) or
    (n_samples, n_train_samples, n_classes)
        Softmax output of a model.

    Returns
    -------
    ArrayLike of shape (n_samples, n_classes)
        Softmax output of a model if the scores all sum to one.

    Raises
    ------
    ValueError
        If the sum of the scores is not equal to one.
    """
    np.testing.assert_allclose(
        np.sum(y_pred_proba, axis=axis),
        1,
        err_msg="The sum of the scores is not equal to one.",
        rtol=1e-5
    )
    return y_pred_proba.astype(np.float64)


def get_last_index_included(
    y_pred_proba_cumsum: NDArray,
    threshold: NDArray,
    include_last_label: Optional[Union[bool, str]]
) -> NDArray:
    """
    Return the index of the last included sorted probability
    depending if we included the first label over the quantile
    or not.

    Parameters
    ----------
    y_pred_proba_cumsum: NDArray of shape (n_samples, n_classes)
        Cumsumed probabilities in the original order.

    threshold: NDArray of shape (n_alpha,) or shape (n_samples_train,)
        Threshold to compare with y_proba_last_cumsum, can be either:

        - the quantiles associated with alpha values when
            ``cv`` == "prefit", ``cv`` == "split"
            or ``agg_scores`` is "mean"

        - the conformity score from training samples otherwise
            (i.e., when ``cv`` is a CV splitter and
            ``agg_scores`` is "crossval")

    include_last_label: Union[bool, str]
        Whether or not include the last label. If 'randomized',
        the last label is included.

    Returns
    -------
    NDArray of shape (n_samples, n_alpha)
        Index of the last included sorted probability.
    """
    if include_last_label or include_last_label == 'randomized':
        y_pred_index_last = (
            np.ma.masked_less(
                y_pred_proba_cumsum
                - threshold[np.newaxis, :],
                -EPSILON
            ).argmin(axis=1)
        )
    else:
        max_threshold = np.maximum(
            threshold[np.newaxis, :],
            np.min(y_pred_proba_cumsum, axis=1)
        )
        y_pred_index_last = np.argmax(
            np.ma.masked_greater(
                y_pred_proba_cumsum - max_threshold[:, np.newaxis, :],
                EPSILON
            ), axis=1
        )
    return y_pred_index_last[:, np.newaxis, :]
