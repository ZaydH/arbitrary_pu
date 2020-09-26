# # python version is 3.6.1
from enum import Enum

import numpy as np

from .pu import decision_function

NEG_LABEL = -1
POS_LABEL = 1


class PUcKernelType(Enum):
    r""" Supported PUc kernels """
    GAUSSIAN = "gauss"
    LINEAR = "linear"


def predict(model: dict, x_te: np.ndarray) -> np.ndarray:
    r"""
    Predicts the labels for \p x_te

    :param model: PUc model parameters
    :param x_te: Test vector
    :return: Predicted labels
    """
    dec_scores = decision_function(model, x_te)
    # Specially handle sign to allow custom labels
    preds = np.full([dec_scores.shape[0]], NEG_LABEL, dtype=np.int32)
    preds[dec_scores >= 0] = POS_LABEL
    return preds
