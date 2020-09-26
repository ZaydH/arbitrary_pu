# -*- utf-8 -*-
r"""
    loss.py
    ~~~~~~~~~~~~~~~

    Implements the positive-negative (PN) risk estimator and generic loss functions.

    :copyright: (c) 2020 by Zayd Hammoudeh.
    :license: MIT, see LICENSE file for more details.
"""


__all__ = ["PN",
           "log_loss", "loss_0_1", "ramp_loss", "sigmoid_loss"]

from typing import Callable

from torch import Tensor
import torch.nn as nn

from .datasets.types import Labels
from .types import LossInfo, RiskEstimator

_sigmoid_module = nn.Sigmoid()
_log_sig_module = nn.LogSigmoid()


def sigmoid_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    r""" Sigmoid loss that takes two arguments instead of default one """
    return _sigmoid_module(_prod_input_targets(inputs.neg(), targets))


def log_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    r""" Sigmoid loss that takes two arguments instead of default one """
    clamp_val = 1e2
    return -_log_sig_module(_prod_input_targets(inputs, targets).clamp(-clamp_val, clamp_val))


def loss_0_1(inputs: Tensor, targets: Tensor) -> Tensor:
    r""" 0-1 loss """
    return (inputs.sign() - targets).abs() / 2


def ramp_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    r""" Ramp loss """
    return ((_prod_input_targets(inputs, targets).neg() + 1) / 2).clamp(0., 1.)


def squared_loss(inputs: Tensor, targets: Tensor) -> Tensor:
    return (_prod_input_targets(inputs, targets).neg() + 1).pow(2.) / 4


def _prod_input_targets(inputs: Tensor, targets: Tensor) -> Tensor:
    r""" Standardizes inputs and targets multiplication to ensure sizes are valid """
    for vec, name in ((inputs, "inputs"), (targets, "targets")):
        assert vec.shape[0] == vec.numel(), f"Weird {name} shape"

    prod_val = inputs * targets
    assert prod_val.numel() == inputs.numel() == targets.numel(), "Change in number of elements"
    assert prod_val.shape[0] == prod_val.numel(), "Product is not a vector"
    return prod_val


class PN(RiskEstimator):
    r""" Positive negative risk """
    def __init__(self, train_loss: Callable, valid_loss: Callable, prior: float, is_train: bool):
        super().__init__(train_loss, valid_loss)

        self._prior = prior
        self._is_train = is_train

    def _loss(self, dec_scores: Tensor, lbls: Tensor, _: Tensor, f_loss: Callable) -> LossInfo:
        r""" Straight forward PN loss -- No weighting by prior & label """
        if len(dec_scores.shape) == 2: dec_scores = dec_scores.squeeze(dim=1)
        if len(lbls.shape) == 2: lbls = lbls.squeeze(dim=1)

        assert len(dec_scores.shape) == 1 and len(lbls.shape) == 1, "Bizarre input shape"

        loss = f_loss(dec_scores, lbls.float())
        p_loss = loss[lbls == Labels.POS].mean()
        n_loss = loss[lbls == Labels.NEG].mean()
        loss = self._prior * p_loss + (1 - self._prior) * n_loss
        return LossInfo(te_loss=loss, grad_var=loss)

    def name(self) -> str:
        r""" Returns name of the trainer """
        return f"{'tr' if self._is_train else 'te'}-PN"

    def is_nn(self) -> bool:
        r""" PN learners are never non-negative"""
        return False
