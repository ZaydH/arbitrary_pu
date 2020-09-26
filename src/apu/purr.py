# -*- utf-8 -*-
r"""
    purr.py
    ~~~~~~~~~~~~~~~

    Implements the PURR risk estimator from the paper "Learning from Positive & Unlabeled Data
    with Arbitrary Positive Shift" published at NeurIPS'20.

    :copyright: (c) 2020 by Zayd Hammoudeh.
    :license: MIT, see LICENSE file for more details.
"""

__all__ = ["PURR"]

from typing import Callable

import torch
from torch import Tensor

from .datasets.types import Labels
from .types import OptBool, RiskEstimator
from .utils import TORCH_DEVICE


class PURR(RiskEstimator):
    class Loss:
        r""" Encapsulates the loss calculated by the loss and the accompanying fields """
        def __init__(self):
            self.r_te_p_plus = self.r_n_plus = self.r_n_minus = None

            self.te_loss = self.grad_var = None

        def is_te_p_plus_invalid(self) -> bool:
            r""" Returns \p True if :math:`R_{\text{te-p}}^{+} < 0` """
            return bool(self.r_te_p_plus < 0)

        def is_n_plus_invalid(self) -> bool:
            r""" Returns \p True if :math:`R_{\text{te-n}}^{+} < 0` """
            return bool(self.r_n_plus < 0)

        def is_n_minus_invalid(self) -> bool:
            r""" Returns \p True if :math:`R_{\text{te-n}}^{-} < 0` """
            return bool(self.r_n_minus < 0)

    def __init__(self, train_prior: float, test_prior: float,
                 train_loss: Callable, valid_loss: Callable, gamma: float = 1.,
                 use_nn: OptBool = True, abs_nn: bool = False):
        assert not abs_nn or use_nn, "Cannot use absolute NN with NN disabled"
        super().__init__(train_loss, valid_loss)

        self._tr_prior, self._te_prior = train_prior, test_prior

        self._pr_ratio = (1 - self._te_prior) / (1 - self._tr_prior)

        self._use_nn, self._abs_nn = use_nn, abs_nn
        self.gamma = gamma

    def _loss(self, dec_scores: Tensor, lbls: Tensor, _: Tensor, f_loss: Callable) -> "PURR.Loss":
        r""" Shared function for validation/training loss """
        p_mask = lbls == Labels.Training.POS.value
        u_te_mask = lbls == Labels.Training.U_TEST.value
        u_tr_mask = lbls == Labels.Training.U_TRAIN.value

        # noinspection PyUnresolvedReferences
        assert bool((p_mask | u_te_mask | u_tr_mask).all()), "Unknown labels"
        # noinspection PyUnresolvedReferences
        assert bool((p_mask ^ u_te_mask ^ u_tr_mask).all()), "Overlapping labels"

        has_p = self._has_any(p_mask)
        has_u_te, has_u_tr = self._has_any(u_te_mask), self._has_any(u_tr_mask)

        ones = torch.ones_like(lbls, device=TORCH_DEVICE)
        plus_lbl_rsk = f_loss(dec_scores, ones)
        minus_lbl_rsk = f_loss(dec_scores, -ones)

        loss = self.Loss()

        def _calc_tr_lbl_risk(lbl_rsk_all: Tensor) -> Tensor:
            r""" Calculates the labeled risk of the TRAINING set"""
            _loss = self._pr_ratio * (_get_mean_loss(lbl_rsk_all, u_tr_mask, has_u_tr)
                                      - self._tr_prior * _get_mean_loss(lbl_rsk_all, p_mask, has_p))
            return _loss.abs() if self._abs_nn else _loss

        loss.r_n_plus = _calc_tr_lbl_risk(plus_lbl_rsk)
        loss.r_n_minus = _calc_tr_lbl_risk(minus_lbl_rsk)

        r_te_u_plus = _get_mean_loss(plus_lbl_rsk, u_te_mask, has_u_te)
        if not self._use_nn:
            loss.r_te_p_plus = r_te_u_plus - loss.r_n_plus
            loss.te_loss = loss.grad_var = loss.r_te_p_plus + loss.r_n_minus
        else:
            loss.r_te_p_plus = r_te_u_plus - loss.r_n_plus.clamp_min(0)
            if self._abs_nn:
                loss.r_te_p_plus = loss.r_te_p_plus.abs()

            loss.te_loss = loss.r_te_p_plus.clamp_min(0) + loss.r_n_minus.clamp_min(0)
            if self._abs_nn:
                loss.grad_var = loss.te_loss
            elif loss.is_n_minus_invalid():
                loss.grad_var = -self.gamma * loss.r_n_minus
            elif loss.is_n_plus_invalid():
                loss.grad_var = -self.gamma * loss.r_n_plus
            elif loss.is_te_p_plus_invalid():
                loss.grad_var = -self.gamma * loss.r_te_p_plus
            else:
                loss.grad_var = loss.te_loss
        return loss

    def name(self) -> str:
        r""" Name of the loss function """
        return "PURR"

    @property
    def is_nn(self) -> bool:
        r""" Return \p True if using non-negative PURR """
        return self._use_nn


def _get_mean_loss(l_tensor: Tensor, mask: Tensor, has_any: bool) -> Tensor:
    r""" Gets the mean loss for the specified \p mask. If the mask is empty, return 0 """
    if not has_any:
        return torch.zeros((), device=TORCH_DEVICE)
    return l_tensor[mask].mean().squeeze()
