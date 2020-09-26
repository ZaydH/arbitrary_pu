# -*- utf-8 -*-
r"""
    two_step.py
    ~~~~~~~~~~~~~~~

    Implements the two-step risk estimators -- wUU and aPNU" from the paper "Learning from
    Positive & Unlabeled Data with Arbitrary Positive Shift" published at NeurIPS'20.

    :copyright: (c) 2020 by Zayd Hammoudeh.
    :license: MIT, see LICENSE file for more details.
"""

__all__ = ["APNU",
           "Step1Method",
           "WUU",
           "is_two_step"]

from abc import ABC
from enum import Enum
from typing import Callable, Optional, Union, Set

import torch
from torch import Tensor

from . import LossInfo
from .datasets.types import Labels
from .nnpu import _NNBase
from .types import RiskEstimator
from .utils import TORCH_DEVICE


class Step1Method(Enum):
    r""" Defines technique for use in step 1 """
    SOFT = "S"
    HARD = "H"
    TOP_K = "TK"


def is_two_step(loss: RiskEstimator) -> bool:
    r""" Checks if the specified \p loss object is a two-step method """
    return isinstance(loss, _WUBase)


# noinspection PyPep8Naming
class _WUBase(_NNBase, ABC):
    r""" Implements key methods and fields for the weighted unlabeled set"""
    def __init__(self, gamma: float, train_prior: float, test_prior: float, abs_nn: bool,
                 train_loss: Callable, valid_loss: Optional[Callable] = None,
                 s1_method: Step1Method = Step1Method.SOFT):
        super().__init__(gamma=gamma, train_loss=train_loss, valid_loss=valid_loss,
                         abs_nn=abs_nn)

        # Sanity check the priors the store them
        assert 0 < train_prior < 1, "Invalid TRAINING prior"
        assert 0 < test_prior < 1, "Invalid TEST prior"
        self._tr_prior = train_prior
        self._te_prior = test_prior
        # Trial code to round the soft weights
        self._s1_method = s1_method

    def calc_train_loss(self, dec_scores: Tensor, lbls: Tensor, sigma_x: Tensor, tk: Tensor):
        r""" Calculates the loss using the TRAINING specific loss function """
        sigma = self._select_s1_sigma(sigma_x=sigma_x, tk=tk)
        return super().calc_train_loss(dec_scores=dec_scores, lbls=lbls, sigma_x=sigma, tk=sigma)

    def calc_validation_loss(self, dec_scores: Tensor, lbls: Tensor, sigma_x: Tensor,
                             tk: Tensor):
        r""" Calculates the loss using the VALIDATION specific loss function """
        sigma = self._select_s1_sigma(sigma_x=sigma_x, tk=tk)
        return super().calc_validation_loss(dec_scores=dec_scores, lbls=lbls, sigma_x=sigma,
                                            tk=sigma)

    def _select_s1_sigma(self, sigma_x: Tensor, tk: Tensor) -> Tensor:
        r""" Standard method to select which sigma is used """
        if self._s1_method == Step1Method.SOFT:
            return sigma_x
        if self._s1_method == Step1Method.HARD:
            return sigma_x.round().clamp(0., 1.)
        if self._s1_method == Step1Method.TOP_K:
            return tk
        raise ValueError("Unknown Step-1 Method")

    @staticmethod
    def _check_weights_valid(dec_scores, weights: Tensor) -> None:
        r""" Verify that the weight information is not corrupted """
        if dec_scores.shape[0] != weights.shape[0]:
            raise ValueError("Weights and decision function scores length mismatch")

        if len(weights.shape) != 1:
            raise ValueError("Weights should be a vector")

        if float(weights.max().item()) > 1:
            raise ValueError("Maximum weight must be less than or equal to 1.")
        if float(weights.min().item()) < 0:
            raise ValueError("Minimum weight must be greater than or equal to 0.")

    def _get_weighted_neg_loss(self, lbl_loss: Tensor, sigma_x: Tensor,
                               u_tr_mask: Tensor) -> Tensor:
        r"""
        Calculates the weighted negative loss using the unlabeled training set.  Weights by
        :math:`1 - \pi_{\text{te}}` to eliminate need to do this later in multiple points in
        the code
        """
        if not self._has_any(u_tr_mask):
            return torch.zeros((), device=TORCH_DEVICE)

        # Weighted loss from unlabeled set
        w_loss = lbl_loss[u_tr_mask] * (sigma_x[u_tr_mask].neg() + 1)
        assert len(w_loss.shape) == 1, "Weighted loss shape should be a vector"
        return (1. - self._te_prior) / (1. - self._tr_prior) * w_loss.mean()

    def _clean_loss_inputs(self, dec_scores: Tensor, lbls: Tensor, sigma_x: Tensor):
        r""" Checks and cleans the loss inputs before running loss """
        if len(dec_scores.shape) == 2: dec_scores = dec_scores.squeeze(dim=1)
        if len(lbls.shape) == 2: lbls = lbls.squeeze(dim=1)
        self._verify_loss_inputs(dec_scores, lbls)

        if len(sigma_x.shape) == 2: sigma_x = sigma_x.squeeze(dim=1)
        self._check_weights_valid(dec_scores, sigma_x)
        return dec_scores, lbls, sigma_x

    def _get_prior_str(self) -> str:
        r""" Helper method used to get a string regarding the prior settings """
        return f"tr{self._tr_prior:.2f}_te{self._te_prior:.2f}".replace(".", "p")

    def _get_s1_method_str(self) -> str:
        r""" Standardized method to add the step 1 method """
        return self._s1_method.value.lower()


# noinspection PyPep8Naming
class WUU(_WUBase):
    def __init__(self, train_prior: float, test_prior: float, u_tr_label: int,
                 u_te_label: Optional[Union[int, Set[int]]],
                 train_loss: Callable, valid_loss: Optional[Callable] = None, gamma: float = 1.,
                 abs_nn: bool = False, s1_method: Step1Method = Step1Method.SOFT):
        super().__init__(gamma=gamma, train_prior=train_prior, test_prior=test_prior, abs_nn=abs_nn,
                         train_loss=train_loss, valid_loss=valid_loss, s1_method=s1_method)

        self._u_tr_labels = u_tr_label
        if isinstance(u_te_label, int):
            u_te_label = {u_te_label}
        self._u_te_labels = u_te_label

    def name(self) -> str:
        r""" Name of the weighted negative-unlabeled learner """
        return "wUU"

    def _loss(self, dec_scores: Tensor, lbls: Tensor, sigma_x: Tensor,
              f_loss: Callable) -> LossInfo:
        r"""
        nnWUU is similar in structure to nnPU, but relies on using a weighted negative set.

        :param f_loss: Surrogate loss function to use in the calculation
        :param dec_scores: Decision function value
        :param lbls: Labels for each sample in \p.
        :param sigma_x: Training unlabeled set's weights, i.e., :math:`\sigma(x)`
        :return: Named tuple with value used for loss and the one used for the gradient
        """
        dec_scores, lbls, sigma_x = self._clean_loss_inputs(dec_scores=dec_scores, lbls=lbls,
                                                            sigma_x=sigma_x)

        u_tr_mask = self._build_mask(lbls, self._u_tr_labels)
        u_te_mask = self._build_mask(lbls, self._u_te_labels)
        has_u_te = self._has_any(u_te_mask)

        ones = torch.ones([dec_scores.shape[0]], device=TORCH_DEVICE)

        pos_lbl_loss, neg_lbl_loss = f_loss(dec_scores, ones), f_loss(dec_scores, -ones)
        # Get the neg risk
        neg_risk = self._get_weighted_neg_loss(lbl_loss=neg_lbl_loss, u_tr_mask=u_tr_mask,
                                               sigma_x=sigma_x)

        if not has_u_te:
            pos_risk = torch.zeros((), device=TORCH_DEVICE)
        else:
            u_pos_risk = pos_lbl_loss[u_te_mask].mean()
            n_pos_risk = self._get_weighted_neg_loss(lbl_loss=pos_lbl_loss, u_tr_mask=u_tr_mask,
                                                     sigma_x=sigma_x)
            pos_risk = u_pos_risk - n_pos_risk

        # Risk terms deliberately swapped versus nnPU since this is based on an NU risk not PU
        return self._calc_loss_info(always_pos_risk=neg_risk, nn_risk=pos_risk)


class APNU(_WUBase):
    """ Non-negative Positive, Negative, Unlabeled learner """

    def __init__(self, train_prior: float, test_prior: float, rho: float,
                 train_loss: Callable, valid_loss: Optional[Callable] = None, gamma: float = 1.,
                 abs_nn: bool = False, s1_method: Step1Method = Step1Method.SOFT):
        r"""
        :param rho: Trade-off parameter between NU and PN losses
        :param gamma: Gamma learning rate attenuating parameter
        :param train_loss: Loss function underlying the classifier
        :param valid_loss: Optional validation loss function.  If not specified, uses
                           \p train_loss for validation.
        """
        super().__init__(gamma=gamma, train_prior=train_prior, test_prior=test_prior, abs_nn=abs_nn,
                         train_loss=train_loss, valid_loss=valid_loss, s1_method=s1_method)
        if not (0 <= rho <= 1):
            raise ValueError("rho must be in the range [0,1]")
        self.rho = rho

    def name(self) -> str:
        r""" Name of the non-negative PNU learner """
        return "aPNU"

    # noinspection DuplicatedCode
    def _loss(self, dec_scores: Tensor, lbls: Tensor, sigma_x: Tensor,
              f_loss: Callable) -> LossInfo:
        r"""
        nnPNU combine the ideas of nnPU and nnWUU.

        :param f_loss: Surrogate loss function to use in the calculation
        :param dec_scores: Decision function value
        :param lbls: Labels for each sample in \p.
        :param sigma_x: Training unlabeled set's weights, i.e., :math:`\sigma(x)`
        :return: Named tuple with value used for loss and the one used for the gradient
        """
        dec_scores, lbls, sigma_x = self._clean_loss_inputs(dec_scores=dec_scores, lbls=lbls,
                                                            sigma_x=sigma_x)

        # Get all labeled risks
        ones = torch.ones([dec_scores.shape[0]], device=TORCH_DEVICE)
        pos_lbl_loss, neg_lbl_loss = f_loss(dec_scores, ones), f_loss(dec_scores, -ones)

        def _get_lbl_rsk(mask: Tensor, lbl_loss: Tensor):
            r""" Consolidates check for any and handles return value"""
            if not self._has_any(mask):
                return torch.zeros((), device=TORCH_DEVICE)
            return lbl_loss[mask].mean()

        # First term: R_{tr-p}^{p}
        p_mask = self._build_mask(lbls, Labels.Training.POS.value)
        pos_risk = (1 - self.rho) * self._te_prior * _get_lbl_rsk(p_mask, pos_lbl_loss)

        # Second Term: R_{n-tr-u}^{-}
        u_tr_mask = self._build_mask(lbls, Labels.Training.U_TRAIN.value)
        n_tr_neg_risk = self._get_weighted_neg_loss(lbl_loss=neg_lbl_loss, sigma_x=sigma_x,
                                                    u_tr_mask=u_tr_mask)

        # Third Term:
        u_te_mask = self._build_mask(lbls, Labels.Training.U_TEST.value)
        u_te_pos_risk = _get_lbl_rsk(u_te_mask, pos_lbl_loss)
        n_tr_pos_risk = self._get_weighted_neg_loss(lbl_loss=pos_lbl_loss, sigma_x=sigma_x,
                                                    u_tr_mask=u_tr_mask)

        # Risk terms deliberately swapped versus nnPU since this is based on an NU risk not PU
        return self._calc_loss_info(always_pos_risk=pos_risk + n_tr_neg_risk,
                                    nn_risk=self.rho * (u_te_pos_risk - n_tr_pos_risk))
