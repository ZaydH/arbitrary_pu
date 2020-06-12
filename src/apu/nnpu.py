__all__ = ["_NNBase", "APNU", "NNPU", "WUU"]

from abc import ABC
from typing import Callable, Optional, Set, Union

import numpy as np

import torch
from torch import Tensor

from .datasets.types import Labels
from .types import LossInfo, RiskEstimator
from .utils import TORCH_DEVICE


class _NNBase(RiskEstimator, ABC):
    class Config:
        # GAMMA = 1
        BETA = 0

    def __init__(self, gamma: float, train_loss: Callable, valid_loss: Optional[Callable] = None,
                 abs_nn: bool = False):
        r"""
        :param gamma: Attenuates non-negative gradient
        :param train_loss: Loss function underlying the classifier
        :param valid_loss: Optional validation loss function.  If not specified, uses
                           \p train_loss for validation.
        """
        # :param use_nnpu: If \p True, use nnPU loss.  Otherwise, use uPU.
        if valid_loss is not None: valid_loss = train_loss
        super().__init__(train_loss, valid_loss)

        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        if self.Config.BETA < 0:
            raise ValueError("beta must be non-negative")

        self.gamma = gamma
        self.beta = self.Config.BETA
        self._abs_nn = abs_nn

    @property
    def is_nn(self) -> bool:
        r""" Returns \p True if the loss is non-negative \p False. """
        # return self._is_nn
        return True

    @classmethod
    def _verify_loss_inputs(cls, dec_scores: Tensor, labels: Tensor) -> None:
        r""" Sanity check the inputs """
        assert len(dec_scores.shape) == 1, "dec_scores should be a vector"
        assert len(labels.shape) == 1, "labels should be a vector"
        assert dec_scores.shape[0] == labels.shape[0], "Batch size mismatch"

        assert dec_scores.dtype == torch.float, "dec_scores tensor must be float"
        # assert labels.dtype == torch.int64, "labels must be integers"

    def _calc_loss_info(self, always_pos_risk: Tensor, nn_risk: Tensor) -> LossInfo:
        r"""
        Standardizes the non-negativity risk
        :param always_pos_risk: Risk term that is exclusively positive-valued.  This is not
                                necessarily the positive labeled risk since for nnWUU the always
                                positive risk term is actually the negative labeled risk.
        :param nn_risk: Risk term that cannot be negative.
        :return: Loss information with loss and gradient variables.
        """
        if self._abs_nn:
            loss = always_pos_risk + nn_risk.abs()
            return LossInfo(te_loss=loss, grad_var=loss)

        min_clamp = -self.beta if self.is_nn else -np.inf
        loss = gradient_var = always_pos_risk + nn_risk.clamp_min(min_clamp)
        if self.is_nn and nn_risk < -self.beta:
            gradient_var = -self.gamma * nn_risk
        return LossInfo(te_loss=loss, grad_var=gradient_var)


class NNPU(_NNBase):
    """ Non-Negative PU learner based on the work of Kiryo et al. """

    def __init__(self, prior: float,
                 train_loss: Callable, valid_loss: Optional[Callable] = None,
                 only_u_train: bool = False, only_u_test: bool = False,
                 gamma: float = 1., special_suffix: str = "",
                 abs_nn: bool = False):
        r"""
        :param prior: Positive class prior probability, i.e., :math:`\Pr[y = +1]`
        :param gamma: Attenuates non-negative gradient
        :param train_loss: Loss function underlying the classifier
        :param valid_loss: Optional validation loss function.  If not specified, uses
                           \p train_loss for validation.
        :param only_u_train: Only use the TRAINING unlabeled set and ignore the TEST unlabeled set.
        :param only_u_test: Only use the TEST unlabeled set and ignore the TRAINING unlabeled set.
        """
        if only_u_train and only_u_test:
            raise ValueError("Cannot specify to use only train AND only test")

        super().__init__(gamma=gamma, train_loss=train_loss, valid_loss=valid_loss,
                         abs_nn=abs_nn)

        if not (0 < prior < 1):
            raise ValueError("The class prior should be in (0, 1)")
        self.prior = prior

        self._only_u_train = only_u_train
        self._only_u_test = only_u_test

        self._special_suffix = special_suffix

    def name(self) -> str:
        r""" Name of the loss.  Includes suffix for unlabeled set used """
        # return "nnPU" if self.is_nn else "uPU"
        fields = ["nnPU", self._get_name_suffix()]
        if self._special_suffix:
            fields.append(self._special_suffix)
        return "_".join(fields)

    def _get_name_suffix(self) -> str:
        r""" Gets suffix name based on the configuration """
        fields = []
        if self._only_u_train:
            fields.append("tr")
        elif self._only_u_test:
            fields.append("te")
        else:
            fields.append("all")
        if self._abs_nn:
            fields.append("abs")
        return "_".join(fields)

    def _loss(self, dec_scores: Tensor, lbls: Tensor, _: Tensor, f_loss: Callable) -> LossInfo:
        r"""
        nnPU uses separate approaches for determining the loss and variable used for calculating
        the gradient.
        :param f_loss: Surrogate loss function to use in the calculation
        :param dec_scores: Decision function value
        :param lbls: Labels for each sample in \p.
        :return: Named tuple with value used for loss and the one used for the gradient
        """
        if len(dec_scores.shape) == 2: dec_scores = dec_scores.squeeze(dim=1)
        if len(lbls.shape) == 2: lbls = lbls.squeeze(dim=1)

        self._verify_loss_inputs(dec_scores, lbls)

        # Mask used to filter the dec_scores tensor and in loss calculations
        p_mask = self._build_mask(y=lbls, lbl_vals=Labels.Training.POS.value)
        u_mask = self._build_u_mask(lbls=lbls)

        has_p, has_u = self._has_any(p_mask), self._has_any(u_mask)

        ones = torch.ones([dec_scores.shape[0]], device=TORCH_DEVICE)
        neg_lbl_risk = f_loss(dec_scores, -ones)
        neg_risk = neg_lbl_risk[u_mask].mean() if has_u else torch.zeros((), device=TORCH_DEVICE)
        if has_p:
            y_pos = f_loss(dec_scores[p_mask], ones[p_mask])
            pos_risk = self.prior * y_pos.mean()

            neg_risk -= self.prior * neg_lbl_risk[p_mask].mean()
        else:
            pos_risk = torch.zeros((), device=TORCH_DEVICE)  # Needs to be have len(shape) == 0

        # Different from wUU.  nnPU always does non-negativity correction on negative risk term
        return self._calc_loss_info(always_pos_risk=pos_risk, nn_risk=neg_risk)

    def _build_u_mask(self, lbls: Tensor) -> Tensor:
        r""" Builds the unlabeled set mask """
        u_lbls = []
        if not self._only_u_test:
            u_lbls.append(Labels.Training.U_TRAIN.value)
        if not self._only_u_train:
            u_lbls.append(Labels.Training.U_TEST.value)
        return self._build_mask(y=lbls, lbl_vals=u_lbls)


class _WUBase(_NNBase, ABC):
    r""" Implements key methods and fields for the weighted unlabeled set"""
    def __init__(self, gamma: float, train_prior: float, test_prior: float, abs_nn: bool,
                 train_loss: Callable, valid_loss: Optional[Callable] = None):
        super().__init__(gamma=gamma, train_loss=train_loss, valid_loss=valid_loss,
                         abs_nn=abs_nn)

        # Sanity check the priors the store them
        assert 0 < train_prior < 1, "Invalid TRAINING prior"
        assert 0 < test_prior < 1, "Invalid TEST prior"
        self._tr_prior = train_prior
        self._te_prior = test_prior

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

    def get_prior_str(self) -> str:
        r""" Helper method used to get a string regarding the prior settings """
        return f"tr{self._tr_prior:.2}_te{self._te_prior:.2}".replace(".", "_")


# noinspection PyPep8Naming
class WUU(_WUBase):
    def __init__(self, train_prior: float, test_prior: float, u_tr_label: int,
                 u_te_label: Optional[Union[int, Set[int]]],
                 train_loss: Callable, valid_loss: Optional[Callable] = None,
                 gamma: float = 1., abs_nn: bool = False):
        super().__init__(gamma=gamma, train_prior=train_prior, test_prior=test_prior,
                         train_loss=train_loss, valid_loss=valid_loss, abs_nn=abs_nn)

        self._u_tr_labels = u_tr_label
        if isinstance(u_te_label, int):
            u_te_label = {u_te_label}
        self._u_te_labels = u_te_label

    def name(self) -> str:
        r""" Name of the weighted negative-unlabeled learner """
        fields = ["wUU", self.get_prior_str()]
        if self._abs_nn:
            fields.append("abs")
        return "_".join(fields)

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


# noinspection PyPep8Naming
class APNU(_WUBase):
    """ Non-negative Positive, Negative, Unlabeled learner """

    def __init__(self, train_prior: float, test_prior: float, rho: float,
                 train_loss: Callable, valid_loss: Optional[Callable] = None,
                 gamma: float = 1., abs_nn: bool = False):
        r"""
        :param rho: Trade-off parameter between NU and PN losses
        :param gamma: Gamma learning rate attenuating parameter
        :param train_loss: Loss function underlying the classifier
        :param valid_loss: Optional validation loss function.  If not specified, uses
                           \p train_loss for validation.
        """
        super().__init__(gamma=gamma, train_prior=train_prior, test_prior=test_prior,
                         train_loss=train_loss, valid_loss=valid_loss, abs_nn=abs_nn)
        if not (0 <= rho <= 1):
            raise ValueError("rho must be in the range [0,1]")
        self.rho = rho

    def name(self) -> str:
        r""" Name of the non-negative PNU learner """
        fields = ["aPNU", f"{self.rho:.2}", self.get_prior_str()]
        if self._abs_nn:
            fields.append("abs")
        return "_".join(fields).replace(".", "_")

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
