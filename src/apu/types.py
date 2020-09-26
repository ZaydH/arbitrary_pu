# -*- utf-8 -*-
r"""
    types.py
    ~~~~~~~~~~~~~~~

    Object types used by the aPU Learners

    :copyright: (c) 2020 by Zayd Hammoudeh.
    :license: , see MIT for more details.
"""

__all__ = ["LearnerParams",
           "ListOrInt", "LossInfo",
           "OptBool", "OptDataBunch", "OptDict", "OptInt",
           "OptFloat", "OptStr", "OptTensor",
           "PathOrStr", "RiskEstimator",
           "TensorTuple", "TorchOrNp"
           ]

from abc import ABC, abstractmethod
from argparse import Namespace
import collections
import dataclasses
from enum import Enum
from pathlib import Path
from typing import Any, Callable, List, Optional, Set, Tuple, Union

import numpy as np
from pandas import DataFrame

from fastai.basic_data import DataBunch
import torch
from torch import Tensor

OptBool = Optional[bool]
OptCallable = Optional[Callable]
OptDataBunch = Optional[DataBunch]
OptDataFrame = Optional[DataFrame]
OptDict = Optional[dict]
OptFloat = Optional[float]
OptInt = Optional[int]
OptListInt = Optional[List[int]]
OptListStr = Optional[List[str]]
OptNamespace = Optional[Namespace]
OptStr = Optional[str]
OptTensor = Optional[Tensor]

ListOrInt = Union[int, List[int]]
SetListOrInt = Union[int, Set[int], List[int]]
SetOrList = Union[List[Any], Set[Any]]

PathOrStr = Union[Path, str]

TensorTuple = Tuple[Tensor, Tensor]
TorchOrNp = Union[Tensor, np.ndarray]

LossInfo = collections.namedtuple("LossInfo", ["te_loss", "grad_var"])


class RiskEstimator(ABC):
    def __init__(self, train_loss: Callable, validation_loss: Callable):
        self.tr_loss = train_loss
        self.val_loss = validation_loss

    def calc_train_loss(self, dec_scores: Tensor, lbls: Tensor, sigma_x: Tensor, tk: Tensor):
        r""" Calculates the loss using the TRAINING specific loss function """
        return self._loss(dec_scores, lbls, sigma_x, self.tr_loss)

    def calc_validation_loss(self, dec_scores: Tensor, lbls: Tensor, sigma_x: Tensor,
                             tk: Tensor):
        r""" Calculates the loss using the VALIDATION specific loss function """
        return self._loss(dec_scores, lbls, sigma_x, self.val_loss)

    @abstractmethod
    def _loss(self, dec_scores: Tensor, lbls: Tensor, sigma_x: Tensor, f_loss: Callable):
        r"""
        Shared function for training & validation loss.
        :param dec_scores: Decision score value(s)
        :param lbls: Labels indicating train, unlabeled train, and unlabeled test.
        :param sigma_x: May not be used by many loss functions.  Represents from the aPU paper
                        :math:`\sigma(x) = \Pr[Y = +1 | x]`.
        :param f_loss: Loss function :math:`\ell`
        :return: Loss tensor
        """

    @abstractmethod
    def name(self) -> str:
        r""" Name of the loss function """

    @property
    @abstractmethod
    def is_nn(self) -> bool:
        r""" Returns \p True if the loss is non-negative \p False. """

    @staticmethod
    def _build_mask(y: Tensor, lbl_vals: Union[int, List[int], Set[int]]) -> Tensor:
        r""" Construct the mask for the indices with a label matching \p lbl_vals """
        assert len(y.shape) == 1 or (len(y.shape) == 2 and y.shape[1] == 1), "Unexpected y shape"
        # If just a single label, simple compare enough
        if isinstance(lbl_vals, int):
            return y == lbl_vals

        # Build labels iteratively
        mask = torch.full([y.shape[0]], False, dtype=torch.bool, device=y.device)
        for lbl in lbl_vals:
            mask |= y == lbl
        return mask

    def change_loss_functions(self, train_loss: Optional[Callable] = None,
                              validation_loss: Optional[Callable] = None) -> None:
        r"""
        Changes either the training or validation loss functions.  If either loss function parameter
        is set to None, it is left unchanged
        """
        if train_loss is None and validation_loss is None:
            raise ValueError("Training and validation losses cannot be None")

        if train_loss is not None: self.tr_loss = train_loss
        if validation_loss is not None: self.val_loss = validation_loss

    @staticmethod
    def _has_any(mask: Tensor) -> bool:
        r""" Checks if the mask has any set to \p True """
        assert mask.dtype == torch.bool, "Mask should be a Boolean Tensor"
        return bool(mask.any().item())


@dataclasses.dataclass(order=True)
class LearnerParams:
    r""" Learner specific parameters """
    class Attribute(Enum):
        LEARNING_RATE = "lr"
        WEIGHT_DECAY = "wd"
        GAMMA = "gamma"
        # NUM_FF_LAYERS = "num_ff_layers"
        # NUM_SIGMA_LAYERS = "num_sigma_layers"

    learner_name: str

    lr: float = None
    wd: float = None
    gamma: float = None

    # num_ff_layers: int = None
    # num_sigma_layers: int = None

    def set_attr(self, attr_name: str, value: Union[int, float]) -> None:
        r""" Enhanced set attribute method that has enhanced checking """
        try:
            # Allow short field name or longer attribute name
            attr_name = attr_name.lower()
            self.__getattribute__(attr_name)
        except AttributeError:
            try:
                attr_name = self.Attribute[attr_name.upper()].value
                self.__getattribute__(attr_name)
            except KeyError:
                raise AttributeError(f"No attribute \"{attr_name}\"")

        for field in dataclasses.fields(self):
            if field.name == attr_name: break
        else:
            raise ValueError(f"Cannot find field \"{attr_name}\"")

        assert isinstance(value, field.type), "Type mismatch when setting"
        self.__setattr__(attr_name, value)

    def get_attr(self, attr_name: str) -> Optional[Union[int, float]]:
        r""" Attribute accessor with more robust handling of attribute name """
        attr_name = attr_name.lower()
        try:
            return self.__getattribute__(attr_name)
        except AttributeError:
            raise AttributeError("No attribute \"attr_name\"")
