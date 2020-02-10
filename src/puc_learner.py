import logging
import time
from typing import Optional, Tuple

import numpy as np

from torch import Tensor

import puc
from puc import PUcKernelType
from apu import config
from apu.datasets.types import TensorGroup
from apu.utils import ViewTo1D


class PUcLearner:
    r""" Encapsulates learning for the PUc learner"""
    BASE_NAME = "puc"

    def __init__(self, prior: Optional[float] = None):
        self._train_start = None
        try:
            self._kernel_type = PUcKernelType[config.KERNEL_TYPE.upper()]
        except KeyError:
            raise ValueError(f"Unknown kernel type: {config.KERNEL_TYPE.upper()}")

        self._model = None
        self._gamma_list = np.mgrid[.1:.9:9j]
        self._lambda_list = np.logspace(-6, 1, 20)
        self._prior = prior if prior is not None else config.TRAIN_PRIOR

        self._flatten = ViewTo1D()

    def _flatten_to_np(self, x: Tensor) -> np.ndarray:
        r""" Takes a \p torch \p Tensor object and flattens it to be 1D for an SVM """
        return self._flatten(x).cpu().numpy()

    @classmethod
    def build_name(cls, prior: float) -> str:
        r""" Helper method standard  """
        return f"{cls.BASE_NAME}_tr{prior:.2}".replace(".", "_")

    def name(self) -> str:
        r""" Name of the learner"""
        return self.build_name(self._prior)

    def fit(self, ts_grp: TensorGroup):
        r""" Fit the PUc learner """
        msg = f"Training PUc with {config.KERNEL_TYPE.upper()} kernel & prior {self._prior:.2}"
        logging.info(f"Starting: {msg}")
        self._train_start = time.time()

        # Since PUc uses an SVM, must be 1D data vector
        p_x = self._flatten_to_np(ts_grp.p_x)
        u_tr_x = self._flatten_to_np(ts_grp.u_tr_x)
        u_te_x = self._flatten_to_np(ts_grp.u_te_x)

        # self._model = puc.sq_pu(xp=p_x, xu=u_tr_x,
        self._model = puc.fit(xp_tr=p_x, xu_tr=u_tr_x, xu_te=u_te_x,
                              gamma_list=self._gamma_list,
                              kertype=self._kernel_type.value,
                              prior=self._prior,
                              lambda_list=self._lambda_list)
        logging.info(f"COMPLETED: {msg}")

    def decision_function(self, x: Tensor) -> np.ndarray:
        r""" Predicts the tensor labels """
        assert self._model is not None, "Model not trained"
        return puc.decision_function(self._model, self._flatten_to_np(x))

    def decision_boundary(self) -> Tuple[float, float]:
        r""" Gets linear decision boundary """
        assert self._model is not None, "Model not trained"
        assert self._kernel_type == PUcKernelType.LINEAR, "Only linear boundary supported"

        alpha = self._model["alpha"]
        m = -alpha[0] / alpha[1]
        b = -alpha[2] / alpha[1]
        return m, b

    def predict(self, x: Tensor) -> np.ndarray:
        r""" Predicts the tensor labels """
        assert self._model is not None, "Model not trained"
        return puc.predict(self._model, self._flatten_to_np(x))
