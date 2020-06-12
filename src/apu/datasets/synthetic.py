__all__ = ["Module", "generate_data"]

import random
from typing import Optional, Set, Tuple

import torch
from torch import Tensor
import torch.nn as nn

from . import utils as ds_tools
from .types import Centroid, Labels, APU_Module, TensorGroup
from .. import _config as config

MAX_EPS = 1E-4  # Maximum difference allowed in probability vector


class Module(APU_Module):
    r""" Module used for training with Synthetic data"""
    LIN_NAME = "Linear"

    def __init__(self, in_dim: int = Centroid.FEATURE_DIM):
        super().__init__()
        lin = nn.Linear(in_dim, 1)
        lin.weight.data.normal_(0, 0.1)
        lin.bias.data.fill_(0)
        self._model.add_module(self.LIN_NAME, lin)

    def decision_boundary(self) -> Tuple[float, float]:
        r"""
        Returns the decision boundary in the form :math:`y = mx + b`
        Decision boundary at :math:`a0 * x0 + a1 x1 + b = 0` so need to negate terms
        """
        linear = self._model.__getattr__(self.LIN_NAME)
        weights, bias = linear.weight, linear.bias
        m = -weights[0, 0] / weights[0, 1]
        b = -bias / weights[0, 1]
        return float(m.item()), float(b.item())


def _generate_single_ds(prior: float, num_sample: int, p_cents: Set[Centroid], p_bias: Tensor,
                        n_cents: Optional[Set[Centroid]],
                        n_bias: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    r"""
    Generates a single X/Y tensor given a pair of centroid sets

    :param prior: Positive class prior probability :math:`\Pr[y = +1]`
    :param num_sample: Total number of samples between the positive and negative class
    :param p_cents: Positive centroid definitions
    :param p_bias: Bias vector used for sampling amongst all POSITIVE elements
    :param n_cents: Negative centroid definitions
    :param n_bias: Bias vector used for sampling amongst all NEGATIVE elements
    :return: Tensors for the centroids X and Y
    """
    if prior > 0 and not p_cents:
        raise ValueError("Non-zero positive prior but no positive centroids")
    if prior < 1 and not n_cents:
        raise ValueError("Positive prior not 1 but no negative centroids")
    k = ds_tools.binom_sample(prior, num_sample)

    fields = ((Labels.POS, k, p_cents, p_bias),
              (Labels.NEG, num_sample - k, n_cents, n_bias))
    points = []
    for lbl, n, cents, bias_vec in fields:
        if n == 0: continue
        assert cents, "Generating points from centroids which do not exist"

        p_sum = float(bias_vec.sum().item())
        assert abs(p_sum - 1) < MAX_EPS, "Invalid probability sample vector"

        n_per_cent = ds_tools.multinomial_sample(n, bias_vec)
        points += [(c.gen_points(1), torch.full([1], lbl, dtype=torch.int32))
                   for c, n_c in zip(cents, n_per_cent) for _ in range(int(n_c))]
    random.shuffle(points)

    # Construct the X and Y tensors respectively
    out = [torch.cat([x[i] for x in points]) for i in range(len(points[0]))]
    return out[0], out[1]


def generate_data() -> TensorGroup:
    r""" Generates the training (including unlabeled) and test data """
    n_cents = {x for x in config.CENTROIDS if x.label == Labels.Synthetic.NEG}
    n_tr_bias = ds_tools.construct_bias_vec(n_cents, "neg_train_bias")
    n_te_bias = ds_tools.construct_bias_vec(n_cents, "neg_test_bias")

    tr_pos = {Labels.Synthetic.POS_TR.value, Labels.Synthetic.POS_BOTH.value}
    p_tr_cents = {x for x in config.CENTROIDS if x.label.value in tr_pos}
    p_tr_bias = ds_tools.construct_bias_vec(p_tr_cents, "pos_train_bias")

    te_pos = {Labels.Synthetic.POS_TE.value, Labels.Synthetic.POS_BOTH.value}
    p_te_cents = {x for x in config.CENTROIDS if x.label.value in te_pos}
    p_te_bias = ds_tools.construct_bias_vec(p_te_cents, "pos_test_bias")

    tg = TensorGroup()
    if config.TRAIN_PRIOR > 0:
        tg.p_x, _ = _generate_single_ds(1., config.N_P, p_tr_cents, p_tr_bias, None, None)
    tg.u_tr_x, tg.u_tr_y = _generate_single_ds(config.TRAIN_PRIOR, config.N_U_TRAIN,
                                               p_tr_cents, p_tr_bias, n_cents, n_tr_bias)
    tg.test_x_tr, tg.test_y_tr = _generate_single_ds(config.TRAIN_PRIOR, config.N_TEST,
                                                     p_tr_cents, p_tr_bias, n_cents, n_tr_bias)

    tg.u_te_x, tg.u_te_y = _generate_single_ds(config.TEST_PRIOR, config.N_U_TEST,
                                               p_te_cents, p_te_bias, n_cents, n_te_bias)
    # For inductive testing
    tg.test_x, tg.test_y = _generate_single_ds(config.TEST_PRIOR, config.N_TEST,
                                               p_te_cents, p_te_bias, n_cents, n_te_bias)

    return tg
