__all__ = ["BASE_DIR",
           "ClassifierBlock", "DATA_DIR", "IS_CUDA",
           "LOG_LEVEL", "LOGGER_NAME", "LOG_DIR",
           "NEWSGROUPS_DIR", "NUM_WORKERS",
           "PLOTS_DIR", "RES_DIR", "TORCH_DEVICE",
           "ViewModule", "ViewTo1D",
           "configure_dataset_args", "construct_filename",
           "export_step1_results",
           "log_decision_boundary", "log_seeds",
           "shuffle_tensors"
           ]

try:
    # noinspection PyUnresolvedReferences
    from matplotlib import use
    use('Agg')
    # noinspection PyUnresolvedReferences
    import matplotlib.pyplot as plt
except ImportError:
    # raise ImportError("Unable to import matplotlib")
    pass

import copy
from enum import Enum
import logging
from pathlib import Path
import random
import re
import socket
import sys
import time
from typing import List, Optional, Set, Tuple

import numpy as np

from fastai.basic_data import DeviceDataLoader
import torch
from torch import Tensor
import torch.backends.cudnn
import torch.nn as nn
from torch.optim.lbfgs import LBFGS

from . import _config as config
from .datasets import cifar, libsvm_ds, mnist, newsgroups, open_ml, spam, synthetic
from .datasets import utils as ds_utils
from .datasets.types import APU_Module, BaseFFModule, SpamFFModule, TensorGroup
from .plot_utils import plot_scatter, plot_histogram
from .types import RiskEstimator

IS_CUDA = torch.cuda.is_available()
TORCH_DEVICE = torch.device("cuda:0" if IS_CUDA else "cpu")
APU_Module.TORCH_DEVICE = TORCH_DEVICE

NUM_WORKERS = 0 if IS_CUDA else 0
LOG_LEVEL = logging.DEBUG
LOGGER_NAME = "apu"

shuffle_tensors = ds_utils.shuffle_tensors


BASE_DIR = Path(".").absolute()
DATA_DIR = BASE_DIR / ".data"
NEWSGROUPS_DIR = DATA_DIR / "20_newsgroups"

LOG_DIR = BASE_DIR / "logs"
PLOTS_DIR = BASE_DIR / "plots"
RES_DIR = BASE_DIR / "res"


def construct_filename(prefix: str, out_dir: Path, file_ext: str,
                       add_timestamp: bool = False) -> Path:
    r""" Standardize naming scheme for the filename """
    def _classes_to_str(cls_set: Set[Enum]) -> str:
        # noinspection PyTypeChecker
        return ",".join([str(x) for x in sorted(cls_set)])

    fields = [prefix] if prefix else []
    fields += [config.DATASET.name.lower().replace("_", "-")]
    # Prior information
    fields += [f"pr-tr={config.TRAIN_PRIOR:.2f}", f"pr-te={config.TEST_PRIOR:.2f}"]
    fields += [f"np={config.N_P}", f"ntr={config.N_U_TRAIN}", f"nte={config.N_U_TEST}"]
    if not config.DATASET.is_synthetic():
        fields += [f"p-tr={_classes_to_str(config.POS_TRAIN_CLASSES)}",
                   f"p-te={_classes_to_str(config.POS_TEST_CLASSES)}",
                   f"neg={_classes_to_str(config.NEG_CLASSES)}"]
    if config.POS_TRAIN_BIAS is not None: fields.append("p-tr-bias")
    if config.POS_TEST_BIAS is not None: fields.append("p-te-bias")

    if config.NEG_TRAIN_BIAS is not None:
        if config.NEG_TRAIN_BIAS == config.NEG_TEST_BIAS:
            fields.append("n-bias")
        else:
            fields.append("n-tr-te-bias")
    elif config.NEG_TEST_BIAS is not None:
        fields.append("n-te-bias")

    if add_timestamp:
        time_str = time.strftime("%Y-%m-%d-%H-%M-%S")
        fields.append(time_str)

    if file_ext[0] != ".": file_ext = "." + file_ext
    fields[-1] += file_ext

    # Add the dataset name to better organize files
    out_dir /= config.DATASET.name.lower()
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / "_".join(fields)


def save_module(module: nn.Module, filepath: Path) -> None:
    r""" Save the specified \p model to disk """
    filepath.parent.mkdir(parents=True, exist_ok=True)
    torch.save(module.state_dict(), str(filepath))


def load_module(module: nn.Module, filepath: Path):
    r"""
    Loads the specified model in file \p filepath into \p module and then returns \p module.

    :param module: \p Module where the module on disk will be loaded
    :param filepath: File where the \p Module is stored
    :return: Loaded model
    """
    # Map location allows for mapping model trained on any device to be loaded
    module.load_state_dict(torch.load(str(filepath), map_location=TORCH_DEVICE))
    module.eval()
    return module


def set_random_seeds(seed: Optional[int] = None) -> None:
    r"""
    Sets random seeds to avoid non-determinism
    :See: https://pytorch.org/docs/stable/notes/randomness.html
    """
    if seed is not None:
        logging.warning(f"Debug mode enabled. Seed {seed} used")
        np_seed = seed
        disable_torch_backend_randomness()
    else:
        seed = torch.initial_seed() & ((0x1 << 63) - 1)
        logging.debug(f"Initial torch seed {seed} used to see all random number generators")

        np_seed = int(abs(seed) & 0x7FFFFFFF)  # Max numpy seed is 2^32 - 1
        logging.debug(f"Initial numpy seed {np_seed} derived from torch seed")

    random.seed(seed)
    np.random.seed(np_seed)
    torch.manual_seed(seed)

    log_seeds()


def disable_torch_backend_randomness():
    r""" Torch backend randomness disabled """
    logging.warning("Torch backend CUDNN randomness disabled")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ViewModule(nn.Module):
    r""" General view layer to flatten to any output dimension """
    def __init__(self, d_out: List[int]):
        super().__init__()
        self._d_out = tuple(d_out)

    def forward(self, x: Tensor) -> Tensor:  # pylint: disable=arguments-differ
        # noinspection PyUnresolvedReferences
        return x.view((x.shape[0], *self._d_out))


class ViewTo1D(ViewModule):
    r""" View layer simplifying to specifically a single dimension """
    def __init__(self):
        super().__init__([-1])


class ClassifierBlock(nn.Module):
    def __init__(self, net: nn.Module, estimator: RiskEstimator):
        super().__init__()
        self.module = copy.deepcopy(net)
        self.loss = estimator
        self.optim = None

        self.train_loss = self.num_batch = self.valid_loss = None
        self.best_loss, self.is_best = np.inf, False

    def forward(self, xs: Tensor) -> Tensor:
        return self.module.forward(xs)

    def name(self) -> str:
        r""" Classifier block name derived from the loss' name """
        return self.loss.name()

    def epoch_start(self):
        r""" Configures the module for the start of an epoch """
        self.train_loss, self.num_batch = torch.zeros((), device=TORCH_DEVICE), 0

        self.valid_loss = np.inf
        self.is_best = False

        self.train()

    def process_batch(self, batch):
        r""" Process a batch including tracking the loss and pushing the gradients """
        xs, ys, ws, tks = batch

        self.optim.zero_grad()

        # Inner closure needed here for LBFGS support
        def closure():
            l_closure = self.loss.calc_train_loss(self.forward(xs), ys, ws, tks)
            self.optim.zero_grad()
            l_closure.grad_var.backward()
            return l_closure.te_loss

        if isinstance(self.optim, LBFGS):
            # noinspection PyNoneFunctionAssignment
            loss = self.optim.step(closure)
        else:
            loss = self.loss.calc_train_loss(self.forward(xs), ys, ws, tks)
            loss.grad_var.backward()
            self.optim.step()
            loss = loss.te_loss

        self.train_loss += loss.detach()
        self.num_batch += 1

    def calc_valid_loss(self, valid: DeviceDataLoader):
        r""" Calculates and stores the validation loss """
        all_scores, all_lbls, all_w, all_tk = [], [], [], []

        self.eval()
        with torch.no_grad():
            for xs, ys, ws, tks in valid:
                all_lbls.append(ys)
                all_w.append(ws)
                all_tk.append(tks)
                all_scores.append(self.forward(xs))

        dec_scores, labels = torch.cat(all_scores, dim=0), torch.cat(all_lbls, dim=0)
        w, tk = torch.cat(all_w, dim=0), torch.cat(all_tk, dim=0)
        val_loss = self.loss.calc_validation_loss(dec_scores, labels, w, tk)
        self.valid_loss = abs(float(val_loss.te_loss.item()))

        if self.valid_loss >= self.best_loss:
            return

        # Update the best loss if appropriate
        self.best_loss = self.valid_loss
        self.is_best = True
        save_module(self, self._build_serialize_name())

    def restore_best(self) -> None:
        r""" Restores the best model (i.e., with the minimum validation error) """
        msg = f"Restoring {self.name()}'s best trained model"
        logging.debug(f"Starting: {msg}")
        load_module(self, self._build_serialize_name())
        logging.debug(f"COMPLETED: {msg}")
        self.eval()

    def logger_field_info(self) -> Tuple[List[str], List[int]]:
        r"""
        Defines the block specific field names and sizes
        :return: Tuple of lists of the field names and widths (in number of characters) respectively
        """
        names = [f"{self.name()} L-Tr", f"{self.name()} L-Val", f"Best"]

        base_sizes = [12, 12, 4]
        sizes = [max(base, len(name)) for name, base in zip(names, base_sizes)]
        return names, sizes

    def epoch_log_fields(self):
        r""" Log the epoch information """
        return [self.train_loss / self.num_batch, self.valid_loss, self.is_best]

    def _build_serialize_name(self) -> Path:
        r""" Constructs the serialized model's name """
        serialize_dir = BASE_DIR / "models"
        serialize_dir.mkdir(parents=True, exist_ok=True)
        return construct_filename(self.name().lower(), serialize_dir, "pth", add_timestamp=False)


def configure_dataset_args() -> Tuple[TensorGroup, APU_Module]:
    r""" Manages generating the source data (if not already serialized to disk """
    use_dropout = False  # Default
    if config.DATASET.is_synthetic():
        tensor_grp = synthetic.generate_data()

    elif config.DATASET.is_cifar():
        cifar_dir = DATA_DIR / "CIFAR10"
        tensor_grp = cifar.load_data(cifar_dir, TORCH_DEVICE)

    elif config.DATASET.is_libsvm():
        tensor_grp = libsvm_ds.load_data(dest=DATA_DIR)

    elif config.DATASET.is_mnist_variant():
        tensor_grp = mnist.load_data(DATA_DIR)

    elif config.DATASET.is_newsgroups():
        tensor_grp = newsgroups.load_data(NEWSGROUPS_DIR)
        use_dropout = True

    elif config.DATASET.is_openml():
        tensor_grp = open_ml.load_data(dest=DATA_DIR)

    elif config.DATASET.is_spam():
        tensor_grp = spam.load_data(dest=DATA_DIR)

    else:
        raise ValueError(f"Dataset generation not supported for {config.DATASET.name}")

    assert list(config.DATASET.value.dim) == list(tensor_grp.p_x.shape[1:]), \
        "Unexpected tensor shape"

    if config.DATASET.is_synthetic():
        module = synthetic.Module()
    elif config.DATASET.is_spam():
        module = SpamFFModule(x=tensor_grp.p_x, num_hidden_layers=config.NUM_FF_LAYERS)
    else:
        module = BaseFFModule(x=tensor_grp.p_x, num_hidden_layers=config.NUM_FF_LAYERS,
                              add_dropout=use_dropout)
    return tensor_grp, module


def export_step1_results(tg: TensorGroup) -> None:
    r"""
    Plot statistics about the step 1 results

    :param tg: Tensor group of data
    :return: Step 1's accuracy rate
    """
    neg_lbl = torch.min(tg.u_tr_y)
    n_u_tr = tg.u_tr_y.numel()  # Number of training samples
    # Create a label tensor similar to label tensor

    p_cls_str = "|".join((str(x) for x in sorted(config.POS_TRAIN_CLASSES)))
    msg = " ".join([f"DS {config.DATASET.name}:", f"Pos {p_cls_str}: ",
                    f"$\\pi_{{\\mathrm{{tr}}}}= {config.TRAIN_PRIOR:.2f}$:"])

    combined_vals = torch.zeros((n_u_tr, 2))
    combined_vals[:, 0] = tg.u_tr_y
    combined_vals[:, 1] = tg.u_tr_sigma

    s1_res_dir = RES_DIR / "step1"
    # Plot the scatter for the step 1 results
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_scatter(ax=ax, x=combined_vals[:, 0], y=2 * combined_vals[:, 1] - 1, title=msg,
                 xlabel=r"$\mathcal{X}_{\mathrm{u-tr}}$ True Label", ylabel=r"$2 \sigma(x) - 1$",
                 xmin=-1.1, xmax=1.1, ymin=-1.1, ymax=1.1)

    name = construct_filename("s1-sigma", out_dir=s1_res_dir, file_ext="pdf", add_timestamp=True)
    name.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(str(name))

    # Export the raw stats
    combined_np = combined_vals.numpy()
    name = construct_filename("s1-raw", out_dir=s1_res_dir, file_ext="txt", add_timestamp=True)
    np.savetxt(str(name), combined_np, delimiter=",")

    # Plot a histogram of the error
    y_norm = tg.u_tr_y.clone()  # Normalizes labels to {0,1} to make a sensical error value
    y_norm[tg.u_tr_y == neg_lbl] = 0
    err = y_norm - tg.u_tr_sigma

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    plot_histogram(ax=ax, vals=err, n_bins=40, title=msg, hist_range=(-1, 1),
                   xlabel=r"$y - \sigma(x)$")

    name = construct_filename("s1-hist", out_dir=s1_res_dir, file_ext="pdf", add_timestamp=True)
    fig.savefig(str(name))
    # Close all plots
    plt.close('all')


def set_debug_mode(seed: int = 42) -> None:
    logging.warning("Debug mode enabled")
    set_random_seeds(seed=seed)
    log_seeds()


def log_seeds():
    r""" Log the seed information """
    logging.debug("Torch Random Seed: %d", torch.initial_seed())
    if "numpy" in sys.modules:
        state_str = re.sub(r"\s+", " ", str(np.random.get_state()))
        logging.debug("NumPy Random Seed: %s", state_str)
    #  Prints a seed way too long for normal use
    # if "random" in sys.modules:
    #     import random
    #     logging.debug("Random (package) Seed: %s", random.getstate())


def log_decision_boundary(module, name: str):
    r""" Logs a linear decision boundary"""
    assert config.DATASET.is_synthetic(), "Decision boundary only supported for synthetic data"
    decision_m, decision_b = module.decision_boundary()
    sign, decision_b = "-" if decision_b < 0 else "+", abs(decision_b)
    logging.debug(f"{name} Dec. Boundary: y = {decision_m:.6} * x + {decision_b:.6}")
