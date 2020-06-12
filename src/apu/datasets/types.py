__all__ = ["APU_Dataset", "APU_Module",
           "BaseFFModule",
           "Centroid",
           "CnnModule",
           "Labels",
           "LibsvmParams",
           "NewsgroupCategory", "NewsgroupCategoryInfo",
           "OpenMLParams",
           "SpamFFModule",
           "TensorGroup",
           "ViewTo1D"]

from collections import namedtuple, Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

import sklearn.datasets

from fastai.basic_data import DeviceDataLoader
import torch
from torch import Tensor
import torch.distributions as distributions
import torch.nn as nn
from torch.utils.data import TensorDataset

DatasetParams = namedtuple("DatasetParams", "name dim")


class Centroid:
    FEATURE_DIM = 2

    r""" Centroid used to generate points under train/test """
    def __init__(self, params: List):
        assert self.FEATURE_DIM + 2 == len(params), "Centroid size mismatch"
        lbl_name = params[0]
        try:
            self.label = Labels.Synthetic[lbl_name]
        except KeyError:
            raise ValueError(f"Unknown synthetic label {lbl_name}")

        self._mean = torch.tensor([float(x) for x in params[1:-1]])
        self._var = float(params[-1])

        cov = self._var * torch.eye(self._mean.numel())
        self._mv_norm = distributions.MultivariateNormal(loc=self._mean, covariance_matrix=cov)

    def gen_points(self, num_points) -> Tensor:
        r""" Generates a point according to this centroid """
        points = self._mv_norm.sample((num_points,))
        return points

    def mean_x(self) -> float:
        r""" X value of mean """
        return float(self._mean[0].item())

    def mean_y(self) -> float:
        r""" Y value of mean """
        return float(self._mean[1].item())

    def variance(self) -> float:
        r""" Y value of mean """
        return float(self._var)

    def __str__(self):
        return "".join([f"Cent({self.label.name},x={self.mean_x():.2f},y={self.mean_y():.2f},",
                        f"var={self.variance():.2f})"])


class NewsgroupCategoryInfo:
    r""" Stores the information about a specific category """
    def __init__(self, ids: List[int]):
        self.ids = ids
        self.rel_id_probs = None  # Relative probability of each category
        self.cat_prob = None

    def config_id_probs(self, lbl_counter: Counter):
        r""" Configures the probability for each ID number in the category """
        lbl_cnts = [lbl_counter[x] for x in self.ids]
        cat_cnt = sum(lbl_cnts)

        tot_cnt = sum(lbl_counter.values())

        # Total probability of the category
        self.cat_prob = cat_cnt / tot_cnt
        self.rel_id_probs = [x / cat_cnt for x in lbl_cnts]

    def get_id_probs(self, cat_prob: Optional[float]) -> dict:
        r"""
        Gets the category bias probabilities given the specified
        :param cat_prob: Probability for the entire category.  If not specified, use actual
                         probability
        :return: Dictionary mapping ID numbers to ID probabilities
        """
        if cat_prob is None:
            cat_prob = self.cat_prob
        return {id_num: cat_prob * id_prob for id_num, id_prob in zip(self.ids, self.rel_id_probs)}


class NewsgroupCategory(Enum):
    r""" Information about the Newsgroup categories """
    ALT = NewsgroupCategoryInfo([0])
    COMP = NewsgroupCategoryInfo([1, 2, 3, 4, 5])
    MISC = NewsgroupCategoryInfo([6])
    REC = NewsgroupCategoryInfo([7, 8, 9, 10])
    SCI = NewsgroupCategoryInfo([11, 12, 13, 14])
    SOC = NewsgroupCategoryInfo([15])
    TALK = NewsgroupCategoryInfo([16, 17, 18, 19])

    def __lt__(self, other: 'NewsgroupCategory') -> bool:
        return min(self.value.ids) < min(other.value.ids)

    @classmethod
    def configure_probs(cls, docs_dir: Path) -> None:
        r""" Set the probability for each category """
        # shuffle=True is used since ElmoEmbedder stores states between sentences so randomness
        # should reduce this effect
        docs_dir.mkdir(parents=True, exist_ok=True)
        # noinspection PyUnresolvedReferences
        bunch = sklearn.datasets.fetch_20newsgroups(subset="train", data_home=docs_dir,
                                                    shuffle=True)
        counter = Counter(bunch.target)
        for cat in cls:
            cat.value.config_id_probs(counter)


class OpenMLParams:
    r""" Stores parameters for datasets retrieved from OpenML """
    def __init__(self, dim: Union[int, List[int]], data_id: int, num_test: int):
        if isinstance(dim, int): dim = [dim]

        self.dim = dim
        self.data_id = data_id
        self.num_test = num_test


@dataclass
class LibsvmParams:
    r"""
    :see: https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/
    """
    dim: List[int]
    train_url: str
    test_url: Optional[str] = None
    num_test: Optional[int] = None


class Labels:
    r""" Encapsulates all labeling standards used in this program """
    POS = 1
    NEG = -1

    class Training(Enum):
        r""" Labels used during training """
        POS = 1
        NEG = -2  # ToDo Remove
        U_TRAIN = 0
        U_TEST = -1

    class Synthetic(Enum):
        r""" Labels used for defining the synthetic 2D centroids """
        POS_BOTH = 2  # Positive for train and test
        POS_TR = 1    # Positive only for training
        POS_TE = 0    # Positive only for testing
        NEG = -1


CIFAR_DIM = [1024]
MNIST_DIM = [1, 28, 28]
NEWSGROUPS_DIM = [9216]


# noinspection PyPep8Naming
class APU_Dataset(Enum):
    r""" Valid datasets for testing the aPU learners"""
    # A9A = OpenMLParams(dim=123, data_id=1430, num_test=16281)
    A9A = LibsvmParams(dim=[123],
                       train_url="binary/a9a",
                       test_url="binary/a9a.t",
                       num_test=16281)

    BANANA = OpenMLParams(dim=2, data_id=1460, num_test=2000)

    CIFAR10 = DatasetParams("CIFAR10", CIFAR_DIM)

    COD_RNA = LibsvmParams(dim=[8],
                           train_url="binary/cod-rna")

    CONNECT4 = LibsvmParams(dim=[126],
                            train_url="multiclass/connect-4")

    COVTYPE_BINARY = LibsvmParams(dim=[54],
                                  train_url="binary/covtype.libsvm.binary.scale.bz2")

    EPSILON = LibsvmParams(dim=[2000],
                           train_url="binary/epsilon_normalized.bz2",
                           test_url="binary/epsilon_normalized.t.bz2",
                           num_test=100000)

    FASHION_MNIST = DatasetParams("FashionMNIST", MNIST_DIM)

    IJCNN1 = OpenMLParams(dim=22, data_id=1575, num_test=91701)

    KMNIST = DatasetParams("KMNIST", MNIST_DIM)
    MNIST = DatasetParams("MNIST", MNIST_DIM)

    NEWSGROUPS = DatasetParams("20 Newsgroups", NEWSGROUPS_DIM)

    PHISHING = LibsvmParams(dim=[68],
                            train_url="binary/phishing")

    SPAM = DatasetParams("spam", NEWSGROUPS_DIM)

    SUSY = LibsvmParams(dim=[18],
                        train_url="binary/SUSY.bz2",
                        test_url=None,
                        num_test=500000)

    SYNTHETIC = DatasetParams("Synthetic", [Centroid.FEATURE_DIM])

    W8A = LibsvmParams(dim=[300],
                       train_url="binary/w8a",
                       test_url="binary/w8a.t",
                       num_test=14951)

    def is_cifar(self) -> bool:
        r""" Returns \p True if dataset is a CIFAR dataset """
        return self == self.CIFAR10

    def is_libsvm(self) -> bool:
        r""" Returns \p True if dataset from LibSVM """
        return isinstance(self.value, LibsvmParams)

    def is_mnist_variant(self) -> bool:
        r""" Returns \p True if dataset is MNIST or one of its variants """
        all_mnist = (self.FASHION_MNIST, self.KMNIST, self.MNIST)
        return any(self == x for x in all_mnist)

    def is_newsgroups(self) -> bool:
        r""" Return \p True if the dataset 20 Newsgroups """
        return self == self.NEWSGROUPS

    def is_openml(self) -> bool:
        r""" Returns \p True if using an OpenML dataset """
        return isinstance(self.value, OpenMLParams)

    def is_spam(self) -> bool:
        r""" Returns \p True if using the SPAM dataset """
        return self == self.SPAM

    def is_synthetic(self) -> bool:
        r""" Return \p True if the dataset is SYNTHETIC """
        return self == self.SYNTHETIC


# noinspection PyPep8Naming
class APU_Module(nn.Module):
    TORCH_DEVICE = None

    def __init__(self):
        super().__init__()
        self._model = nn.Sequential()

    def forward(self, x: Tensor) -> Tensor:
        if self.TORCH_DEVICE is not None: x.to(self.TORCH_DEVICE)
        y_hat = self._model(x).squeeze(dim=1)
        return y_hat

    def predict(self, x) -> Tensor:
        r""" Provides a class prediction, i.e., positive or negative """
        y_hat = self.forward(x)
        lbls = torch.full((x.shape[0]), Labels.NEG, dtype=torch.int32)
        lbls[y_hat >= 0] = Labels.POS

        if self.TORCH_DEVICE is not None:
            lbls.to(self.TORCH_DEVICE)
        return lbls


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


class BaseFFModule(APU_Module):

    class Config:
        r""" Configuration settings for the 20 newsgroups learner """
        # FF_HIDDEN_DEPTH = 2
        FF_HIDDEN_DIM = 300
        FF_ACTIVATION = nn.ReLU

    def __init__(self, x: Tensor, num_hidden_layers: int, add_dropout: bool = False):
        super().__init__()

        self._model.add_module("View1D", ViewTo1D())
        in_dim = x[0].numel()

        self.num_hidden_layers = num_hidden_layers
        self._ff = nn.Sequential()
        # Special to prevent overfitting on spam dataset by reducing input features
        if add_dropout:
            self._ff.add_module(f"Input_Dropout", nn.Dropout(p=0.5))

        for i in range(1, self.num_hidden_layers + 1):
            ff_block = nn.Sequential()
            ff_block.add_module(f"FF_Lin", nn.Linear(in_dim, self.Config.FF_HIDDEN_DIM))
            ff_block.add_module(f"FF_Act", self.Config.FF_ACTIVATION())
            ff_block.add_module(f"FF_BatchNorm", nn.BatchNorm1d(self.Config.FF_HIDDEN_DIM))
            if add_dropout:
                ff_block.add_module(f"FF_Dropout", nn.Dropout(p=0.5))

            self._ff.add_module(f"Hidden_Block_{i}", ff_block)
            in_dim = self.Config.FF_HIDDEN_DIM

        # Add output layer
        self._model.add_module("FF", self._ff)
        self._model.add_module("FF_Out", nn.Linear(in_dim, 1))


class SpamFFModule(BaseFFModule):
    r""" Feedforward module for spam with dropout """
    def __init__(self, x: Tensor, num_hidden_layers: int):
        super().__init__(x=x, num_hidden_layers=num_hidden_layers, add_dropout=True)


@dataclass
class TensorGroup:
    r""" Encapsulates a group of tensors used by the learner """
    p_x: Optional[Tensor] = None
    p_sigma: Optional[Tensor] = None

    u_tr_x: Optional[Tensor] = None
    u_tr_sigma: Optional[Tensor] = None
    u_tr_y: Optional[Tensor] = None

    u_te_x: Optional[Tensor] = None
    u_te_sigma: Optional[Tensor] = None
    u_te_y: Optional[Tensor] = None

    # test has no sigma since sigma is a training only parameter
    test_x: Optional[Tensor] = None
    test_y: Optional[Tensor] = None
    # Train test set
    test_x_tr: Optional[Tensor] = None
    test_y_tr: Optional[Tensor] = None

    def has_sigmas(self) -> bool:
        r""" Returns \p True if the \p TensorGroup has weights """
        is_none = [w is None for w in (self.p_sigma, self.u_tr_sigma, self.u_te_sigma)]
        assert all(is_none) or not any(is_none), "Mix of weights not allowed"
        return not any(is_none)

    def set_sigmas(self, sigma_module: nn.Module, config, device: torch.device) -> None:
        r"""
        Sets all weights in the \p TensorGroup

        :param sigma_module: Module that represents :math:`\sigma(x) = \Pr[Y = +1 | x]` in the
                             aPU paper.
        :param config: Configuration settings for the learner
        :param device: Device where to run the weight calculations
        """
        sigma_module.eval()

        # Iterate through each X vector and build the weights
        all_priors = (1., config.TRAIN_PRIOR, config.TEST_PRIOR)
        for ds_name, prior in zip(("p", "u_tr", "u_te"), all_priors):
            x = self.__getattribute__(f"{ds_name}_x")
            assert x is not None, f"{ds_name}_x cannot be None"

            dl = DeviceDataLoader.create(dataset=TensorDataset(x), shuffle=False,
                                         drop_last=False, bs=config.BATCH_SIZE,
                                         num_workers=0, device=device)
            all_sigma = []
            with torch.no_grad():
                for xs, in dl:
                    sig_vals = sigma_module.calc_cal_weights(xs, prior=prior)
                    all_sigma.append(sig_vals)

            w = torch.cat(all_sigma).detach().cpu()
            if len(w.shape) > 1: w = w.squeeze(dim=1)

            # Sanity check the weights information
            assert float(w.min().item()) >= 0, "Minimum sigma must be greater than or equal to 0"
            assert float(w.max().item()) <= 1, "Maximum sigma must be less than or equal to 1"
            assert w.numel() == x.shape[0], "Number of weights does not match number of elements"
            assert len(w.shape) == 1, "Strange size for sigma vector"

            assert self.__getattribute__(f"{ds_name}_sigma") is None, f"{ds_name}_sigma is not None"
            self.__setattr__(f"{ds_name}_sigma", w)

    def reset_sigmas(self) -> None:
        r""" DEBUG ONLY.  Reset the sigmas back to None"""
        self.p_sigma = self.u_tr_sigma = self.u_te_sigma = None


class CnnModule(APU_Module):
    ACTIVATION = nn.ReLU

    CONV_LAYER_FILTERS_OUT = (3 * [96]) + (5 * [192]) + (1 * [10])
    CONV_LAYER_KERNEL_SIZES = (7 * [3]) + (2 * [1])
    CONV_LAYER_STRIDE = (2 * [1]) + (1 * [2]) + (2 * [1]) + (1 * [2]) + (3 * [1])
    CONV_LAYER_PAD_SIZE = (7 * [1]) + (2 * [0])

    NUM_HIDDEN_FF_LAYER = 2
    FF_HIDDEN_DIM = 1000

    def __init__(self, x: Tensor):
        if len(x.shape) != 4:
            raise ValueError("Dimension of input x appears incorrect")
        super().__init__()

        # Verify the convolutional settings
        self._num_conv_layers = len(self.CONV_LAYER_FILTERS_OUT)
        self._verify_conv_sizes(x)

        self._base_mod = nn.Sequential()
        # Constructs the convolutional 2D
        flds = (self.CONV_LAYER_FILTERS_OUT, self.CONV_LAYER_KERNEL_SIZES,
                self.CONV_LAYER_STRIDE, self.CONV_LAYER_PAD_SIZE)
        input_dim = x.shape[1]
        for i, (out_dim, k_size, stride, pad) in enumerate(zip(*flds)):
            conv_seq = nn.Sequential(nn.Conv2d(input_dim, out_dim, k_size, stride, pad),
                                     self.ACTIVATION(),
                                     nn.BatchNorm2d(out_dim))
            self._base_mod.add_module("Conv2D_%02d" % i, conv_seq)
            input_dim = out_dim
        self._base_mod.add_module("Flatten", ViewTo1D())

        # Find the size of the tensor input into the FF block
        self._base_mod.eval()
        x = x.cpu()  # Base module still on CPU at this point
        with torch.no_grad():
            ff_in = self._base_mod.forward(x).shape[1]
        self._base_mod.train()
        # Constructs the FF block
        for i in range(1, self.NUM_HIDDEN_FF_LAYER + 1):
            ff_seq = nn.Sequential(nn.Linear(ff_in, self.FF_HIDDEN_DIM),
                                   self.ACTIVATION())
            ff_in = self.FF_HIDDEN_DIM
            self._base_mod.add_module("FF_%02d" % i, ff_seq)

        self._model.add_module("Base Module", self._base_mod)
        self._model.add_module("FF_Out", nn.Linear(ff_in, 1))

    def _verify_conv_sizes(self, x: Tensor):
        r""" Sanity check the dimensions of the input tensor and convolutional block """
        assert len(x.shape) == 4, "X tensor should be 2D"

        assert self._num_conv_layers == len(self.CONV_LAYER_FILTERS_OUT), "# Filters mismatch"
        assert self._num_conv_layers == len(self.CONV_LAYER_KERNEL_SIZES), "# Kernels mismatch"
        assert self._num_conv_layers == len(self.CONV_LAYER_STRIDE), "# strides mismatch"
        assert self._num_conv_layers == len(self.CONV_LAYER_PAD_SIZE), "# paddings mismatch"
