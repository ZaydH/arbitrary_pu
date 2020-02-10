__all__ = ["load_data"]

from pathlib import Path
from typing import Optional

import fastai.vision
from fastai.basic_data import DeviceDataLoader
import torch
from torch import Tensor
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from .types import APU_Module, TensorGroup, ViewTo1D
from .utils import shared_tensor_dataset_importer

USE_TRANSFER = True

# MODEL = fastai.vision.models.resnet34
# MODEL = fastai.vision.models.resnet50
# MODEL = torchvision.models.vgg16_bn
MODEL = fastai.vision.models.densenet121


# class CnnModule(PurplModule):
#     ACTIVATION = nn.ReLU
#
#     CONV_LAYER_FILTERS_OUT = (3 * [96]) + (5 * [192]) + (1 * [10])
#     CONV_LAYER_KERNEL_SIZES = (7 * [3]) + (2 * [1])
#     CONV_LAYER_STRIDE = (2 * [1]) + (1 * [2]) + (2 * [1]) + (1 * [2]) + (3 * [1])
#     CONV_LAYER_PAD_SIZE = (7 * [1]) + (2 * [0])
#
#     NUM_HIDDEN_FF_LAYER = 2
#     FF_HIDDEN_DIM = 1000
#
#     def __init__(self, x: Tensor):
#         if len(x.shape) != 4:
#             raise ValueError("Dimension of input x appears incorrect")
#         super().__init__()
#
#         # Verify the convolutional settings
#         self._num_conv_layers = len(self.CONV_LAYER_FILTERS_OUT)
#         self._verify_conv_sizes(x)
#
#         self._base_mod = nn.Sequential()
#         # Constructs the convolutional 2D
#         flds = (self.CONV_LAYER_FILTERS_OUT, self.CONV_LAYER_KERNEL_SIZES,
#                 self.CONV_LAYER_STRIDE, self.CONV_LAYER_PAD_SIZE)
#         input_dim = x.shape[1]
#         for i, (out_dim, k_size, stride, pad) in enumerate(zip(*flds)):
#             conv_seq = nn.Sequential(nn.Conv2d(input_dim, out_dim, k_size, stride, pad),
#                                      self.ACTIVATION(),
#                                      nn.BatchNorm2d(out_dim))
#             self._base_mod.add_module("Conv2D_%02d" % i, conv_seq)
#             input_dim = out_dim
#         self._base_mod.add_module("Flatten", ViewTo1D())
#
#         # Find the size of the tensor input into the FF block
#         self._base_mod.eval()
#         x = x.cpu()  # Base module still on CPU at this point
#         with torch.no_grad():
#             ff_in = self._base_mod.forward(x).shape[1]
#         self._base_mod.train()
#         # Constructs the FF block
#         for i in range(1, self.NUM_HIDDEN_FF_LAYER + 1):
#             ff_seq = nn.Sequential(nn.Linear(ff_in, self.FF_HIDDEN_DIM),
#                                    self.ACTIVATION())
#             ff_in = self.FF_HIDDEN_DIM
#             self._base_mod.add_module("FF_%02d" % i, ff_seq)
#
#         self._model.add_module("Base Module", self._base_mod)
#         self._model.add_module("FF_Out", nn.Linear(ff_in, 1))
#
#     def _verify_conv_sizes(self, x: Tensor):
#         r""" Sanity check the dimensions of the input tensor and convolutional block """
#         assert len(x.shape) == 4, "X tensor should be 2D"
#
#         assert self._num_conv_layers == len(self.CONV_LAYER_FILTERS_OUT), "# Filters mismatch"
#         assert self._num_conv_layers == len(self.CONV_LAYER_KERNEL_SIZES), "# Kernels mismatch"
#         assert self._num_conv_layers == len(self.CONV_LAYER_STRIDE), "# strides mismatch"
#         assert self._num_conv_layers == len(self.CONV_LAYER_PAD_SIZE), "# paddings mismatch"


class AdaptiveConcatPool2d(nn.Module):
    r"""Layer that concats `AdaptiveAvgPool2d` and `AdaptiveMaxPool2d`."""

    def __init__(self, sz: Optional[int] = None):
        r"""Output will be 2*sz or 2 if sz is None"""
        super().__init__()
        sz = sz or 1
        self.ap, self.mp = nn.AdaptiveAvgPool2d(sz), nn.AdaptiveMaxPool2d(sz)

    def forward(self, x):
        return torch.cat([self.mp(x), self.ap(x)], 1)


def _convert_dataloader_to_tensor(output_file: Path, dl: DataLoader):
    r"""
    Converts the dataset in \p DataLoader to X/y \p torch tensors that are serialized to
    \p output_file.

    :param output_file: Path to serialize the tensor for faster reload
    :param dl: \p DataLoader to export to disk
    """
    # If the serialized file already exists, no need to convert to tensors.
    if output_file.exists(): return
    x, y = [], []
    for _x, _y in dl:
        x.append(_x.cpu())
        y.append(_y)
    x, y = torch.cat(x, dim=0), torch.cat(y, dim=0)
    assert x.dtype == torch.float32, "Wrong CIFAR datatype"
    assert list(x.shape[1:]) == [3, 32, 32], "Wrong shape CIFAR tensor"
    # Reorder the dimensions as torch requires NCHW
    output_file.parent.mkdir(exist_ok=True, parents=True)
    # noinspection PyUnresolvedReferences
    torch.save((x.cpu(), y.cpu()), output_file)


def _flatten_cifar(config, tensor_path: Path, dest_dir: Path, device: torch.device):
    r""" Flattens CIFAR into preprocessed vectors """
    # `body` is the base layers of the specified model
    body = fastai.vision.create_body(MODEL)
    body.add_module("Flatten", ViewTo1D())
    body.eval()
    body.to(device)

    # Path to write the processed tensor
    tensor_ds = TensorDataset(*torch.load(tensor_path))
    with torch.no_grad():
        dl = DeviceDataLoader.create(tensor_ds, bs=config.BATCH_SIZE, num_workers=0,
                                     shuffle=False, drop_last=False, device=device)
        flat_x, flat_y = [], []
        for xs, ys in dl:
            flat_x.append(body.forward(xs).cpu())
            flat_y.append(ys.cpu())

    # Concatenate all objects
    dest_dir.mkdir(exist_ok=True, parents=True)
    dest_path = dest_dir / tensor_path.name
    flat_x, flat_y = torch.cat(flat_x, dim=0).cpu(), torch.cat(flat_y, dim=0).cpu()
    torch.save((flat_x, flat_y), dest_path)


def load_data(config, cifar_dir: Path, device: torch.device) -> TensorGroup:
    r""" Loads the CIFAR10 dataset """
    tfms = [transforms.ToTensor()]

    processed_dir = cifar_dir / "processed"
    flat_dir = cifar_dir / "FLAT" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    for is_training in [True, False]:
        # Path to write the processed tensor
        out_name = processed_dir / f"{'training' if is_training else 'test'}.pt"
        if out_name.exists():
            continue

        # noinspection PyTypeChecker
        ds = torchvision.datasets.cifar.CIFAR10(cifar_dir, transform=transforms.Compose(tfms),
                                                train=is_training, download=True)
        dl = DataLoader(ds, batch_size=config.BATCH_SIZE, num_workers=0, drop_last=False,
                        pin_memory=False, shuffle=False)

        _convert_dataloader_to_tensor(out_name, dl)

        _flatten_cifar(config, out_name, flat_dir, device)

    tensor_dir = flat_dir if USE_TRANSFER else processed_dir
    return shared_tensor_dataset_importer(config, dest=tensor_dir)
