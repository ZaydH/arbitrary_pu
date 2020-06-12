__all__ = ["load_data"]

from pathlib import Path
from typing import Optional

import fastai.vision
from fastai.basic_data import DeviceDataLoader
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torchvision
import torchvision.transforms as transforms

from .. import _config as config
from .types import TensorGroup, ViewTo1D
from .utils import shared_tensor_dataset_importer

USE_TRANSFER = True

# MODEL = fastai.vision.models.resnet34
# MODEL = fastai.vision.models.resnet50
# MODEL = torchvision.models.vgg16_bn
MODEL = fastai.vision.models.densenet121


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


def _flatten_cifar(tensor_path: Path, dest_dir: Path, device: torch.device):
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


def load_data(cifar_dir: Path, device: torch.device) -> TensorGroup:
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

        _flatten_cifar(out_name, flat_dir, device)

    tensor_dir = flat_dir if USE_TRANSFER else processed_dir
    return shared_tensor_dataset_importer(dest=tensor_dir)
