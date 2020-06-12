__all__ = ["load_data"]

import bz2
import logging
from pathlib import Path
from typing import Tuple, Union

import sklearn
from scipy.sparse import csr_matrix

import torch
from torch import Tensor

from .. import _config as config
from .types import TensorGroup
from .utils import build_puc_style_dataset, download_file

BASE_URL = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/"


def load_data(dest: Union[Path, str]) -> TensorGroup:
    r""" Constructs the dataset objects """
    ds = config.DATASET.value
    all_x, all_y = [], []
    for url in (ds.train_url, ds.test_url):
        if url is None:
            continue
        x, y = _get_tensor(dest=dest, url=url)
        all_x.append(x)
        all_y.append(y)

    # Combine train and if applicable test
    x, y = torch.cat(all_x, dim=0), torch.cat(all_y, dim=0)
    assert x.shape[1] == ds.dim[0], "Unexpected X feature dimension"
    return build_puc_style_dataset(x, y)


def _build_download_path(dest_dir: Path, url: Path) -> Path:
    r""" Download path to write the downloaded file """
    dir_pth = dest_dir / config.DATASET.name.lower()
    dir_pth.mkdir(exist_ok=True, parents=True)
    return dir_pth / url.name


def _build_data_path(file_path: Path) -> Path:
    r""" Download path to write the downloaded file """
    file_path.parent.mkdir(parents=True, exist_ok=True)
    if not _is_compressed(file_path):
        return file_path
    return file_path.parent / file_path.stem


def _build_tensor_path(uncompressed_path: Path) -> Path:
    r""" Build the path to write the tensor """
    processed_dir = uncompressed_path.parent / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    file_name = uncompressed_path.stem + ".pt"
    return processed_dir / file_name


def _is_compressed(file_path: Path) -> bool:
    r""" Returns \p True if file path is a compressed file """
    return file_path.suffix in [".bz2"]


def _decompress(file_path: Path) -> Path:
    r""" Decompressed the file """
    if not _is_compressed(file_path):
        return file_path

    new_path = _build_data_path(file_path)
    if new_path.exists():
        logging.info(f"Decompressed file \"{new_path}\" exists. Skipping...")
        return file_path

    msg = f"Decompressing file \"{file_path}\" to \"{new_path}\""
    logging.info(f"Starting: {msg}...")
    with open(str(new_path), 'wb') as new_file, bz2.BZ2File(file_path, 'rb') as file:
        for data in iter(lambda: file.read(100 * 1024), b''):
            new_file.write(data)
    logging.info(f"COMPLETED: {msg}...")
    return file_path


def _get_tensor(dest: Path, url: str) -> Tuple[Tensor, Tensor]:
    r""" Constructs a tensor from the LibSVM file """
    download_path = _build_download_path(dest, Path(url))
    tensor_path = _build_tensor_path(download_path)
    if tensor_path.exists():
        logging.info(f"Tensor file \"{tensor_path}\" already exists. Skipping...")
    else:
        url = "".join([BASE_URL, url])  # Base URL appended to front of url
        download_file(url, download_path)

        file_path = _decompress(download_path)
        assert file_path == download_path, "Mismatch of file and download path"

        msg = f"Creating tensor file \"{tensor_path}\" from \"{file_path}\""
        logging.info(f"Starting: {msg}...")
        x, y = sklearn.datasets.load_svmlight_file(str(file_path))

        if isinstance(x, csr_matrix): x = x.toarray()
        if isinstance(y, csr_matrix): y = y.toarray()

        x, y = torch.from_numpy(x).float().cpu(), torch.from_numpy(y).int().cpu()
        torch.save((x, y), tensor_path)
        logging.info(f"COMPLETED: {msg}")

    msg = f"Loading tensor file \"{tensor_path}\""
    logging.info(f"Starting: {msg}...")
    tensors = torch.load(tensor_path)
    logging.info(f"COMPLETED: {msg}")
    return tensors
