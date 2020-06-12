__all__ = ["load_data"]

from pathlib import Path
from typing import Union

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.datasets import fetch_openml

import torch
from torch import Tensor

from .. import _config as config
from .types import APU_Dataset, TensorGroup
from .utils import build_puc_style_dataset


def load_data(dest: Union[Path, str]) -> TensorGroup:
    r"""
    Constructs the dataset for the specified OpenML experiments

    :param dest: Base directory to store all downloaded OpenML data
    :return: TensorGroup for the experiment
    """
    ds_name, ml_ds = config.DATASET.name, config.DATASET.value

    dest = Path(dest) / ds_name.lower()
    dest.mkdir(parents=True, exist_ok=True)

    x, y = fetch_openml(data_id=ml_ds.data_id, data_home=dest, return_X_y=True)

    # Configure the data to run with torch
    x = _convert_x_tensor(x)
    y = _convert_y_vector(y)
    assert x.shape[1] == ml_ds.dim[0], "Dataset dimension mismatch"

    return build_puc_style_dataset(x, y)


def _convert_x_tensor(x: np.ndarray) -> Tensor:
    r""" Convert the \p X tensor to a \p torch tensor """
    if isinstance(x, csr_matrix):
        x = x.toarray()

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float().cpu()

    raise ValueError("Unable to convert X tensor to torch")


def _convert_y_vector(y: np.ndarray) -> Tensor:
    r""" Configures y vector for the learner """
    base_set = (APU_Dataset.A9A, APU_Dataset.BANANA, APU_Dataset.IJCNN1, APU_Dataset.SUSY)
    if config.DATASET in base_set:
        y = y.astype(np.int)
    else:
        raise ValueError(f"Unknown dataset {config.DATASET.name}")
    return torch.from_numpy(y)
