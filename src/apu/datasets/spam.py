from collections import namedtuple
from enum import Enum
import logging
from pathlib import Path
# import tarfile
from typing import List, Optional, Tuple

# import numpy as np

import torch
from torch import Tensor
from torchvision.datasets.utils import download_file_from_google_drive

from .. import _config as config
from .utils import binom_sample
# from .utils import binom_sample, download_nltk_tokenizer, make_elmo_embedders,
#     use_elmo_to_process_doc
from .types import Labels, TensorGroup


SPAM_SET_SIZE = 10000


class SpamLabels(Enum):
    SPAM = 1
    HAM = -1


# pth_file_id: Contains the preprocessed tensors
# tgz_file_id: Contains the complete TREC dataset
SpamDsParams = namedtuple("SpamDsParams", "name pth_file_id tgz_file_id")


class SpamDataset(Enum):
    TREC2005 = SpamDsParams(name="trec05p-1",
                            pth_file_id="16_YC3yQbvp2QtcheVihDd_Fb6RevcabG",
                            tgz_file_id="1SL5IOKqtyHN0cZ-h6Qsc3EJbCXE1P4F9")
    TREC2007 = SpamDsParams(name="trec07p",
                            pth_file_id="1JeHrjz-_owvanezgqU-CfPzBC9AfJLw0",
                            tgz_file_id="1_eFFm4DojnTHw3g6eTFnodzxNjlpQ4uv")


# def _extract_tgz(tar_file: Path, folder: Path):
#     r""" Extract the tgz spam files"""
#     tar = tarfile.open(str(tar_file), 'r')
#     for item in tar:
#         tar.extract(item, folder)
#         # if item.name.find(".tgz") != -1 or item.name.find(".tar") != -1:
#         #     extract(item.name, "./" + item.name[:item.name.rfind('/')])
#
#
# def _generate_preprocessed_vectors(data_dir: Path, index_file: Path,
#                                    ds_name: str) -> Tuple[np.ndarray, np.ndarray]:
#     r"""
#     Constructs the preprocessed vectors for either the test or train datasets.
#     :param index_file: File defining whether each file is spam or ham
#     :param ds_name: Either "test" or "training"
#     """
#     assert ds_name == "training" or ds_name == "test"
#
#     # First learner use CUDA, second does not
#     elmos = make_elmo_embedders(data_dir)
#
#     with open(str(index_file), "r") as f_in:
#         lines = f_in.read().splitlines()
#
#     # ToDo Remove part only using subset
#     lines = lines[:min(len(lines), SPAM_SET_SIZE)]
#     n = len(lines)
#     # First learner use CUDA, second does not
#     x, y = np.zeros([n, 9216]), np.zeros(n)
#     index_dir = str(index_file.parent)
#
#     msg = f"Creating the SPAM preprocessed vectors for \"{ds_name}\" set"
#     logging.info(f"Starting: {msg}")
#     file_enc_err_cnt = 0
#     # Has to be out of for loop or stdout overwrite messes up
#     if not torch.cuda.is_available(): logging.info('CUDA unavailable for ELMo encoding')
#     for i, index_line in enumerate(lines):
#         split = index_line.split(" ", maxsplit=1)
#         # Extract the label
#         lbl = SpamLabels.SPAM if split[0].lower() == "spam" else SpamLabels.HAM
#         y[i] = lbl.value
#
#         print(f"Processing {ds_name} document {i+1}/{n}", end="", flush=True)
#         try:
#             with open(str(f"{index_dir}/{split[1]}"), "r") as f_in:
#                 file_contents = f_in.read()
#         except UnicodeDecodeError:
#             with open(str(f"{index_dir}/{split[1]}"), "r", encoding="ISO-8859-1") as f_in:
#                 file_contents = f_in.read()
#             file_enc_err_cnt += 1
#         x[i] = use_elmo_to_process_doc(elmos, file_contents)
#
#         # Go back to beginning of the line. Weird formatting due to PyCharm issues
#         print('\r', end="")
#
#     logging.info(f"{ds_name} dataset: Encoding read error count: {file_enc_err_cnt:,}")
#     logging.info(f"COMPLETED: {msg}")
#     return x, y
#
#
# def _construct_spam_dataset(dest: Path) -> Tuple[Path, Path]:
#     r""" Use ELMo to construct the dataset """
#     # Define the output files
#     processed_dir = dest / "processed"
#     lst_path = (processed_dir / "training.pt", processed_dir / "test.pt")
#     if processed_dir.exists():
#         return lst_path
#
#     raw = dest / "raw"
#     raw.mkdir(parents=True, exist_ok=True)
#
#     download_nltk_tokenizer(dest)
#
#     for out_file, ds_info in zip(lst_path, (SpamDataset.TREC2005, SpamDataset.TREC2007)):
#         filename = f"{ds_info.value.name}.tgz"
#         tar_file = raw / str(filename)
#         if not tar_file.exists():
#             logging.debug(f"Downloading file {str(tar_file)}...")
#             download_file_from_google_drive(file_id=ds_info.value.tgz_file_id,
#                                             root=str(raw),
#                                             filename=filename)
#         else:
#             logging.debug(f"{str(tar_file)} already exists. Skipping download...")
#
#         full_folder = raw / Path(filename).stem  # Extension not in folder name
#         if not full_folder.exists():
#             _extract_tgz(tar_file, raw)
#
#         index_file = full_folder / "full/index"
#         x, y = _generate_preprocessed_vectors(data_dir=dest,
#                                               index_file=index_file,
#                                               ds_name=out_file.stem)  # ds_name only for logging
#         # Convert to torch
#         x, y = torch.from_numpy(x).float(), torch.from_numpy(y).view([-1]).int()
#
#         out_file.parent.mkdir(parents=True, exist_ok=True)
#         torch.save((x, y), out_file)
#     return tuple(lst_path)


def _construct_spam_dataset(dest: Path) -> Tuple[Path, Path]:
    r""" Use ELMo to construct the dataset """
    # Define the output files
    processed_dir = dest / "processed"
    lst_path = processed_dir / "training.pt", processed_dir / "test.pt"
    spam_datasets = SpamDataset.TREC2005.value, SpamDataset.TREC2007.value

    processed_dir.mkdir(parents=True, exist_ok=True)
    # Downloads the processed tensors
    for tensor_pth, ds_info in zip(lst_path, spam_datasets):
        if tensor_pth.exists():
            logging.debug(f"Spam tensor \"{str(tensor_pth)}\" already exists. Skipping download...")
            continue
        download_file_from_google_drive(file_id=ds_info.pth_file_id,
                                        root=str(tensor_pth.parent), filename=tensor_pth.name)
    return lst_path


def _get_elements_from_tensor(n: int, labels: List[int], x: Tensor, y: Tensor,
                              y_out_lbl: int) -> Tuple[Tensor, Tensor]:
    r""" Select \p n elements in \p X/\p y tensors with labels in \p labels """
    # Select all elements with the right label
    keep = torch.full_like(y, fill_value=False).bool()
    for lbl in labels:
        keep |= y == lbl
    x, y = x[keep], y[keep]
    assert x.shape[0] == y.shape[0]
    # Select n elements
    assert y.numel() >= n, f"Insufficient elements {n} needed but {y.size()} available"
    perm = torch.randperm(y.numel())[:n]  # Indices to keep from reduced array

    out_y = torch.full((n,), fill_value=y_out_lbl).int().view([-1])
    return x[perm], out_y


def _get_x_y_from_file(prior: Optional[float], n_ele: int, file_path: Path, pos_labels: List[int],
                       neg_labels: Optional[List[int]] = None) -> Tuple[Tensor, Tensor]:
    r""" Get the X/y Tensor given the specified information"""
    assert prior is None or 0 < prior <= 1, "Invalid positive prior"
    assert n_ele > 0, "Number of elements must be positive"

    # Read the tensors from disk
    all_x, all_y = torch.load(file_path)

    # Define the size of the positive and negative sets
    if prior is None:
        assert neg_labels is None, "Negative labels invalid for trivial prior"
        n_pos = n_ele
    else:
        assert neg_labels is not None, "Non-trivial prior but no negative labels specified"
        n_pos = binom_sample(prior=prior, n=n_ele)
    n_neg = n_ele - n_pos

    x, y = _get_elements_from_tensor(n=n_pos, labels=pos_labels, x=all_x, y=all_y,
                                     y_out_lbl=Labels.POS)
    if n_neg > 0:
        n_x, n_y = _get_elements_from_tensor(n=n_neg, labels=neg_labels, x=all_x, y=all_y,
                                             y_out_lbl=Labels.NEG)
        x, y = torch.cat([x, n_x], dim=0), torch.cat([y, n_y], dim=0).int()
    return x, y


def load_data(dest: Path):
    dest /= "spam"
    tr_path, te_path = _construct_spam_dataset(dest=dest)

    tg = TensorGroup()
    tg.p_x, _ = _get_x_y_from_file(prior=None, n_ele=config.N_P, file_path=tr_path,
                                   pos_labels=config.POS_TRAIN_CLASSES, neg_labels=None)
    tg.u_tr_x, tg.u_tr_y = _get_x_y_from_file(prior=config.TRAIN_PRIOR, n_ele=config.N_U_TRAIN,
                                              file_path=tr_path,
                                              pos_labels=config.POS_TRAIN_CLASSES,
                                              neg_labels=config.NEG_CLASSES)
    # Inductive set from the train distribution
    tg.test_x_tr, tg.test_y_tr = _get_x_y_from_file(prior=config.TRAIN_PRIOR,
                                                    n_ele=min(config.N_U_TRAIN, config.N_TEST),
                                                    file_path=tr_path,
                                                    pos_labels=config.POS_TRAIN_CLASSES,
                                                    neg_labels=config.NEG_CLASSES)

    tg.u_te_x, tg.u_te_y = _get_x_y_from_file(prior=config.TEST_PRIOR, n_ele=config.N_U_TEST,
                                              file_path=te_path,
                                              pos_labels=config.POS_TRAIN_CLASSES,
                                              neg_labels=config.NEG_CLASSES)

    tg.test_x, tg.test_y = _get_x_y_from_file(prior=config.TEST_PRIOR, n_ele=config.N_TEST,
                                              file_path=te_path,
                                              pos_labels=config.POS_TRAIN_CLASSES,
                                              neg_labels=config.NEG_CLASSES)
    return tg
