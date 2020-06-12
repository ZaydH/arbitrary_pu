__all__ = ["binom_sample",
           "build_puc_style_dataset",
           "construct_bias_vec",
           "download_file",
           "download_nltk_tokenizer",
           "make_elmo_embedders",
           "shared_tensor_dataset_importer",
           "shuffle_tensors",
           "use_elmo_to_process_doc"
           ]

import logging
import os
from pathlib import Path
import random
import requests
from typing import Collection, List, Optional, Set, Tuple, Union

import nltk
import nltk.tokenize
import numpy as np

from allennlp.commands.elmo import ElmoEmbedder
from allennlp.common.file_utils import cached_path
import torch
from torch import Tensor
import torch.distributions as distributions

from .. import _config as config
from .types import Labels, TensorGroup


def binom_sample(prior: float, n: int) -> int:
    r""" Binomial distribution sample """
    assert 0 <= prior <= 1., "Invalid prior"
    assert n > 0, "Number of samples must be positive"
    binom = distributions.Binomial(n, torch.tensor([prior]))
    return int(binom.sample())


def multinomial_sample(n: int, p_vec: Tensor) -> Tensor:
    r""" Multinomial distribution sample """
    assert p_vec.shape[0] > 0, "Multinomial size doesn't make sense"

    n_per_category = distributions.Multinomial(n, p_vec).sample().int()

    assert p_vec.shape == n_per_category.shape, "Dimension mismatch"
    assert int(n_per_category.sum().item()) == n, "Number of elements mismatch"
    return n_per_category


def shuffle_tensors(*args) -> Union[Tensor, Tuple[Tensor, ...]]:
    r"""
    Performs row-wise shuffle of all tensors in \p args.  All tensors in \p args must have the same
    number of rows.
    """
    assert len(args) > 0, "At least one tensor required"
    assert all(isinstance(arg, Tensor) for arg in args), "Not all Tensors"

    num_rows = args[0].shape[0]
    assert all(arg.shape[0] == num_rows for arg in args), "Mismatch is number of rows"

    idx = torch.randperm(num_rows)
    shuffled = [arg[idx] for arg in args]
    if len(args) == 1:
        return shuffled[0]
    return tuple(shuffled)


def shared_tensor_dataset_importer(dest: Union[Path, str],
                                   normalize_factor: Optional[int] = None,
                                   view_size: Optional[Union[int, Tuple[int, ...]]] = None) \
        -> TensorGroup:
    r"""
    General function used to import MNIST-like datasets downloaded from the \p torchvision dataset.
    :return: \p TensorGroup of extracted data
    """
    # Sanity check setup
    for cls_set in (config.POS_TRAIN_CLASSES, config.POS_TEST_CLASSES, config.NEG_CLASSES):
        assert cls_set is not None and cls_set, "No class IDs specified"

    assert set(config.POS_TRAIN_CLASSES).union(set(config.NEG_CLASSES)), "Trn. pos/neg not disjoint"
    assert set(config.POS_TEST_CLASSES).union(set(config.NEG_CLASSES)), "Test pos/neg not disjoint"

    dest.mkdir(parents=True, exist_ok=True)
    training, test = torch.load(dest / "training.pt"), torch.load(dest / "test.pt")
    if normalize_factor is not None:
        # Convert to floats and normalize to range [0,1]
        training = (training[0].unsqueeze(dim=1).float().div_(normalize_factor), training[1])
        test = (test[0].unsqueeze(dim=1).float().div_(normalize_factor), test[1])

    if view_size is not None:
        if isinstance(view_size, int): view_size = [view_size]
        view_size = list(view_size)
        training = (training[0].view([training[0].shape[0]] + view_size), training[1])
        test = (test[0].view([test[0].shape[0]] + view_size), test[1])

    ts_grp = TensorGroup()
    # Positive labeled set
    p_tr_bias = construct_bias_vec(config.POS_TRAIN_CLASSES, "pos_train_bias")
    # noinspection PyTypeChecker
    ts_grp.p_x, _ = _build_group_tensor(*training, config.TRAIN_PRIOR, config.N_P,
                                        config.POS_TRAIN_CLASSES, p_tr_bias, None, None)

    neg_tr_bias = construct_bias_vec(config.NEG_CLASSES, "neg_train_bias")
    neg_te_bias = construct_bias_vec(config.NEG_CLASSES, "neg_test_bias")
    # Unlabeled training distribution samples
    # noinspection PyTypeChecker
    ts_grp.u_tr_x, ts_grp.u_tr_y = _build_group_tensor(*training, config.TRAIN_PRIOR,
                                                       config.N_U_TRAIN,
                                                       config.POS_TRAIN_CLASSES, p_tr_bias,
                                                       config.NEG_CLASSES, neg_tr_bias)
    # noinspection PyTypeChecker
    ts_grp.test_x_tr, ts_grp.test_y_tr = _build_group_tensor(*training, config.TRAIN_PRIOR,
                                                             min(config.N_U_TRAIN, config.N_TEST),
                                                             config.POS_TRAIN_CLASSES, p_tr_bias,
                                                             config.NEG_CLASSES, neg_tr_bias)

    p_te_bias = construct_bias_vec(config.POS_TEST_CLASSES, "pos_test_bias")
    # Unlabeled test (transductive) distribution samples
    # noinspection PyTypeChecker
    ts_grp.u_te_x, ts_grp.u_te_y = _build_group_tensor(*training, config.TEST_PRIOR,
                                                       config.N_U_TEST,
                                                       config.POS_TEST_CLASSES, p_te_bias,
                                                       config.NEG_CLASSES, neg_te_bias)
    # Test (inductive) distribution samples
    # noinspection PyTypeChecker
    ts_grp.test_x, ts_grp.test_y = _build_group_tensor(*test, config.TEST_PRIOR,
                                                       config.N_TEST,
                                                       config.POS_TEST_CLASSES, p_te_bias,
                                                       config.NEG_CLASSES, neg_te_bias)
    return ts_grp


def _build_group_tensor(x: Tensor, y: Tensor, prior: float, n_sample: int,
                        p_cls: List[int], p_bias: Optional[Tensor],
                        n_cls: Optional[List[int]],
                        n_bias: Optional[Tensor]) -> Tuple[Tensor, Tensor]:
    r"""
    Samples a single tensor given u.a.r. sampling and a pre-defined prior
    :param x: Feature tensor
    :param y: Labels tensor
    :param prior: :math:`\Pr[Y = +1]`
    :param n_sample: Number of samples in tensor
    :param p_cls: IDs of the POSITIVE classes
    :param n_cls: IDs of the NEGATIVE classes
    :return: Group tensor set
    """
    if n_cls is not None:
        num_pos = binom_sample(prior, n_sample)
    else:
        num_pos = n_sample

    p_idx = _get_tensor_indices(y, p_cls)

    out_x = _get_tensor_subset(x, p_idx, p_bias, num_pos)
    if num_pos != n_sample:
        n_idx = _get_tensor_indices(y, n_cls)
        x_n = _get_tensor_subset(x, n_idx, n_bias, n_sample - num_pos)
        out_x = torch.cat([out_x, x_n], dim=0)

    out_y = torch.cat([torch.full([num_pos], fill_value=Labels.POS, dtype=torch.int32),
                       torch.full([n_sample - num_pos], fill_value=Labels.NEG, dtype=torch.int32)],
                      dim=0)

    # Randomize the elements
    # noinspection PyTypeChecker
    return shuffle_tensors(out_x, out_y)


def _get_tensor_indices(y: Tensor, cls_ids: List[int]) -> List[Set[int]]:
    r""" Get all tensor indices with the specified class IDs in ascending order """
    return [{i for i in range(y.shape[0]) if int(y[i].item()) == lbls} for lbls in cls_ids]


def _get_tensor_subset(tensor: Tensor, idx: List[Set[int]], bias: Tensor,
                       count: Optional[int] = None) -> Tensor:
    r"""
    Slices \p tensor and get \p count rows with the set of valid rows in \p idx.  If no \p count is
    specified, use all rows in \p idx.

    :param tensor: Tensor to slice
    :param idx: List of indices that can be included in the sliced tensor
    :param bias: Sampling probability vector
    :param count: Number of indices to include in subset
    :return: Subset of Tensor
    """
    if count is None: count = len(idx)

    assert count > 0, "Count must be positive"
    assert count <= sum(len(cls_size) for cls_size in idx), "Need more elements than in set"

    assert len(idx) == bias.numel(), "Bias/class elements mismatch"
    n_per_category = multinomial_sample(count, bias)
    filt_idx = []
    for n_idx, idx_grp in zip(n_per_category, idx):
        n_idx, idx_grp = int(n_idx.item()), list(idx_grp)
        assert n_idx <= len(idx_grp), "Requesting more elements than in the group"
        random.shuffle(idx_grp)
        filt_idx.extend(idx_grp[:n_idx])

    assert len(filt_idx) == count, "Filter list missing elements"
    idx = torch.tensor(filt_idx, dtype=torch.int64)
    return tensor[idx]


def construct_bias_vec(items: Collection, attr_name: Optional[str] = None) -> Tensor:
    r""" Constructs a bias vector given an attribute name """
    attr_name = attr_name.upper()

    if attr_name is None or getattr(config, attr_name) is None:
        return torch.full([len(items)], 1 / len(items))

    bias_vec = getattr(config, attr_name)
    assert len(bias_vec) == len(items), "Bias/items size mismatch"
    return torch.as_tensor(bias_vec)


def build_puc_style_dataset(x: Tensor, y: Tensor) -> TensorGroup:
    r"""
    Build the \p TensorGroup object according to the definition in the PUc paper

    :param x: X feature data tensor
    :param y: Label information
    :return: Dataset tensor group
    """
    # Divide x into negative, biased positive (above/below) sets
    neg_x = _filter_tensor_by_labels(x, y, config.NEG_CLASSES)
    pos_lt_med, pos_ge_med = _split_pos_values(x, y)

    # Sanity check splits
    # ZSH -- Won't necessary have all classes in PUC due to 20 newsgroups
    # tot_el = neg_x.shape[0] + pos_lt_med.shape[0] + pos_ge_med.shape[0]
    # assert tot_el == x.shape[0] == y.shape[0], "Element missing after split"
    assert x.shape[1] == neg_x.shape[1] == pos_lt_med.shape[1] == pos_ge_med.shape[1], "Mismatch"

    tg = TensorGroup()
    tg.p_x, _ = _build_output_tensor(pos_x_lt=pos_lt_med, pos_x_ge=pos_ge_med,
                                     n_sample=config.N_P, is_test=False)

    tg.u_tr_x, tg.u_tr_y = _build_output_tensor(pos_x_lt=pos_lt_med, pos_x_ge=pos_ge_med,
                                                n_sample=config.N_U_TRAIN, is_test=False,
                                                neg_x=neg_x, prior=config.TRAIN_PRIOR)

    tg.test_x_tr, tg.test_y_tr = _build_output_tensor(pos_x_lt=pos_lt_med, pos_x_ge=pos_ge_med,
                                                      n_sample=config.N_TEST, is_test=False,
                                                      neg_x=neg_x, prior=config.TRAIN_PRIOR)

    tg.u_te_x, tg.u_te_y = _build_output_tensor(pos_x_lt=pos_lt_med, pos_x_ge=pos_ge_med,
                                                n_sample=config.N_U_TEST, is_test=True,
                                                neg_x=neg_x, prior=config.TEST_PRIOR)

    tg.test_x, tg.test_y = _build_output_tensor(pos_x_lt=pos_lt_med, pos_x_ge=pos_ge_med,
                                                n_sample=config.N_TEST, is_test=True,
                                                neg_x=neg_x, prior=config.TEST_PRIOR)

    _print_ds_info(y)
    return tg


def _build_output_tensor(pos_x_lt: Tensor, pos_x_ge, n_sample: int, is_test: bool,
                         neg_x: Optional[Tensor] = None,
                         prior: Optional[float] = None) -> Tuple[Tensor, Tensor]:
    r"""
    Samples a single tensor given u.a.r. sampling and a pre-defined prior
    :param pos_x_lt: Positive-valued examples LESS THAN to centroid
    :param pos_x_ge: Positive-valued examples GREATER THAN OR EQUAL to centroid
    :param n_sample: Number of elements to put in tensor
    :param is_test: If \p True, then use test probability on positive set
    :param neg_x: \p X tensor for negative valued elements
    :param prior: :math:`\Pr[Y = +1]`
    :return: Tensor with specified parameters
    """
    assert (neg_x is not None and prior is not None) or (neg_x is None and prior is None), \
        "Both or neither neg_x and prior must be None"

    # Determine how positive/negative elements split
    if neg_x is not None:
        num_pos = binom_sample(prior, n_sample)
    else:
        num_pos = n_sample
    num_neg = n_sample - num_pos

    def _select_subset(x: Tensor, n: int) -> Tensor:
        r""" Select a subset of the tensor elements """
        assert n <= x.shape[0], "Trying to take more elements than in tensor"
        return shuffle_tensors(x)[:n]

    def _get_y_tensor(n: int, lbl: int) -> Tensor:
        r""" Construct the y tensor """
        return torch.full([n], lbl, dtype=torch.int)

    all_x, all_y = [], []
    # Some instances may have no negative elements
    if num_neg > 0:
        # noinspection PyTypeChecker
        all_x.append(_select_subset(neg_x, num_neg))
        all_y.append(_get_y_tensor(num_neg, Labels.NEG))

    # Split according to distribution in PUc paper
    lt_prob = 0.1 if is_test else 0.9
    num_lt = binom_sample(lt_prob, num_pos)
    all_x.append(_select_subset(pos_x_lt, num_lt))
    all_x.append(_select_subset(pos_x_ge, num_pos - num_lt))

    all_y.append(_get_y_tensor(num_pos, Labels.POS))

    out_x, out_y = torch.cat(all_x, dim=0).cpu(), torch.cat(all_y, dim=0).int().cpu()
    assert out_x.shape[0] == out_y.shape[0] == n_sample, "Not all elements present"
    assert out_x.shape[1] == pos_x_lt.shape[1], "Feature dimension mismatch"
    # Randomize the elements
    # noinspection PyTypeChecker
    return shuffle_tensors(out_x, out_y)


def _print_ds_info(y: Tensor) -> None:
    r""" Prints information about the dataset """
    ds_name = config.DATASET.name

    num_pos = y[y == Labels.POS].numel()
    tot_ele = y.numel()
    logging.debug(f"{ds_name} Dataset: Positive Prior: {num_pos / tot_ele:.2}")
    logging.debug(f"{ds_name} Dataset: Total Number of Instances: {tot_ele}")


def _filter_tensor_by_labels(x: Tensor, y: Tensor, lbls: List[int]) -> Tensor:
    r"""
    Filters the \p x tensor to keep only the examples with the specified labels

    :param x: Feature data
    :param y: Example labels
    :param lbls: Label set used to filter X
    :return: X for all elements with label in \p lbls
    """
    # Build a mask for all positive labeled examples
    mask = torch.full([x.shape[0]], False, dtype=torch.bool)
    for cls_id in lbls:
        mask[y == cls_id] = True
    return x[mask]  # Positive X


def _split_pos_values(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
    r"""
    Splits the positive set according to the norm formula in the PUc formula

    :param x: Raw tensor values
    :param y: Labels for each element
    :return: Tuple of Tensors less than median and greater than median respectively
    """
    x = _filter_tensor_by_labels(x, y, config.POS_TRAIN_CLASSES)

    x_mean = torch.mean(x, dim=0)
    assert x_mean.shape[0] == x.shape[1], "Mean is along value dimension"

    c = (x - x_mean).norm(p=2, dim=1)
    assert c.shape[0] == x.shape[0], "Each sample should have a norm value"
    med = c.median()
    return x[c < med], x[c >= med]


CHUNK_SIZE = 128 * 1024  # BYTES


def download_file(url: str, file_path: Path) -> None:
    r""" Downloads the specified file """
    if file_path.exists():
        logging.info(f"File \"{file_path}\" already downloaded. Skipping...")
        return

    # Store the download file to a temporary directory
    tmp_file = file_path.parent / f"tmp_{file_path.stem}.download"

    msg = f"Downloading file at \"{url}\""
    logging.info(f"Starting: {msg}...")
    # NOTE the stream=True parameter below
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(str(tmp_file), 'wb+') as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:  # filter out keep-alive new chunks
                    f.write(chunk)
                    # f.flush()
    logging.info(f"COMPLETED: {msg}")

    msg = f"Renaming temporary file \"{tmp_file}\" to \"{file_path}\""
    logging.info(f"Starting: {msg}...")
    tmp_file.rename(file_path)
    logging.info(f"COMPLETED: {msg}")

    assert file_path.exists(), "Specified file path does not exist"


def download_nltk_tokenizer(newsgroups_dir: Path):
    r""" NLTK uses 'punkt' tokenizer which needs to be downloaded """
    # Download the nltk tokenizer
    nltk_path = newsgroups_dir / "nltk"
    nltk_path.mkdir(parents=True, exist_ok=True)
    nltk.data.path.append(str(nltk_path))
    nltk.download("punkt", download_dir=str(nltk_path))


OPTION_FILE = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B" \
              "/elmo_2x4096_512_2048cnn_2xhighway_5.5B_options.json"
WEIGHT_FILE = "https://allennlp.s3.amazonaws.com/models/elmo/2x4096_512_2048cnn_2xhighway_5.5B" \
              "/elmo_2x4096_512_2048cnn_2xhighway_5.5B_weights.hdf5"


def make_elmo_embedders(data_dir: Path) -> List[ElmoEmbedder]:
    r""" Make and return a simple ELMo embedder.  May need to download the ELMo weights first """
    allennlp_dir = data_dir / "allennlp"
    allennlp_dir.mkdir(parents=True, exist_ok=True)
    os.putenv('ALLENNLP_CACHE_ROOT', str(allennlp_dir))

    def _make_elmo(n_device: int) -> ElmoEmbedder:
        # noinspection PyTypeChecker
        return ElmoEmbedder(cached_path(OPTION_FILE, allennlp_dir),
                            cached_path(WEIGHT_FILE, allennlp_dir), n_device)
    elmos = []
    if torch.cuda.is_available():
        elmos.append(_make_elmo(n_device=0))
    # CPU only
    elmos.append(_make_elmo(n_device=-1))
    return elmos


def use_elmo_to_process_doc(elmos: List[ElmoEmbedder], document: str) -> np.ndarray:
    r""" Use ELMo to process the document """
    item = [nltk.tokenize.word_tokenize(document)]
    with torch.no_grad():
        try:
            em = elmos[0].embed_batch(item)
        except RuntimeError:
            em = elmos[1].embed_batch(item)
    em = np.concatenate(
        [np.mean(em[0], axis=1).flatten(),
         np.min(em[0], axis=1).flatten(),
         np.max(em[0], axis=1).flatten()])
    return em
