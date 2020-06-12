__all__ = ["load_data"]

import logging
from pathlib import Path

import h5py
import nltk
import nltk.tokenize
import numpy as np
import sklearn.datasets
from sklearn import preprocessing
# noinspection PyProtectedMember
from sklearn.utils import Bunch

import torch

from .types import TensorGroup
from .utils import download_nltk_tokenizer, make_elmo_embedders, shared_tensor_dataset_importer, \
    use_elmo_to_process_doc

PREPROCESSED_FIELD = "data"

NEWSGROUPS_NORMALIZE_FACTOR = 1
NEWSGROUPS_DIM = (9216,)


def _build_elmo_file_path(newsgroups_dir: Path, ds_name: str) -> Path:
    r"""
    Constructs the file path to store the preprocessed vector h5py file.
    :param newsgroups_dir: Path to where the newsgroups data is stored
    :param ds_name: Either "test" or "train"
    :return: Path to the elmo file directory
    """
    return newsgroups_dir / f"20newsgroups_elmo_mmm_{ds_name}.hdf5"


def _generate_preprocessed_vectors(ng_dir: Path, ds_name: str,
                                   newsgroups: Bunch, path: Path) -> None:
    r"""
    Constructs the preprocessed vectors for either the test or train datasets.
    :param ng_dir: Path to where the newsgroups data is stored
    :param ds_name: Either "test" or "train"
    :param newsgroups: Scikit-Learn object containing the 20 newsgroups dataset
    :param path: Location to write serialized vectors
    """
    assert ds_name == "train" or ds_name == "test"
    n = len(newsgroups.data)

    # First learner use CUDA, second does not
    elmos = make_elmo_embedders(ng_dir)

    data = np.zeros([n, 9216])

    msg = f"Creating the preprocessed vectors for \"{ds_name}\" set"
    logging.info(f"Starting: {msg}")
    # Has to be out of for loop or stdout overwrite messes up
    if not torch.cuda.is_available(): logging.info('CUDA unavailable for ELMo encoding')
    for i in range(n):
        print(f"Processing {ds_name} document {i+1}/{n}", end="", flush=True)
        data[i] = use_elmo_to_process_doc(elmos, newsgroups.data[i])
        # Go back to beginning of the line. Weird formatting due to PyCharm issues
        print('\r', end="")

    path.parent.mkdir(parents=True, exist_ok=True)
    f = h5py.File(str(path), 'w')
    f.create_dataset(PREPROCESSED_FIELD, data=data)
    f.close()
    logging.info(f"COMPLETED: {msg}")


def _create_serialized_20newsgroups_preprocessed(ng_dir: Path, processed_dir: Path) -> None:
    r""" Serializes the 20 newsgroups as preprocessed vectors """
    for ds_name, is_training in (("train", True), ("test", False)):
        out_path = processed_dir / f"{'training' if is_training else 'test'}.pt"
        if out_path.exists():
            continue

        # shuffle=True is used since ElmoEmbedder stores states between sentences so randomness
        # should reduce this effect
        docs_dir = ng_dir / "text"
        docs_dir.mkdir(parents=True, exist_ok=True)
        # noinspection PyUnresolvedReferences
        bunch = sklearn.datasets.fetch_20newsgroups(subset=ds_name, data_home=docs_dir,
                                                    shuffle=True)

        download_nltk_tokenizer(ng_dir)

        path = _build_elmo_file_path(ng_dir, ds_name)
        if not path.exists():
            _generate_preprocessed_vectors(ng_dir, ds_name, bunch, path)

        vecs = h5py.File(str(path), 'r')
        x = preprocessing.scale(vecs[PREPROCESSED_FIELD][:])
        x, y = torch.from_numpy(x).float(), torch.from_numpy(bunch.target).int()

        torch.save((x.cpu(), y.cpu()), out_path)


def load_data(dest: Path) -> TensorGroup:
    processed_dir = dest / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    _create_serialized_20newsgroups_preprocessed(dest, processed_dir)

    return shared_tensor_dataset_importer(dest=processed_dir,
                                          normalize_factor=NEWSGROUPS_NORMALIZE_FACTOR,
                                          view_size=NEWSGROUPS_DIM)
