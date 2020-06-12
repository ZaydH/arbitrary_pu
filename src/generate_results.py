import copy
from dataclasses import dataclass
from enum import Enum
import logging
from pathlib import Path
import re
from types import ModuleType
from typing import ClassVar, List, Optional, Union

import numpy as np
from sklearn.metrics import confusion_matrix, average_precision_score, f1_score

from fastai.basic_data import DeviceDataLoader
from fastai.metrics import auc_roc_score
import torch
from torch import Tensor
from torch.utils.data import TensorDataset

from apu import config, LearnerParams
from apu.datasets.types import APU_Module, TensorGroup
from apu.utils import RES_DIR, TORCH_DEVICE, construct_filename, log_decision_boundary
from puc_learner import PUcLearner


@dataclass(order=True)
class LearnerResults:
    r""" Encapsulates ALL results for a single NLP learner model """
    FIELD_SEP: ClassVar[str] = ","

    @dataclass(init=True, order=True)
    class DatasetResult:
        r""" Encapsulates results of a model on a SINGLE dataset """
        ds_size: int
        accuracy: float = None
        auroc: float = None
        auprc: float = None
        f1: float = None

    loss_name = None
    valid_loss = None

    decision_m: float = None
    decision_b: float = None

    unlabel_train = None
    tr_test = None
    unlabel_test = None
    test = None


def calculate_results(tg: TensorGroup, our_learner, puc_learners: List[PUcLearner],
                      dest_dir: Optional[Union[Path, str]] = None,
                      exclude_puc: bool = False) -> dict:
    r"""
    Calculates and writes to disk the model's results

    :param tg: Tensor group containing the test conditions
    :param our_learner: PURR, PU2aPNU, or PU2wUU
    :param puc_learners: Learner(s) implementing the PUc algorithm
    :param dest_dir: Location to write the results
    :param exclude_puc: DEBUG ONLY. Exclude the PUc results.
    :return: Dictionary containing results of all experiments
    """
    if dest_dir is None: dest_dir = RES_DIR
    dest_dir = Path(dest_dir)

    our_learner.eval()

    all_res = dict()
    ds_flds = (("unlabel_train", TensorDataset(tg.u_tr_x, tg.u_tr_y)),
               ("tr_test", TensorDataset(tg.test_x_tr, tg.test_y_tr)),
               ("unlabel_test", TensorDataset(tg.u_te_x, tg.u_te_y)),
               ("test", TensorDataset(tg.test_x, tg.test_y)))

    for block_name, block in our_learner.blocks():
        res = LearnerResults()
        res.loss_name = block.loss.name()
        res.valid_loss = block.best_loss

        for ds_name, ds in ds_flds:
            # noinspection PyTypeChecker
            dl = DeviceDataLoader.create(ds, shuffle=False, drop_last=False, bs=config.BATCH_SIZE,
                                         num_workers=0, device=TORCH_DEVICE)
            all_y, dec_scores = [], []
            with torch.no_grad():
                for xs, ys in dl:
                    all_y.append(ys)
                    dec_scores.append(block.forward(xs))

            # Iterator transforms label so transform it back
            y = torch.cat(all_y, dim=0).squeeze().cpu().numpy()
            dec_scores = torch.cat(dec_scores, dim=0).squeeze().cpu()
            y_hat, dec_scores = dec_scores.sign().cpu().numpy(), dec_scores.cpu().numpy()
            # Store for name "unlabel" or "test"
            res.__setattr__(ds_name, _single_ds_results(block, ds_name, y, y_hat, dec_scores))

        if config.DATASET.is_synthetic():
            log_decision_boundary(block.module, name=block_name)
        all_res[block_name] = res

    if not exclude_puc:
        for puc in puc_learners:
            all_res[puc.name()] = _build_puc_results(puc, ds_flds)
            if config.DATASET.is_synthetic():
                log_decision_boundary(puc, name=puc.name())

    config.print_configuration()
    _write_results_to_disk(dest_dir, our_learner.train_start_time(), all_res)

    return all_res


def _single_ds_results(block: "Union[PUcLearner, APU_Module]",
                       ds_name: str, y: np.ndarray, y_hat: np.ndarray,
                       dec_scores: np.ndarray) -> LearnerResults.DatasetResult:
    r""" Logs and returns the results on a single dataset """
    res = LearnerResults.DatasetResult(y.shape[0])

    str_prefix = f"{block.name()} {ds_name}:"

    logging.debug(f"{str_prefix} Dataset Size: {res.ds_size:,}")
    # Pre-calculate fields needed in other calculations
    res.conf_matrix = confusion_matrix(y, y_hat)
    assert np.sum(res.conf_matrix) == res.ds_size, "Verify size matches"

    # Calculate prior information
    res.accuracy = np.trace(res.conf_matrix) / res.ds_size
    logging.debug(f"{str_prefix} Accuracy: {100. * res.accuracy:.3}%")

    res.auroc = auc_roc_score(torch.tensor(dec_scores).cpu(), torch.tensor(y).cpu())
    logging.debug(f"{str_prefix} AUROC: {res.auroc:.6}")

    res.auprc = average_precision_score(y, dec_scores)
    logging.debug(f"{str_prefix} AUPRC: {res.auprc:.6}")

    res.f1 = float(f1_score(y, y_hat))
    logging.debug(f"{str_prefix} F1-Score: {res.f1:.6f}")

    logging.debug(f"{str_prefix} Confusion Matrix:\n{res.conf_matrix}")
    res.conf_matrix = re.sub(r"\s+", " ", str(res.conf_matrix))

    return res


def _build_puc_results(puc_learner: PUcLearner, ds_flds) -> LearnerResults:
    r""" Construct the PUc learner results """
    res = LearnerResults()
    res.loss_name, res.valid_loss = puc_learner.name(), None
    for ds_name, ds in ds_flds:
        y = ds.tensors[1].cpu().numpy()
        y_hat = puc_learner.predict(ds.tensors[0])
        dec_scores = puc_learner.decision_function(ds.tensors[0])
        res.__setattr__(ds_name, _single_ds_results(puc_learner, ds_name, y, y_hat, dec_scores))
    return res


def _write_results_to_disk(dest_dir: Path, start_time: str, all_res: dict) -> None:
    r""" Logs the results to disk for later analysis """
    def _log_val(_v) -> str:
        if isinstance(_v, str): return _v
        if isinstance(_v, bool): return str(_v)
        if isinstance(_v, int): return f"{_v:d}"
        if isinstance(_v, Tensor): _v = float(_v.item())
        if isinstance(_v, float): return f"{_v:.15f}"
        if isinstance(_v, Enum): return _v.name
        if isinstance(_v, set):
            return ",".join([_log_val(_x) for _x in sorted(_v)])
        if _v is None: return "NA"
        if isinstance(_v, list):
            lst_str = ", ".join([str(ele) for ele in _v])
            return f"\"[{lst_str}]\""
        raise ValueError(f"Unknown value type \"{type(_v)}\" to log")

    classifier_name_idx = 1
    loss_name_idx = classifier_name_idx + 1
    num_epoch_idx = None  # Start of fields to ignore for PUc learner
    kernel_type_idx = None

    header = ["start-time", "classifier-name", "loss-name"]
    base_fields = [start_time, None, None]
    for key, val in vars(config).items():
        # Exclude dunders in all modules
        if key.startswith("__") and key.endswith("__"): continue
        # Exclude any functions in config
        if callable(val): continue
        # Exclude any imported modules
        if isinstance(val, ModuleType): continue
        if key.lower().endswith("_key"): continue
        if key.upper() not in config.__all__:
            logging.debug(f"Skipping {key} as not in config.__all__")
            continue
        if key.lower() == config.NUM_EPOCH_KEY.lower():
            assert num_epoch_idx is None, f"Number of epochs field should be None by default"
            num_epoch_idx = len(base_fields)

        header.append(key.replace("_", "-"))
        if key == "bias" and isinstance(val, list):
            base_fields.append(",".join([f"{x:.2f}" for _, x in val]))
            continue
        if key.lower() == config.KERNEL_KEY.lower():
            kernel_type_idx = len(base_fields)
            base_fields.append("N/A")
            continue
        base_fields.append(_log_val(val))

    all_fields = []
    for i, (block_name, block_res) in enumerate(all_res.items()):
        fields = copy.deepcopy(base_fields)
        fields[classifier_name_idx] = block_name
        if PUcLearner.BASE_NAME.lower() != block_name.lower():
            fields[loss_name_idx] = block_res.loss_name
        else:
            for itr in range(num_epoch_idx, len(fields)):
                fields[itr] = "N/A"
            fields[loss_name_idx] = "squared_loss"
            fields[kernel_type_idx] = config.KERNEL_TYPE

        # Block identifier
        fields[classifier_name_idx] = block_name
        # Add learner specific parameters
        for attr in LearnerParams.Attribute:
            if i == 0: header.append(attr.name)
            attr_val = config.get_learner_val(block_name, attr)
            assert attr_val is not None, "Attribute value unset"
            fields.append(_log_val(attr_val))

        for field_name in ("valid_loss", "decision_m", "decision_b"):
            if i == 0: header.append(field_name)
            fields.append(_log_val(block_res.__getattribute__(field_name)))

        for res_name in ("unlabel_train", "tr_test", "unlabel_test", "test"):
            res_val = block_res.__getattribute__(res_name)
            for fld_name, fld_val in vars(res_val).items():
                if i == 0: header.append(f"{res_name}-{fld_name}".replace("_", "-"))
                fields.append(_log_val(fld_val))
        # Add the results
        all_fields.append(fields)

    # Writes the file
    filename = construct_filename(prefix="res", out_dir=dest_dir, file_ext="csv",
                                  add_timestamp=True)
    with open(str(filename), "w+") as f_out:
        f_out.write(LearnerResults.FIELD_SEP.join(header))
        for block_fields in all_fields:
            f_out.write("\n")
            f_out.write(LearnerResults.FIELD_SEP.join(block_fields))
