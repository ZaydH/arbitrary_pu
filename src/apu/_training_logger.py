# -*- utf-8 -*-
r"""
    _training_logger.py
    ~~~~~~~~~~~~~~~

    Provides utilities to simplify and standardize logging in particular for training for training
    with \p torch.

    :copyright: (c) 2019 by Zayd Hammoudeh.
    :license: MIT, see LICENSE for more details.
"""

__all__ = ["TrainingLogger", "create_stdout_handler", "setup_logger"]

try:
    # noinspection PyUnresolvedReferences
    import matplotlib
    # noinspection PyUnresolvedReferences
    from matplotlib import use
    use('Agg')
except ImportError:
    # raise ImportError("Unable to import matplotlib")
    pass

from _pydecimal import Decimal
import copy
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import List, Optional, Any

from tensorboardX import SummaryWriter
import torch
from torch import nn as nn, Tensor

from . import utils
from .types import ListOrInt, PathOrStr, OptDict, OptStr, OptInt, TorchOrNp
from .utils import log_seeds

FORMAT_STR = '%(asctime)s -- %(levelname)s -- %(message)s'


class TrainingLogger:
    r""" Helper class used for standardizing logging """
    FIELD_SEP = " "
    DEFAULT_WIDTH = 12
    EPOCH_WIDTH = 7

    DEFAULT_FIELD = None

    LOG = logging.info

    tb = None

    @classmethod
    def create_tensorboard(cls, file_path: PathOrStr, hparams: OptDict = None):
        r""" Creates the \p Tensorboard for logging """
        if cls.has_tensorboard():
            # raise RuntimeError("Already has a tensorboard. Cannot recreate")
            return

        if not isinstance(file_path, Path):
            file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        cls.tb = SummaryWriter(str(file_path))

        if hparams is not None:
            for key in sorted(hparams.keys()):
                tags = "/".join(["hyperparam", key])
                cls.tb.add_text(tags, str(hparams[key]))

    @classmethod
    def add_graph(cls, model: nn.Module, x: Tensor):
        r""" Add the computational graph to the \p Tensorboard """
        cls._check_tensorboard_exists()
        cls.tb.add_graph(model, x)

    @classmethod
    def has_tensorboard(cls) -> bool:
        r""" Returns \p True if \p TrainingLogger has a \p Tensorboard object """
        return cls.tb is not None

    @classmethod
    def _check_tensorboard_exists(cls) -> None:
        r""" Check whether the \p Tensorboard object has been created """
        assert cls.has_tensorboard(), "Trying to use tensorboard but no tensorboard created"

    def __init__(self, fld_names: List[str], fld_widths: Optional[List[int]] = None,
                 logger_name: OptStr = None, tb_grp_name: OptStr = None):
        r"""
        :param fld_names: Names of the flds to log
        :param fld_widths: Width of the field in monospace count
        :param tb_grp_name: Optional group name that can be used to automatically add data to
                            a tensorboard
        """
        if fld_widths is None: fld_widths = len(fld_names) * [TrainingLogger.DEFAULT_WIDTH]
        if len(fld_widths) != len(fld_names):
            raise ValueError("Mismatch in the length of field names and widths")
        if tb_grp_name and self.tb is None:
            msg = "Tensorboard group name %s specified but no tensorboard created" % tb_grp_name
            raise ValueError(msg)

        logger = logging.getLogger(logger_name)
        self._log = logger.info
        self._fld_names = fld_names
        self._fld_widths = fld_widths
        self._grp_name = tb_grp_name  # Used for tensorboard

        # Print the column headers
        combined_names = ["Epoch"] + fld_names
        combined_widths = [TrainingLogger.EPOCH_WIDTH] + fld_widths
        fmt_str = TrainingLogger.FIELD_SEP.join(["{:^%d}" % _d for _d in combined_widths])
        self._log(fmt_str.format(*combined_names))
        # Line of separators under the headers (default value is hyphen)
        sep_line = TrainingLogger.FIELD_SEP.join(["{:-^%d}" % _w for _w in combined_widths])
        # pylint: disable=logging-format-interpolation
        self._log(sep_line.format(*(len(combined_widths) * [""])))

    @property
    def num_fields(self) -> int:
        r""" Number of fields to log """
        return len(self._fld_widths)

    def log(self, epoch: int, values: List[Any]) -> None:
        r""" Log the list of values.  If it has been created, the tensorboard is also updated """
        if len(values) > self.num_fields:
            raise ValueError("More values to log than fields known by the logger")
        if self.tb is not None:
            self._add_to_tensorboard(epoch, values)

        values = self._clean_values_list(values)
        format_str = self._build_values_format_str(values).format(epoch, *values)
        self._log(format_str)

    def _add_to_tensorboard(self, epoch: int, vals: List[Any]) -> None:
        r""" Automatically add values to the \p Tensorboard """
        self._check_tensorboard_exists()

        for tag_name, val in zip(self._fld_names, vals):
            if isinstance(val, str): continue
            if isinstance(val, bool): val = 1 if val else 0
            if isinstance(val, Tensor): val = float(val.item())
            # if isinstance(val, Decimal): val = float(val)

            if self._grp_name: tag_name = "/".join([self._grp_name, tag_name])
            tag_name = tag_name.replace(" ", "_")

            self.tb.add_scalar(tag_name, val, epoch)

    def _build_values_format_str(self, values: List[Any]) -> str:
        r""" Constructs a format string based on the values """
        def _get_fmt_str(_w: int, fmt: str) -> str:
            return "{:^%d%s}" % (_w, fmt)

        frmt = [_get_fmt_str(self.EPOCH_WIDTH, "d")]
        for width, v in zip(self._fld_widths, values):
            if isinstance(v, (str, bool)):
                fmt_str = "s"
            elif isinstance(v, Decimal):
                fmt_str = ".3E"
            elif isinstance(v, int):
                fmt_str = "d"
            elif isinstance(v, float):
                fmt_str = ".4f"
            else:
                raise ValueError("Unknown value type")

            frmt.append(_get_fmt_str(width, fmt_str))
        return TrainingLogger.FIELD_SEP.join(frmt)

    def _clean_values_list(self, values: List[Any]) -> List[Any]:
        r""" Modifies values in the \p values list to make them straightforward to log """
        values = copy.deepcopy(values)
        # Populate any missing fields
        while len(values) < self.num_fields:
            values.append(TrainingLogger.DEFAULT_FIELD)

        new_vals = []
        for v in values:
            if isinstance(v, bool): v = "+" if v else ""
            if v is None: v = "N/A"
            elif isinstance(v, Tensor): v = v.item()

            # Must be separate since v can be a float due to a Tensor
            if isinstance(v, float) and (v <= 1E-3 or v >= 1E5): v = Decimal(v)
            new_vals.append(v)
        return new_vals

    def add_figure(self, tag: str, fig, step: OptInt = None):
        r""" Add a figure to the tensorboard """
        self.tb.add_figure(tag, fig, global_step=step)

    def log_pr_curve(self, set_name: str, labels: TorchOrNp, dec_scores: TorchOrNp,
                     epoch: OptInt = None) -> None:
        r"""
        Log the precision recall curve

        :param set_name: Set name, e.g., "positive", "negative", "unlabeled", "test", etc.
        :param labels: Label
        :param dec_scores: Decision score value
        :param epoch: Optional epoch number
        """
        self._check_tensorboard_exists()

        name = "-".join([set_name, "conf-matrix"])
        if self._grp_name:
            name = "/".join([self._grp_name, name])
        self.tb.add_pr_curve(name, labels, dec_scores, epoch)

    def log_confidence_matrix(self, epoch: int, set_name: str, con_mat: TorchOrNp) -> None:
        r"""
        Logs a confidence matrix
        :param epoch: Epoch number
        :param set_name: Dataset name
        :param con_mat: Confidence matrix
        """
        self._check_tensorboard_exists()
        assert list(con_mat.shape) == [2, 2], "Only 2x2 supported"

        name = "-".join([set_name, "conf-matrix"])
        if self._grp_name: name = "/".join([self._grp_name, name])

        conf_fields = dict()
        for row in range(2):
            for col in range(2):
                correct = "True" if row == col else "False"
                label = "Positive" if col == 1 else "Negative"
                conf_fields["_".join([correct, label])] = con_mat[row][col]
        self.tb.add_scalars(name, conf_fields, epoch)


def setup_logger(quiet_mode: bool, log_level: int = logging.DEBUG,
                 log_dir: Optional[PathOrStr] = None, job_id: Optional[ListOrInt] = None) -> None:
    r"""
    Logger Configurator

    Configures the test logger.

    :param quiet_mode: True if quiet mode (i.e., disable logging to stdout) is used
    :param log_level: Level to log
    :param log_dir: Directory to write the logs (if any)
    :param job_id: Identification number for the job
    """
    date_format = '%m/%d/%Y %I:%M:%S %p'  # Example Time Format - 12/12/2010 11:46:36 AM

    job_str, time_str = "", str(datetime.now()).replace(" ", "-")
    if job_id is not None:
        if isinstance(job_id, int): job_id = [job_id]
        job_str = "_j=%s_" % "-".join("%05d" % x for x in job_id)
    filename = "log%s_%s.log" % (job_str, time_str)

    if log_dir is None:
        log_dir = utils.LOG_DIR
    if not isinstance(log_dir, Path): log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = log_dir / filename

    logging.basicConfig(filename=filename, level=log_level, format=FORMAT_STR, datefmt=date_format)

    # Print to stdout on root logger if no handlers exist
    if not quiet_mode and not hasattr(setup_logger, "_HAS_ROOT"):
        create_stdout_handler(log_level, format_str=FORMAT_STR)
        setup_logger._HAS_ROOT = True

    # Matplotlib clutters the logger so change its log level
    if "matplotlib" in sys.modules:
        # noinspection PyProtectedMember
        matplotlib._log.setLevel(logging.WARNING)  # pylint: disable=protected-access
    # Disable logging in PyTorch ignite
    logging.getLogger("ignite.engine.engine.Engine").setLevel(logging.WARNING)

    logging.info("******************* New Run Beginning *****************")
    # noinspection PyUnresolvedReferences
    logging.debug("Torch Version: %s", torch.__version__)
    logging.debug("Torch CUDA: %s", "ENABLED" if torch.cuda.is_available() else "Disabled")
    # noinspection PyUnresolvedReferences
    logging.debug("Torch cuDNN Enabled: %s", "YES" if torch.backends.cudnn.is_available() else "NO")
    logging.info(" ".join(sys.argv))
    log_seeds()


def create_stdout_handler(log_level, format_str: str = FORMAT_STR,
                          logger_name: OptStr = None) -> None:
    r"""
    Creates and adds a handler for logging to stdout.  If \p logger_name is specified, the handler
    is added to that logger.  Otherwise it is added to the root logger.

    :param log_level: Level at which to log
    :param format_str: Format of the logs
    :param logger_name: Optional logger to which to add the handler
    :return: Logger object
    """
    logger = logging.getLogger(logger_name)  # Gets logger if exists. Otherwise creates new one

    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(log_level)
    formatter = logging.Formatter(format_str)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
