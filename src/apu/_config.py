__all__ = ["BATCH_SIZE", "CALIBRATION_SPLIT_RATIO",
           "CENTROIDS", "DATASET",
           # "GAMMA",
           "KERNEL_TYPE",
           # "LEARNING_RATE",
           "N_P", "N_U_TEST", "N_U_TRAIN", "N_TEST",
           "NEG_TEST_BIAS", "NEG_TRAIN_BIAS", "NEG_CLASSES",
           "NUM_EPOCH",
           "NUM_FF_LAYERS", "NUM_SIGMA_LAYERS",
           "PN_TRAIN_BATCH_SIZE", "PN_TEST_BATCH_SIZE",
           "POS_TEST_BIAS", "POS_TEST_CLASSES",
           "POS_TRAIN_BIAS", "POS_TRAIN_CLASSES",
           "SIGMA_BATCH_SIZE",
           "TEST_PRIOR", "TRAIN_PRIOR",
           "USE_ABS",
           "VALIDATION_SPLIT_RATIO",
           # "WEIGHT_DECAY",
           "parse", "print_configuration",
           "set_gamma", "set_layer_counts", "set_learning_rate",
           "set_neg_train_bias", "set_pos_train_bias",
           "set_weight_decay"
           ]

import copy
import logging
from pathlib import Path
import re
from typing import Callable, Collection, List, Optional, Union

from ruamel.yaml import YAML

from .datasets.types import Centroid, NewsgroupCategory, APU_Dataset
from .types import LearnerParams, PathOrStr
from .utils import NEWSGROUPS_DIR

USE_ABS = True
ABS_KEY = "use_abs"

DATASET = None  # type: Optional[APU_Dataset]
DATASET_KEY = "dataset"

TRAIN_PRIOR = 0.5
TEST_PRIOR = 0.5

N_P = 300
N_U_TRAIN = 700
N_U_TEST = 700
# Test samples for inductive verification
N_TEST = 1000

POS_TRAIN_CLASSES = None
POS_TEST_CLASSES = None
NEG_CLASSES = None

POS_TRAIN_20NEWS = None
POS_TEST_20NEWS = None
NEG_20NEWS = None

NEG_BIAS = None
NEG_TRAIN_BIAS = None
NEG_TEST_BIAS = None

POS_TRAIN_BIAS = None
POS_TEST_BIAS = None

CENTROIDS = None
CENTROIDS_KEY = "centroids"

NUM_FF_LAYERS = None
NUM_SIGMA_LAYERS = None

NUM_EPOCH = 500
NUM_EPOCH_KEY = "num_epoch"
SIGMA_BATCH_SIZE = None
BATCH_SIZE = 250
PN_TRAIN_BATCH_SIZE = None
PN_TEST_BATCH_SIZE = None

LEARNING_RATE = 1E-3
WEIGHT_DECAY = 1E-4
GAMMA = 0

# Fraction of training samples used for
VALIDATION_SPLIT_RATIO = 1 / 6
CALIBRATION_SPLIT_RATIO = None

# ===  Start PUc only fields here
KERNEL_TYPE = None
KERNEL_KEY = "kernel_type"

LEARNER_CONFIGS = dict()


def parse(config_file: PathOrStr) -> None:
    r""" Parses the configuration """
    config_file = Path(config_file)
    if not config_file.exists():
        raise FileNotFoundError(f"Unable to find config file {config_file}")
    if not config_file.is_file():
        raise FileExistsError(f"Configuration {config_file} does not appear to be a file")

    with open(str(config_file), 'r') as f_in:
        all_yaml_docs = YAML().load_all(f_in)

        base_config = next(all_yaml_docs)
        _parse_general_settings(base_config)
        _parse_learner_specific_settings(all_yaml_docs)


def _parse_general_settings(config) -> None:
    r"""
    Parses general settings for the learning configuration including the dataset, priors, positive
    & negative class information.  It also extracts general learner information
    """
    module_dict = globals()
    for key, val in config.items():
        if key.lower() == CENTROIDS_KEY:
            _parse_centroids(module_dict, val)
        elif key.lower() == DATASET_KEY:
            ds_name = val.upper()
            try:
                module_dict[key.upper()] = APU_Dataset[ds_name]
            except KeyError:
                raise ValueError(f"Unknown dataset {ds_name}")
        elif key.lower() == ABS_KEY:
            module_dict[key.upper()] = bool(val)
        elif _is_invalid_neg_bias_key(key):
            raise ValueError(f"{key.upper()} is an invalid key for a configuration file")
        # Removed to allow string class names for 20 newsgroups
        # elif key.lower().endswith("_classes"):
        #     module_dict[key.upper()] = list(int(x) for x in val)
        # Drop in replacement field
        else:
            key = key.upper()
            if key not in module_dict:
                raise ValueError(f"Unknown configuration field \"{key}\"")
            module_dict[key] = val

    # Copy all the biases
    module_dict["NEG_TRAIN_BIAS"] = module_dict["NEG_TEST_BIAS"] = NEG_BIAS

    if DATASET.is_newsgroups():
        _parse_newsgroups(module_dict)
    # Set other batch sizes if unset
    base_bs_key = "BATCH_SIZE"
    other_keys = ("SIGMA_BATCH_SIZE", "PN_TRAIN_BATCH_SIZE", "PN_TEST_BATCH_SIZE")
    for other_key in other_keys:
        if module_dict[other_key] is None:
            module_dict[other_key] = module_dict[base_bs_key]

    # Sanity checks nothing is out of whack
    _verify_configuration(module_dict)


def _parse_learner_specific_settings(all_yaml_docs) -> None:
    r""" Gets the learner specific settings to allow for different settings per learner """
    # Each additional doc is a learner
    for learner_config in all_yaml_docs:
        # Handle case where multiple learners in same doc
        for name, learn_val in learner_config.items():
            name = name.lower()
            params = LearnerParams(learner_name=name)
            if name in LEARNER_CONFIGS:
                raise ValueError(f"Duplicate learner settings for learner \"{name}\"")
            LEARNER_CONFIGS[name] = params

            for key, val in learn_val.items():
                params.set_attr(key, val)


def get_learner_val(learner_name: str, param: LearnerParams.Attribute) -> Union[int, float]:
    r""" Gets learner specific values """
    orig_learner_name = learner_name
    name = learner_name.lower()
    while True:
        try:
            learner_params = LEARNER_CONFIGS[name]  # type: LearnerParams

            learner_val = learner_params.get_attr(param.value)
            if learner_val is not None:
                p_name = param.name
                logging.debug(f"{orig_learner_name} Parameter {p_name}: Loaded {learner_val:.1E}")
                return learner_val
            break
        except KeyError:
            split_str = name.split("_")
            if split_str[0] == name:
                break
            # name = split_str[0]
            name = "_".join(split_str[:-1])

    # Get the general key value
    key = param.name.upper()
    try:
        val = globals()[key]
    except KeyError:
        raise ValueError(f"Unknown general learner parameter \"{key}\"")
    return val


def _parse_centroids(module_dict: dict, raw_centroids: List):
    r""" Parses the centroid list """
    centroids = []
    for cent in raw_centroids:
        centroids.append(Centroid(cent))
    module_dict[CENTROIDS_KEY.upper()] = centroids


def _is_invalid_neg_bias_key(key: str) -> bool:
    r""" Returns \p True if \p key specifies an invalid negative bias """
    return bool(re.match("neg_(train|test)_bias", key.lower()))


def _parse_newsgroups(module_dict: dict) -> None:
    r""" Parse the file information for 20 newsgroups """
    assert DATASET.is_newsgroups(), "Only valid for 20 Newsgroups parsing"

    # Store the original categories
    for ds_name in ("POS_TRAIN", "POS_TEST", "NEG"):
        module_dict[f"{ds_name}_20NEWS"] = copy.deepcopy(module_dict[f"{ds_name}_CLASSES"])

    # Convert category information into standard class information
    docs_dir = NEWSGROUPS_DIR / "text"
    NewsgroupCategory.configure_probs(docs_dir)
    _config_newsgroups_cat(module_dict, "POS_TRAIN_CLASSES", "POS_TRAIN_BIAS")
    _config_newsgroups_cat(module_dict, "POS_TEST_CLASSES", "POS_TEST_BIAS")

    # Store a copy of neg_classes as it gets set inside configure category function
    tmp_neg_classes = copy.deepcopy(NEG_CLASSES)
    _config_newsgroups_cat(module_dict, "NEG_CLASSES", "NEG_TRAIN_BIAS")

    module_dict["NEG_CLASSES"] = tmp_neg_classes
    _config_newsgroups_cat(module_dict, "NEG_CLASSES", "NEG_TEST_BIAS")


def _config_newsgroups_cat(module_dict: dict, cat_lst_name: str, bias_lst_name: str) -> None:
    r"""
    Converts 20 Newsgroups category information into class specific information

    :param module_dict: Dictionary containing reference to all objects in this module
    :param cat_lst_name: Name of the module variable containing the category information
    :param bias_lst_name: Name of the module variable containing the associated bias information
    """
    # Get the module level variables that will be updated
    cat_list = module_dict[cat_lst_name]
    bias_list = module_dict[bias_lst_name]

    # Sanity check the variables
    if not all(isinstance(cls, str) for cls in cat_list):
        raise ValueError("20 newsgroups only supports categories not classes")
    if len(cat_list) != len(set(cat_list)):
        raise ValueError(f"Duplicate category names in {cat_lst_name}")
    assert len(cat_list) == len(set(cat_list)), "Duplicate category names"
    _check_bias_vectors(cat_list, bias_list)

    # Convert classes to categories
    try:
        cat_list = [NewsgroupCategory[cls.upper()] for cls in cat_list]
    except KeyError:
        raise ValueError(f"Unknown 20 Newsgroups class in {cat_list}")

    # If bias not specified, use true class probabilities
    if bias_list is None:
        bias_list = [None] * len(cat_list)

    # Convert category probabilities to individual class probabilities
    new_cls, new_bias = [], []
    for cat, cat_prob in zip(cat_list, bias_list):
        items = cat.value.get_id_probs(cat_prob)
        new_cls.extend(items.keys())
        new_bias.extend(items.values())
    # Ensure total bias probability is normalized 1
    tot_bias_prob = sum(new_bias)
    new_bias = [bias / tot_bias_prob for bias in new_bias]

    assert len(new_cls) == len(new_bias), "Mismatch in length of new arrays"
    assert abs(sum(new_bias) - 1) < 1E-4, "Bias does not sum to close to 1"

    # Update the module variables with the new values
    module_dict[cat_lst_name] = new_cls
    module_dict[bias_lst_name] = new_bias


def _verify_configuration(module_dict: dict):
    r""" Sanity checks the configuration """
    if DATASET is None: raise ValueError("A dataset must be specified")
    if KERNEL_TYPE is None: raise ValueError("A kernel must be specified")

    if VALIDATION_SPLIT_RATIO <= 0 or VALIDATION_SPLIT_RATIO >= 1:
        raise ValueError("Validation split ratio must be in range (0,1)")

    if LEARNING_RATE <= 0: raise ValueError("Learning rate must be positive")
    if GAMMA <= 0: raise ValueError("Gamma must be positive")
    if NUM_EPOCH <= 0: raise ValueError("Number of training epochs must be positive")
    if WEIGHT_DECAY < 0: raise ValueError("Weight decay must be non-negative")

    if N_P < 0:
        raise ValueError("Positive labeled set size must be positive")

    # Verify unlabeled sizes
    for name in ("N_U_TRAIN", "N_U_TEST", "N_TEST"):
        raw_u_size = module_dict[name]
        if raw_u_size <= 0: raise ValueError(f"{raw_u_size} set size must be positive")
        if name != "N_TEST":
            u_size = (1 - VALIDATION_SPLIT_RATIO) * raw_u_size
            if u_size < 0: raise ValueError(f"{name}'s unlabeled set size must be positive")
            v_size = VALIDATION_SPLIT_RATIO * raw_u_size
            if v_size < 0: raise ValueError(f"{name}'s validation set size must be positive")

    _check_batch_sizes()

    bias_vec = [POS_TRAIN_BIAS, POS_TEST_BIAS, NEG_TRAIN_BIAS, NEG_TEST_BIAS]
    cls_grps = [POS_TRAIN_CLASSES, POS_TEST_CLASSES, NEG_CLASSES, NEG_CLASSES]

    if DATASET.is_synthetic():
        if NUM_FF_LAYERS is not None: raise ValueError("Synthetic data does not use a FF")
        if NUM_SIGMA_LAYERS is not None: raise ValueError("Synthetic data does not use a Ssigma FF")

        if CENTROIDS is None: raise ValueError("Centroids must be specified for SYNTHETIC dataset")
        assert all(v is None for v in cls_grps), "Class groups must be empty for SYNTHETIC dataset"
        assert all(bias is None for bias in bias_vec), "Bias vector not supported for SYNTHETIC"
    else:
        # noinspection PyTypeChecker
        if NUM_SIGMA_LAYERS is None or NUM_SIGMA_LAYERS < 0:
            raise ValueError("Number of sigma learner layers must be non-negative")
        # noinspection PyTypeChecker
        if NUM_FF_LAYERS is None or NUM_FF_LAYERS < 0:
            raise ValueError("Number of FF layers must be non-negative")

        if CENTROIDS is not None:
            raise ValueError("Centroids only valid for SYNTHETIC dataset")
        if any(v is None for v in cls_grps):
            raise ValueError("One class ID group is not defined")
        if any(not v for v in cls_grps):
            raise ValueError("One class ID group is empty")

        # Verify the bias information versus class lists
        for items, bias in zip(cls_grps, bias_vec):
            # noinspection PyTypeChecker
            _check_bias_vectors(items, bias)

        # noinspection PyTypeChecker
        if set(POS_TRAIN_CLASSES).intersection(set(NEG_CLASSES)):
            raise ValueError("Pos train overlap")
        # noinspection PyTypeChecker
        if set(POS_TEST_CLASSES).intersection(set(NEG_CLASSES)):
            raise ValueError("Pos test overlap")

    if DATASET.is_openml():
        # noinspection PyTypeChecker
        if set(POS_TRAIN_CLASSES).difference(set(POS_TEST_CLASSES)):
            raise ValueError("Positive training and set sets must be identical for OpenML datasets")
        if not all(bias is None for bias in bias_vec):
            raise ValueError("All biases must be None for OpenML datasets")


def _check_batch_sizes() -> None:
    r""" Verifies the batch sizes are valid """
    # Verify batch size versus dataset sizes
    cal_ratio = 0 if CALIBRATION_SPLIT_RATIO is None else CALIBRATION_SPLIT_RATIO
    additional_offset = 5  # Correct for torch doing weird things to prevent single element batches
    cal_tr_size = int((1 - VALIDATION_SPLIT_RATIO - cal_ratio) * (N_P + N_U_TRAIN))
    # noinspection PyTypeChecker
    if cal_tr_size - additional_offset < SIGMA_BATCH_SIZE:
        raise ValueError(f"Sigma learner training set size ({cal_tr_size}) too small "
                         f"for sigma batch size ({SIGMA_BATCH_SIZE})")
    tr_size = cal_tr_size + int((1 - VALIDATION_SPLIT_RATIO - cal_ratio) + N_U_TEST)
    if tr_size - additional_offset < BATCH_SIZE:
        raise ValueError(f"Training set size ({tr_size}) smaller than batch size ({BATCH_SIZE})")

    # noinspection PyTypeChecker
    if N_U_TRAIN - additional_offset < PN_TRAIN_BATCH_SIZE:
        raise ValueError(f"Unlabeled training set size ({N_U_TRAIN}) smaller than "
                         f"PN train batch size ({PN_TRAIN_BATCH_SIZE})")

    # noinspection PyTypeChecker
    if N_U_TEST - additional_offset < PN_TEST_BATCH_SIZE:
        raise ValueError(f"Unlabeled test set size ({N_U_TEST}) smaller than "
                         f"PN test batch size ({PN_TEST_BATCH_SIZE})")


def _check_bias_vectors(items: Collection, bias: Optional[List[float]]):
    r""" Standardizes the bias vector checks to prevent code duplication """
    if bias is None: return
    if all(not isinstance(x, float) for x in bias):
        raise ValueError("All bias vector elements must be non empty")
    if abs(sum(bias) - 1) > 1E-4:
        raise ValueError(f"Bias vector probability does not sum to 1")
    if len(items) != len(bias):
        raise ValueError(f"Items and bias vectors lengths do not match")


def print_configuration(log: Callable = logging.info) -> None:
    r""" Print the configuration settings """
    log(f"Dataset: {DATASET.name}")
    if not DATASET.is_synthetic():
        flds = [("Pos Train", POS_TRAIN_CLASSES, POS_TRAIN_BIAS),
                ("Pos Test", POS_TEST_CLASSES, POS_TEST_BIAS)]
        if NEG_BIAS == NEG_TRAIN_BIAS == NEG_TEST_BIAS:
            flds.append(("Negative Classes", NEG_CLASSES, NEG_BIAS))
        else:
            for setup_name, bias in (("Train", NEG_TRAIN_BIAS), ("Test", NEG_TEST_BIAS)):
                flds.append((f"Neg {setup_name} Classes", NEG_CLASSES, bias))

        for name, cls, bias in flds:
            # noinspection PyTypeChecker
            log(f"{name} Classes: %s" % ", ".join(str(x) for x in sorted(cls)))
            if bias is None:
                log(f"{name} BIAS: NONE")
            else:
                # noinspection PyTypeChecker
                log(f"{name} BIAS: %s" % ", ".join(f"{float(x):.3}" for x in sorted(bias)))
    log(f"Train Prior: {TRAIN_PRIOR:.3}")
    log(f"Test Prior: {TEST_PRIOR:.3}")

    log(f"Positive (labeled) Training Set Size: {N_P:,}")
    log(f"Unlabeled Training Set Size: {N_U_TRAIN:,}")
    log(f"Unlabeled Test Set Size: {N_U_TEST:,}")
    log(f"Inductive Test Set Size: {N_TEST:,}")

    log(f"# Sigma Layers: {NUM_SIGMA_LAYERS}")
    log(f"# FF Layers: {NUM_FF_LAYERS}")

    log(f"# Epoch: {NUM_EPOCH}")
    log(f"Sigma Batch Size: {SIGMA_BATCH_SIZE}")
    log(f"Batch Size: {BATCH_SIZE}")
    log(f"PN Train Batch Size: {PN_TRAIN_BATCH_SIZE}")
    log(f"PN Test Batch Size: {PN_TEST_BATCH_SIZE}")
    log(f"Learning Rate: {LEARNING_RATE:.0E}")
    log(f"Weight Decay: {WEIGHT_DECAY:.0E}")


def reset_biases() -> None:
    r""" DEBUG ONLY.  Use to reset bias in hyperparameter tests """
    global NEG_BIAS, NEG_TEST_BIAS, NEG_TRAIN_BIAS
    global POS_TRAIN_BIAS, POS_TEST_BIAS

    NEG_BIAS = NEG_TEST_BIAS = NEG_TRAIN_BIAS = None
    POS_TRAIN_BIAS = POS_TEST_BIAS = None


def reset_learner_settings() -> None:
    r""" DEBUG ONLY.  Reset the settings specific to individual learners/loss functions """
    global LEARNER_CONFIGS
    LEARNER_CONFIGS = dict()


def set_layer_counts(ff_layers: Optional[int] = None, sigma_layers: Optional[int] = None) -> None:
    r""" Set the number of learner layers """
    assert ff_layers is not None or sigma_layers is not None, "Must set at least one layer count"

    if ff_layers is not None:
        global NUM_FF_LAYERS
        NUM_FF_LAYERS = ff_layers

    if sigma_layers is not None:
        global NUM_SIGMA_LAYERS
        NUM_SIGMA_LAYERS = sigma_layers


def set_pos_train_bias(new_bias: List[float]) -> None:
    r""" Manually configures the positive training bias """
    # noinspection PyTypeChecker
    _verify_new_bias(POS_TRAIN_CLASSES, new_bias)

    global POS_TRAIN_BIAS
    POS_TRAIN_BIAS = new_bias


def set_neg_train_bias(new_bias: List[float]) -> None:
    r""" Manually configures the negative set bias """
    # noinspection PyTypeChecker
    _verify_new_bias(NEG_CLASSES, new_bias)

    if DATASET is None:
        raise ValueError(f"Dataset appears to be unset")
    if DATASET.is_openml() or DATASET.is_libsvm():
        raise ValueError(f"Dataset \"{DATASET.name}\" does not support negative train bias")

    global NEG_TRAIN_BIAS
    NEG_TRAIN_BIAS = new_bias


def set_weight_decay(weight_decay: float) -> None:
    r""" Set the DEFAULT weight decay value """
    global WEIGHT_DECAY
    WEIGHT_DECAY = weight_decay


def set_learning_rate(learning_rate: float) -> None:
    r""" Set the DEFAULT learning rate value """
    global LEARNING_RATE
    LEARNING_RATE = learning_rate


def set_gamma(gamma: float) -> None:
    r""" Set the DEFAULT gamma value """
    global GAMMA
    GAMMA = gamma


def set_priors(tr_prior: float, te_prior: float):
    for prior, name in ((tr_prior, "train"), (te_prior, "test")):
        assert 0 < prior < 1, f"Invalid {name} prior"

    global TRAIN_PRIOR
    TRAIN_PRIOR = tr_prior
    global TEST_PRIOR
    TEST_PRIOR = te_prior


def _verify_new_bias(cls_list: List[Union[int, str]], new_bias: List[float]):
    r"""
    Verifies basic parameters for the new bias versus the corresponding class list.
    :param cls_list: List of elements in the class
    :param new_bias: New bias values
    """
    if cls_list is None:
        raise ValueError("Class list is None")
    if DATASET.is_synthetic():
        raise ValueError("Updating centroid bias not currently supported")

    # noinspection PyTypeChecker
    if len(cls_list) != len(new_bias):
        raise ValueError("Mismatch between length of new bias and the positive class list")
    if not all(x >= 0 for x in new_bias):
        raise ValueError("Bias probabilities must be non-negative")
    if abs(1 - sum(new_bias)) > 1E-4:
        raise ValueError("New bias does not sum to 1")
