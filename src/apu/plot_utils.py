__all__ = ["plot_centroids",
           "plot_histogram",
           "plot_scatter"]

try:
    # noinspection PyUnresolvedReferences
    from matplotlib import use

    use('Agg')
    # noinspection PyUnresolvedReferences
    import matplotlib.pyplot as plt
    # noinspection PyUnresolvedReferences
    import matplotlib.ticker
except ImportError:
    # raise ImportError("Unable to import matplotlib")
    pass


import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns

from torch import Tensor
import torch

from .datasets.types import TensorGroup
from .types import TorchOrNp, PathOrStr, OptStr


def plot_scatter(ax, x: TorchOrNp, y: TorchOrNp, title: str,
                 # legend: Optional[List[str]] = None,
                 xlabel: str = "", ylabel: str = "",
                 log_x: bool = False, log_y: bool = False,
                 xmin: Optional[float] = None, xmax: Optional[float] = None,
                 ymin: Optional[float] = None, ymax: Optional[float] = None):
    # Automatically convert to numpy
    if isinstance(x, Tensor): x = x.numpy()
    if isinstance(y, Tensor): y = y.numpy()

    ax.scatter(x, y)
    # if legend is not None:
    #     ax.legend(legend)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title)
    if log_x:
        ax.set_xscale("log")
    if log_y:
        ax.set_yscale("log")
    # Set the minimum and maximum values if specified
    ax.set_xlim(xmin=xmin, xmax=xmax)
    ax.set_ylim(ymin=ymin, ymax=ymax)


def plot_histogram(ax, vals: TorchOrNp, n_bins: int, title: str,
                   hist_range: Optional[Tuple[float, float]] = None, xlabel: str = ""):
    if isinstance(vals, Tensor): vals = vals.numpy()

    n_ele = vals.shape[0]
    # N is the count in each bin, bins is the lower-limit of the bin
    _, bins, patches = ax.hist(vals, bins=n_bins, range=hist_range, weights=np.ones(n_ele) / n_ele)

    # Now we format the y-axis to display percentage
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))

    ax.set_title(title)
    if xlabel:
        ax.set_xlabel(xlabel)


def plot_centroids(filename: PathOrStr, ts_grp: TensorGroup, title: OptStr = None,
                   decision_boundary: Optional[Tuple[float, float]] = None) -> None:
    filename = Path(filename)
    msg = f"Generating centroid plot to path \"{str(filename)}\""
    logging.debug(f"Starting: {msg}")

    # Combine all the data into one tensor
    x, y = [ts_grp.p_x], [torch.zeros([ts_grp.p_x.shape[0]], dtype=ts_grp.u_tr_y[1].dtype)]

    flds = [(1, (ts_grp.u_tr_x, ts_grp.u_tr_y)), (3, (ts_grp.u_te_x, ts_grp.u_te_y))]
    for y_offset, (xs, ys) in flds:
        x.append(xs)
        y.append(ys.clamp_min(0) + y_offset)
    x, y = torch.cat(x, dim=0), torch.cat(y, dim=0)

    def _build_labels(lbl: int) -> str:
        if lbl == 0: return "Pos"
        sgn = "+" if lbl % 2 == 0 else "-"
        ds = "U-te" if (lbl - 1) // 2 == 1 else "U-tr"
        return f"{ds} {sgn}"

    y_str = [_build_labels(_y) for _y in y.cpu().numpy()]
    # Residual code from t-SNE plotting.  Just list class counts
    unique, counts = torch.unique(y.squeeze(), return_counts=True)
    for uniq, cnt in zip(unique.squeeze(), counts.squeeze()):
        logging.debug("Centroids %s: %d elements", _build_labels(uniq.item()), int(cnt.item()))

    label_col_name = "Label"
    data = {'x1': x[:, 0].squeeze(), 'x2': x[:, 1].squeeze(), label_col_name: y_str}
    flatui = ["#0d14e0", "#4bf05b", "#09ac15", "#e6204e", "#ba141d"]
    markers = ["P", "s", "D", "X", "^"]
    width, height = 8, 8
    plt.figure(figsize=(8, 8))
    ax = sns.lmplot(x="x1", y="x2",
                    hue="Label",
                    hue_order=["Pos", "U-tr +", "U-te +", "U-tr -", "U-te -"],
                    # palette=sns.color_palette("bright", unique.shape[0]),
                    palette=sns.color_palette(flatui),
                    data=pd.DataFrame(data),
                    # legend="full",
                    # s=20,
                    # alpha=0.4,
                    markers=markers,
                    height=height,
                    aspect=width / height,
                    legend=False,
                    fit_reg=False,
                    scatter_kws={"s": 20, "alpha": 0.4}
                    )

    ax.set(xlabel='', ylabel='')
    if title is not None: ax.set(title=title)
    plt.legend(title='')

    if decision_boundary is not None:
        _add_line_from_slope_and_intercept(*decision_boundary)

    xmin = ymin = np.inf
    xmax = ymax = -np.inf
    for tensor in (ts_grp.p_x, ts_grp.u_tr_x, ts_grp.u_te_x, ts_grp.test_x):
        xmin, xmax = min(xmin, float(tensor[:, 0].min())), max(xmax, float(tensor[:, 0].max()))
        ymin, ymax = min(ymin, float(tensor[:, 1].min())), max(ymax, float(tensor[:, 1].max()))
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)

    filename.parent.mkdir(exist_ok=True, parents=True)
    plt.savefig(str(filename))
    plt.close('all')

    # Export a csv of the raw t-SNE data
    csv_path = filename.with_suffix(".csv")
    data["y"] = y.squeeze().cpu().numpy()
    df = pd.DataFrame(data)
    df.drop(columns=label_col_name, inplace=True)  # Label column not used by LaTeX
    df.to_csv(str(csv_path), index=False, encoding="utf-8", float_format='%.3f')

    logging.debug(f"COMPLETED: {msg}")


def _add_line_from_slope_and_intercept(slope, intercept):
    r"""
    Assuming an existing \p matplotlib object is active, this function adds a line in the
    form :math:`y = m x + b`.

    :param slope: :math:`m` plot slope
    :param intercept: :math:`b` -- y-intercept
    """
    r"""Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color="black")
