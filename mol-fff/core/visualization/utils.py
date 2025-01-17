from typing import Iterable, Optional
from matplotlib.figure import Figure

import numpy as np
import torch
from numpy import ndarray
import matplotlib.pyplot as plt

import concurrent.futures


def figure_to_array(fig: Figure) -> ndarray:
    """Converts a matplotlib figure to a numpy array"""
    fig.canvas.draw()
    width, height = fig.canvas.get_width_height()
    pixel_array = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    return pixel_array.reshape((height, width, 4))


def make_grid(range: list[float], n: int):
    grid_range = torch.linspace(range[0], range[1], n)
    return torch.stack(torch.meshgrid(grid_range, grid_range), dim=-1).view(-1, 2)


def plot_grid(
    plot_fn: callable,
    *iterables: Iterable,
    ncols: int = 4,
    box_size: tuple[int] = (3.3, 2.5),
    max_workers: Optional[int] = None,
    **kwargs
) -> Figure:
    """Plots a grid of input_iterable objects using matplotlib.pyplot.
    :param plot_fn: function to plot each input object. Must accept an ax argument.
    :param input_iterable: list of input objects to be plotted.
    :param ncols: number of columns in the grid.
    :param box_size: size of each box in the grid.
    :param args: additional arguments to be passed to the plot function.
    :param kwargs: additional keyword arguments to be passed to the plot function.
    :return: matplotlib figure."""
    rows = int(np.ceil(len(iterables[0]) / ncols))
    fig = plt.figure(figsize=(box_size[0] * ncols, box_size[1] * rows))

    def plot(i, *inputs):
        ax = fig.add_subplot(rows, ncols, i + 1)
        plot_fn(*inputs, **kwargs, ax=ax)

    with concurrent.futures.ThreadPoolExecutor(max_workers) as executor:
        executor.map(
            plot,
            range(len(iterables[0])),
            *iterables,
        )

    return fig
