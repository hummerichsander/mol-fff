from typing import Callable

import torch
from numpy import ndarray
from sklearn.manifold import TSNE


@torch.no_grad()
def project_embedding(
    embedding_fn: Callable, *args, projection_dim: int = 2, **kwargs
) -> ndarray:
    """
    Projects the output of an embedding function to a lower-dimensional space.

    :param embedding_fn: Function returning a tensor of shape (..., projection_dim).
    :param projection_dim: Dimension of the projection space.
    :return: A tensor of shape (..., projection_dim)."""

    embedding = embedding_fn(*args)
    if embedding.device.type == "cuda":
        embedding = embedding.cpu()
    return TSNE(n_components=projection_dim, **kwargs).fit_transform(embedding)
