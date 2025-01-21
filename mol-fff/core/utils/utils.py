import os
from importlib import import_module
from contextlib import contextmanager

import logging
from typing import Tuple, Type

import torch
from torch import nn


def import_from_string(model_name: str) -> Type[nn.Module]:
    """Imports a module from a string. E.g. "torch.nn.Linear" -> torch.nn.Linear"""
    module_name, model_name = model_name.rsplit(".", 1)
    module = import_module(module_name)
    return getattr(module, model_name)


def add_parameter(
        model: nn.Module,
        name: str,
        shape: Tuple[int],
        initialization: str = "torch.nn.init.xavier_uniform_",
) -> None:
    """Initializes a parameter of a given shape and initialization and registers
    it under a given name in the model."""
    param = nn.Parameter(torch.empty(*shape))
    initialization = import_from_string(initialization)
    initialization(param)
    model.register_parameter(name, param)


def get_latest_run(
        model_type: Type[nn.Module], checkpoint_path: str, **kwargs
) -> Tuple[nn.Module, int]:
    """Returns the latest run from the checkpoint path
    :param checkpoint_path: the path to the checkpoint
    :return: the model and the step of the latest run"""

    latest = 0
    latest_name = None
    for file in os.listdir(checkpoint_path):
        if file.endswith(".ckpt"):
            ckpt_name = file.split(".")[0]
            try:
                ckpt_step = int(ckpt_name.split("step=")[-1])
            except ValueError:
                continue
            if ckpt_step > latest:
                latest = ckpt_step
                latest_name = ckpt_name
    model = model_type.load_from_checkpoint(
        os.path.join(checkpoint_path, latest_name + ".ckpt"), **kwargs
    )
    return model, latest
