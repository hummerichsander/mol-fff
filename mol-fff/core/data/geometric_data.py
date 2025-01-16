from typing import Callable

from abc import ABC

from hydrantic.data import PyTorchData
from hydrantic.data.hparams import DataHparams
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.transforms import Compose

from ..utils.utils import import_from_string


class GeometricDataHparams(DataHparams):
    """A class for storing PyTorch Geometric data hparams."""

    transform: list[tuple[str, dict]] | Callable | None = None
    pre_transform: list[tuple[str, dict]] | Callable | None = None
    pre_filter: str | Callable | None = None
    root: str = "./data"


class GeometricData(PyTorchData, ABC):
    """A class for storing PyTorch Geometric data. It overwrites the _configure_dataloader method to utilize the
    torch_geometric DataLoader."""

    hparams_schema = GeometricDataHparams

    def __init__(self, hparams: GeometricDataHparams):
        if hparams.transform is not None and not isinstance(hparams.transform, Compose):
            hparams.transform = self._get_transform_composition(hparams.transform)

        if hparams.pre_transform is not None and not isinstance(
            hparams.pre_transform, Compose
        ):
            hparams.pre_transform = self._get_transform_composition(
                hparams.pre_transform
            )

        if hparams.pre_filter is not None and isinstance(hparams.pre_filter, str):
            hparams.pre_filter = import_from_string(hparams.pre_filter)

        super().__init__(hparams)

    def _configure_dataloader(self, dataset: Dataset) -> DataLoader:
        """Configures the dataloader from the module name and kwargs specified in the hparams.

        :dataset: The dataset to use for the dataloader.
        :return: The configured dataloader."""

        return DataLoader(
            dataset,
            batch_size=self.hparams.loader.batch_size,
            shuffle=self.hparams.loader.shuffle,
            pin_memory=self.hparams.loader.pin_memory,
            num_workers=self.hparams.loader.num_workers,
            persistent_workers=self.hparams.loader.persistent_workers,
            drop_last=self.hparams.loader.drop_last,
        )

    @staticmethod
    def _get_transform_composition(
        transforms: list[tuple[str, dict]] | Compose
    ) -> Compose:
        """Converts a list of transform strings + kwargs to a transform composition.
        :param transforms: A list of transform strings + kwargs.

        :return: The transform composition."""

        if isinstance(transforms, Callable):
            return transforms

        transform_modules: list[Callable] = []
        for transform, kwargs in transforms:
            transform_modules.append(import_from_string(transform)(**kwargs))
        return Compose(transform_modules)
