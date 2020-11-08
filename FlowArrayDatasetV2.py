from torch.utils.data import Dataset as _TorchDataset
from monai.data import Dataset, ZipDataset
from monai.transforms import Randomizable, apply_transform
from monai.utils import get_seed
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union
import numpy as np

MAX_SEED = np.iinfo(np.uint32).max + 1


class OurDataset(_TorchDataset):
    
    def __init__(self, data: Sequence, transform: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.
        """
        self.data = data
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int):
        data = self.data[index]
        if self.transform is not None:
            data = apply_transform(self.transform, data, map_items=False)

        return data