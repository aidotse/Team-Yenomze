from torch.utils.data import Dataset as _TorchDataset
from monai.data import Dataset, ZipDataset
from monai.transforms import Randomizable
from monai.utils import get_seed
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Union
import numpy as np

MAX_SEED = np.iinfo(np.uint32).max + 1


class FlowArrayDataset(Randomizable, _TorchDataset):
    """
    Dataset for segmentation and classification tasks based on array format input data and transforms.
    It ensures the same random seeds in the randomized transforms defined for image, segmentation and label.
    The `transform` can be :py:class:`monai.transforms.Compose` or any other callable object.
    For example:
    If train based on Nifti format images without metadata, all transforms can be composed::

        img_transform = Compose(
            [
                LoadNifti(image_only=True),
                AddChannel(),
                RandAdjustContrast()
            ]
        )
        FlowArrayDataset(img_file_list, img_transform=img_transform)

    If training based on images and the metadata, the array transforms can not be composed
    because several transforms receives multiple parameters or return multiple values. Then Users need
    to define their own callable method to parse metadata from `LoadNifti` or set `affine` matrix
    to `Spacing` transform::

        class TestCompose(Compose):
            def __call__(self, input_):
                img, metadata = self.transforms[0](input_)
                img = self.transforms[1](img)
                img, _, _ = self.transforms[2](img, metadata["affine"])
                return self.transforms[3](img), metadata
        img_transform = TestCompose(
            [
                LoadNifti(image_only=False),
                AddChannel(),
                Spacing(pixdim=(1.5, 1.5, 3.0)),
                RandAdjustContrast()
            ]
        )
        FlowArrayDataset(img_file_list, img_transform=img_transform)
    """

    def __init__(
        self,
        inputZ01: Sequence,
        inputZ02: Sequence,
        inputZ03: Sequence,
        inputZ04: Sequence,
        inputZ05: Sequence,
        inputZ06: Sequence,
        inputZ07: Sequence,
        targetC01: Sequence,
        targetC02: Sequence,
        targetC03: Sequence,
        inputZ01_transform: Optional[Callable] = None,
        inputZ02_transform: Optional[Callable] = None,
        inputZ03_transform: Optional[Callable] = None,
        inputZ04_transform: Optional[Callable] = None,
        inputZ05_transform: Optional[Callable] = None,
        inputZ06_transform: Optional[Callable] = None,
        inputZ07_transform: Optional[Callable] = None,
        targetC01_transform: Optional[Callable] = None,
        targetC02_transform: Optional[Callable] = None,
        targetC03_transform: Optional[Callable] = None

    ) -> None:
        """
        Initializes the dataset with the filename lists. The transform `img_transform` is applied
        to the images and `seg_transform` to the segmentations.
        """

        items = [(inputZ01, inputZ01_transform), (inputZ02, inputZ02_transform), (inputZ03, inputZ03_transform),
                 (inputZ04, inputZ04_transform), (inputZ05, inputZ05_transform), (inputZ06, inputZ06_transform),
                 (inputZ07, inputZ07_transform),
                 (targetC01, targetC01_transform), (targetC02, targetC02_transform), (targetC03, targetC03_transform)]
        self.set_random_state(seed=get_seed())
        datasets = [Dataset(x[0], x[1]) for x in items if x[0] is not None]
        self.dataset = datasets[0] if len(datasets) == 1 else ZipDataset(datasets)

        self._seed = 0  # transform synchronization seed

    def __len__(self) -> int:
        return len(self.dataset)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()
        if isinstance(self.dataset, ZipDataset):
            # set transforms of each zip component
            for dataset in self.dataset.data:
                transform = getattr(dataset, "transform", None)
                if isinstance(transform, Randomizable):
                    print("Here")
                    transform.set_random_state(seed=self._seed)
        transform = getattr(self.dataset, "transform", None)
        if isinstance(transform, Randomizable):
            transform.set_random_state(seed=self._seed)
        return self.dataset[index]
    
    
    
class FlowArrayDatasetV2(Randomizable, _TorchDataset):
    """
    Dataset for segmentation and classification tasks based on array format input data and transforms.
    It ensures the same random seeds in the randomized transforms defined for image, segmentation and label.
    The `transform` can be :py:class:`monai.transforms.Compose` or any other callable object.
    For example:
    If train based on Nifti format images without metadata, all transforms can be composed::

        img_transform = Compose(
            [
                LoadNifti(image_only=True),
                AddChannel(),
                RandAdjustContrast()
            ]
        )
        FlowArrayDataset(img_file_list, img_transform=img_transform)

    If training based on images and the metadata, the array transforms can not be composed
    because several transforms receives multiple parameters or return multiple values. Then Users need
    to define their own callable method to parse metadata from `LoadNifti` or set `affine` matrix
    to `Spacing` transform::

        class TestCompose(Compose):
            def __call__(self, input_):
                img, metadata = self.transforms[0](input_)
                img = self.transforms[1](img)
                img, _, _ = self.transforms[2](img, metadata["affine"])
                return self.transforms[3](img), metadata
        img_transform = TestCompose(
            [
                LoadNifti(image_only=False),
                AddChannel(),
                Spacing(pixdim=(1.5, 1.5, 3.0)),
                RandAdjustContrast()
            ]
        )
        FlowArrayDataset(img_file_list, img_transform=img_transform)
    """

    def __init__(
        self,
        inputZ01: Sequence,
        inputZ02: Sequence,
        inputZ03: Sequence,
        inputZ04: Sequence,
        inputZ05: Sequence,
        inputZ06: Sequence,
        inputZ07: Sequence,
        targetC01: Sequence,
        targetC02: Sequence,
        targetC03: Sequence,
        inputZ01_transform: Optional[Callable] = None,
        inputZ02_transform: Optional[Callable] = None,
        inputZ03_transform: Optional[Callable] = None,
        inputZ04_transform: Optional[Callable] = None,
        inputZ05_transform: Optional[Callable] = None,
        inputZ06_transform: Optional[Callable] = None,
        inputZ07_transform: Optional[Callable] = None,
        targetC01_transform: Optional[Callable] = None,
        targetC02_transform: Optional[Callable] = None,
        targetC03_transform: Optional[Callable] = None

    ) -> None:
        """
        Initializes the dataset with the filename lists. The transform `img_transform` is applied
        to the images and `seg_transform` to the segmentations.
        """

        items = [(inputZ01, inputZ01_transform), (inputZ02, inputZ02_transform), (inputZ03, inputZ03_transform),
                 (inputZ04, inputZ04_transform), (inputZ05, inputZ05_transform), (inputZ06, inputZ06_transform),
                 (inputZ07, inputZ07_transform),
                 (targetC01, targetC01_transform), (targetC02, targetC02_transform), (targetC03, targetC03_transform)]
        self.set_random_state(seed=get_seed())
        datasets = [Dataset(x[0], x[1]) for x in items if x[0] is not None]
        self.dataset = datasets[0] if len(datasets) == 1 else ZipDataset(datasets)

        self._seed = 0  # transform synchronization seed

    def __len__(self) -> int:
        return len(self.dataset)

    def randomize(self, data: Optional[Any] = None) -> None:
        self._seed = self.R.randint(MAX_SEED, dtype="uint32")

    def __getitem__(self, index: int):
        self.randomize()
        if isinstance(self.dataset, ZipDataset):
            # set transforms of each zip component
            for dataset in self.dataset.data:
                transform = getattr(dataset, "transform", None)
                if isinstance(transform, Randomizable):
                    transform.set_random_state(seed=self._seed)
        transform = getattr(self.dataset, "transform", None)
        if isinstance(transform, Randomizable):
            transform.set_random_state(seed=self._seed)
        return self.dataset[index]