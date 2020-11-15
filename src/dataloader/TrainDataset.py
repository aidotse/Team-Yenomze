import numpy as np
from torch.utils.data import Dataset as _TorchDataset
from monai.transforms import apply_transform, LoadImage, RandSpatialCropSamples

from typing import Any, Callable, Hashable, Optional, Sequence, Tuple, Union

from src.util.DataUtils import get_mag_level, preprocess

class OurDataset(_TorchDataset):
    
    def __init__(self, 
                 data: Sequence, 
                 samples_per_image: int,
                 roi_size: int,
                 data_reader: Callable,
                 transform: Optional[Callable] = None) -> None:
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.
        """
        self.samples_per_image = samples_per_image
        self.roi_size = (10, roi_size, roi_size)
        self.data = data
        self.image_reader = LoadImage(data_reader, image_only=True)
        self.transform = transform
        self.sampler = RandSpatialCropSamples(roi_size=self.roi_size, 
                                              num_samples=self.samples_per_image, 
                                              random_center=True, 
                                              random_size=False)

    def __len__(self) -> int:
        return len(self.data) * self.samples_per_image

    def __getitem__(self, index: int):
        image_id = int(index / self.samples_per_image)
        image_paths = self.data[image_id]        
        images = np.expand_dims(np.stack([self.image_reader(x) for x in image_paths]), axis=0)
        
        # Get mag level of file
        mag_level = get_mag_level(image_paths[0])
        
        patches = self.sampler(images)
        
        if len(patches) != self.samples_per_image:
            raise RuntimeWarning(
                f"`patch_func` must return a sequence of length: samples_per_image={self.samples_per_image}.")

        patch_id = (index - image_id * self.samples_per_image) * (-1 if index < 0 else 1)
        patch = patches[patch_id]
        if self.transform is not None:
            # Preprocessing - 1,10,256,256
            patch[0,7,:,:] = preprocess(patch[0,7,:,:], mag_level, "C01")
            patch[0,8,:,:] = preprocess(patch[0,8,:,:], mag_level, "C02")
            patch[0,9,:,:] = preprocess(patch[0,9,:,:], mag_level, "C03")
            patch[0,:7,:,:] = preprocess(patch[0,:7,:,:], mag_level, "C04")
            
            patch = apply_transform(self.transform, patch, map_items=False)
        return patch