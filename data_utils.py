import torch
from torch.utils.data import IterableDataset, Dataset as _TorchDataset
from monai.transforms import Compose, Randomizable, apply_transform, LoadImage, RandSpatialCropSamples
from monai.utils import NumpyPadMode
from monai.data.utils import iter_patch
import numpy as np

from typing import Any, Callable, Hashable, Optional, Sequence, Tuple, Union

import math


def find_first_occurance(tuple_list, target_str):
    for i, t in enumerate(tuple_list):
        if target_str in t[0]:
            return i
        
        
def split_train_val(data_list, N_valid_per_magn=1):
    indexes = [
        find_first_occurance(data_list, mag_lev)
        for mag_lev in ["20x", "40x", "60x"]
    ]
    indexes = [i for initial in indexes for i in range(initial, initial+N_valid_per_magn)]
    train_split = [data_list[i] for i in range(len(data_list)) if i not in indexes]
    val_split = [data_list[i] for i in indexes]
    return train_split, val_split


class MozartTheComposer(Compose):
    def __call__(self, input_):
        # read images
#         vol = np.stack([self.transforms[0](x) for x in input_], axis=0)
        # apply magical mapping
        vol = (np.log(1 + input_) - 5.5)/5.5
        # apply transforms
        for t in self.transforms:
            vol = t(vol)
        return vol
    

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
        
        patches = self.sampler(images)
        
        if len(patches) != self.samples_per_image:
            raise RuntimeWarning(
                f"`patch_func` must return a sequence of length: samples_per_image={self.samples_per_image}.")

        patch_id = (index - image_id * self.samples_per_image) * (-1 if index < 0 else 1)
        patch = patches[patch_id]
        if self.transform is not None:
            patch = apply_transform(self.transform, patch, map_items=False)
        return patch
    

class OurGridyDataset(IterableDataset):
    
    def __init__(self, 
                 data: Sequence, 
                 patch_size: int,
                 data_reader: Callable):
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.
        """
        self.patch_size = (None,) + (10, patch_size, patch_size)
        self.start_pos = ()
        self.mode = NumpyPadMode.WRAP
        
        self.data = data
        self.image_reader = LoadImage(data_reader, image_only=True)

    def __len__(self) -> int:
        return len(self.data)
    
    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        iter_start = 0
        iter_end = len(self.data)

        if worker_info is not None:
            # split workload
            per_worker = int(math.ceil((iter_end - iter_start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = iter_start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, iter_end)

        for index in range(iter_start, iter_end):
            img_paths = self.data[index]
            arrays = np.expand_dims(np.stack([self.image_reader(x) for x in img_paths]), axis=(0,1))
            arrays = (np.log(1 + arrays) - 5.5)/5.5

            iters = [iter_patch(a, self.patch_size, self.start_pos, False, self.mode) for a in arrays]

            yield from zip(*iters)
            
            
class OverlappyGridyDataset(IterableDataset):
    
    def __init__(self, 
                 data: Sequence, 
                 patch_size: int,
                 overlap_ratio: float,
                 data_reader: Callable):
        """
        Args:
            data: input data to load and transform to generate dataset for model.
            transform: a callable data transform on input data.
        """
        self.patch_size = patch_size
        self.overlap_ratio = overlap_ratio
        self.overlap_pix = int(overlap_ratio*patch_size)
        self.nonoverlap_pix = int((1-overlap_ratio)*patch_size)
        
        self.start_pos = ()
        self.mode = NumpyPadMode.WRAP
        
        self.data = data
        self.image_reader = LoadImage(data_reader, image_only=True)
        
        self.img = np.expand_dims(np.stack([self.image_reader(x) for x in self.data]), axis=0)
        self.img = (np.log(1 + self.img) - 5.5)/5.5
        
        self.img_h, self.img_w = self.img.shape[-2:]
        self.num_grids_h = math.ceil(self.img_h/self.nonoverlap_pix)
        self.num_grids_w = math.ceil(self.img_w/self.nonoverlap_pix)

    def __len__(self) -> int:
        return self.get_num_patches()
    
    def get_num_patches(self):
        return self.num_grids_h*self.num_grids_w
    
    def merge_patches(self, patches):
        
        num_pred_matrix = np.zeros(self.img.shape[-2:])
        img_merged = np.zeros(self.img.shape[-2:])
        
        i = 0
        for h in range(self.num_grids_h):
            for w in range(self.num_grids_w):
                slice_h_start = 0
                slice_w_start = 0

                if h == (self.num_grids_h-1) and w == (self.num_grids_w-1):
                    slice_h_start = self.img_h
                    slice_w_start = self.img_w
                elif h == (self.num_grids_h-1):
                    slice_h_end = self.img_h
                    slice_w_end = min(self.nonoverlap_pix*w + self.patch_size, self.img_w)
                elif w == (self.num_grids_w-1):
                    slice_h_end = min(self.nonoverlap_pix*h + self.patch_size, self.img_h)
                    slice_w_end = self.img_w
                else:
                    slice_h_end = min(self.nonoverlap_pix*h + self.patch_size, self.img_h)
                    slice_w_end = min(self.nonoverlap_pix*w + self.patch_size, self.img_w)

                slice_h_start = slice_h_end - self.patch_size
                slice_w_start = slice_w_end - self.patch_size

                img_merged[slice_h_start: slice_h_end, slice_w_start: slice_w_end] = img_merged[slice_h_start: slice_h_end, slice_w_start: slice_w_end] + patches[i].to('cpu').numpy()
                num_pred_matrix[slice_h_start: slice_h_end, slice_w_start: slice_w_end] = num_pred_matrix[slice_h_start: slice_h_end, slice_w_start: slice_w_end] + 1.0
                i += 1

        img_merged = img_merged / num_pred_matrix
        return img_merged
    
    def __iter__(self):

        for h in range(self.num_grids_h):
            for w in range(self.num_grids_w):
                slice_h_start = 0
                slice_w_start = 0

                if h == (self.num_grids_h-1) and w == (self.num_grids_w-1):
                    slice_h_start = self.img_h
                    slice_w_start = self.img_w
                elif h == (self.num_grids_h-1):
                    slice_h_end = self.img_h
                    slice_w_end = min(self.nonoverlap_pix*w + self.patch_size, self.img_w)
                elif w == (self.num_grids_w-1):
                    slice_h_end = min(self.nonoverlap_pix*h + self.patch_size, self.img_h)
                    slice_w_end = self.img_w
                else:
                    slice_h_end = min(self.nonoverlap_pix*h + self.patch_size, self.img_h)
                    slice_w_end = min(self.nonoverlap_pix*w + self.patch_size, self.img_w)

                slice_h_start = slice_h_end - self.patch_size
                slice_w_start = slice_w_end - self.patch_size

                img_patch = self.img[:, :, slice_h_start: slice_h_end, slice_w_start: slice_w_end]
                yield img_patch
                    