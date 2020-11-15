import cv2
import math
import numpy as np

from monai.utils import NumpyPadMode
from monai.transforms import LoadImage
from torch.utils.data import IterableDataset

from src.util.DataUtils import get_mag_level, preprocess

from typing import Any, Callable, Hashable, Optional, Sequence, Tuple, Union

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
        self.data = data
        
        # Get mag level of file
        self.mag_level = get_mag_level(self.data[0])
        
        if self.mag_level == "20x":
            self.sample_patch_size = self.patch_size // 2
            self.resize=True
        else:
            self.sample_patch_size = self.patch_size
            self.resize=False
            
        self.overlap_pix = int(overlap_ratio*self.sample_patch_size)
        self.nonoverlap_pix = int((1-overlap_ratio)*self.sample_patch_size)
        
        self.start_pos = ()
        self.mode = NumpyPadMode.WRAP
        
        
        self.image_reader = LoadImage(data_reader, image_only=True)
        
        self.img = np.expand_dims(np.stack([self.image_reader(x) for x in self.data]), axis=0)
        
        #self.img = self.img / 30000.0
        #self.img = (np.log(1 + self.img) - 5.5)/5.5

        # Preprocessing - 1,10,256,256
#         if not is_test:
#             self.img[0][7,:,:] = preprocess(self.img[0,7,:,:], self.mag_level, "C01")
#             self.img[0][8,:,:] = preprocess(self.img[0,8,:,:], self.mag_level, "C02")
#             self.img[0][9,:,:] = preprocess(self.img[0,9,:,:], self.mag_level, "C03")
        
        self.img[0][:7,:,:] = preprocess(self.img[0,:7,:,:], self.mag_level, "C04")
        
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
                if self.resize:
                    patch = cv2.resize(patches[i].numpy(), (self.sample_patch_size, self.sample_patch_size), interpolation = cv2.INTER_CUBIC)
                else:
                    patch = patches[i].numpy()
                    
                slice_h_start = 0
                slice_w_start = 0

                if h == (self.num_grids_h-1) and w == (self.num_grids_w-1):
                    slice_h_start = self.img_h
                    slice_w_start = self.img_w
                elif h == (self.num_grids_h-1):
                    slice_h_end = self.img_h
                    slice_w_end = min(self.nonoverlap_pix*w + self.sample_patch_size, self.img_w)
                elif w == (self.num_grids_w-1):
                    slice_h_end = min(self.nonoverlap_pix*h + self.sample_patch_size, self.img_h)
                    slice_w_end = self.img_w
                else:
                    slice_h_end = min(self.nonoverlap_pix*h + self.sample_patch_size, self.img_h)
                    slice_w_end = min(self.nonoverlap_pix*w + self.sample_patch_size, self.img_w)

                slice_h_start = slice_h_end - self.sample_patch_size
                slice_w_start = slice_w_end - self.sample_patch_size

                img_merged[slice_h_start: slice_h_end, slice_w_start: slice_w_end] = img_merged[slice_h_start: slice_h_end, slice_w_start: slice_w_end] + patch
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
                    slice_w_end = min(self.nonoverlap_pix*w + self.sample_patch_size, self.img_w)
                elif w == (self.num_grids_w-1):
                    slice_h_end = min(self.nonoverlap_pix*h + self.sample_patch_size, self.img_h)
                    slice_w_end = self.img_w
                else:
                    slice_h_end = min(self.nonoverlap_pix*h + self.sample_patch_size, self.img_h)
                    slice_w_end = min(self.nonoverlap_pix*w + self.sample_patch_size, self.img_w)

                slice_h_start = slice_h_end - self.sample_patch_size
                slice_w_start = slice_w_end - self.sample_patch_size

                img_patch = self.img[:, :, slice_h_start: slice_h_end, slice_w_start: slice_w_end]
                
                if self.resize:
                    img_resized = []
                    
                    for i, img_patch_slice in enumerate(img_patch[0]):
                        img_resized.append(cv2.resize(img_patch_slice, (self.patch_size, self.patch_size), interpolation = cv2.INTER_CUBIC))
                    
                    img_patch = np.expand_dims(np.stack(img_resized, axis=0), axis=0)
                    
                yield img_patch
                    