import torch
import numpy as np
from torch.utils.data import IterableDataset

from monai.utils import NumpyPadMode
from monai.transforms import LoadImage
from monai.data.utils import iter_patch

from typing import Any, Callable, Hashable, Optional, Sequence, Tuple, Union

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
            
            #arrays = arrays / 30000.0
            #arrays = (np.log(1 + arrays) - 5.5)/5.5
            
            # Get mag level of file
            mag_level = get_mag_level(img_paths[0])
            
            # Preprocessing - 1,1,10,256,256
            arrays[0,0,7,:,:] = preprocess(arrays[0,0,7,:,:], mag_level, "C01")
            arrays[0,0,8,:,:] = preprocess(arrays[0,0,8,:,:], mag_level, "C02")
            arrays[0,0,9,:,:] = preprocess(arrays[0,0,9,:,:], mag_level, "C03")
            arrays[0,0,:7,:,:] = preprocess(arrays[0,0,:7,:,:], mag_level, "C04")

            iters = [iter_patch(a, self.patch_size, self.start_pos, False, self.mode) for a in arrays]

            yield from zip(*iters)