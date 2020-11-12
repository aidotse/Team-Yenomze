import os
import sys
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from Generator import GeneratorUnet
from monai.data import PILReader, DataLoader

from data_utils import postprocess

class TestHandler():
    def __init__(self,
                 patch_iterator: torch.utils.data.IterableDataset,
                 model: GeneratorUnet,
                 output_dir: str="./output",
                ) -> None:
        self.patch_iterator = patch_iterator
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
        self.model = model.to(self.device)
        self.model.eval()
    
    def generate_patch_iterator(self, image_list):
        for img_tuple in image_list:
            fname = img_tuple[0].split("/")[-1]
            img_prefix = fname[:-16]
            dataset = self.patch_iterator(data=img_tuple,
                                          patch_size=256,
                                          overlap_ratio=0.5,
                                          data_reader=PILReader())
            data_loader = DataLoader(dataset,
                                     batch_size=8,
                                     shuffle=False)
            yield img_prefix, data_loader

    def run_test(self, image_list, mag_level):
        merged_images_list = []
        
        with torch.no_grad():
            for  img_prefix, data_loader in tqdm(self.generate_patch_iterator(image_list),
                                                 total=len(image_list),
                                                 file=sys.stdout):
                patchesC01, patchesC02, patchesC03 = [], [], []
                for batch_index, batch in enumerate(data_loader):
                    # unpack the inputs
                    inpZ01, inpZ02, inpZ03, inpZ04, inpZ05, inpZ06, inpZ07 = \
                        batch[:,:,0,:,:].to(self.device), \
                        batch[:,:,1,:,:].to(self.device), \
                        batch[:,:,2,:,:].to(self.device), \
                        batch[:,:,3,:,:].to(self.device), \
                        batch[:,:,4,:,:].to(self.device), \
                        batch[:,:,5,:,:].to(self.device), \
                        batch[:,:,6,:,:].to(self.device)
                    # predict with model
                    outC01, outC02, outC03 = self.model(inpZ01, inpZ02, inpZ03, 
                                                        inpZ04, inpZ05, inpZ06, inpZ07)
                    outC01, outC02, outC03 = [p[0] for p in outC01.data.cpu()], \
                                             [p[0] for p in outC02.data.cpu()], \
                                             [p[0] for p in outC03.data.cpu()]
                    patchesC01.extend(outC01)
                    patchesC02.extend(outC02)
                    patchesC03.extend(outC03)
                # (3,256,256)
                merged_images = []
                channels = ["C01", "C02", "C03"]
                
                for i, patches in enumerate([patchesC01, patchesC02, patchesC03]):
                    #print(len(patches))
                    #print(patches[0].shape)
                    merged_img = data_loader.dataset.merge_patches(patches)
                    merged_img = postprocess(merged_img, mag_level, channels[i])
                    merged_images.append(merged_img)
                    
                    out_dir = os.path.join(self.output_dir)
                    if not os.path.exists(out_dir):
                        os.makedirs(out_dir)
                    self.save_img(merged_img, 
                                  os.path.join(out_dir,
                                               f"{img_prefix}L01A0{i+1}Z01C0{i+1}.tif"))
                # merged_images_list.append(np.stack(merged_images))
        # return merged_images_list
                
    @staticmethod
    def save_img(img,
                 output_path):
        # write 16-bit TIF image
        Image.fromarray(img.astype(np.uint16)).save(output_path)