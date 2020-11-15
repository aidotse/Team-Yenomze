import os
import glob
import torch
import argparse

import numpy as np

from src.model_handler.TestHandler import TestHandler
from src.dataloader.TestDataset import OverlappyGridyDataset
from src.model.Generator import GeneratorUnet

BEST_MODELS = {
    "20x": "checkpoints/model20x/G_epoch_548.pth",
    "40x": "checkpoints/model40x/G_epoch_125.pth",
    "60x": "checkpoints/model60x/G_epoch_228.pth"
}

def parse_args():
    parser = argparse.ArgumentParser(description="Adipocyte Fluorescence Predictor CLI Tool")
    parser.add_argument("-c", "--model_checkpoint", type=str,
                        help="checkpoint path for UNet model.")
    parser.add_argument("-m", "--magnification", type=str,
                        help="magnification level of input images.", required=True, choices=['20x', '40x', '60x'])
    parser.add_argument("-i", "--input_dir", type=str,
                        help="input directory that has brightfield images.", required=True)
    parser.add_argument("-o", "--output_dir", type=str,
                        help="output directory to save predictions as 16bit TIF files.", required=True)

    args = parser.parse_args()
    print(args)
    return args
    

def main():
    args = parse_args()
    # get image paths
    inputs = [
        sorted(glob.glob(os.path.join(args.input_dir, f"*A04Z0{i}*.tif"), recursive=True))
        for i in range(1,8)
    ]
    inputs = list(zip(*inputs))
        
    # Load Model
    model = GeneratorUnet(split=True)
    model_chkp_path = BEST_MODELS[args.magnification] if args.model_checkpoint is None else args.model_checkpoint
    print(f'Loading checkpoint: {model_chkp_path}')
    chkp = torch.load(model_chkp_path)
    model.load_state_dict(chkp["model_state_dict"])
    
    # Patch Iterator Dataset
    patch_iterator = OverlappyGridyDataset
    
    # Test Handler
    test_handler = TestHandler(patch_iterator,
                               model=model,
                               output_dir=args.output_dir)
    # Run test
    test_handler.run_test(inputs, mag_level=args.magnification)
    

if __name__ == "__main__":
    main()
