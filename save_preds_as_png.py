import os
import cv2
import glob
import argparse
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="TIF (16bit) to PNG (8bit) Tool for Predictions")
    parser.add_argument("-n", "--number", type=str,
                        help="number of png images to save.")
    parser.add_argument("-i", "--input_dir", type=str,
                        help="input directory that has predictions for specific magnification.", required=True)
    parser.add_argument("-g", "--gt_dir", type=str,
                        help="ground truth directory for specific magnification.", required=True)
    parser.add_argument("-o", "--output_dir", type=str,
                        help="output directory to save predictions as 16bit TIF files.", required=True)

    args = parser.parse_args()
    print(args)
    return args


def get_gt_path(img_path, gt_dir):
    img_name = img_path.split("/")[-1]
    return os.path.join(gt_dir, img_name)


def main():
    # get arguments
    args = parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    gt_dir = args.gt_dir
    N = args.number
    # get image paths
    preds = [
        sorted(glob.glob(os.path.join(input_dir, f"*A0{i}Z01C0{i}*.tif"), recursive=True))
        for i in range(1,4)
    ]
    preds = list(zip(*preds))
    # sample from the list
    if N is not None:
        sample_indexes = np.random.choice(np.arange(len(preds)), size=int(N), replace=False)
        preds = [preds[i] for i in sample_indexes]
    # get the ground truths
    gts = [tuple([get_gt_path(img_path, gt_dir) for img_path in channels]) for channels in preds]
    
    # construct directories
    pred_dir = os.path.join(output_dir, "pred")
    gt_dir = os.path.join(output_dir, "gt")
    for dir_path in [pred_dir, gt_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            
    # save images
    pred_dir = os.path.join(output_dir, "pred")
    gt_dir = os.path.join(output_dir, "gt")
    for dir_path in [pred_dir, gt_dir]:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)

    for pred_paths, gt_paths in tqdm(zip(preds, gts), total=int(N)):
        # read 16 bit images
        pred16 = np.stack([cv2.imread(path, cv2.IMREAD_ANYDEPTH) for path in pred_paths], axis=2)
        gt16 = np.stack([cv2.imread(path, cv2.IMREAD_ANYDEPTH) for path in gt_paths], axis=2)
        # scale them to 8bit
        pred8 = (pred16 / (pred16.max()/256)).astype(np.uint8)
        gt8 = (gt16 / (gt16.max()/256)).astype(np.uint8)
        # save single channels
        for i in range(3):
            img_name = pred_paths[i].split("/")[-1][:-4] + ".png"
            cv2.imwrite(os.path.join(pred_dir, img_name), pred8[:,:,i])
            cv2.imwrite(os.path.join(gt_dir, img_name), gt8[:,:,i])
        # save rgb images
        rgb_name = pred_paths[0].split("/")[-1][:-16] + "_RGB.png"
        cv2.imwrite(os.path.join(pred_dir, rgb_name), pred8)
        cv2.imwrite(os.path.join(gt_dir, rgb_name), gt8)

    
if __name__ == "__main__":
    main()
