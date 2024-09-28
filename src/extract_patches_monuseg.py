"""extract_patches.py

Patch extraction script.
"""

import re
import glob
import os
import os.path as osp
import tqdm
from pathlib import Path

import numpy as np
import cv2

import sys
proj_dir = os.path.join(*Path(os.path.abspath(__file__)).parts[:-2])
sys.path.append(proj_dir)
sys.path.remove(os.path.join(proj_dir,'src'))

print(sys.path)



from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir

from PIL import Image
import blobfile as bf

from src.config import get_dataset_info
from src.dataset import get_dataset
# from config import get_dataset_info
# from dataset import get_dataset

import argparse

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="/mnt/dataset/MoNuSeg/dataset")
    parser.add_argument("--window_size", type=int, default=128)
    parser.add_argument("--step_size", type=int, default=128)
    args = parser.parse_args()

    return args



# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    args = get_args()

    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = False
    out_format = '.png'

    # Name of dataset - use Kumar, CPM17 or CoNSeP.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "monuseg"

    win_size = [args.window_size, args.window_size]
    step_size = [args.step_size, args.step_size]
    # win_size = [128, 128]
    # step_size = [128, 128]
    # resize = [256, 256]
    extract_type = "valid"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # data_root = "/mnt/dataset/MoNuSeg/dataset"
    # data_root = "/mnt/dataset/MoNuSeg/TCGA/"

    data_root = args.data_path
    save_root = f"{data_root}/patches_{extract_type}_inst"

    # a dictionary to specify where the dataset path should be
    dataset_info = get_dataset_info(root=data_root, extra="")

    patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
    parser = get_dataset(dataset_name)
    xtractor = PatchExtractor(win_size, step_size, debug=False)
    for split_name, split_desc in dataset_info.items():
        img_ext, img_dir = split_desc["img"]
        ann_ext, ann_dir = split_desc["ann"]
        mask_ext, mask_dir = split_desc["mask"]

        out_dir = "%s_%dx%d_%dx%d/%s/" % (
            save_root,
            win_size[0],
            win_size[1],
            step_size[0],
            step_size[1],
            split_name
        )

        print(out_dir)
        file_list = glob.glob(patterning("%s/*%s" % (img_dir, img_ext)))
        file_list.sort()  # ensure same ordering across platform
        if len(file_list) == 0: continue

        rm_n_mkdir(out_dir)
        if not osp.exists(osp.join(out_dir, "images")): os.makedirs(osp.join(out_dir, "images"))
        if not osp.exists(osp.join(out_dir, "bin_masks")) and ann_ext is not None: os.makedirs(osp.join(out_dir, "bin_masks"))
        if not osp.exists(osp.join(out_dir, "inst_masks")) and mask_ext is not None: os.makedirs(osp.join(out_dir, "inst_masks"))

        pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
        pbarx = tqdm.tqdm(
            total=len(file_list), bar_format=pbar_format, ascii=True, position=0
        )

        for file_idx, file_path in enumerate(file_list):
            base_name = Path(file_path).stem

            img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
            if ann_ext is not None:
                ann = parser.load_ann("%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification)
            else:
                ann = np.empty(img.shape[:2])[..., np.newaxis]
            if mask_ext is not None:
                inst_mask = parser.load_mask("%s/%s%s" % (mask_dir, base_name, mask_ext), type_classification)
                print(inst_mask.dtype)
            else:
                inst_mask = np.empty(img.shape[:2])[..., np.newaxis]

            # *
            img = np.concatenate([img, ann, inst_mask], axis=-1)
            sub_patches = xtractor.extract(img, extract_type)

            pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbar = tqdm.tqdm(
                total=len(sub_patches),
                leave=False,
                bar_format=pbar_format,
                ascii=True,
                position=1,
            )

            for idx, patch in enumerate(sub_patches):
                if out_format == '.png':
                    img_patch, ann_patch, mask_patch = patch[:,:,0:3], patch[:,:,3], patch[:,:,4]
                    out_img = Image.fromarray(img_patch.astype(np.uint8)).resize(resize)
                    out_img.save(f"{out_dir}/images/{base_name}_{idx:03d}.png")
                    if ann_ext is not None:
                        Image.fromarray(ann_patch.astype(np.uint8)).resize(resize, Image.NEAREST).save(f"{out_dir}/bin_masks/{base_name}_{idx:03d}.png")
                    if mask_ext is not None:
                        Image.fromarray(mask_patch.astype(np.int16)).resize(resize, Image.NEAREST).save(f"{out_dir}/inst_masks/{base_name}_{idx:03d}.tif")
                    # print(mask_patch.dtype)
                    print(out_img.size)

                elif out_format == '.npy':
                    np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                pbar.update()
            pbar.close()

            pbarx.update()
        pbarx.close()
