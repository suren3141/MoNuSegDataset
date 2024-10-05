# A python function to convert H&E images and xml annotations 
# Creates binary masks, class-based masks, and overlay from annotation
# Based on the code from the following publication:
# "A Dataset and a Technique for Generalized Nuclear Segmentation for 
# Computational Pathology," in IEEE Transactions on Medical Imaging, 
# vol. 36, no. 7, pp. 1550-1560, July 2017

import sys, os, glob
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
import random
from tqdm import tqdm
import argparse

def he_to_mask(im_file, xml_file, out_path=None):

    if out_path is not None:
        os.makedirs(os.path.join(out_path, "rgb_image"), exist_ok =True)
        os.makedirs(os.path.join(out_path, "bin_masks"), exist_ok =True)
        os.makedirs(os.path.join(out_path, "inst_masks"), exist_ok =True)
        # os.makedirs(os.path.join(out_path, "masks"), exist_ok =True)
        os.makedirs(os.path.join(out_path, "overlay"), exist_ok =True)

    file_name = Path(im_file).stem

    img = Image.open(im_file)
    rgbimg = Image.new("RGBA", img.size)
    rgbimg.paste(img)

    # img.show()

    img_arr = np.array(rgbimg.convert('RGB'))
    binary_mask= np.zeros(img_arr.shape[:2], dtype=np.uint8)
    color_mask= np.zeros(img_arr.shape[:2], dtype=np.uint16)
    overlay = np.array(img_arr).astype(np.uint8)


    tree = ET.parse(xml_file)
    root = tree.getroot()

    regions = root.findall('.//Region')

    # instance_mask = np.zeros((img_arr.shape[0], img_arr.shape[1], len(regions)), dtype=np.bool_)

    for c, r in enumerate(regions):
        vertices = r.findall('.//Vertex')
        pts = [(float(v.get('X')), float(v.get('Y'))) for v in vertices]
        pts = np.array(pts).astype(np.int32)

        cv2.drawContours(binary_mask, [pts], contourIdx=-1, color=255, thickness=cv2.FILLED)
        cv2.drawContours(color_mask, [pts], contourIdx=-1, color=int(c+1), thickness=cv2.FILLED)
        cv2.drawContours(overlay, [pts], contourIdx=-1, color=(np.random.rand(3)*255).astype(np.uint8).tolist(), thickness=3)

        # instance_mask[:,:,c] = np.where(color_mask == c, True, False)


    rgbimg.save(os.path.join(out_path, "rgb_image", file_name + ".png"))

    Image.fromarray(binary_mask).save(os.path.join(out_path, "bin_masks", file_name + ".png"))
    Image.fromarray(color_mask).save(os.path.join(out_path, "inst_masks", file_name + ".tif"))
    Image.fromarray(overlay).save(os.path.join(out_path, "overlay", file_name + ".png"))

    # cv2.imwrite(os.path.join(out_path, "bin_masks", file_name + ".png"), binary_mask)
    # cv2.imwrite(os.path.join(out_path, "inst_masks", file_name + ".png"), color_mask)
    # cv2.imwrite(os.path.join(out_path, "overlay", file_name + ".png"), overlay)

    return binary_mask, color_mask

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_path", type=str, default='/mnt/dataset/MoNuSeg/dataset/MoNuSegTrainingData')
    parser.add_argument("--test_path", type=str, default='/mnt/dataset/MoNuSeg/dataset/MoNuSegTestData')
    args = parser.parse_args()

    return args



if __name__ == "__main__":

    args = get_args()

    split = {
        'train' : args.train_path,
        'test' : args.test_path
    }

    for s in split:
        data_path = split[s]

        im_files = sorted(glob.glob(os.path.join(data_path, "images", "*.tif")))
        ann_files = sorted(glob.glob(os.path.join(data_path, "annotations", "*.xml")))

        for (im_file, ann_file) in tqdm(zip(im_files, ann_files), total=len(im_files)):
            he_to_mask(im_file, ann_file, out_path=data_path)

    







