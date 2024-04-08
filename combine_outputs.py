from PIL import Image
import os, glob
from glob import glob
from pathlib import Path
import argparse

def combine_output_single(img_file, label_file, sample_file):

    file_path, file_name = os.path.split(img_file)
    out_path = os.path.join(os.path.dirname(file_path), "combined")

    if not os.path.exists(out_path): os.mkdir(out_path)

    out_name = os.path.join(out_path, file_name)

    images = [Image.open(x) for x in [img_file, label_file, sample_file]]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))
    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(out_name)

    for img in images:
        img.close()

    new_im.close()

parser = argparse.ArgumentParser()
parser.add_argument("path")
args = parser.parse_args()

# path = "/mnt/dataset/MoNuSeg/out_sdm_128CH_hv/patches_256x256_128x128/names_37_14/TCGA-18-5592-01Z-00-DX1_unc_guidance_100k/TCGA-AC-A2FO-01A-01-TS1/output_s1.5_ema_0.9999_090000/"
path = args.path
print(path)

img_files = sorted(glob(os.path.join(path, "images", "*.png")))
label_files = sorted(glob(os.path.join(path, "labels", "*.png")))
sample_files = sorted(glob(os.path.join(path, "samples", "*.png")))

for img_file, label_file, sample_file in zip(img_files, label_files, sample_files):
    combine_output_single(img_file, label_file, sample_file)
