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

from tifffile import imread, imwrite

def he_to_mask(im_file, xml_file, out_path=None):

    if out_path is not None:
        os.makedirs(os.path.join(out_path, "bin_masks"), exist_ok =True)
        # os.makedirs(os.path.join(out_path, "instance_masks"), exist_ok =True)
        os.makedirs(os.path.join(out_path, "masks"), exist_ok =True)
        os.makedirs(os.path.join(out_path, "overlay"), exist_ok =True)

    file_name = Path(im_file).stem

    img = Image.open(im_file)
    # img.show()

    img_arr = np.array(img)
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



    cv2.imwrite(os.path.join(out_path, "bin_masks", file_name + ".png"), binary_mask)
    # cv2.imwrite(os.path.join(out_path, "masks", file_name + ".png"), color_mask)
    cv2.imwrite(os.path.join(out_path, "overlay", file_name + ".png"), overlay)


    # imwrite(os.path.join(out_path, "instance_masks", file_name + ".tif"), instance_mask)#, planarconfig='CONTIG')

    return binary_mask, color_mask

'''
function [xy,binary_mask,color_mask]=he_to_binary_mask_final(filename)
im_file=strcat(filename,'.svs');

xml_file=strcat(filename,'.xml'); 
 
xDoc = xmlread(xml_file);
Regions=xDoc.getElementsByTagName('Region'); % get a list of all the region tags
for regioni = 0:Regions.getLength-1
    Region=Regions.item(regioni);  % for each region tag
 
    %get a list of all the vertexes (which are in order)
    verticies=Region.getElementsByTagName('Vertex');
    xy{regioni+1}=zeros(verticies.getLength-1,2); %allocate space for them
    for vertexi = 0:verticies.getLength-1 %iterate through all verticies
        %get the x value of that vertex
        x=str2double(verticies.item(vertexi).getAttribute('X'));
       
        %get the y value of that vertex
        y=str2double(verticies.item(vertexi).getAttribute('Y'));
        xy{regioni+1}(vertexi+1,:)=[x,y]; % finally save them into the array
    end
   
end
im_info=imfinfo(im_file);

 
nrow=im_info.Height;
ncol=im_info.Width;
binary_mask=zeros(nrow,ncol); %pre-allocate a mask
color_mask = zeros(nrow,ncol,3);
%mask_final = [];
for zz=1:length(xy) %for each region
    fprintf('Processing object # %d \n',zz);
    smaller_x=xy{zz}(:,1); 
    smaller_y=xy{zz}(:,2);
   
    %make a mask and add it to the current mask
    %this addition makes it obvious when more than 1 layer overlap each
    %other, can be changed to simply an OR depending on application.
    polygon = poly2mask(smaller_x,smaller_y,nrow,ncol);
    binary_mask=binary_mask+zz*(1-min(1,binary_mask)).*polygon;%
    color_mask = color_mask + cat(3, rand*polygon, rand*polygon,rand*polygon);
    %binary mask for all objects
    %imshow(color_mask)
end

figure;imshow(binary_mask)
figure;imshow(color_mask)

'''

if __name__ == "__main__":

    split = {
        'train' : '/mnt/dataset/MoNuSeg/dataset/MoNuSegTrainingData',
        'test' : '/mnt/dataset/MoNuSeg/dataset/MoNuSegTestData'
    }

    for s in split:
        data_path = split[s]

        im_files = sorted(glob.glob(os.path.join(data_path, "images", "*.tif")))
        ann_files = sorted(glob.glob(os.path.join(data_path, "annotations", "*.xml")))

        for (im_file, ann_file) in tqdm(zip(im_files, ann_files), total=len(im_files)):
            he_to_mask(im_file, ann_file, out_path=data_path)

    







