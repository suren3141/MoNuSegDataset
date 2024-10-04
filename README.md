# [MoNuSeg grand-challenge](https://monuseg.grand-challenge.org/) organized at [MICCAI 2018](https://www.miccai2018.org/en/)

This repository is an extension of the [Mask R-CNN implementation](https://github.com/ruchikaverma-iitg/MoNuSeg) with functionality extended to python.

## Dataset

### Quick run
Dataset with extracted patches can be found in the following link:
[Dataset](https://drive.google.com/drive/folders/1hUj9ToCYhIeIDOzfmj8BNp-BgLRRxbJX?usp=sharing)

### Step-by-step instruction
1. The original dataset can be downloaded from the following link: [https://monuseg.grand-challenge.org/Data/]

The dataset should have the following structure:
```
dataset
├── MoNuSegTrainingData
│   ├── images
│   │   ├──TCGA-18-5592-01Z-00-DX1.tif
│   │   ├──TCGA-21-5784-01Z-00-DX1.tif
│   │
│   └── annotations
│       ├──TCGA-18-5592-01Z-00-DX1.xml
│       ├──TCGA-21-5784-01Z-00-DX1.xml
│
└── MoNuSegTestgData
    ├── images
    └── annotations
```

2. Convert `.xml` files to 2D binary and instance masks of format `.png` and `.tif`

```bash
python src/he_to_binary_mask.py --train_path $PATH_TO_TRAIN_SET --test_path $PATH_TO_TEST_SET
```

The dataset should then have the following format
```
dataset
├── MoNuSegTrainingData
│   ├── images
│   ├── bin_masks
│   ├── inst_masks
│   ├── overlay
│   └── annotations
│
└── MoNuSegTestgData
    ├── images
    ├── bin_masks
    ├── inst_masks
    ├── overlay
    └── annotations
```

3. Extract patches from generated images and masks:

```bash
python src/extract_patches_monuseg.py --data_path $PATH_TO_DATASET --window_size 128 --step_size 128
```


## Citations

Please cite the following papers if you use this repository-

[Kumar, N., Verma R. et al., "A Multi-organ Nucleus Segmentation Challenge," in IEEE Transactions on Medical Imaging 2019](https://ieeexplore.ieee.org/document/8880654)

[Kumar, N., Verma, R., Sharma, S., Bhargava, S., Vahadane, A., & Sethi, A. (2017). A dataset and a technique for generalized nuclear segmentation for computational pathology. IEEE transactions on medical imaging, 36(7), 1550-1560](https://ieeexplore.ieee.org/document/7872382)




