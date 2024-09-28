def get_dataset_info(root="/mnt/dataset/MoNuSeg/dataset", img_path="images", bin_path="bin_masks", inst_path="inst_masks", extra=""):

    
    dataset_info = {
        "MoNuSegTrainingData": {
            "img": (".tif", f"{root}/MoNuSegTrainingData/{img_path}/"),
            "ann": (".png", f"{root}/MoNuSegTrainingData/{bin_path}/"),
            "mask": (".tif", f"{root}/MoNuSegTrainingData/{inst_path}/"),
        },
        "MoNuSegTestData": {
            "img": (".tif", f"{root}/MoNuSegTestData/{img_path}/"),
            "ann": (".png", f"{root}/MoNuSegTestData/{bin_path}/"),
            "mask": (".tif", f"{root}/MoNuSegTestData/{inst_path}/"),
        },
        extra: {
            "img": (".png", f"{root}/{extra}/{img_path}/"),
            "ann": (".png", f"{root}/{extra}/{bin_path}/"),
            "mask": (".tif", f"{root}/{extra}/{inst_path}/"),
        },
    }
    return dataset_info


