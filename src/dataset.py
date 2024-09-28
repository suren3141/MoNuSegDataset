import numpy as np
from PIL import Image, TiffImagePlugin
import blobfile as bf

####
class __MoNuSeg:
    """
    """

    def load_img(self, path):

        with bf.BlobFile(path, "rb") as f:
            pil_image = Image.open(f)
            pil_image.load()
        pil_image = pil_image.convert("RGB")

        return np.array(pil_image)

    def load_ann(self, path, with_type=False):
        assert not with_type, "Not support"
        # assumes that ann is HxW
        with bf.BlobFile(path, "rb") as f:
            pil_class = Image.open(f)
            pil_class.load()
        pil_class = pil_class.convert("L")
        ann_inst = np.array(pil_class)
        ann = np.expand_dims(ann_inst, -1)
        return ann

    def load_mask(self, path, with_type=False):
        assert not with_type, "Not support"
        # assumes that ann is HxW
        with bf.BlobFile(path, "rb") as f:
            pil_class = Image.open(f)
            pil_class.load()
        # pil_class = pil_class.convert("L")
        ann_inst = np.array(pil_class)
        ann = np.expand_dims(ann_inst, -1)
        return ann


####
def get_dataset(name):
    """Return a pre-defined dataset object associated with `name`."""
    name_dict = {
        "monuseg": lambda: __MoNuSeg(),
    }
    if name.lower() in name_dict:
        return name_dict[name]()
    else:
        assert False, "Unknown dataset `%s`" % name
