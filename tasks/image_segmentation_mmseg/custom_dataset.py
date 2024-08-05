from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class CustomDataset(BaseSegDataset):
    METAINFO = dict(classes=(), palette=[])

    def _init_(self, **kwargs):
        super()._init_(img_suffix=".jpg", seg_map_suffix=".png", **kwargs)
