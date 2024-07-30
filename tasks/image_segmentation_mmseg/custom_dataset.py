from mmseg.registry import DATASETS
from mmseg.datasets import BaseSegDataset


@DATASETS.register_module()
class CustomDataset(BaseSegDataset):
    METAINFO = dict(
        classes=("sky", "tree", "road", "grass", "water", "bldg", "mntn", "fg obj"),
        palette=[
            [128, 128, 128],
            [129, 127, 38],
            [120, 69, 125],
            [53, 125, 34],
            [0, 11, 123],
            [118, 20, 12],
            [122, 81, 25],
            [241, 134, 51],
        ],
    )

    def _init_(self, **kwargs):
        super()._init_(img_suffix=".jpg", seg_map_suffix=".png", **kwargs)
