from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class HematoDataset(CustomDataset):
    """Hemato dataset for binary WBC segmentation.

    In segmentation map annotation for Hemato dataset:
    - 0 stands for background
    - 1 stands for WBC (any white blood cell type)
    
    This config assumes binary masks.
    """
    CLASSES = (
        'Background', 
        'WBC'
    )

    PALETTE = [
        [0, 0, 0],        # Background (black)
        [255, 255, 255]   # WBC (white)
    ]

    def __init__(self, **kwargs):
        super(HematoDataset, self).__init__(
            img_suffix='.png',
            seg_map_suffix='_mask.png',  # Mask file suffix
            reduce_zero_label=False,     # Keep background class as 0
            **kwargs)
