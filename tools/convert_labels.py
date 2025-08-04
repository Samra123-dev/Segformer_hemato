
from PIL import Image
import numpy as np

mask = np.array(Image.open('/media/iml/cv-lab/Datasets_B_cells/Hemato_Data/Train/masks_binary/img (1)_mask.png'))
print(np.unique(mask))  # Should print: [0 1]

