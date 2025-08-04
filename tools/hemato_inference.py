import os
import numpy as np
from PIL import Image
from mmseg.apis import init_segmentor, inference_segmentor
from scipy.ndimage import label

# === Config and checkpoint paths ===
config_file = '/media/iml/cv-lab/Segformer_hemato_binary/SegFormer/local_configs/segformer/B1/segformer.b1.512x512.hemato.160k.py'
checkpoint_file = '/media/iml/cv-lab/Segformer_hemato_binary/SegFormer/work_dirs/segformer.b1.512x512.hemato.160k/latest.pth'

# === Input and output paths ===
input_dir = '/media/iml/cv-lab/Datasets_B_cells/Hemato_Data/inference'
output_dir = 'inference_results_hemato_binary'
os.makedirs(output_dir, exist_ok=True)

# === Function to generate consistent random colors ===
def random_color_map(n):
    np.random.seed(42)
    return [tuple(np.random.randint(50, 256, size=3)) for _ in range(n)]

# === Function to color connected cells with black background ===
def color_cells_with_black_background(mask, alpha=1.0):
    labeled_mask, num_cells = label(mask == 1)
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    colors = random_color_map(num_cells + 1)
    for i in range(1, num_cells + 1):
        cell_mask = labeled_mask == i
        for c in range(3):
            colored_mask[:, :, c][cell_mask] = int(colors[i][c] * alpha)
    return colored_mask

# === Combine original and segmented images side by side ===
def concat_side_by_side(img1, img2):
    arr1 = np.array(img1)
    arr2 = np.array(img2)
    h = max(arr1.shape[0], arr2.shape[0])
    arr1 = np.array(Image.fromarray(arr1).resize((arr1.shape[1], h)))
    arr2 = np.array(Image.fromarray(arr2).resize((arr2.shape[1], h)))
    combined = np.concatenate((arr1, arr2), axis=1)
    return Image.fromarray(combined)

# === Load model ===
model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

# === Run inference on all images ===
for filename in os.listdir(input_dir):
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        continue

    img_path = os.path.join(input_dir, filename)
    result = inference_segmentor(model, img_path)
    mask = result[0].astype(np.uint8)  # Binary mask: 0 = BG, 1 = WBC

    original_img = Image.open(img_path).convert('RGB')
    colored_mask = color_cells_with_black_background(mask, alpha=1.0)
    overlay = concat_side_by_side(original_img, Image.fromarray(colored_mask))

    save_path = os.path.join(output_dir, f'{os.path.splitext(filename)[0]}_seg.png')
    overlay.save(save_path)
    print(f"Saved: {save_path}")
