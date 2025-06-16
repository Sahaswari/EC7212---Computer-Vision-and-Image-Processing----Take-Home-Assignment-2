import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.feature import peak_local_max
from skimage import img_as_ubyte

# Region Growing (Colored)
def region_growing_colored(image, seeds, threshold=25):
    h, w = image.shape[:2]
    segmented = np.zeros((h, w, 3), dtype=np.uint8)  # Color image
    visited = np.zeros((h, w), dtype=bool)
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    colors = plt.get_cmap('hsv', len(seeds))

    for idx, seed in enumerate(seeds):
        stack = [seed]
        seed_color = image[seed].astype(float)
        region_color = (np.array(colors(idx)[:3]) * 255).astype(np.uint8)  # Convert cmap to RGB

        while stack:
            y, x = stack.pop()
            if visited[y, x]:
                continue
            visited[y, x] = True

            color_dist = np.linalg.norm(image[y, x].astype(float) - seed_color)
            if color_dist <= threshold:
                segmented[y, x] = region_color
                for dy, dx in neighbors:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                        stack.append((ny, nx))

    return segmented

# Load image using dialog
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(
    title="Select an image file",
    filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")]
)
org_image = cv2.imread(file_path)
image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB)

# Preprocessing (blur)
smoothed_image = gaussian(image, sigma=1, channel_axis=-1)
smoothed_image = img_as_ubyte(smoothed_image)

# Get multiple seed points
gray_image = rgb2gray(smoothed_image)
seed_points = peak_local_max(gray_image, min_distance=15, num_peaks=6)
seed_coords = [tuple(coord) for coord in seed_points]

# Run region growing with colors
segmented = region_growing_colored(smoothed_image, seed_coords, threshold=25)

# Show original and result
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original Image")
plt.imshow(image)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("Colored Region-Grown Segmentation")
plt.imshow(segmented)
plt.axis('off')

plt.tight_layout()
plt.show()
