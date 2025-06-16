import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import filedialog
from skimage import io

#region_growing function
def region_growing (image, seeds, threshold=40):
    if image.ndim == 2:
        h, w = image.shape
        segmented = np.zeros((h,w), dtypes=np.uint8)
        visited = np.zeros((h, w), dtype=bool)
        neighbors = [(-1,0), (1,0), (0,-1), (0,1)]
        for seed in seeds:
            stack = [seed]
            seed_value = image[seed]
            while stack:
                y, x = stack.pop()
                if visited[y, x]:
                    continue
                visited[y, x] = True
                if abs(int(image[y, x]) - int(seed_value)) <= threshold:
                    segmented[y, x] = 255
                    for dy, dx in neighbors:
                        ny, nx = y + dy, x+ dx
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                            stack.append((ny, nx))
        return segmented
    elif image.ndim == 3:
        h, w, c = image.shape
        segmented = np.zeros((h, w), dtype=np.uint8)
        visited = np.zeros((h, w), dtype=bool)
        neighbors = [(-1,0), (1,0), (1,-1), (0,1)]
        for seed in seeds:
            stack = [seed]
            seed_color = image[seed]
            while stack:
                y, x = stack.pop()
                if visited[y, x]:
                    continue
                visited[y, x] = True
                color_dist = np.linalg.norm(image[y, x].astype(float) - seed_color.astype(float))
                if color_dist <= threshold:
                    segmented[y, x] = 255
                    for dy, dx in neighbors:
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < h and 0 <= nx < w and not visited[ny, nx]:
                            stack.append((ny, nx))
        return segmented
    else:
        raise ValueError("Unsupported image format")

#open file dialogbox for uploading image
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select an image file", filetypes = [("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
org_image = cv2.imread(file_path)
image = cv2.cvtColor(org_image, cv2.COLOR_BGR2RGB) #disply and color processing

#choose a seed
seed = (image.shape[0] //2, image.shape[1] //2)
segmented = region_growing(image, [seed], threshold=30)

#show results
plt.subplot(1,2,1)
plt.title("Original Image")
plt.imshow(image if image.ndim==3 else image, cmap='gray')
plt.axis('off')
plt.subplot(1,2,2)
plt.title("Region Grown")
plt.imshow(segmented, cmap='gray')
plt.axis('off')
plt.show()