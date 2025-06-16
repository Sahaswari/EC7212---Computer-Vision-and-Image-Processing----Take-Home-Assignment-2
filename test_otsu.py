import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from skimage import io
from skimage import data, filters, util
from skimage.filters import gaussian
from scipy.ndimage import median_filter
from skimage import exposure

#upload a image
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select an image file", filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp")])
image = io.imread(file_path, as_gray=True)

#create image (two-objects + background)
# image = np.zeros((100,100), dtype=np.uint8)
# image[20:50:, 20:50] = 85
# image[60:90,60:90] = 170

#Add Gaussian noice
noisy_image = util.random_noise(image, mode='gaussian', var=0.01)
noisy_image = (noisy_image*255).astype(np.uint8)

#Pre-processing
denoised_image = gaussian(noisy_image, sigma=1) #clean the image
denoised_image = median_filter(noisy_image, size=3)
#enhanced_image = exposure.equalize_hist(denoised_image)

#Apply Otsu's thresholding
thereshould = filters.threshold_otsu(denoised_image)
binary_image = noisy_image > thereshould

#Display results
plt.figure(figsize=(10,3))
plt.subplot(1,4,1); plt.title('Original Image'); plt.imshow(image)
plt.subplot(1,4,2);plt.title('Noisy Image'); plt.imshow(noisy_image, cmap='gray')
plt.subplot(1,4,3);plt.title('Otsu Thresholding'); plt.imshow(binary_image, cmap='gray')
plt.subplot(1,4,4);plt.title('Resulting Image');plt.imshow(binary_image, cmap='gray')
plt.show()
