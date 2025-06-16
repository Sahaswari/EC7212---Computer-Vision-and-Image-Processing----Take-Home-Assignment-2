import numpy as np
import cv2
import matplotlib.pyplot as plt
import tkinter as tk

from tkinter import filedialog
from skimage import io

#open file dialogbox for uploading image
root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename(title="Select an image file")
org_image = io.imread(file_path)