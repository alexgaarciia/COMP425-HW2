# import other necessary libaries
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.feature import canny
from skimage.color import rgb2gray
from utils import create_line, create_mask


# load the input image and convert to grayscale
im = imread('road.jpg').astype('float')
im = im / 255
image_gray = rgb2gray(im)

# run Canny edge detector to find edge points
edges = canny(image_gray)
plt.figure(figsize=(8, 6))
plt.imshow(edges, cmap='gray')
plt.axis('off')
plt.title('Edges')
plt.show()

# create a mask for ROI by calling create_mask
H, W = image_gray.shape
mask = create_mask(H, W)
plt.figure(figsize=(8, 6))
plt.imshow(mask, cmap='gray')
plt.axis('off')
plt.title('Mask')
plt.show()

# extract edge points in ROI by multiplying edge map with the mask
masked_edges = edges * mask
plt.figure(figsize=(8, 6))
plt.imshow(masked_edges, cmap='gray')
plt.axis('off')
plt.title('Edges in ROI')
plt.show()

# perform Hough transform

# find the right lane by finding the peak in hough space

# zero out the values in accumulator around the neighborhood of the peak

# find the left lane by finding the peak in hough space

# plot the results
