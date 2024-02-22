# import other necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage.feature import canny
from skimage.color import rgb2gray
from utils import create_line, create_mask
from skimage.transform import hough_line, hough_line_peaks


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
hough_space, angles, distances = hough_line(masked_edges)

# find the right lane by finding the peak in hough space
accum, angles_peaks, dists_peaks = hough_line_peaks(hough_space, angles, distances)
rho, theta = dists_peaks[0], angles_peaks[0]
xs, ys = create_line(rho, theta, im)

# zero out the values in accumulator around the neighborhood of the peak
nms_radius = 100

# Convert peak coordinates to accumulator array indices
peak_index = (np.argmin(np.abs(distances - rho)), np.argmin(np.abs(angles - theta)))

# Apply NMS
for r_offset in range(-nms_radius, nms_radius + 1):
    for theta_offset in range(-nms_radius, nms_radius + 1):
        if 0 <= peak_index[0] + r_offset < hough_space.shape[0] and 0 <= peak_index[1] + theta_offset < hough_space.shape[1]:
            hough_space[peak_index[0] + r_offset, peak_index[1] + theta_offset] = 0

# find the left lane by finding the peak in hough space
accum, angles_peaks, dists_peaks = hough_line_peaks(hough_space, angles, distances)
rho_orange, theta_orange = dists_peaks[1], angles_peaks[1]
xs_orange, ys_orange = create_line(rho_orange, theta_orange, im)

# plot the results
# Plotting the original image
plt.figure(figsize=(10, 7))
plt.imshow(image_gray, cmap='gray')
plt.axis('off')
plt.title('Detected Lanes')

# Plot the first lane (Blue lane)
plt.plot(xs, ys, '-b', label='Blue Lane')

# Plot the second lane (Orange lane)
plt.plot(xs_orange, ys_orange, color='orange', label='Orange Lane')
plt.show()
