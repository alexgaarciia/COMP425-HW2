import numpy as np
import cv2


def matchPics(I1, I2):
    # Convert images to 8-bit unsigned integer format
    I1_uint8 = (I1 * 255).astype(np.uint8)
    I2_uint8 = (I2 * 255).astype(np.uint8)

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect SIFT features and compute descriptors.
    keypoints1, descriptors1 = sift.detectAndCompute(I1_uint8, None)
    keypoints2, descriptors2 = sift.detectAndCompute(I2_uint8, None)

    # Initialize and use BFMatcher to match descriptors
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    # Apply Lowe's ratio test to filter matches
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # Prepare data for return. Extract locations of good matches.
    locs1 = np.array([keypoints1[m.queryIdx].pt for m in good_matches])
    locs2 = np.array([keypoints2[m.trainIdx].pt for m in good_matches])

    # Convert locs1 and locs2 from (x, y) to (row, column), which means flipping x and y
    locs1 = np.flip(locs1, axis=1)
    locs2 = np.flip(locs2, axis=1)

    # Prepare matches array in the format specified (index in locs1 to index in locs2)
    matches_array = np.array([[i, i] for i in range(len(good_matches))])

    return matches_array, locs1, locs2


def computeHomography(locs1, locs2):
    # Initialize an empty list A, which will be used to build up the matrix A. It will be a system of linear equations
    # that emerges from the correspondences between points in locs1 and locs2.
    A = []
    for i in range(len(locs1)):
        # Extract the x and y components of the i-th point in locs1 and locs2
        x1, y1 = locs1[i, :]
        x2, y2 = locs2[i, :]

        A.append([-x1, -y1, -1, 0, 0, 0, x1 * x2, y1 * x2, x2])
        A.append([0, 0, 0, -x1, -y1, -1, -x1 * y2, y1 * y2, y2])

    # Convert the list into a NumPy array to perform SVD
    A = np.array(A)
    U, S, Vh = np.linalg.svd(A)

    # The homography is the last column of V or the last row of Vh
    H = Vh[-1].reshape((3, 3))

    # Normalize the homography matrix by dividing it by the element in the third row and third column so the scale of
    # the transformation is correctly set to assume the points are in homogeneous coordinates with the third coordinate
    # set to 1
    H = H / H[-1, -1]
    return H


def computeH_ransac(matches, locs1, locs2):
    # Compute the best fitting homography using RANSAC given a list of matching pairs
    # Define parameters
    bestH = None
    max_inliers = []  # Largest set of match indices
    num_iters = 1000
    threshold = 5

    for _ in range(1):
        # Select four feature points (at random):
        points = np.random.choice(len(matches), 4, replace=False)
        selected_matches = matches[points]

        # Extract the matching points
        points1 = locs1[selected_matches[:, 0]]
        points2 = locs2[selected_matches[:, 1]]

        # Compute the homography using the selected points
        H = computeHomography(points1, points2)

        # Calculate the inliers for this homography
        inliers = []
        
    return bestH, inliers


def compositeH(H, template, img):
    # Create a composite image after warping the template image on top
    # of the image using homography

    # Create mask of same size as template

    # Warp mask by appropriate homography

    # Warp template by appropriate homography

    # Use mask to combine the warped template and the image

    return composite_img
