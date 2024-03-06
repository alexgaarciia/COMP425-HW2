import numpy as np
import cv2 as cv


def matchPics(I1, I2):
    # Convert images to uint8
    I1 = (I1 * 255).astype(np.uint8)
    I2 = (I2 * 255).astype(np.uint8)

    # Initialize SIFT detector
    sift = cv.SIFT_create()

    # Detect SIFT features and compute descriptors.
    keypoints1, descriptors1 = sift.detectAndCompute(I1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(I2, None)

    # Initialize and use BFMatcher to match descriptors
    bf = cv.BFMatcher()
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

    # Convert locs1 and locs2 from (row, column) to (x, y)
    locs1 = np.flip(locs1, axis=1)
    locs2 = np.flip(locs2, axis=1)

    # First, let's define some parameters
    max_inliers = []  # Largest set of match indices
    num_iters = 2000
    threshold = 5

    for _ in range(num_iters):
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
        for i, match in enumerate(matches):
            # Retrieve the coordinates of a point from the first and second image, and convert these coordinates to
            # homogeneous coordinates
            point1_homogeneous = np.append(locs1[match[0]], 1)
            point2_homogeneous = np.append(locs2[match[1]], 1)

            # Use the current homography to project point1_homogeneous from the first image into the coordinate frame of the second
            # image, and normalize it
            projected_point1 = np.dot(H, point1_homogeneous)
            projected_point1 = projected_point1 / projected_point1[2]

            # Compute SSD between the transformed point1 and point2
            ssd = np.sum((projected_point1[:2] - point2_homogeneous[:2]) ** 2)
            if ssd < threshold:
                inliers.append(i)

        # Keep the largest set of inliers
        if len(inliers) > len(max_inliers):
            max_inliers = inliers

    # Re-compute the best homography using all the inliers
    bestH = computeHomography(locs1[matches[max_inliers, 0]], locs2[matches[max_inliers, 1]])

    return bestH, max_inliers


def compositeH(H, template, img):
    # Create a composite image after warping the template image on top of the image using homography

    # In order to do this, there are some steps that must be carried out:
    # STEP 1: Create a mask of the same size as template. The mask indicates where the template exists within its own
    # frame of reference. When the mask is later transformed by homography, it will indicate where the template should
    # be placed on the target image.
    mask = np.ones(template.shape, dtype=np.uint8)

    # STEP 2: Warp mask by appropriate homography. In this part, the functon "warpPerspetive" applies the homography to
    # every point in the mask, transforming it to the target image's perspective. It is necessary to ensure that it
    # correctly lines up with the features on the target image. The transformed mask will be used to blend the template
    # with the target image accurately.
    warped_mask = cv.warpPerspective(mask, H, (img.shape[1], img.shape[0]))

    # STEP 3: Warp template by appropriate homography. This applies the geometric transformation to the template image
    # so that it aligns with the perspective of the target image. This step is crucial for aligning the template image
    # with the target image in a way that conforms to the perspective transformation described by the homography.
    # This is what makes the template appear as though it's part of the target scene.
    warped_template = cv.warpPerspective(template, H, (img.shape[1], img.shape[0]))

    # STEP 4: Use mask to combine the warped template and the image. Once we avoid changing the original image, the
    # pixels in composite_img where the warped mask is True are replaced with the corresponding pixels from the warped
    # template image.
    composite_img = img.copy()
    composite_img[warped_mask == 1] = warped_template[warped_mask == 1]

    return composite_img
