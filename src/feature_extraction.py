import cv2
import numpy as np
import os
from skimage.feature import hog


# --------------------------------------------------
# HARRIS CORNER DETECTION
# --------------------------------------------------

def harris_corner_detection(gray_image):
    """
    Detect corners using the Harris Corner Detector.
    """

    gray_float = np.float32(gray_image)

    corners = cv2.cornerHarris(
        gray_float,
        blockSize=2,
        ksize=3,
        k=0.04
    )

    corners = cv2.dilate(corners, None)

    return corners


def mark_harris_corners(image, corners):
    """
    Mark detected Harris corners on the original image.
    """

    output = image.copy()

    threshold = 0.01 * corners.max()

    output[corners > threshold] = [0, 0, 255]

    return output


def save_harris_result(output):
    """
    Save the Harris corner detection result.
    """

    os.makedirs("output/corners", exist_ok=True)

    cv2.imwrite(
        "output/corners/harris_corners.jpg",
        output
    )

    print("Harris corner detection result saved successfully.")


# --------------------------------------------------
# SIFT FEATURE EXTRACTION
# --------------------------------------------------

def sift_feature_extraction(gray_image):
    """
    Detect SIFT keypoints and calculate feature descriptors.
    """

    sift = cv2.SIFT_create()

    keypoints, descriptors = sift.detectAndCompute(
        gray_image,
        None
    )

    return keypoints, descriptors


def save_sift_result(image, keypoints):
    """
    Draw and save SIFT keypoints on the image.
    """

    os.makedirs("output/features", exist_ok=True)

    sift_image = cv2.drawKeypoints(
        image,
        keypoints,
        None,
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    cv2.imwrite(
        "output/features/sift_keypoints.jpg",
        sift_image
    )

    print("SIFT feature extraction result saved successfully.")
    print("Number of SIFT keypoints:", len(keypoints))


# --------------------------------------------------
# HOG FEATURE EXTRACTION
# --------------------------------------------------

def hog_feature_extraction(gray_image):
    """
    Extract Histogram of Oriented Gradients (HOG) features.
    """

    features, hog_image = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        visualize=True
    )

    return features, hog_image


def save_hog_result(hog_image):
    """
    Save the HOG visualization.
    """

    os.makedirs("output/features", exist_ok=True)

    # Convert HOG image into an 8-bit image
    hog_image = np.uint8(
        255 * (hog_image - hog_image.min()) /
        (hog_image.max() - hog_image.min() + 1e-8)
    )

    cv2.imwrite(
        "output/features/hog_visualization.jpg",
        hog_image
    )

    print("HOG feature extraction result saved successfully.")