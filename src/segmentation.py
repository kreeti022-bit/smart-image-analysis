import cv2
import numpy as np
import os


# --------------------------------------------------
# THRESHOLDING SEGMENTATION
# --------------------------------------------------

def threshold_segmentation(gray_image):
    _, binary = cv2.threshold(
        gray_image,
        127,
        255,
        cv2.THRESH_BINARY
    )

    return binary


# --------------------------------------------------
# REGION-BASED SEGMENTATION
# --------------------------------------------------

def region_segmentation(gray_image):
    # Otsu thresholding automatically selects threshold
    _, region = cv2.threshold(
        gray_image,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return region


# --------------------------------------------------
# CONTOUR-BASED OBJECT ANALYSIS
# --------------------------------------------------

def contour_detection(binary_image, original_image):
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    output = original_image.copy()

    valid_contours = []

    for contour in contours:
        area = cv2.contourArea(contour)

        # Ignore very small regions/noise
        if area > 500:
            valid_contours.append(contour)

            x, y, w, h = cv2.boundingRect(contour)

            cv2.rectangle(
                output,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.drawContours(
                output,
                [contour],
                -1,
                (0, 0, 255),
                2
            )

    return output, valid_contours


# --------------------------------------------------
# SAVE SEGMENTATION RESULTS
# --------------------------------------------------

def save_segmentation_results(
    threshold_image,
    region_image,
    contour_image,
    contours
):

    os.makedirs("output/segmentation", exist_ok=True)

    cv2.imwrite(
        "output/segmentation/threshold.jpg",
        threshold_image
    )

    cv2.imwrite(
        "output/segmentation/region_segmentation.jpg",
        region_image
    )

    cv2.imwrite(
        "output/segmentation/contours.jpg",
        contour_image
    )

    print("Segmentation results saved successfully.")
    print("Number of significant regions detected:", len(contours))