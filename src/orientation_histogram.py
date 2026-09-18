import cv2
import numpy as np
import os


# --------------------------------------------------
# ORIENTATION HISTOGRAM
# --------------------------------------------------

def calculate_orientation_histogram(gray_image):

    # Calculate horizontal and vertical gradients
    gradient_x = cv2.Sobel(
        gray_image,
        cv2.CV_64F,
        1,
        0,
        ksize=3
    )

    gradient_y = cv2.Sobel(
        gray_image,
        cv2.CV_64F,
        0,
        1,
        ksize=3
    )

    # Calculate gradient magnitude
    magnitude = cv2.magnitude(
        gradient_x.astype(np.float32),
        gradient_y.astype(np.float32)
    )

    # Calculate gradient orientation in degrees
    orientation = cv2.phase(
        gradient_x.astype(np.float32),
        gradient_y.astype(np.float32),
        angleInDegrees=True
    )

    # Create 36 orientation bins
    bins = np.linspace(0, 360, 37)

    histogram, _ = np.histogram(
        orientation,
        bins=bins,
        weights=magnitude
    )

    return histogram


# --------------------------------------------------
# SAVE HISTOGRAM DATA
# --------------------------------------------------

def save_orientation_histogram(histogram):

    os.makedirs(
        "output/features",
        exist_ok=True
    )

    output_file = (
        "output/features/"
        "orientation_histogram.txt"
    )

    with open(output_file, "w") as file:

        file.write(
            "ORIENTATION HISTOGRAM\n"
        )

        file.write(
            "=====================\n\n"
        )

        file.write(
            "Number of orientation bins: 36\n\n"
        )

        for index, value in enumerate(histogram):

            start_angle = index * 10
            end_angle = start_angle + 10

            file.write(
                f"{start_angle:3d}-{end_angle:3d} degrees: "
                f"{value:.2f}\n"
            )

    print(
        "Orientation histogram saved successfully."
    )

    print(
        "Number of orientation bins:",
        len(histogram)
    )