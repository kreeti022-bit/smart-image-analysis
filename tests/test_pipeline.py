import os
import sys
import cv2


# --------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# --------------------------------------------------

project_root = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.insert(0, project_root)


# --------------------------------------------------
# IMPORT PROJECT MODULES
# --------------------------------------------------

from src.preprocessing import preprocess_image

from src.edge_detection import (
    canny_edge_detection,
    log_edge_detection,
    dog_edge_detection
)

from src.feature_extraction import (
    harris_corner_detection,
    sift_feature_extraction,
    hog_feature_extraction
)

from src.segmentation import (
    threshold_segmentation,
    region_segmentation,
    contour_detection
)

from src.object_analysis import (
    analyze_objects,
    calculate_object_statistics
)

from src.orientation_histogram import (
    calculate_orientation_histogram
)


# --------------------------------------------------
# TEST INPUT
# --------------------------------------------------

IMAGE_PATH = "input/sample.jpg"


# --------------------------------------------------
# TEST 1: IMAGE PREPROCESSING
# --------------------------------------------------

def test_preprocessing():

    image, gray, blurred = preprocess_image(
        IMAGE_PATH
    )

    assert image is not None
    assert gray is not None
    assert blurred is not None

    assert image.shape[0] == 600
    assert image.shape[1] == 800

    print("PASS: Image preprocessing")


# --------------------------------------------------
# TEST 2: EDGE DETECTION
# --------------------------------------------------

def test_edge_detection():

    _, gray, blurred = preprocess_image(
        IMAGE_PATH
    )

    canny = canny_edge_detection(
        blurred
    )

    log = log_edge_detection(
        blurred
    )

    dog = dog_edge_detection(
        blurred
    )

    assert canny is not None
    assert log is not None
    assert dog is not None

    assert canny.shape == gray.shape
    assert log.shape == gray.shape
    assert dog.shape == gray.shape

    print("PASS: Edge detection")


# --------------------------------------------------
# TEST 3: HARRIS CORNER DETECTION
# --------------------------------------------------

def test_harris():

    _, _, blurred = preprocess_image(
        IMAGE_PATH
    )

    corners = harris_corner_detection(
        blurred
    )

    assert corners is not None
    assert corners.shape == blurred.shape

    print(
        "PASS: Harris corner detection"
    )


# --------------------------------------------------
# TEST 4: SIFT FEATURE EXTRACTION
# --------------------------------------------------

def test_sift():

    _, _, blurred = preprocess_image(
        IMAGE_PATH
    )

    keypoints, descriptors = sift_feature_extraction(
        blurred
    )

    assert keypoints is not None
    assert len(keypoints) > 0

    print(
        "PASS: SIFT feature extraction"
    )


# --------------------------------------------------
# TEST 5: HOG FEATURE EXTRACTION
# --------------------------------------------------

def test_hog():

    _, _, blurred = preprocess_image(
        IMAGE_PATH
    )

    features, hog_image = hog_feature_extraction(
        blurred
    )

    assert features is not None
    assert len(features) > 0
    assert hog_image is not None

    print(
        "PASS: HOG feature extraction"
    )


# --------------------------------------------------
# TEST 6: IMAGE SEGMENTATION
# --------------------------------------------------

def test_segmentation():

    image, _, blurred = preprocess_image(
        IMAGE_PATH
    )

    threshold = threshold_segmentation(
        blurred
    )

    region = region_segmentation(
        blurred
    )

    contour_image, contours = contour_detection(
        region,
        image
    )

    assert threshold is not None
    assert region is not None
    assert contour_image is not None
    assert contours is not None

    print(
        "PASS: Image segmentation"
    )


# --------------------------------------------------
# TEST 7: OBJECT ANALYSIS
# --------------------------------------------------

def test_object_analysis():

    image, _, blurred = preprocess_image(
        IMAGE_PATH
    )

    region = region_segmentation(
        blurred
    )

    _, contours = contour_detection(
        region,
        image
    )

    object_data = analyze_objects(
        contours
    )

    statistics = calculate_object_statistics(
        object_data
    )

    assert object_data is not None
    assert statistics is not None

    assert "total_objects" in statistics
    assert "total_area" in statistics
    assert "average_area" in statistics

    print(
        "PASS: Object analysis"
    )


# --------------------------------------------------
# TEST 8: ORIENTATION HISTOGRAM
# --------------------------------------------------

def test_orientation_histogram():

    _, _, blurred = preprocess_image(
        IMAGE_PATH
    )

    histogram = calculate_orientation_histogram(
        blurred
    )

    assert histogram is not None
    assert len(histogram) == 36

    print(
        "PASS: Orientation histogram"
    )


# --------------------------------------------------
# RUN ALL TESTS
# --------------------------------------------------

if __name__ == "__main__":

    print(
        "\nRunning Smart Image Analysis tests...\n"
    )

    test_preprocessing()

    test_edge_detection()

    test_harris()

    test_sift()

    test_hog()

    test_segmentation()

    test_object_analysis()

    test_orientation_histogram()

    print(
        "\nAll tests passed successfully!"
    )