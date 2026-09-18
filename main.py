from src.preprocessing import preprocess_image

from src.edge_detection import (
    canny_edge_detection,
    log_edge_detection,
    dog_edge_detection,
    save_edge_images
)

from src.line_detection import (
    hough_line_detection,
    save_line_detection
)

from src.feature_extraction import (
    harris_corner_detection,
    mark_harris_corners,
    save_harris_result,
    sift_feature_extraction,
    save_sift_result,
    hog_feature_extraction,
    save_hog_result
)

from src.segmentation import (
    threshold_segmentation,
    region_segmentation,
    contour_detection,
    save_segmentation_results
)

from src.object_analysis import (
    analyze_objects,
    calculate_object_statistics,
    save_object_analysis
)

from src.orientation_histogram import (
    calculate_orientation_histogram,
    save_orientation_histogram
)


# --------------------------------------------------
# INPUT IMAGE
# --------------------------------------------------

image_path = "input/sample.jpg"


try:

    # --------------------------------------------------
    # STEP 1: IMAGE PREPROCESSING
    # --------------------------------------------------

    image, gray, blurred = preprocess_image(
        image_path
    )

    print("Image preprocessing completed.")


    # --------------------------------------------------
    # STEP 2: EDGE DETECTION
    # --------------------------------------------------

    canny = canny_edge_detection(
        blurred
    )

    log = log_edge_detection(
        blurred
    )

    dog = dog_edge_detection(
        blurred
    )

    save_edge_images(
        canny,
        log,
        dog
    )

    print(
        "Edge detection completed successfully."
    )


    # --------------------------------------------------
    # STEP 3: HOUGH LINE DETECTION
    # --------------------------------------------------

    line_image, lines = hough_line_detection(
        image,
        canny
    )

    save_line_detection(
        line_image,
        lines
    )

    print(
        "Hough line detection completed successfully."
    )


    # --------------------------------------------------
    # STEP 4: HARRIS CORNER DETECTION
    # --------------------------------------------------

    corners = harris_corner_detection(
        blurred
    )

    corner_image = mark_harris_corners(
        image,
        corners
    )

    save_harris_result(
        corner_image
    )

    print(
        "Harris corner detection completed successfully."
    )


    # --------------------------------------------------
    # STEP 5: SIFT FEATURE EXTRACTION
    # --------------------------------------------------

    keypoints, descriptors = sift_feature_extraction(
        blurred
    )

    save_sift_result(
        image,
        keypoints
    )

    print(
        "SIFT feature extraction completed successfully."
    )

    print(
        "Number of SIFT keypoints:",
        len(keypoints)
    )


    # --------------------------------------------------
    # STEP 6: HOG FEATURE EXTRACTION
    # --------------------------------------------------

    hog_features, hog_image = hog_feature_extraction(
        blurred
    )

    save_hog_result(
        hog_image
    )

    print(
        "HOG feature extraction completed successfully."
    )

    print(
        "Number of HOG features:",
        len(hog_features)
    )


    # --------------------------------------------------
    # STEP 7: IMAGE SEGMENTATION
    # --------------------------------------------------

    print(
        "\nStarting image segmentation..."
    )

    threshold_image = threshold_segmentation(
        blurred
    )

    region_image = region_segmentation(
        blurred
    )

    contour_image, contours = contour_detection(
        region_image,
        image
    )

    save_segmentation_results(
        threshold_image,
        region_image,
        contour_image,
        contours
    )

    print(
        "Image segmentation completed successfully."
    )


    # --------------------------------------------------
    # STEP 8: OBJECT ANALYSIS
    # --------------------------------------------------

    print(
        "\nStarting object analysis..."
    )

    object_data = analyze_objects(
        contours
    )

    statistics = calculate_object_statistics(
        object_data
    )

    save_object_analysis(
        object_data,
        statistics
    )

    print(
        "Object analysis completed successfully."
    )


    # --------------------------------------------------
    # STEP 9: ORIENTATION HISTOGRAM
    # --------------------------------------------------

    print(
        "\nStarting orientation histogram analysis..."
    )

    orientation_histogram = calculate_orientation_histogram(
        gray
    )

    save_orientation_histogram(
        orientation_histogram
    )

    print(
        "Orientation histogram analysis completed successfully."
    )


    # --------------------------------------------------
    # PROJECT STATUS
    # --------------------------------------------------

    print(
        "\nAll current Computer Vision modules "
        "completed successfully."
    )


except FileNotFoundError as error:

    print(error)