import cv2
import numpy as np
import os


def canny_edge_detection(gray_image):
    edges = cv2.Canny(gray_image, 100, 200)
    return edges


def log_edge_detection(gray_image):
    blurred = cv2.GaussianBlur(gray_image, (5, 5), 0)

    laplacian = cv2.Laplacian(
        blurred,
        cv2.CV_64F
    )

    edges = np.uint8(np.absolute(laplacian))

    return edges


def dog_edge_detection(gray_image):
    blur1 = cv2.GaussianBlur(
        gray_image,
        (5, 5),
        1
    )

    blur2 = cv2.GaussianBlur(
        gray_image,
        (5, 5),
        3
    )

    dog = blur1.astype(np.float32) - blur2.astype(np.float32)

    edges = np.uint8(np.absolute(dog))

    return edges


def save_edge_images(canny, log, dog):

    os.makedirs("output/edges", exist_ok=True)

    cv2.imwrite(
        "output/edges/canny.jpg",
        canny
    )

    cv2.imwrite(
        "output/edges/log.jpg",
        log
    )

    cv2.imwrite(
        "output/edges/dog.jpg",
        dog
    )

    print("Edge detection results saved successfully.")