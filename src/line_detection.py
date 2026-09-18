import cv2
import numpy as np
import os


def hough_line_detection(image, edges):
    """
    Detect straight lines using Probabilistic Hough Line Transform.
    """

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=80,
        minLineLength=50,
        maxLineGap=10
    )

    output = image.copy()

    if lines is not None:

        # Convert lines into a simple (N, 4) format
        lines = np.asarray(lines).reshape(-1, 4)

        for x1, y1, x2, y2 in lines:

            cv2.line(
                output,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 255, 0),
                2
            )

    return output, lines


def save_line_detection(output, lines):

    os.makedirs("output/lines", exist_ok=True)

    cv2.imwrite(
        "output/lines/hough_lines.jpg",
        output
    )

    if lines is not None:
        print("Number of detected lines:", len(lines))
    else:
        print("No lines detected.")

    print("Hough line detection result saved successfully.")