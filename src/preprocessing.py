import cv2
import os


def preprocess_image(image_path):
    # Read image
    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    # Resize image
    image = cv2.resize(image, (800, 600))

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    return image, gray, blurred


def save_preprocessed_images(image, gray, blurred):
    os.makedirs("output/preprocessing", exist_ok=True)

    cv2.imwrite("output/preprocessing/resized.jpg", image)
    cv2.imwrite("output/preprocessing/grayscale.jpg", gray)
    cv2.imwrite("output/preprocessing/blurred.jpg", blurred)

    print("Preprocessed images saved successfully.")