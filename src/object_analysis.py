import cv2
import os


# --------------------------------------------------
# OBJECT ANALYSIS
# --------------------------------------------------

def analyze_objects(contours):

    object_data = []

    for index, contour in enumerate(contours, start=1):

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(
            contour,
            True
        )

        x, y, width, height = cv2.boundingRect(
            contour
        )

        moments = cv2.moments(contour)

        if moments["m00"] != 0:
            center_x = int(
                moments["m10"] / moments["m00"]
            )

            center_y = int(
                moments["m01"] / moments["m00"]
            )
        else:
            center_x = 0
            center_y = 0

        object_info = {
            "id": index,
            "area": area,
            "perimeter": perimeter,
            "x": x,
            "y": y,
            "width": width,
            "height": height,
            "center_x": center_x,
            "center_y": center_y
        }

        object_data.append(object_info)

    return object_data


# --------------------------------------------------
# OBJECT STATISTICS
# --------------------------------------------------

def calculate_object_statistics(object_data):

    total_objects = len(object_data)

    if total_objects == 0:
        return {
            "total_objects": 0,
            "total_area": 0,
            "average_area": 0
        }

    total_area = sum(
        obj["area"]
        for obj in object_data
    )

    average_area = total_area / total_objects

    statistics = {
        "total_objects": total_objects,
        "total_area": total_area,
        "average_area": average_area
    }

    return statistics


# --------------------------------------------------
# SAVE OBJECT ANALYSIS
# --------------------------------------------------

def save_object_analysis(object_data, statistics):

    os.makedirs(
        "output/object_analysis",
        exist_ok=True
    )

    output_file = (
        "output/object_analysis/"
        "object_statistics.txt"
    )

    with open(output_file, "w") as file:

        file.write(
            "SMART IMAGE ANALYSIS\n"
        )

        file.write(
            "OBJECT ANALYSIS RESULTS\n"
        )

        file.write(
            "========================\n\n"
        )

        file.write(
            f"Total objects detected: "
            f"{statistics['total_objects']}\n"
        )

        file.write(
            f"Total detected area: "
            f"{statistics['total_area']:.2f}\n"
        )

        file.write(
            f"Average object area: "
            f"{statistics['average_area']:.2f}\n\n"
        )

        file.write(
            "INDIVIDUAL OBJECT INFORMATION\n"
        )

        file.write(
            "------------------------------\n"
        )

        for obj in object_data:

            file.write(
                f"\nObject {obj['id']}\n"
            )

            file.write(
                f"Area: {obj['area']:.2f}\n"
            )

            file.write(
                f"Perimeter: {obj['perimeter']:.2f}\n"
            )

            file.write(
                f"Bounding Box: "
                f"({obj['x']}, {obj['y']}, "
                f"{obj['width']}, {obj['height']})\n"
            )

            file.write(
                f"Centroid: "
                f"({obj['center_x']}, "
                f"{obj['center_y']})\n"
            )

    print(
        "Object analysis results saved successfully."
    )

    print(
        "Total objects analyzed:",
        statistics["total_objects"]
    )