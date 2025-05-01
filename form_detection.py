import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import maximum_filter as filter_max
from scipy.ndimage import minimum_filter as filter_min
import scipy.ndimage as ndimage
import scipy.spatial


def load_and_preprocess_image(file_name):
    """Load and preprocess the image."""
    img = cv2.imread(f"Image\\{file_name}.jpg")
    img = cv2.resize(img, (800, 800))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return img, gray


def threshold_image(gray):
    """Apply thresholding to the grayscale image."""
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(f"Binary_Image\\{file_name}_binary_version.jpg", thresh)
    return thresh


def find_connected_components(thresh):
    """Find connected components in the thresholded image."""
    analysis = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
    return analysis


def draw_bounding_boxes_and_centroids(img, stats, numLabels, image_crop):
    """Draw bounding boxes and centroids on the image."""
    pt1, pt2 = (0, 0), (0, 0)
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 1000:
            new_img = img.copy()
            x1, y1, w, h = (
                stats[i, cv2.CC_STAT_LEFT],
                stats[i, cv2.CC_STAT_TOP],
                stats[i, cv2.CC_STAT_WIDTH],
                stats[i, cv2.CC_STAT_HEIGHT],
            )
            pt1, pt2 = (x1, y1), (x1 + w, y1 + h)
            cropped_image, image_crop = crop_image(pt1, pt2, new_img, i, image_crop)

    return cropped_image, image_crop


def detect_contours(img, contours):
    """Detect and draw contours on the image."""
    detected_img = img.copy()
    for cnt in contours:
        epsilon = 0.0001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(detected_img, [approx], 0, (0, 255, 0), 2)
    return detected_img


def display_images(images, titles):
    """Display images with titles."""
    for i in range(len(images)):
        plt.subplot(int(len(images) / 2), 3, i + 1)
        plt.imshow(images[i], "gray")
        plt.title(titles[i])
        plt.xticks([]), plt.yticks([])
    plt.show()


def HarrisCornerDetection(image):
    crop_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    crop_gray = cv2.GaussianBlur(crop_gray, (5, 5), 0)
    gray_32 = np.float32(crop_gray)
    dst = cv2.cornerHarris(gray_32, 5, 3, 0.04)
    dst = cv2.dilate(dst, None)
    image[dst > 0.06 * dst.max()] = [255, 0, 0]
    coords = np.argwhere(dst > 0.06 * dst.max())
    yx = coords[:, ::-1]

    return image, yx


def crop_image(pt1, pt2, image, numLabels, image_crop):
    print(f"{pt1[1]} : {pt2[1]}, {pt1[0]} : {pt2[0]}")
    cropped_image = image[pt1[1] : pt2[1], pt1[0] : pt2[0]]
    cv2.imwrite(f"piece_library\\{file_name}_piece{numLabels}.jpg", cropped_image)
    image_crop.append(cropped_image)
    return cropped_image, image_crop


def preserve_points(img, gray, thresh, connected_analysis, yx, file_name):
    """
    Preserve and combine points from multiple detection methods.

    Parameters:
    - img: Original image
    - gray: Grayscale image
    - thresh: Thresholded binary image
    - connected_analysis: Result from cv2.connectedComponentsWithStats()
    - yx: Corner points from Harris detection

    Returns:
    - Comprehensive set of preserved points
    - Visualization of preserved points
    """
    # Unpack connected component analysis
    numLabels, labels, stats, centroids = connected_analysis

    # Create a copy of the image for visualization
    preserved_img = img.copy()

    # Collect points from different methods
    all_points = []

    # 1. Centroids from connected components
    for i in range(1, numLabels):  # Skip background (index 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 1000:  # Filter small components
            centroid = centroids[i]
            all_points.append(
                {
                    "point": (int(centroid[0]), int(centroid[1])),
                    "type": "centroid",
                    "color": (255, 0, 0),  # Blue
                }
            )

    # 2. Harris Corner Detection Points
    for point in yx:
        point_coords = (int(point[0]), int(point[1]))
        all_points.append(
            {
                "point": point_coords,
                "type": "harris_corner",
                "color": (0, 255, 0),  # Green
            }
        )

    # 3. Contour Extreme Points
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        # Extract extreme points
        if len(cnt) > 0:
            extreme_points = [
                tuple(cnt[cnt[:, :, 0].argmin()][0]),  # Leftmost
                tuple(cnt[cnt[:, :, 0].argmax()][0]),  # Rightmost
                tuple(cnt[cnt[:, :, 1].argmin()][0]),  # Topmost
                tuple(cnt[cnt[:, :, 1].argmax()][0]),  # Bottommost
            ]

            for point in extreme_points:
                all_points.append(
                    {
                        "point": point,
                        "type": "contour_extreme",
                        "color": (0, 0, 255),  # Red
                    }
                )

    # 4. Contour Vertices
    simplified_contours = []
    for cnt in contours:
        epsilon = 0.04 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        simplified_contours.append(approx)

    for cnt in simplified_contours:
        for point in cnt:
            point = tuple(point[0])
            all_points.append(
                {
                    "point": point,
                    "type": "contour_vertex",
                    "color": (255, 255, 0),  # Cyan
                }
            )

    # Remove duplicate points
    unique_points = {}
    for point_info in all_points:
        point = point_info["point"]
        if point not in unique_points:
            unique_points[point] = point_info

    # Visualize preserved points
    for point_info in unique_points.values():
        cv2.circle(preserved_img, point_info["point"], 5, point_info["color"], -1)

    # Convert to list of point coordinates
    preserved_point_list = [
        list(point_info["point"]) for point_info in unique_points.values()
    ]

    # Save visualization
    cv2.imwrite(f"Preserved_Points\\{file_name}_preserved_points.jpg", preserved_img)

    return preserved_point_list, preserved_img


def prepare_piece_library():
    folder = "piece_library"
    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            os.remove(file_path)
        except Exception as e:
            print("Failed to delete %s. Reason: %s" % (file_path, e))


# Modify the main execution in the original script

# Use preserved_points instead of internal_points in corner_filter


# Main execution
prepare_piece_library()

file_name = "pieces_puzzle_multiple"
image_crop = []
corner = []
img, gray = load_and_preprocess_image(file_name)
thresh = threshold_image(gray)
numLabels, labels, stats, centroids = find_connected_components(thresh)
cropped_image, image_crop = draw_bounding_boxes_and_centroids(
    img, stats, numLabels, image_crop
)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
detected_img = detect_contours(img, contours)
for i in image_crop:
    corner_detection, yx = HarrisCornerDetection(i.copy())
    corner.append(corner_detection)

preserved_points, preserved_points_img = preserve_points(
    img, gray, thresh, (numLabels, labels, stats, centroids), yx, file_name
)


titles = [
    "Original Image",
    "Binary Image",
    "Detected Form",
    "Cropped Image",
    "Corner Detection",
    "Preserved Points",
]
images = [
    img,
    thresh,
    detected_img,
    image_crop[0],
    corner[0],
    preserved_points_img,
]

display_images(images, titles)
