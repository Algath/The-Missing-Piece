import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.ndimage import maximum_filter as filter_max
from scipy.ndimage import minimum_filter as filter_min
import scipy.ndimage as ndimage
import scipy.spatial
import random
from typing import List, Tuple
from itertools import combinations


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


def first_process(file_name):
    img, gray = load_and_preprocess_image(file_name)
    thresh = threshold_image(gray)
    numLabels, labels, stats, centroids = find_connected_components(thresh)
    return img, numLabels, labels, stats, centroids


def second_process(image_crop, thresh_crop, gray_crop):
    """Apply thresholding to the cropped images."""

    gray = cv2.cvtColor(image_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    thresh_crop.append(thresh)
    gray_crop.append(gray)
    return thresh_crop, gray_crop


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


def detect_contours(img, thresh_crop, contours_crop):
    """Detect and draw contours on the image."""
    detected_img = img.copy()

    contours_crop, _ = cv2.findContours(
        thresh_crop[-1], cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    for cnt in contours_crop:
        epsilon = 0.0001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(detected_img, [approx], 0, (0, 255, 0), 2)

    return detected_img, contours_crop


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
    """Crop the image based on the bounding box + buffer when possible"""

    if pt1[0] - 10 < 0:
        pt1 = (0, pt1[1])
    else:
        pt1 = (pt1[0] - 10, pt1[1])

    if pt1[1] - 10 < 0:
        pt1 = (pt1[0], 0)
    else:
        pt1 = (pt1[0], pt1[1] - 10)

    if pt2[0] + 10 > image.shape[1]:
        pt2 = (image.shape[1], pt2[1])
    else:
        pt2 = (pt2[0] + 10, pt2[1])

    if pt2[1] + 10 > image.shape[0]:
        pt2 = (pt2[0], image.shape[0])
    else:
        pt2 = (pt2[0], pt2[1] + 10)

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

    preserved_point_list = []

    # 1. Centroids from connected components
    for i in range(1, numLabels):  # Skip background (index 0)
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 1000:  # Filter small components
            centroid = centroids[i]
            all_points.append(
                {
                    "point": (int(centroid[0]), int(centroid[1])),
                    "type": "centroid",
                    "color": (0, 0, 255),  # Blue
                }
            )

    # 3. Contour Extreme Points
    for i in range(len(thresh)):
        contours, _ = cv2.findContours(
            thresh[i], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )

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
                            "color": (255, 0, 0),  # Red
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
                        "color": (255, 255, 0),  # Jaune
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


def corner_detection(image, yx, contours, numLabels):
    """Detect corners in the image."""
    moitie = yx.shape[0] // 2
    angle_droit = set()
    tolerance = 1e-2  # Tolérance pour la comparaison d'angle
    segment_len = 0
    norme_ACont1 = []
    test = img.copy()

    for j in range(len(yx)):
        A = yx[j]
        for i in range(len(contours[0])):
            # print(f"yx: {j}, contours: {i}")
            ACont = contours[0][i] - A
            norme_ACont1.append(np.linalg.norm(ACont))
            """ if norme_ACont1 == 504.9831680363218:
                cv2.circle(test, tuple(contours[0][i]), 10, (255, 0, 0), -1)
                cv2.dilate(test, None)
                cv2.imshow("Angle droit", test)
                cv2.waitKey(0)
                cv2.destroyAllWindows() """
            if np.any(contours[0][i] == A):
                if i < 10:
                    B = contours[0][i + 10]
                    C = contours[0][len(contours[0]) - 10 + i]
                elif i > len(contours[0]) - 10 or i + 10 == len(contours[0]):
                    B = contours[0][10 - len(contours[0]) + i]
                    C = contours[0][i - 10]
                else:
                    B = contours[0][i + 10]
                    C = contours[0][i - 10]
                if B is not None and C is not None:
                    """print("A:", A)
                    print("B:", B)
                    print("C:", C)"""

                    AB = np.array(B) - np.array(A)
                    AC = np.array(C) - np.array(A)

                    """ print("AB:", AB)
                    print("AC:", AC) """

                    AB = AB.flatten()
                    AC = AC.flatten()

                    """ print("AB post flatten:", AB)
                    print("AC post flatten:", AC) """

                    norme_AB = np.linalg.norm(AB)
                    norme_AC = np.linalg.norm(AC)

                    if norme_AB == 0 or norme_AC == 0:
                        continue  # Éviter la division par zéro

                    produit_scalaire = AB @ AC

                    angle_radiant = np.arccos(produit_scalaire / (norme_AB * norme_AC))
                    angle_degre = np.degrees(angle_radiant)

                    if abs(angle_degre - 90) < tolerance:
                        angle_droit.add(tuple(A))
    # print("Norme ACont min:", min(norme_ACont1))
    print(f"yx len: {len(yx)}")
    print(f"contours len: {len(contours[0])}")
    for point in angle_droit:
        print("Angle droit:", point)
        cv2.circle(image, tuple(point), 5, (0, 255, 0), -1)
    cv2.imshow("Angle droit", image)
    cv2.imwrite(f"corner_detection/Angle_droit{numLabels}.jpg", image)
    cv2.waitKey(0)


# Main execution
prepare_piece_library()

file_name = "pieces_puzzle_multiple"
image_crop = []
corner = []
thresh_crop = []
gray_crop = []
contours_crop = []

img, numLabels, labels, stats, centroids = first_process(file_name)
cropped_image, image_crop = draw_bounding_boxes_and_centroids(
    img, stats, numLabels, image_crop
)
for i in range(len(image_crop)):
    thresh_crop, gray_crop = second_process(image_crop[i], thresh_crop, gray_crop)
    # print(thresh_crop)

    detected_img, contours_crop = detect_contours(
        image_crop[i], thresh_crop, contours_crop
    )

    corner_detected, yx = HarrisCornerDetection(image_crop[i].copy())
    corner.append(corner_detected)

    # print(contours[5])
    corner_detection(image_crop[i].copy(), yx, contours_crop, i)

    preserved_points, preserved_points_img = preserve_points(
        img,
        gray_crop,
        thresh_crop,
        (numLabels, labels, stats, centroids),
        yx,
        file_name,
    )

# corner_detection(image_crop[9], yx)

titles = [
    "Original Image",
    "Binary Image",
    "Detected Form",
    "Cropped Image",
    "Corner Detection",
    "Preserved Points",
    "Max Rectangle",
]
images = [
    img,
    thresh_crop[5],
    detected_img,
    image_crop[5],
    corner[5],
    preserved_points_img,
]

display_images(images, titles)
