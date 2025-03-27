import cv2
import numpy as np
import matplotlib.pyplot as plt
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


def draw_bounding_boxes_and_centroids(img, stats, centroids, numLabels):
    """Draw bounding boxes and centroids on the image."""
    output = np.zeros(img.shape[:2], dtype="uint8")
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
            (X, Y) = centroids[i]

            cv2.rectangle(new_img, pt1, pt2, (0, 255, 0), 3)
            cv2.circle(new_img, (int(X), int(Y)), 4, (0, 0, 255), -1)

            componentMask = (labels == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)
    return output, pt1, pt2


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
    gray_32 = np.float32(gray)
    dst = cv2.cornerHarris(gray_32, 5, 5, 0.04)
    dst = cv2.dilate(dst, None)
    image[dst > 0.06 * dst.max()] = [255, 0, 0]
    coords = np.argwhere(dst > 0.06 * dst.max())
    yx = coords[:, ::-1]
    """ data = dst.copy()
     data[data < 0.3 * dst.max()] = 0

    data_max = filter_max(data, 5)
    maxima = data == data_max
    data_min = filter_min(data, 5)
    diff = (data_max - data_min) > 100
    maxima[diff == 0] = 0

    labeled, num_objects = ndimage.label(maxima)
    slices = ndimage.find_objects(labeled)
    yx = np.array(ndimage.center_of_mass(data, labeled, range(1, num_objects + 1))) """
    return image, yx


def point_filter(xy, pt1, pt2, image):
    image = cv2.rectangle(image, pt1, pt2, (0, 255, 0), 3)
    internal_points = []
    external_points = []

    for i in range(len(xy)):
        point = (int(xy[i][0]), int(xy[i][1]))
        if pt1[0] <= xy[i][0] <= pt2[0] and pt1[1] <= xy[i][1] <= pt2[1]:
            image = cv2.circle(image, point, 4, (255, 0, 0), -1)
            internal_points.append(point)
        else:
            image = cv2.circle(image, point, 4, (0, 0, 255), -1)
            external_points.append(point)
    return image, internal_points, external_points


def corner_filter(xy, d_threshold=30, angle_thresh=30, verbose=0):
    """Efficiently detect rectangular shapes among points using vectorized operations.

    Parameters:
    - xy: List or array of points to analyze
    - d_threshold: Minimum distance between points
    - angle_thresh: Threshold for angle deviation
    - verbose: Verbosity level for debugging

    Returns:
    - Best fitting rectangle points or None"""

    # Convert to numpy array and ensure 2D
    xy = np.asarray(xy)

    # Quick exit conditions
    if len(xy) < 4:
        return None

    # Compute pairwise distances efficiently
    dist_matrix = scipy.spatial.distance_matrix(xy, xy)

    # Compute angle matrix more efficiently
    def fast_angle_matrix(points):
        """Compute angle matrix using vectorized operations."""
        dx = points[:, 0][:, np.newaxis] - points[:, 0]
        dy = points[:, 1][:, np.newaxis] - points[:, 1]

        # Use numpy's arctan2 for efficient angle calculation
        angles = np.arctan2(dy, dx) * 180 / np.pi
        return np.abs(angles)

    angle_matrix = fast_angle_matrix(xy)

    def is_perpendicular(angle1, angle2, thresh=angle_thresh):
        """Check if two angles are close to perpendicular."""
        # Compute the absolute angular difference
        diff = np.abs(np.abs(angle1 - angle2) - 90)
        return diff <= thresh

    def find_rectangles():
        """Find potential rectangles using efficient search."""
        rectangles = []

        # Prune distance matrix to only close points
        close_mask = (dist_matrix > 0) & (
            dist_matrix < d_threshold * 3
        )  # Limit search space

        for i in range(len(xy)):
            # Find potential first points
            first_points = np.where(close_mask[i])[0]

            for j in first_points:
                if j <= i:
                    continue

                # Find points forming first side
                side1 = dist_matrix[i, j]

                # Iterate through potential perpendicular points
                for k in first_points:
                    if k <= j:
                        continue

                    # Check if points form a close to perpendicular angle
                    if is_perpendicular(angle_matrix[i, j], angle_matrix[j, k]):
                        # Check third point is far enough
                        side2 = dist_matrix[j, k]

                        # Find fourth point to close rectangle
                        for l in first_points:
                            if l <= k:
                                continue

                            # Check if points form a rectangle
                            if (
                                is_perpendicular(angle_matrix[j, k], angle_matrix[k, l])
                                and np.abs(dist_matrix[k, l] - side1) < side1 * 0.2
                                and dist_matrix[l, i] > 0
                            ):
                                rectangle = [i, j, k, l]
                                rectangles.append(rectangle)

        return rectangles

    def score_rectangles(rectangles):
        """Score rectangles based on area and shape."""
        if not rectangles:
            return None

        def polygon_area(rectangle):
            """Compute polygon area using shoelace formula."""
            points = xy[rectangle]
            x = points[:, 0]
            y = points[:, 1]
            return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

        def rectangularness(rectangle):
            """Compute how close the shape is to a perfect rectangle."""
            points = xy[rectangle]
            angles = np.abs(
                np.diff(
                    np.arctan2(
                        np.diff(points[:, 1], append=points[0, 1]),
                        np.diff(points[:, 0], append=points[0, 0]),
                    )
                )
                * 180
                / np.pi
            )
            return np.mean(np.abs(angles - 90))

        # Compute scores
        areas = np.array([polygon_area(rect) for rect in rectangles])
        rectangularness_scores = np.array(
            [rectangularness(rect) for rect in rectangles]
        )

        # Combined scoring
        scores = areas / (rectangularness_scores + 1e-5)
        best_idx = np.argmax(scores)

        return xy[rectangles[best_idx]]

    # Main processing
    potential_rectangles = find_rectangles()
    best_rectangle = score_rectangles(potential_rectangles)

    return best_rectangle


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


# Modify the main execution in the original script

# Use preserved_points instead of internal_points in corner_filter


# Main execution
file_name = "pieces_puzzle_multiple"
img, gray = load_and_preprocess_image(file_name)
thresh = threshold_image(gray)
numLabels, labels, stats, centroids = find_connected_components(thresh)
output, pt1, pt2 = draw_bounding_boxes_and_centroids(img, stats, centroids, numLabels)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
detected_img = detect_contours(img, contours)
corner_detection, yx = HarrisCornerDetection(img.copy())
point_filtered, internal_points, external_points = point_filter(
    yx, pt1, pt2, img.copy()
)

preserved_points, preserved_points_img = preserve_points(
    img, gray, thresh, (numLabels, labels, stats, centroids), yx, file_name
)

intersections = corner_filter(preserved_points)
""" intersections = corner_filter(internal_points)

if intersections is None:
    raise RuntimeError("No rectangle found")

img_intersection = img.copy()
img_intersection = cv2.rectangle(
    img_intersection,
    intersections[0].astype(tuple).astype(int),
    intersections[2].astype(tuple).astype(int),
    (0, 255, 0),
    3,
) """

cv2.imshow("Piece Edges", preserved_points)
cv2.waitKey(0)

titles = [
    "Original Image",
    "Binary Image",
    "Detected Form",
    "Individual Component",
    "Corner Detection",
    "Filtered Points",
    "piece_edges",
]
images = [
    img,
    thresh,
    detected_img,
    output,
    corner_detection,
    point_filtered,
    preserve_points,
]

display_images(images, titles)
