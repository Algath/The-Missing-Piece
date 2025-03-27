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
    image[dst > 0.1 * dst.max()] = [255, 0, 0]
    coords = np.argwhere(dst > 0.1 * dst.max())
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


def corner_filter(xy, d_threshold=30, perp_angle_thresh=30, verbose=0):
    N = len(xy)

    dist = scipy.spatial.distance.cdist(xy, xy)
    dist[dist < d_threshold] = 0
    # print(dist[dist < d_threshold])

    def calculate_angle(xy):
        angles = np.zeros((N, N))

        for i in range(N):
            for j in range(i + 1, N):
                point_i, point_j = xy[i], xy[j]
                if point_i[0] == point_j[0]:
                    angle = 90
                else:
                    angle = (
                        np.arctan2(point_j[1] - point_i[1], point_j[0] - point_i[0])
                        * 180
                        / np.pi
                    )

                angles[i, j] = angle
                angles[j, i] = angle
        return angles

    angles = calculate_angle(xy)
    possible_rectangles = []

    def check_possible_rectangle(idx, prev_points=[], dist=dist, angles=angles):
        curr_point = xy[idx]
        depth = len(prev_points)

        if depth == 0:
            right_points_idx = np.nonzero(
                np.logical_and(xy[:, 0] > curr_point[0], dist[idx] > 0)
            )[0]

            if verbose >= 2:
                print("point", idx, curr_point)

            for right_point_idx in right_points_idx:
                check_possible_rectangle(right_point_idx, [idx])

            if verbose >= 2:
                print("---")
            return

        last_angle = angles[idx, prev_points[-1]]
        perp_angle = last_angle - 90
        if perp_angle < 0:
            perp_angle += 180

        if depth in (1, 2):
            if verbose >= 2:
                print(
                    "\t" * depth,
                    "point",
                    idx,
                    curr_point,
                    "angle",
                    last_angle,
                    "perp",
                    perp_angle,
                )
            diff0 = np.abs(angles[idx] - perp_angle) <= perp_angle_thresh
            diff180_0 = np.abs(angles[idx] - (perp_angle + 180)) <= perp_angle_thresh
            diff180_1 = np.abs(angles[idx] - (perp_angle - 180)) <= perp_angle_thresh
            all_diffs = np.logical_or(diff0, np.logical_or(diff180_0, diff180_1))

            diff_to_explore = np.nonzero(np.logical_and(all_diffs, dist[idx] > 0))[0]

            if verbose >= 2:
                print(
                    "\t" * depth,
                    "diff0:",
                    np.nonzero(diff0)[0],
                    "diff180:",
                    np.nonzero(diff180_0)[0],
                    "diff_to_explore:",
                    diff_to_explore,
                )

            for dte_idx in diff_to_explore:
                if (
                    dte_idx not in prev_points
                ):  # unlickly to happen but just to be certain
                    next_points = prev_points[::]
                    next_points.append(idx)

                    check_possible_rectangle(dte_idx, next_points)

        if depth == 3:
            angle41 = angles[idx, prev_points[0]]

            diff0 = np.abs(angle41 - perp_angle) <= perp_angle_thresh
            diff180_0 = np.abs(angle41 - (perp_angle + 180)) <= perp_angle_thresh
            diff180_1 = np.abs(angle41 - (perp_angle - 180)) <= perp_angle_thresh
            dist = dist[idx, prev_points[0]] > 0

            if dist and (diff0 or diff180_0 or diff180_1):
                rect_points = prev_points[::]
                rect_points.append(idx)

                if verbose == 2:
                    print("We have a rectangle:", rect_points)

                already_present = False
                for possible_rectangle in possible_rectangles:
                    if set(possible_rectangle) == set(rect_points):
                        already_present = True
                        break

                if not already_present:
                    possible_rectangles.append(rect_points)

    if verbose >= 2:
        print("Coords")
        print(xy)
        print
        print("Distances")
        print(dist)
        print
        print("Angles")
        print(angles)
        print

    for i in range(N):
        check_possible_rectangle(i)

    if len(possible_rectangles) == 0:
        return None

    def PolyArea(x, y):
        return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

    areas = []
    rectangularness = []
    diff_angles = []

    for r in possible_rectangles:
        points = xy[r]
        areas.append(PolyArea(points[:, 0], points[:, 1]))

        mse = 0
        da = []
        for i1, i2, i3 in [(0, 1, 2), (1, 2, 3), (2, 3, 0), (3, 0, 1)]:
            diff_angle = abs(angles[r[i1], r[i2]] - angles[r[i2], r[i3]])
            da.append(abs(diff_angle - 90))
            mse += (diff_angle - 90) ** 2

        diff_angles.append(da)
        rectangularness.append(mse)

    areas = np.array(areas)
    rectangularness = np.array(rectangularness)

    scores = areas * scipy.stats.norm(0, 150).pdf(rectangularness)
    best_fitting_idxs = possible_rectangles[np.argmax(scores)]
    return xy[best_fitting_idxs]


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
intersections = corner_filter(internal_points)

if intersections is None:
    raise RuntimeError("No rectangle found")
"""
print(intersections[0], intersections[2])

print(yx[0][0], yx[0][1])

img_t = img.copy()
for i in range(len(yx)):
    img_t = cv2.circle(img_t, (int(yx[i][0]), int(yx[i][1])), 4, (255, 0, 0), -1)
"""
img_intersection = img.copy()
img_intersection = cv2.rectangle(
    img_intersection,
    intersections[0].astype(tuple).astype(int),
    intersections[2].astype(tuple).astype(int),
    (0, 255, 0),
    3,
)

titles = [
    "Original Image",
    "Binary Image",
    "Detected Form",
    "Individual Component",
    "Corner Detection",
    "Filtered Points",
    "Intersection",
]
images = [
    img,
    thresh,
    detected_img,
    output,
    corner_detection,
    point_filtered,
    img_intersection,
]

display_images(images, titles)
