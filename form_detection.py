import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    for i in range(1, numLabels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area > 1000:
            new_img = img.copy()
            x1, y1, w, h = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            pt1, pt2 = (x1, y1), (x1 + w, y1 + h)
            (X, Y) = centroids[i]

            cv2.rectangle(new_img, pt1, pt2, (0, 255, 0), 3)
            cv2.circle(new_img, (int(X), int(Y)), 4, (0, 0, 255), -1)

            componentMask = (labels == i).astype("uint8") * 255
            output = cv2.bitwise_or(output, componentMask)
    return output

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

def HarrisCornerDetection(gray, img):
    """Perform Harris Corner Detection."""
    gray_32 = np.float32(gray)
    dst = cv2.cornerHarris(gray, 5, 5, 0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]
    cv2.imshow("Image", img)
    cv2.waitKey(0)

# Main execution
file_name = "pieces_puzzle_multiple"
img, gray = load_and_preprocess_image(file_name)
thresh = threshold_image(gray)
numLabels, labels, stats, centroids = find_connected_components(thresh)
output = draw_bounding_boxes_and_centroids(img, stats, centroids, numLabels)
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
detected_img = detect_contours(img, contours)

titles = [
    "Original Image",
    "Binary Image",
    "Detected Form",
    "Individual Component",
    "Filtered Components",
]
images = [img, thresh, detected_img, output]

display_images(images, titles)
HarrisCornerDetection(gray, img)
