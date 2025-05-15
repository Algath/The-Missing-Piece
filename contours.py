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


def detect_contours(img, contours):
    """Detect and draw contours on the image."""
    detected_img = img.copy()
    for cnt in contours:
        epsilon = 0.0001 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(detected_img, [approx], 0, (255, 0, 0), 2)
    return detected_img


# Main execution
file_name = "duo_piece_1"
img, gray = load_and_preprocess_image(file_name)
thresh = threshold_image(gray)
gray_32 = np.float32(gray)

contours_edges, hierarchy = cv2.findContours(
    thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)
contours_all, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

detected_edges = cv2.drawContours(img.copy(), contours_edges, -1, (0, 255, 0), 3)
detected_all = cv2.drawContours(img.copy(), contours_all, -1, (0, 255, 0), 3)


images = [detected_edges, detected_all, gray_32]
titles = ["Contours Edges", "Contours All", "gray_32"]

for i in range(len(images)):
    plt.subplot(int(len(images) / 2), 3, i + 1)
    plt.imshow(images[i])
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
