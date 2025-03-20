import cv2
import numpy as np
import matplotlib.pyplot as plt

file_name = "duo_piece_1"

img = cv2.imread(f"Image\\{file_name}.jpg")
img = cv2.resize(img, (800, 800))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
gray = cv2.resize(gray, (800, 800))


ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite(f"Binary_Image\\{file_name}_binary_version.jpg", thresh)

analysis = cv2.connectedComponentsWithStats(thresh, 4, cv2.CV_32S)
(numLabels, labels, stats, centroids) = analysis

output = np.zeros(gray.shape, dtype="uint8")
contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


def HarrisCornerDetection():
    gray_32 = np.float32(gray)
    dst = cv2.cornerHarris(gray, 5, 5, 0.04)
    dst = cv2.dilate(dst, None)

    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow("Image", img)
    cv2.waitKey(0)


for i in range(1, numLabels):
    area = stats[i, cv2.CC_STAT_AREA]

    if area > 1000:
        new_img = img.copy()

        x1 = stats[i, cv2.CC_STAT_LEFT]
        y1 = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]

        pt1 = (x1, y1)
        pt2 = (x1 + w, y1 + h)
        (X, Y) = centroids[i]

        cv2.rectangle(new_img, pt1, pt2, (0, 255, 0), 3)
        cv2.circle(new_img, (int(X), int(Y)), 4, (0, 0, 255), -1)

        component = np.zeros(gray.shape, dtype="uint8")
        componentMask = (labels == i).astype("uint8") * 255

        component = cv2.bitwise_or(component, componentMask)
        output = cv2.bitwise_or(output, componentMask)

        """ cv2.imshow("Image", img)
        cv2.imshow("Individual Component", component)
        cv2.imshow("filtered componnets", output)
        cv2.imshow("Cropped Image", cropped_img)
        cv2.waitKey(0) """

detected_img = img.copy()
for cnt in contours:
    epsilon = 0.0001 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    cv2.drawContours(detected_img, [approx], 0, (0, 255, 0), 2)
    cv2.imwrite(f"Output\\{file_name}_detection.jpg", detected_img)

titles = [
    "Original Image",
    "Binary Image",
    "Detected Form",
    "Individual Component",
    "Filtered Components",
]
images = [img, thresh, detected_img, component, output]

for i in range(len(images)):
    plt.subplot(int(len(images) / 2), 3, i + 1)
    plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()

HarrisCornerDetection()
