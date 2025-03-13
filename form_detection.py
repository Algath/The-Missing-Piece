import cv2
import numpy as np
import matplotlib.pyplot as plt

file_name = "pieces_puzzle_multiple"

img = cv2.imread(f"Image\\{file_name}.jpg")
img = cv2.resize(img, (800, 800))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (800, 800))


ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite(f"Binary_Image\\{file_name}_binary_version.jpg", thresh)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


for cnt in contours:
    epsilon = 0.0001 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
    cv2.imwrite(f"Output\\{file_name}_detection.jpg", img)

img_original = cv2.imread(f"Image\\{file_name}.jpg")
img_original = cv2.resize(img_original, (800, 800))
titles = ["Original Image", "Binary Image", "Detected Form"]
images = [img_original, thresh, img]

for i in range(len(images)):
    plt.subplot(1, 3, i + 1)
    plt.imshow(images[i], "gray")
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])

plt.show()
