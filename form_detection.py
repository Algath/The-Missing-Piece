import cv2
import numpy as np

img = cv2.imread("Image\\piece_puzzle_blackBackGround.jpg")
img = cv2.resize(img, (800, 800))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.resize(gray, (800, 800))


ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
cv2.imwrite("Binary_Image\\piece_puzzle_blackBackGround_binary_version.jpg", thresh)

contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)


for cnt in contours:
    epsilon = 0.0001 * cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, epsilon, True)

    cv2.drawContours(img, [approx], 0, (0, 255, 0), 2)
    cv2.imwrite("Output\\piece_detection.jpg", img)

    """ n = approx.ravel()
    i = 0

    for j in n:
        if i % 2 == 0:
            x = n[i]
            y = n[i + 1]

            string = str(x) + " " + str(y)
            if i == 0:
                cv2.putText(
                    img, "Arrow", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0)
                )
            else:
                cv2.putText(
                    img, string, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255)
                )
        i = i + 1

    cv2.imwrite("Output\\piece_detection_after_flatten.jpg", img) """
