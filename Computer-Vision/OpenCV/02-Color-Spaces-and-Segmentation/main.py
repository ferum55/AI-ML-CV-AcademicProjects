import cv2
import numpy as np

image = cv2.imread("original.jpg")

if image is None:
    print("Image not found")
    exit()

pixel = image[100, 150]
print("Pixel BGR:", pixel)

image[200:400, 300:600] = [0, 0, 255]

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

lower_red1 = np.array([0, 100, 100])
upper_red1 = np.array([10, 255, 255])

lower_red2 = np.array([160, 100, 100])
upper_red2 = np.array([179, 255, 255])

mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

mask = mask1 + mask2

result = cv2.bitwise_and(image, image, mask=mask)

scale = 0.5

image_small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
mask_small = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
result_small = cv2.resize(result, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

cv2.imshow("Original", image_small)
cv2.imshow("Mask", mask_small)
cv2.imshow("Segmented", result_small)

cv2.waitKey(0)
cv2.destroyAllWindows()