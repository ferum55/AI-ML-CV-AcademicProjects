import cv2

image = cv2.imread("original.jpg")

if image is None:
    print("Image not found")
    exit()

print("Image shape:", image.shape)

height, width, channels = image.shape
print("Height:", height)
print("Width:", width)
print("Channels:", channels)

cv2.imwrite("copy.jpg", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

cv2.namedWindow("Original", cv2.WINDOW_NORMAL)
cv2.namedWindow("Grayscale", cv2.WINDOW_NORMAL)

cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)

cv2.waitKey(0)
cv2.destroyAllWindows()
