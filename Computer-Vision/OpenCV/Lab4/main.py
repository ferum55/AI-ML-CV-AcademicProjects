import cv2
import numpy as np

def resize_keep_aspect(img, max_w=1200, max_h=800):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def show(title, img):
    cv2.imshow(title, resize_keep_aspect(img))

def stack_h(images, gap=6):
    imgs = []
    for im in images:
        if im.ndim == 2:
            im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
        imgs.append(im)
    h = min(i.shape[0] for i in imgs)
    resized = [cv2.resize(i, (int(i.shape[1] * (h / i.shape[0])), h)) for i in imgs]
    sep = np.zeros((h, gap, 3), dtype=np.uint8)
    out = resized[0]
    for i in resized[1:]:
        out = np.hstack([out, sep, i])
    return out

img1 = cv2.imread("image1.jpg")
img2 = cv2.imread("image2.jpg")

if img1 is None:
    raise FileNotFoundError("Не знайдено image1.jpg")

gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)


_, thresh_simple = cv2.threshold(gray1, 127, 255, cv2.THRESH_BINARY)


thresh_adapt = cv2.adaptiveThreshold(
    gray1, 255,
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    cv2.THRESH_BINARY,
    31,
    5
)


_, thresh_otsu = cv2.threshold(
    gray1, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

cmp_thresh = stack_h([gray1, thresh_simple, thresh_adapt, thresh_otsu])


Z = img1.reshape((-1, 3))
Z = np.float32(Z)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
_, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

center = np.uint8(center)
segmented = center[label.flatten()]
segmented = segmented.reshape((img1.shape))


cmp_seg = stack_h([img1, segmented])


gray_float = np.float32(gray1)
harris = cv2.cornerHarris(gray_float, 2, 3, 0.04)

harris_vis = img1.copy()
harris_vis[harris > 0.01 * harris.max()] = [0, 0, 255]


orb = cv2.ORB_create()

kp1, des1 = orb.detectAndCompute(gray1, None)


orb_vis1 = cv2.drawKeypoints(
    img1, kp1, None,
    color=(0, 255, 0),
    flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
)


if img2 is not None:
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    kp2, des2 = orb.detectAndCompute(gray2, None)


    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    match_vis = cv2.drawMatches(
        img1, kp1,
        img2, kp2,
        matches[:50],
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    show("9-10 ORB Matching", match_vis)


show("1-3 Thresholding compare", cmp_thresh)
show("4-5 KMeans segmentation", cmp_seg)
show("6 Harris corners", harris_vis)
show("7-8 ORB keypoints", orb_vis1)

cv2.waitKey(0)
cv2.destroyAllWindows()