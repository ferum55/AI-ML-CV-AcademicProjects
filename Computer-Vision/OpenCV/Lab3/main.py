import cv2
import numpy as np

def resize_keep_aspect(img, max_w=1200, max_h=800):
    h, w = img.shape[:2]
    scale = min(max_w / w, max_h / h, 1.0)
    nw, nh = int(w * scale), int(h * scale)
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA)

def show(title, img):
    cv2.imshow(title, resize_keep_aspect(img))

def to_gray(bgr):
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)


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

path = "input.jpg"
img = cv2.imread(path)
if img is None:
    raise FileNotFoundError(f"Не знайдено зображення: {path}")

gray = to_gray(img)


blur_k = 7
blur_box = cv2.blur(img, (blur_k, blur_k))

gauss_k = 7
gauss = cv2.GaussianBlur(img, (gauss_k, gauss_k), sigmaX=0)

median_k = 7
median = cv2.medianBlur(img, median_k)

cmp_smooth = stack_h([img, blur_box, gauss, median])

kernel_sizes = [3, 5, 9, 15]
gauss_variants = [cv2.GaussianBlur(img, (k, k), 0) for k in kernel_sizes]
cmp_kernels = stack_h([img] + gauss_variants)

sobelx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_16S, 0, 1, ksize=3)
abs_sobelx = cv2.convertScaleAbs(sobelx)
abs_sobely = cv2.convertScaleAbs(sobely)
sobel_mag = cv2.addWeighted(abs_sobelx, 0.5, abs_sobely, 0.5, 0)

lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=3)
lap_abs = cv2.convertScaleAbs(lap)

cmp_grad = stack_h([gray, abs_sobelx, abs_sobely, sobel_mag, lap_abs])

gray_blur = cv2.GaussianBlur(gray, (5, 5), 0)
canny1 = cv2.Canny(gray_blur, 50, 150)
canny2 = cv2.Canny(gray_blur, 100, 200)
cmp_canny = stack_h([gray, canny1, canny2])

cmp_edges = stack_h([sobel_mag, lap_abs, canny2])

contours, hierarchy = cv2.findContours(canny2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


contour_vis = img.copy()
cv2.drawContours(contour_vis, contours, -1, (0, 255, 0), 2)



_, thr_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
thr_adapt = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 5
)
cmp_thr = stack_h([gray, thr_otsu, thr_adapt])


k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
eroded = cv2.erode(thr_otsu, k, iterations=1)
dilated = cv2.dilate(thr_otsu, k, iterations=1)
cmp_erode_dilate = stack_h([thr_otsu, eroded, dilated])


opened = cv2.morphologyEx(thr_otsu, cv2.MORPH_OPEN, k, iterations=1)
closed = cv2.morphologyEx(thr_otsu, cv2.MORPH_CLOSE, k, iterations=1)
cmp_open_close = stack_h([thr_otsu, opened, closed])

mgrad = cv2.morphologyEx(thr_otsu, cv2.MORPH_GRADIENT, k)
cmp_mgrad = stack_h([thr_otsu, mgrad])


denoise_bin = cv2.morphologyEx(thr_otsu, cv2.MORPH_OPEN, k, iterations=1)
denoise_bin = cv2.morphologyEx(denoise_bin, cv2.MORPH_CLOSE, k, iterations=1)
cmp_denoise = stack_h([thr_otsu, denoise_bin])

show("1-3 Smoothing compare: original | box | gaussian | median", cmp_smooth)
show("5 Kernel size impact (Gaussian): original + k=3,5,9,15", cmp_kernels)
show("6 Gradients: gray | SobelX | SobelY | SobelMag | Laplacian", cmp_grad)
show("7 Canny: gray | (50,150) | (100,200)", cmp_canny)
show("8 Compare edges: SobelMag | Laplacian | Canny", cmp_edges)
show("9 Contours on Canny", contour_vis)
show("11 Threshold: gray | Otsu | Adaptive", cmp_thr)
show("12 Erode/Dilate: Otsu | Erode | Dilate", cmp_erode_dilate)
show("13 Open/Close: Otsu | Open | Close", cmp_open_close)
show("14 Morph gradient: Otsu | Gradient", cmp_mgrad)
show("15 Binary denoise: Otsu | Denoised", cmp_denoise)

cv2.waitKey(0)
cv2.destroyAllWindows()