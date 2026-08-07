import cv2
import numpy as np


cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    raise RuntimeError("Не вдалося відкрити камеру")


cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

ret, prev_frame = cap.read()
if not ret:
    cap.release()
    raise RuntimeError("Не вдалося зчитати перший кадр")


h, w = prev_frame.shape[:2]
roi_x, roi_y = int(w * 0.25), int(h * 0.25)
roi_w, roi_h = int(w * 0.5), int(h * 0.5)

prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
prev_gray = cv2.GaussianBlur(prev_gray, (5, 5), 0)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = 30.0
out_size = (prev_frame.shape[1], prev_frame.shape[0])
writer = cv2.VideoWriter("output.mp4", fourcc, fps, out_size)

if not writer.isOpened():
    cap.release()
    raise RuntimeError("Не вдалося відкрити VideoWriter для output.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)


    roi_prev = prev_gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]
    roi_curr = gray[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w]

    diff = cv2.absdiff(roi_prev, roi_curr)

    _, motion_mask = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel, iterations=1)
    motion_mask = cv2.dilate(motion_mask, kernel, iterations=1)

    contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    vis = frame.copy()


    cv2.rectangle(vis, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 0), 2)


    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        x, y, cw, ch = cv2.boundingRect(c)

        cv2.rectangle(
            vis,
            (roi_x + x, roi_y + y),
            (roi_x + x + cw, roi_y + y + ch),
            (0, 0, 255),
            2
        )


    motion_vis = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
    motion_vis_full = vis.copy()
    motion_vis_full[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w] = cv2.addWeighted(
        motion_vis_full[roi_y:roi_y + roi_h, roi_x:roi_x + roi_w],
        0.6,
        motion_vis,
        0.4,
        0
    )

    cv2.imshow("Motion detection (ROI overlay)", motion_vis_full)
    writer.write(vis)


    prev_gray = gray

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
writer.release()
cv2.destroyAllWindows()