import cv2
import numpy as np

def resize_keep_aspect(img, max_w=1200, max_h=800):
    h, w = img.shape[:2]

    scale = min(max_w / w, max_h / h, 1.0)

    new_w = int(w * scale)
    new_h = int(h * scale)

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

net = cv2.dnn.readNet(
    "yolo/yolov4.weights",
    "yolo/yolov4.cfg"
)

with open("yolo/coco.names") as f:
    classes = [line.strip() for line in f]

image = cv2.imread("images/test.jpg")

height, width = image.shape[:2]

blob = cv2.dnn.blobFromImage(
    image,
    1/255.0,
    (416,416),
    swapRB=True,
    crop=False
)

net.setInput(blob)

layer_names = net.getUnconnectedOutLayersNames()
outputs = net.forward(layer_names)

boxes = []
confidences = []
class_ids = []

for output in outputs:
    for detection in output:

        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]

        if confidence > 0.5:

            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)

            w = int(detection[2] * width)
            h = int(detection[3] * height)

            x = int(center_x - w/2)
            y = int(center_y - h/2)

            boxes.append([x,y,w,h])
            confidences.append(float(confidence))
            class_ids.append(class_id)

indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

for i in indices:

    x,y,w,h = boxes[i]

    label = classes[class_ids[i]]

    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)

    cv2.putText(
        image,
        label,
        (x,y-10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0,255,0),
        2
    )

result = resize_keep_aspect(image, 1200, 800)

cv2.imshow("Detection", result)
cv2.waitKey(0)
cv2.destroyAllWindows()