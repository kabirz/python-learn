from yolov8 import YOLOv8
import onnxruntime as ort
import cv2


def face_detect(img, session, yolo):
    model_inputs = session.get_inputs()[0]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_data = yolo.preprocess(img_rgb, model_inputs.shape[2:])
    outputs = session.run(None, {model_inputs.name: img_data})
    return yolo.postprocess(img, outputs)


session = ort.InferenceSession('zhp.onnx')
yolo = YOLOv8(0.5, 0.5)
cam = cv2.VideoCapture(0)

while cam.isOpened():
    ret, img = cam.read()
    if not ret:
        break
    out_img = face_detect(img, session, yolo)
    cv2.imshow("face detect", out_img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()