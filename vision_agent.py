# vision_agent.py
import cv2
import numpy as np
from ultralytics import YOLO
import pytesseract

class VisionAgent:
    def __init__(self, model_path="yolov8n.pt"):
        # Load YOLOv8n pre-trained model
        self.yolo_model = YOLO(model_path)  # Specify task='detect' for object detection

    def preprocess(self, frame):
        # Resize and preprocess the frame for YOLO
        shape = frame.shape[:2]
        new_shape = (480, 480)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # Ensure padding is divisible by 32
        dw /= 2
        dh /= 2

        img = cv2.resize(frame, new_unpad, interpolation=cv2.INTER_LINEAR)
        img = cv2.copyMakeBorder(
            img, int(round(dh - 0.1)), int(round(dh + 0.1)),
            int(round(dw - 0.1)), int(round(dw + 0.1)),
            cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )

        return img, shape, new_shape

    def detect(self, frame):
        # Preprocess the frame
        processed_frame, ori_shape, processed_shape = self.preprocess(frame)

        # Run YOLO inference
        results = self.yolo_model(processed_frame)

        # Process YOLO results
        objects = []
        boxes = results[0].boxes.xywh.cpu().numpy()  # Bounding boxes in xywh format
        confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
        class_ids = results[0].boxes.cls.cpu().numpy()  # Class IDs
        class_names = results[0].names  # Class names dictionary

        for box, conf, cls_id in zip(boxes, confidences, class_ids):
            if conf > 0.5:  # Confidence threshold
                x, y, w, h = box
                # Scale coordinates back to original frame
                gain = min(processed_shape[0] / ori_shape[0], processed_shape[1] / ori_shape[1])
                pad = (processed_shape[1] - ori_shape[1] * gain) / 2, (processed_shape[0] - ori_shape[0] * gain) / 2
                x = (x - pad[0]) / gain
                y = (y - pad[1]) / gain
                w = w / gain
                h = h / gain
                obj_name = class_names[int(cls_id)]
                objects.append({
                    "object": obj_name,
                    "position": f"x={int(x)}, y={int(y)}",
                    "confidence": float(conf)
                })

        # Detect text using OCR
        text = pytesseract.image_to_string(frame).strip()
        return objects, text

if __name__ == "__main__":
    vision_agent = VisionAgent()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        objects, text = vision_agent.detect(frame)
        print("Objects:", objects)
        print("Text:", text)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()