from ultralytics import YOLO

# Load pretrained YOLOv8 model (downloads automatically if missing)
model = YOLO("yolov8n.pt")  # nano = small, fast

def detect_fall(frame):
    """
    Runs YOLOv8 detection on a frame.
    Returns (fall_detected: bool, bbox: (x1,y1,x2,y2) or None).
    """
    results = model(frame, verbose=False)
    fall_detected, bbox = False, None

    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            if label == "person":
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                w, h = x2 - x1, y2 - y1

                # Simple fall heuristic: lying down (width > height)
                if w > h:
                    fall_detected = True
                    bbox = (x1, y1, x2, y2)

    return fall_detected, bbox