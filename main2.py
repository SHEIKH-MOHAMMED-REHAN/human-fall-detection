
from ultralytics import YOLO
import cv2
import pywhatkit
from datetime import datetime
import geocoder
import time
import numpy as np
import argparse

# Load model
model = YOLO("yolov8s.pt")
number = "+919901181363"

# Tracking data
person_boxes = {}          
fall_timers = {}           
stillness_timers = {}      
alert_sent = set()         

# Thresholds
fall_velocity_threshold = 80    
stillness_movement_threshold = 8  
critical_stillness_time = 15    
fall_detection_time = 8         

def send_whatsapp_message(phone_number, message):
    try:
        loc = geocoder.ip('me')
        location_str = f"📍 Location: {loc.city}, {loc.country}" if loc.city else "📍 Location: Unknown"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        full_message = f"🚨 EMERGENCY ALERT\n{timestamp}\n{message}\n{location_str}"
        
        pywhatkit.sendwhatmsg_instantly(
            phone_number,
            full_message,
            wait_time=15,
            tab_close=True
        )
        print(f"✅ WHATSAPP Alert sent to {phone_number}")
    except Exception as e:
        print(f"❌ Failed to send WhatsApp: {e}")

def show_alert_message(message, mode="video"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"🚨 ALERT TRIGGERED [{mode.upper()}]: {message} at {timestamp}")

def calculate_movement(track_id, current_box):
    if track_id not in person_boxes or len(person_boxes[track_id]) < 2:
        return float('inf')
    
    prev_box, _ = person_boxes[track_id][-2]
    
    curr_center = np.array([(current_box[0] + current_box[2])/2, (current_box[1] + current_box[3])/2])
    prev_center = np.array([(prev_box[0] + prev_box[2])/2, (prev_box[1] + prev_box[3])/2])
    
    movement = np.linalg.norm(curr_center - prev_center)
    return movement

def is_sudden_fall(track_id, current_box, frame_shape):
    h, w = frame_shape[:2]
    
    if track_id not in person_boxes:
        person_boxes[track_id] = []
    
    person_boxes[track_id].append((current_box, time.time()))
    
    if len(person_boxes[track_id]) > 15:
        person_boxes[track_id].pop(0)

    if len(person_boxes[track_id]) < 3:
        return False

    current_time = time.time()
    recent = [(box, t) for box, t in person_boxes[track_id] if current_time - t <= 0.3]
    
    if len(recent) < 2:
        return False

    first_box, first_time = recent[0]
    last_box, last_time = recent[-1]
    
    if last_time - first_time == 0:
        return False

    first_center_y = (first_box[1] + first_box[3]) / 2
    last_center_y = (last_box[1] + last_box[3]) / 2
    
    velocity = (last_center_y - first_center_y) / (last_time - first_time)
    return velocity > fall_velocity_threshold

# Parse command line arguments
parser = argparse.ArgumentParser(description="Human Fall Detection System")
parser.add_argument("--video", type=str, help="Path to video file or '0' for webcam")
parser.add_argument("--send-alerts", action="store_true", help="Send real WhatsApp alerts (use with caution)")
args = parser.parse_args()

# Choose input source
if args.video:
    if args.video == "0":
        cap = cv2.VideoCapture(0)
        mode = "webcam"
    else:
        cap = cv2.VideoCapture(args.video)
        mode = "video"
    print(f"🚀 Running on {mode}: {args.video if args.video != '0' else 'webcam'}")
else:
    cap = cv2.VideoCapture(0)
    mode = "webcam"
    print("🚀 Running on webcam (live detection)")

print("Press 'q' to quit.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        if args.video:
            print("🎥 Video ended.")
        else:
            print("❌ Camera not found or disconnected.")
        break

    results = model.track(frame, persist=True, conf=0.5, classes=[0])

    if results[0].boxes is not None:
        boxes = results[0].boxes.xyxy.cpu().numpy()
        ids = results[0].boxes.id.cpu().numpy().astype(int) if results[0].boxes.id is not None else []

        for i, box in enumerate(boxes):
            track_id = ids[i] if i < len(ids) else 0

            if is_sudden_fall(track_id, box, frame.shape):
                if track_id not in fall_timers:
                    print(f"⚠️ Person {track_id} showing fall pattern. Monitoring stillness...")
                    fall_timers[track_id] = time.time()

            movement = calculate_movement(track_id, box)
            
            if track_id in fall_timers:
                if movement < stillness_movement_threshold:
                    if track_id not in stillness_timers:
                        stillness_timers[track_id] = time.time()
                else:
                    if track_id in fall_timers:
                        del fall_timers[track_id]
                    if track_id in stillness_timers:
                        del stillness_timers[track_id]
                    if track_id in alert_sent:
                        alert_sent.remove(track_id)
                    print(f"✅ Person {track_id} moved after fall. Resetting timers.")

            if track_id in stillness_timers:
                still_time = time.time() - stillness_timers[track_id]
                if still_time > critical_stillness_time and track_id not in alert_sent:
                    alert_msg = "⚠️ Person fell and is motionless for 15+ seconds. Possible serious injury!"
                    show_alert_message(alert_msg, mode)
                    if args.send_alerts and mode == "webcam":
                        send_whatsapp_message(number, alert_msg)
                    alert_sent.add(track_id)

            if track_id in fall_timers:
                fall_duration = time.time() - fall_timers[track_id]
                if fall_duration > fall_detection_time:
                    if movement < stillness_movement_threshold and track_id not in alert_sent:
                        alert_msg = "⚠️ Person showing fall pattern and minimal movement. Possible unconsciousness!"
                        show_alert_message(alert_msg, mode)
                        if args.send_alerts and mode == "webcam":
                            send_whatsapp_message(number, alert_msg)
                        alert_sent.add(track_id)

    annotated_frame = results[0].plot()
    cv2.imshow("Advanced Emergency Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("👋 Detection ended.")