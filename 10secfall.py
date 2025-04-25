import cv2
import cvzone
import math
from datetime import datetime
from ultralytics import YOLO
import geocoder
import pywhatkit
import pyautogui
import time

# Initialize the YOLO model
model = YOLO('yolov8s.pt')

# Load class names

classnames = []
with open('classes.txt', 'r') as f:
    classnames = f.read().splitlines()

# Start capturing frames from the webcam
cap = cv2.VideoCapture(0)

# Dictionary to store the last fall detected time for each person
fall_times = {}

# Function to get the current location
def get_current_location():
    # Get location using geocoder
    g = geocoder.ip('me')
    return g.latlng

# Function to send WhatsApp message and location
def send_whatsapp_message(phone_number, message):
    # Send the message instantly
    pywhatkit.sendwhatmsg_instantly(phone_number, message)
    time.sleep(10)  # Wait for WhatsApp web to open and message to load
    pyautogui.press('enter')  # Press enter to send the message

    time.sleep(5)  # Give some time before proceeding to send location

    # Get the current location
    current_location = get_current_location()
    latitude, longitude = current_location

    # Create Google Maps link
    google_maps_link = f"https://www.google.com/maps/search/?api=1&query={latitude},{longitude}"

    # Send the Google Maps link
    location_message = f"Current Location: {google_maps_link}"
    pywhatkit.sendwhatmsg_instantly(phone_number, location_message)
    time.sleep(10)
    pyautogui.press('enter')

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (980, 740))

    # Perform object detection with YOLO
    results = model(frame)

    for info in results:
        parameters = info.boxes
        for box in parameters:
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            confidence = box.conf[0]
            class_detect = box.cls[0]
            class_detect = int(class_detect)
            class_detect = classnames[class_detect]
            conf = math.ceil(confidence * 100)

            # Implement fall detection using the coordinates x1, y1, x2
            height = y2 - y1
            width = x2 - x1
            threshold = height - width

            if conf > 80 and class_detect == 'person':
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)
                
                # Check if person is in fallen state
                if threshold < 0:
                    current_time = datetime.now()
                    if (x1, y1, x2, y2) in fall_times:
                        # Check if the person has been in fallen state for more than 10 seconds
                        if (current_time - fall_times[(x1, y1, x2, y2)]).total_seconds() >= 10:
                            # Send WhatsApp message with location
                            send_whatsapp_message("+917411369136", "Fall Detected! Help needed.")

                            # Update fall time to current time
                            fall_times[(x1, y1, x2, y2)] = current_time
                    else:
                        # If person is in fallen state, record the fall time
                        fall_times[(x1, y1, x2, y2)] = current_time
            else:
                # If person is not detected or confidence is low, remove them from fall_times
                fall_times.pop((x1, y1, x2, y2), None)

    # Display the frame
    cv2.imshow('frame', frame)
    
    # Exit the loop if 't' is pressed
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

