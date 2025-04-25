import cv2
import cvzone
import math
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

# List of phone numbers to send alerts to
phone_numbers = ["-------","-------"]

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, (980, 740))

    # Perform object detection with YOLO
    results = model(frame)

    person_detected = False

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
                person_detected = True
                cvzone.cornerRect(frame, [x1, y1, width, height], l=30, rt=6)
                cvzone.putTextRect(frame, f'{class_detect}', [x1 + 8, y1 - 12], thickness=2, scale=2)

            if person_detected and threshold < 0:
                cvzone.putTextRect(frame, 'Fall Detected', [height, width], thickness=2, scale=2)

                # Send WhatsApp alert with location
                alert_message = "Fall detected! Please check the location."
                for number in phone_numbers:
                    send_whatsapp_message(number, alert_message)
                    time.sleep(10)  # Wait before sending to the next number

    # Display the frame
    cv2.imshow('frame', frame)

    # Exit the loop if 't' is pressed
    if cv2.waitKey(1) & 0xFF == ord('t'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

