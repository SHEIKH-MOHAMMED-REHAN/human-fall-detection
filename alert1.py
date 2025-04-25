import geocoder
import pywhatkit
import pyautogui
import time

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

# List of phone numbers
phone_numbers = ["+917795721008"]

# Message to send
message = "Hey! This message will be sent instantly."

for number in phone_numbers:
    send_whatsapp_message(number, message)
    time.sleep(10)  # Wait before sending to the next number

if __name__ =="__main__":
    main()

