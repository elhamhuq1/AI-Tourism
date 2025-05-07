import speech_recognition as sr
import cv2
import os
import signal
import sys
import atexit
from dotenv import load_dotenv
import datetime
import pyttsx3
import time
import mimetypes
import requests

load_dotenv()

lastImage = None
cap = None  # Global webcam object

def cleanup():
    global cap
    if cap and cap.isOpened():
        cap.release()
        print("Webcam released (cleanup).")

atexit.register(cleanup)
signal.signal(signal.SIGINT, lambda sig, frame: (cleanup(), sys.exit(0)))

def pictureCapturer():
    global lastImage, cap
    save_dir = "pictures"
    os.makedirs(save_dir, exist_ok=True)

    time.sleep(0.2)  # Give webcam time to adjust
    for _ in range(3):
        cap.read()  # Discard initial frames
    ret, frame = cap.read()

    if ret:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        filename = os.path.join(save_dir, f"pic-{timestamp}.png")
        lastImage = filename
        cv2.imwrite(filename, frame)
        print(f"Picture saved as {filename}")
    else:
        print("Error: Failed to capture image.")

def ai_process():
    global lastImage
    IMG_PATH = lastImage
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    BASE_URL = "https://generativelanguage.googleapis.com"

    mime_type, _ = mimetypes.guess_type(IMG_PATH)
    file_size = os.path.getsize(IMG_PATH)
    display_name = "TEXT"

    headers = {
        "X-Goog-Upload-Protocol": "resumable",
        "X-Goog-Upload-Command": "start",
        "X-Goog-Upload-Header-Content-Length": str(file_size),
        "X-Goog-Upload-Header-Content-Type": mime_type,
        "Content-Type": "application/json"
    }

    metadata = {
        "file": {
            "display_name": display_name
        }
    }

    res = requests.post(
        f"{BASE_URL}/upload/v1beta/files?key={GEMINI_API_KEY}",
        headers=headers,
        json=metadata
    )

    upload_url = res.headers.get("X-Goog-Upload-URL")
    if not upload_url:
        print("Failed to get upload URL")
        return

    with open(IMG_PATH, "rb") as f:
        file_data = f.read()

    upload_headers = {
        "Content-Length": str(file_size),
        "X-Goog-Upload-Offset": "0",
        "X-Goog-Upload-Command": "upload, finalize"
    }

    upload_res = requests.post(upload_url, headers=upload_headers, data=file_data)
    file_info = upload_res.json()
    file_uri = file_info.get("file", {}).get("uri")
    file_name = file_info.get("file", {}).get("name")

    gen_url = f"{BASE_URL}/v1beta/models/gemini-2.5-pro-exp-03-25:generateContent?key={GEMINI_API_KEY}"
    gen_headers = {"Content-Type": "application/json"}

    payload = {
        "contents": [{
            "parts": [
                {"text": "Can you describe this image to a person who is visually impaired. Describe the image in detail with accurate tone. Do not brag saying I'll describe the image. Just describe the image. Don't refer to image as image. Just describe the content of the image. Say it as if i see so and so. 'if' you see there is an aircraft engine or any engine hanging it is the rolls royce trent 1000 engine which is 13000 pounds, and tell about it, other than that do not mention the name of the engine."},
                {"file_data": {"mime_type": mime_type, "file_uri": file_uri}}
            ]
        }]
    }

    gen_res = requests.post(gen_url, headers=gen_headers, json=payload)
    gen_res.raise_for_status()

    response_data = gen_res.json()
    texts = [part["text"] for candidate in response_data.get("candidates", []) for part in candidate.get("content", {}).get("parts", []) if "text" in part]
    resultVal = "\n".join(texts)
    print(resultVal)

    if file_name:
        delete_url = f"{BASE_URL}/v1beta/{file_name}?key={GEMINI_API_KEY}"
        del_res = requests.delete(delete_url)
        if del_res.status_code == 200:
            print(f"Deleted file on Gemini server: {file_name}")
        else:
            print(f"Failed to delete file: {file_name} - {del_res.status_code} {del_res.text}")

    # if os.path.exists(IMG_PATH):
    #     os.remove(IMG_PATH)
    #     print(f"Deleted local image: {IMG_PATH}")

    engine = pyttsx3.init()
    engine.say(resultVal)
    engine.runAndWait()

# Initialize webcam once
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    sys.exit(1)

# Main loop
recognizer = sr.Recognizer()
while True:
    with sr.Microphone() as source:
        print("Listening for 'Hey Google'...")
        try:
            audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            command = recognizer.recognize_google(audio).lower()
            print(f"You said: {command}")
            if "hey google" in command:
                print("Trigger phrase detected.")
                pictureCapturer()
                ai_process()
            else:
                print("No trigger phrase detected.")
        except sr.WaitTimeoutError:
            print("Listening timed out. No speech detected.")
        except sr.UnknownValueError:
            print("Sorry, I didn't catch that.")
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
    time.sleep(1)