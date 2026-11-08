
import os
import cv2
from PIL import Image
from google import genai

# Check API key
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise RuntimeError("GEMINI_API_KEY not found!")

# Connect to Gemini
client = genai.Client(api_key=api_key)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam!")

print("Webcam started.")
print("Press S to ask Gemini to identify objects.")
print("Press Q to quit.")

while True:
    success, frame = cap.read()

    if not success:
        print("Could not read webcam frame.")
        break

    cv2.imshow("Gemini Object Recognition", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
        print("\nSending image to Gemini...")

        # Convert OpenCV image to RGB
        small_frame = cv2.resize(frame, (640, 480))

        rgb_frame = cv2.cvtColor(
            small_frame,
            cv2.COLOR_BGR2RGB
        )

        image = Image.fromarray(rgb_frame)

        try:
            response = client.models.generate_content(
                model="gemini-3.5-flash-lite",
                contents=[
                    image,
                    """
                    Identify all clearly visible physical objects.

                    For each object, provide:
                    1. Object name
                    2. Approximate location in the image
                    3. A short description of its appearance.

                    Do not guess if an object is unclear.
                    """
                ]
            )

            print("\n--- GEMINI OBJECT DETECTION ---")
            print(response.text)
            print("--------------------------------\n")

        except Exception as error:
            print("Gemini API error:", error)

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
