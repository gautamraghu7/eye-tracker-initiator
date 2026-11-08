
import os
import cv2
import time

from PIL import Image
from ultralytics import YOLO
from google import genai


# ==========================================
# 1. GEMINI SETUP
# ==========================================

api_key = os.getenv("GEMINI_API_KEY")

client = None

if api_key:
    client = genai.Client(api_key=api_key)
    print("Gemini connected.")
else:
    print("Gemini API key not found. YOLO will still work.")


# ==========================================
# 2. LOAD YOLO MODEL
# ==========================================

print("Loading YOLO model...")

yolo = YOLO("yolo11n.pt")

print("YOLO loaded successfully!")


# ==========================================
# 3. OPEN WEBCAM
# ==========================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam!")

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nWebcam started.")
print("YOLO is detecting objects continuously.")
print("Press S to ask Gemini to identify objects.")
print("Press Q to quit.\n")


# ==========================================
# 4. MAIN LOOP
# ==========================================

previous_time = time.time()

try:

    while True:

        success, frame = cap.read()

        if not success:
            print("Could not read webcam frame.")
            break


        # ==================================
        # YOLO CONTINUOUS OBJECT DETECTION
        # ==================================

        results = yolo.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=640,
            conf=0.35,
            verbose=False
        )

        # Draw bounding boxes and tracking IDs
        annotated_frame = results[0].plot()


        # ==================================
        # DISPLAY OBJECT INFORMATION
        # ==================================

        boxes = results[0].boxes

        if boxes is not None:

            for box in boxes:

                class_id = int(box.cls[0])

                object_name = yolo.names[class_id]

                x1, y1, x2, y2 = map(
                    int,
                    box.xyxy[0].tolist()
                )

                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2

                confidence = float(box.conf[0])

                # Display object center coordinates
                print(
                    f"\rDetected: {object_name} | "
                    f"Center: ({center_x}, {center_y}) | "
                    f"Confidence: {confidence:.2f}    ",
                    end=""
                )


        # ==================================
        # CALCULATE FPS
        # ==================================

        current_time = time.time()

        fps = 1 / max(
            current_time - previous_time,
            0.0001
        )

        previous_time = current_time

        cv2.putText(
            annotated_frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )


        # ==================================
        # DISPLAY VIDEO
        # ==================================

        cv2.imshow(
            "YOLO + Gemini Object Recognition",
            annotated_frame
        )

        key = cv2.waitKey(1) & 0xFF


        # ==================================
        # GEMINI ON DEMAND
        # ==================================

        if key == ord("s"):

            if client is None:
                print("\nGemini unavailable: set GEMINI_API_KEY.")
                continue

            print("\nSending image to Gemini...")

            small_frame = cv2.resize(
                frame,
                (640, 480)
            )

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
                        3. Short description of its appearance.

                        Do not guess if an object is unclear.
                        """
                    ]
                )

                print("\n--- GEMINI OBJECT RECOGNITION ---")
                print(response.text)
                print("--------------------------------\n")

            except Exception as error:

                print("Gemini API error:", error)


        # ==================================
        # QUIT
        # ==================================

        elif key == ord("q"):
            print("\nClosing program...")
            break


finally:

    cap.release()
    cv2.destroyAllWindows()

    print("Webcam stopped.")
