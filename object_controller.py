
import cv2
import mediapipe as mp
import numpy as np
import time

# ==========================================
# CONFIGURATION
# ==========================================

CAMERA_INDEX = 0
MODEL_PATH = "face_landmarker.task"

# Set this to your ESP32 port later, e.g. "COM3".
# Keep as None while testing without the ESP32.
SERIAL_PORT = None
BAUD_RATE = 115200

# Blue object HSV range.
# These values may need adjustment for your lighting.
LOWER_RED_1 = np.array([0, 100, 70])
UPPER_RED_1 = np.array([10, 255, 255])

LOWER_RED_2 = np.array([170, 100, 70])
UPPER_RED_2 = np.array([180, 255, 255])


MIN_OBJECT_AREA = 500

# Target region as proportions of camera frame.
# This example places it in the centre.
TARGET_WIDTH_RATIO = 0.25
TARGET_HEIGHT_RATIO = 0.25

# Eye closure threshold.
EYE_CLOSED_THRESHOLD = 0.55

# ==========================================
# ESP32 SERIAL CONNECTION
# ==========================================

serial_connection = None

if SERIAL_PORT is not None:
    try:
        import serial
        serial_connection = serial.Serial(
            SERIAL_PORT,
            BAUD_RATE,
            timeout=0.1
        )
        time.sleep(2)
        print("ESP32 connected.")
    except Exception as e:
        print("Serial connection failed:", e)


last_command = None


def send_command(command):
    """Send a command only when the command changes."""
    global last_command

    if command == last_command:
        return

    last_command = command
    print("ESP32 SIGNAL:", command)

    if serial_connection is not None:
        try:
            serial_connection.write((command + "\n").encode())
        except Exception as e:
            print("Serial write error:", e)


# ==========================================
# MEDIAPIPE FACE LANDMARKER
# ==========================================

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    output_face_blendshapes=True
)

landmarker = FaceLandmarker.create_from_options(options)


def get_eye_closure(result):
    """
    Returns average left/right eye-blink score.
    Higher scores indicate more closed eyes.
    """

    if not result.face_blendshapes:
        return None

    categories = result.face_blendshapes[0]

    scores = {
        item.category_name: item.score
        for item in categories
    }

    left = scores.get("eyeBlinkLeft", 0.0)
    right = scores.get("eyeBlinkRight", 0.0)

    return (left + right) / 2.0


# ==========================================
# OBJECT DETECTION
# ==========================================

def detect_object(frame):

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Detect the first red range.
    mask1 = cv2.inRange(
        hsv,
        LOWER_RED_1,
        UPPER_RED_1
    )

    # Detect the second red range.
    mask2 = cv2.inRange(
        hsv,
        LOWER_RED_2,
        UPPER_RED_2
    )

    # Combine both red masks.
    mask = cv2.bitwise_or(mask1, mask2)

    # Remove small noise.
    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, mask

    # Find largest red object.
    largest = max(
        contours,
        key=cv2.contourArea
    )

    if cv2.contourArea(largest) < MIN_OBJECT_AREA:
        return None, mask

    x, y, w, h = cv2.boundingRect(largest)

    center_x = x + w // 2
    center_y = y + h // 2

    return {
        "box": (x, y, w, h),
        "center": (center_x, center_y)
    }, mask

# ==========================================
# CAMERA
# ==========================================

cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

frame_number = 0

# Start in a safe state.
mode = "GRAB"
eyes_closed = False

print("--------------------------------")
print("G = GRAB MODE")
print("R = UNGRAB MODE")
print("O = Disable action mode")
print("Q = Quit")
print("--------------------------------")


try:
    while True:

        ret, frame = cap.read()

        if not ret:
            send_command("STOP")
            continue

        frame_number += 1

        height, width = frame.shape[:2]

        # ----------------------------------
        # TARGET REGION
        # ----------------------------------

        target_w = int(width * TARGET_WIDTH_RATIO)
        target_h = int(height * TARGET_HEIGHT_RATIO)

        target_x1 = width // 2 - target_w // 2
        target_y1 = height // 2 - target_h // 2

        target_x2 = target_x1 + target_w
        target_y2 = target_y1 + target_h

        # ----------------------------------
        # OBJECT TRACKING
        # ----------------------------------

        detected_object, mask = detect_object(frame)

        object_aligned = False

        if detected_object is not None:

            x, y, w, h = detected_object["box"]
            cx, cy = detected_object["center"]

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            cv2.circle(
                frame,
                (cx, cy),
                5,
                (0, 0, 255),
                -1
            )

            # Object centre must be inside target region.
            object_aligned = (
                target_x1 <= cx <= target_x2
                and target_y1 <= cy <= target_y2
            )

        # Draw target region.
        target_color = (
            (0, 255, 0)
            if object_aligned
            else (0, 165, 255)
        )

        cv2.rectangle(
            frame,
            (target_x1, target_y1),
            (target_x2, target_y2),
            target_color,
            2
        )

        # ----------------------------------
        # EYE TRACKING
        # ----------------------------------

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp_ms = frame_number * 33

        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        eye_score = get_eye_closure(result)

        if eye_score is None:
            eyes_closed = False
        else:
            eyes_closed = eye_score >= EYE_CLOSED_THRESHOLD

        # ----------------------------------
        # DECISION SYSTEM
        # ----------------------------------

        command = "STOP"

        # Only allow action if object is detected
        # and positioned correctly.
        if (
            mode in ["GRAB", "UNGRAB"]
            and detected_object is not None
            and object_aligned
            and eyes_closed
            and eye_score is not None
        ):

            if mode == "GRAB":
                command = "GRAB"

            elif mode == "UNGRAB":
                command = "UNGRAB"

        send_command(command)

        # ----------------------------------
        # DISPLAY INFORMATION
        # ----------------------------------

        cv2.putText(
            frame,
            "MODE: " + mode,
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "OBJECT: " + (
                "ALIGNED" if object_aligned
                else "NOT ALIGNED"
            ),
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            target_color,
            2
        )

        eye_text = (
            "CLOSED" if eyes_closed else "OPEN"
        )

        cv2.putText(
            frame,
            "EYES: " + eye_text,
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        score_text = (
            "N/A" if eye_score is None
            else f"{eye_score:.2f}"
        )

        cv2.putText(
            frame,
            "BLINK SCORE: " + score_text,
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            "COMMAND: " + command,
            (20, 175),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2
        )

        cv2.imshow("Object and Eye Controller", frame)
        cv2.imshow("Object Detection Mask", mask)

        # ----------------------------------
        # KEYBOARD CONTROLS
        # ----------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("g"):
            mode = "GRAB"
            send_command("STOP")

        elif key == ord("r"):
            mode = "UNGRAB"
            send_command("STOP")

        elif key == ord("o"):
            mode = "OFF"
            send_command("STOP")

        elif key == ord("q"):
            break

finally:
    send_command("STOP")
    cap.release()
    landmarker.close()
    cv2.destroyAllWindows()

    if serial_connection is not None:
        serial_connection.close()
