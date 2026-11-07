
import cv2
import mediapipe as mp
import serial
import time
import math


# ==========================================
# 1. SETTINGS
# ==========================================

SERIAL_PORT = "COM3"  # Change to your ESP32 port
BAUD_RATE = 115200

CAMERA_ID = 0

CLOSED_THRESHOLD = 0.18
OPEN_THRESHOLD = 0.23

CLOSE_CONFIRM_TIME = 0.7
OPEN_CONFIRM_TIME = 0.15

HEARTBEAT_INTERVAL = 0.1


# ==========================================
# 2. CONNECT TO ESP32
# ==========================================

print("Connecting to ESP32...")

esp32 = serial.Serial(
    SERIAL_PORT,
    BAUD_RATE,
    timeout=0.1
)

time.sleep(2)

print("ESP32 connected!")


# ==========================================
# 3. MEDIAPIPE SETUP
# ==========================================

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)


# ==========================================
# 4. EYE LANDMARKS
# ==========================================

LEFT_EYE = [362, 385, 387, 263, 373, 380]

RIGHT_EYE = [33, 160, 158, 133, 153, 144]


def calculate_ear(landmarks, eye_points, width, height):

    points = []

    for index in eye_points:

        landmark = landmarks[index]

        points.append((
            int(landmark.x * width),
            int(landmark.y * height)
        ))

    vertical_1 = math.dist(points[1], points[5])
    vertical_2 = math.dist(points[2], points[4])

    horizontal = math.dist(points[0], points[3])

    if horizontal == 0:
        return 0

    ear = (
        vertical_1 + vertical_2
    ) / (2.0 * horizontal)

    return ear


# ==========================================
# 5. ESP32 COMMAND CONTROL
# ==========================================

pinch_active = False
last_command_time = 0


def send_command(pinch):

    global pinch_active, last_command_time

    current_time = time.time()

    if pinch:
        command = "PINCH\n"
    else:
        command = "RELEASE\n"

    # Send when state changes or heartbeat is due
    if (
        pinch != pinch_active
        or current_time - last_command_time >= HEARTBEAT_INTERVAL
    ):

        esp32.write(command.encode())

        pinch_active = pinch
        last_command_time = current_time

        print("ESP32:", command.strip())


# ==========================================
# 6. START WEBCAM
# ==========================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():
    send_command(False)
    raise RuntimeError("Could not open webcam!")

print("\nEye control started!")
print("Close eyes = PINCH")
print("Open eyes = RELEASE")
print("Press Q to quit.\n")


# ==========================================
# 7. EYE CONTROL VARIABLES
# ==========================================

closed_since = None
open_since = None

running = True


# ==========================================
# 8. MAIN LOOP
# ==========================================

try:

    # Always start in RELEASE state
    send_command(False)

    while running:

        success, frame = cap.read()

        if not success:

            print("Camera error!")

            send_command(False)
            break

        height, width = frame.shape[:2]

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        results = face_mesh.process(rgb_frame)

        current_time = time.time()

        status = "NO FACE DETECTED"

        if results.multi_face_landmarks:

            landmarks = results.multi_face_landmarks[0].landmark

            # Calculate both eye aspect ratios
            left_ear = calculate_ear(
                landmarks,
                LEFT_EYE,
                width,
                height
            )

            right_ear = calculate_ear(
                landmarks,
                RIGHT_EYE,
                width,
                height
            )

            # Both eyes must be closed
            eyes_closed = (
                left_ear < CLOSED_THRESHOLD
                and right_ear < CLOSED_THRESHOLD
            )

            # Both eyes must be open
            eyes_open = (
                left_ear > OPEN_THRESHOLD
                and right_ear > OPEN_THRESHOLD
            )

            # ----------------------------------
            # EYES CLOSED = PINCH
            # ----------------------------------

            if eyes_closed:

                open_since = None

                if closed_since is None:
                    closed_since = current_time

                if (
                    current_time - closed_since
                    >= CLOSE_CONFIRM_TIME
                ):

                    send_command(True)
                    status = "EYES CLOSED - PINCH"

                else:

                    status = "CONFIRMING EYE CLOSURE"

            # ----------------------------------
            # EYES OPEN = RELEASE
            # ----------------------------------

            elif eyes_open:

                closed_since = None

                if open_since is None:
                    open_since = current_time

                if (
                    current_time - open_since
                    >= OPEN_CONFIRM_TIME
                ):

                    send_command(False)
                    status = "EYES OPEN - RELEASE"

                else:

                    status = "CONFIRMING EYES OPEN"

            # ----------------------------------
            # UNCERTAIN EYE STATE = RELEASE
            # ----------------------------------

            else:

                closed_since = None
                open_since = None

                send_command(False)

                status = "UNCERTAIN - RELEASE"

            # Display eye measurements
            cv2.putText(
                frame,
                f"Left EAR: {left_ear:.3f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"Right EAR: {right_ear:.3f}",
                (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

        else:

            # If face disappears, immediately release
            closed_since = None
            open_since = None

            send_command(False)

        # ----------------------------------
        # DISPLAY STATUS
        # ----------------------------------

        color = (
            (0, 0, 255)
            if pinch_active
            else (0, 255, 0)
        )

        cv2.putText(
            frame,
            status,
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )

        cv2.imshow(
            "Eye Controlled Mechanical System",
            frame
        )

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            running = False


finally:

    # Always attempt to release the mechanism
    try:
        send_command(False)
        esp32.write(b"RELEASE\n")
        esp32.flush()
    except Exception:
        pass

    cap.release()
    face_mesh.close()

    cv2.destroyAllWindows()

    esp32.close()

    print("System stopped.")
