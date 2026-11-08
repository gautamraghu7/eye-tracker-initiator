import cv2
import mediapipe as mp
import serial
import time
import math


# ==========================================
# 1. SETTINGS
# ==========================================

BLUETOOTH_PORT = "COM7"  # Change to your Bluetooth COM port
BAUD_RATE = 115200

CAMERA_ID = 0

# Eye detection thresholds
CLOSED_THRESHOLD = 0.18
OPEN_THRESHOLD = 0.23

# Blink detection
BLINK_MIN_DURATION = 0.06
BLINK_MAX_DURATION = 0.35

# Maximum time between two blinks
DOUBLE_BLINK_WINDOW = 0.5

# Bluetooth heartbeat
HEARTBEAT_INTERVAL = 0.1


# ==========================================
# 2. CONNECT TO ESP32 BLUETOOTH
# ==========================================

print("Connecting to ESP32 Bluetooth...")

esp32 = serial.Serial(
    BLUETOOTH_PORT,
    BAUD_RATE,
    timeout=0.1,
    write_timeout=0.5
)

time.sleep(1)

print("Bluetooth connected!")


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
# 5. BLUETOOTH COMMAND CONTROL
# ==========================================

pinch_active = False

last_command_time = 0


def send_command(pinch):

    global pinch_active
    global last_command_time

    current_time = time.monotonic()

    command = "PINCH\n" if pinch else "RELEASE\n"

    # Send when state changes or heartbeat is due

    if (
        pinch != pinch_active
        or current_time - last_command_time >= HEARTBEAT_INTERVAL
    ):

        esp32.write(command.encode("ascii"))

        esp32.flush()

        pinch_active = pinch

        last_command_time = current_time

        print("BLUETOOTH TX:", command.strip())


# ==========================================
# 6. WEBCAM SETUP
# ==========================================

cap = cv2.VideoCapture(CAMERA_ID)

if not cap.isOpened():

    raise RuntimeError("Could not open webcam!")


# ==========================================
# 7. BLINK CONTROL VARIABLES
# ==========================================

# Current mechanism state
grip_active = False

# Blink detection
blink_start = None

eye_was_closed = False

# Time of first blink
last_blink_time = None

# Display status
status = "WAITING FOR BLINK"


# ==========================================
# 8. MAIN PROGRAM
# ==========================================

try:

    # Always start with mechanism released

    send_command(False)

    print("\n================================")
    print("BLUETOOTH EYE CONTROL STARTED")
    print("================================")

    print("Double blink = Toggle GRIP / RELEASE")

    print("Single blink = No action")

    print("Press Q to quit.")

    print("================================\n")


    while True:

        success, frame = cap.read()

        current_time = time.monotonic()

        if not success:

            print("CAMERA ERROR!")

            send_command(False)

            break


        height, width = frame.shape[:2]


        # Convert webcam image to RGB

        rgb_frame = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )


        # Detect face and landmarks

        results = face_mesh.process(rgb_frame)


        status = "NO FACE DETECTED"


        # ==================================
        # 9. FACE DETECTED
        # ==================================

        if results.multi_face_landmarks:

            landmarks = results.multi_face_landmarks[0].landmark


            # Calculate left eye EAR

            left_ear = calculate_ear(
                landmarks,
                LEFT_EYE,
                width,
                height
            )


            # Calculate right eye EAR

            right_ear = calculate_ear(
                landmarks,
                RIGHT_EYE,
                width,
                height
            )


            # Both eyes must close

            eyes_closed = (
                left_ear < CLOSED_THRESHOLD
                and right_ear < CLOSED_THRESHOLD
            )


            # Both eyes must open

            eyes_open = (
                left_ear > OPEN_THRESHOLD
                and right_ear > OPEN_THRESHOLD
            )


            # ==================================
            # 10. DETECT BLINK START
            # ==================================

            if eyes_closed:

                if not eye_was_closed:

                    blink_start = current_time

                    eye_was_closed = True


                status = "EYES CLOSED"


            # ==================================
            # 11. DETECT BLINK END
            # ==================================

            elif eyes_open:

                if eye_was_closed:

                    blink_duration = (
                        current_time - blink_start
                    )

                    eye_was_closed = False

                    blink_start = None


                    # Accept valid blink duration

                    if (
                        BLINK_MIN_DURATION
                        <= blink_duration
                        <= BLINK_MAX_DURATION
                    ):

                        print(
                            "BLINK DETECTED:",
                            round(blink_duration, 3),
                            "seconds"
                        )


                        # ==================================
                        # 12. DOUBLE BLINK DETECTION
                        # ==================================

                        if (
                            last_blink_time is not None
                            and current_time - last_blink_time
                            <= DOUBLE_BLINK_WINDOW
                        ):

                            print("DOUBLE BLINK DETECTED!")


                            # Toggle mechanism state

                            grip_active = not grip_active


                            if grip_active:

                                send_command(True)

                                status = "GRIP ACTIVATED"

                                print("MECHANISM: GRIP")

                            else:

                                send_command(False)

                                status = "RELEASE ACTIVATED"

                                print("MECHANISM: RELEASE")


                            # Reset double blink detection

                            last_blink_time = None


                        else:

                            # Wait for a second blink

                            last_blink_time = current_time

                            status = "WAITING FOR SECOND BLINK"


                    else:

                        status = "BLINK IGNORED"


                else:

                    status = "EYES OPEN"


            # ==================================
            # 13. EXPIRE SINGLE BLINK CANDIDATE
            # ==================================

            if (
                last_blink_time is not None
                and current_time - last_blink_time
                > DOUBLE_BLINK_WINDOW
            ):

                last_blink_time = None

                status = "WAITING FOR BLINK"


            # ==================================
            # 14. DISPLAY EYE MEASUREMENTS
            # ==================================

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


        # ==================================
        # 15. NO FACE DETECTED
        # ==================================

        else:

            blink_start = None

            eye_was_closed = False

            last_blink_time = None

            grip_active = False

            send_command(False)

            status = "NO FACE - RELEASE"


        # ==================================
        # 16. DISPLAY MECHANISM STATE
        # ==================================

        if grip_active:

            mechanism_status = "MECHANISM: GRIP"

            color = (0, 0, 255)

        else:

            mechanism_status = "MECHANISM: RELEASE"

            color = (0, 255, 0)


        cv2.putText(
            frame,
            status,
            (10, 100),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )


        cv2.putText(
            frame,
            mechanism_status,
            (10, 135),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            color,
            2
        )


        # ==================================
        # 17. DISPLAY CAMERA
        # ==================================

        cv2.imshow(
            "Bluetooth Eye Controlled Mechanism",
            frame
        )


        key = cv2.waitKey(1) & 0xFF


        if key == ord("q"):

            print("Exiting program...")

            break


# ==========================================
# 18. ERROR HANDLING
# ==========================================

except KeyboardInterrupt:

    print("\nProgram interrupted.")


except (serial.SerialException, OSError) as error:

    print("Bluetooth communication error:", error)


# ==========================================
# 19. SAFETY CLEANUP
# ==========================================

finally:

    print("Sending RELEASE command...")

    try:

        esp32.write(b"RELEASE\n")

        esp32.flush()

    except Exception:

        print("Could not send RELEASE. ESP32 watchdog will handle timeout.")


    cap.release()

    face_mesh.close()

    cv2.destroyAllWindows()

    esp32.close()

    print("System stopped safely.")