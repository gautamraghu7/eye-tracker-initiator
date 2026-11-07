
import cv2
import mediapipe as mp
import numpy as np
import pyautogui

# Load calibration
data = np.load("eye_calibration.npz")

coefficients = data["coefficients"]

SCREEN_W = int(data["screen_width"])
SCREEN_H = int(data["screen_height"])

pyautogui.PAUSE = 0

# MediaPipe setup
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path="face_landmarker.task"
    ),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

def get_point(landmarks, index, w, h):
    p = landmarks[index]
    return np.array([p.x * w, p.y * h])

def get_features(landmarks, w, h):

    ri = get_point(landmarks, 468, w, h)
    ra = get_point(landmarks, 33, w, h)
    rb = get_point(landmarks, 133, w, h)
    rt = get_point(landmarks, 159, w, h)
    rd = get_point(landmarks, 145, w, h)

    li = get_point(landmarks, 473, w, h)
    la = get_point(landmarks, 362, w, h)
    lb = get_point(landmarks, 263, w, h)
    lt = get_point(landmarks, 386, w, h)
    ld = get_point(landmarks, 374, w, h)

    rh = (ri[0] - min(ra[0], rb[0])) / max(
        abs(rb[0] - ra[0]), 1
    )

    rv = (ri[1] - min(rt[1], rd[1])) / max(
        abs(rd[1] - rt[1]), 1
    )

    lh = (li[0] - min(la[0], lb[0])) / max(
        abs(lb[0] - la[0]), 1
    )

    lv = (li[1] - min(lt[1], ld[1])) / max(
        abs(ld[1] - lt[1]), 1
    )

    return np.array([1, rh, rv, lh, lv])


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam unavailable")

smooth_x = None
smooth_y = None

alpha = 0.10
frame_number = 0

print("EYE CONTROL ACTIVE")
print("Move your eyes to control the cursor.")
print("Press Q to quit.")

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:

        success, frame = cap.read()

        if not success:
            break

        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = frame_number * 33

        result = landmarker.detect_for_video(
            image, timestamp
        )

        frame_number += 1

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            if len(landmarks) >= 478:

                features = get_features(landmarks, w, h)

                # Predict normalized screen position
                prediction = features @ coefficients

                # Increase vertical sensitivity
                prediction[1] = 0.5 + (prediction[1] - 0.5) * 1.35

                # Convert to screen coordinates
                target_x = int(np.clip(
                    prediction[0], 0, 1
                ) * (SCREEN_W - 1))

                target_y = int(np.clip(
                    prediction[1], 0, 1
                ) * (SCREEN_H - 1))

                # Smooth cursor movement
                if smooth_x is None:
                    smooth_x = target_x
                    smooth_y = target_y

                smooth_x += alpha * (target_x - smooth_x)
                smooth_y += alpha * (target_y - smooth_y)

                pyautogui.moveTo(
                    int(smooth_x),
                    int(smooth_y)
                )

                cv2.putText(
                    frame,
                    f"Gaze: {target_x}, {target_y}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

        cv2.imshow("Eye Controlled Mouse", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
