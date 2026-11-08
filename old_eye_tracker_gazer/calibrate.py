
import cv2
import mediapipe as mp
import numpy as np
import ctypes
import time

MODEL = "face_landmarker.task"

# Get Windows screen resolution
user32 = ctypes.windll.user32
SCREEN_W = user32.GetSystemMetrics(0)
SCREEN_H = user32.GetSystemMetrics(1)

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

def get_point(landmarks, index, w, h):
    p = landmarks[index]
    return np.array([p.x * w, p.y * h])

def get_features(landmarks, w, h):
    # Right iris
    ri = get_point(landmarks, 468, w, h)
    ra = get_point(landmarks, 33, w, h)
    rb = get_point(landmarks, 133, w, h)
    rt = get_point(landmarks, 159, w, h)
    rd = get_point(landmarks, 145, w, h)

    # Left iris
    li = get_point(landmarks, 473, w, h)
    la = get_point(landmarks, 362, w, h)
    lb = get_point(landmarks, 263, w, h)
    lt = get_point(landmarks, 386, w, h)
    ld = get_point(landmarks, 374, w, h)

    rh = (ri[0] - min(ra[0], rb[0])) / max(abs(rb[0]-ra[0]), 1)
    rv = (ri[1] - min(rt[1], rd[1])) / max(abs(rd[1]-rt[1]), 1)

    lh = (li[0] - min(la[0], lb[0])) / max(abs(lb[0]-la[0]), 1)
    lv = (li[1] - min(lt[1], ld[1])) / max(abs(ld[1]-lt[1]), 1)

    return [rh, rv, lh, lv]


targets = [
    (0.15, 0.15), (0.50, 0.15), (0.85, 0.15),
    (0.15, 0.50), (0.50, 0.50), (0.85, 0.50),
    (0.15, 0.85), (0.50, 0.85), (0.85, 0.85)
]

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not detected")

#cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
#cv2.setWindowProperty(
#    "Calibration",
#    cv2.WND_PROP_FULLSCREEN,
#    cv2.WINDOW_FULLSCREEN
#)

cv2.namedWindow("Calibration", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Calibration", 1000, 700)


samples = []
screen_points = []

frame_number = 0

print("LOOK AT EACH DOT.")
print("Press SPACE when ready.")
print("Press Q to cancel.")

with FaceLandmarker.create_from_options(options) as landmarker:

    for i, target in enumerate(targets):

        tx = int(target[0] * 1000)
        ty = int(target[1] * 700)

        canvas = np.zeros(
        (700, 1000, 3),
        dtype=np.uint8
        )

        cv2.circle(canvas, (tx, ty), 18, (0, 255, 0), -1)

        cv2.putText(
            canvas,
            f"Calibration {i+1}/9 - Look at dot and press SPACE",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        while True:
            cv2.imshow("Calibration", canvas)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                raise SystemExit("Calibration cancelled")

            if key == 32:
                break

        print(f"Recording point {i+1}/9...")

        point_features = []

        for j in range(45):
            success, frame = cap.read()

            if not success:
                continue

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
                lm = result.face_landmarks[0]

                if len(lm) >= 478:
                    features = get_features(lm, w, h)

                    if np.all(np.isfinite(features)):
                        point_features.append(features)

            # Show recording progress
            progress = canvas.copy()
            cv2.putText(
                progress,
                f"Recording: {j+1}/45",
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 255),
                2
            )
            cv2.imshow("Calibration", progress)
            cv2.waitKey(1)

        if len(point_features) < 20:
            cap.release()
            cv2.destroyAllWindows()
            raise RuntimeError(
                "Not enough eye detections. Improve lighting and retry."
            )

        samples.append(np.median(point_features, axis=0))
        screen_points.append(target)

        print(f"Point {i+1} saved!")

cap.release()
cv2.destroyAllWindows()

# Train linear calibration model
X = np.array(samples)
Y = np.array(screen_points)

X = np.column_stack([np.ones(len(X)), X])

coefficients, _, _, _ = np.linalg.lstsq(
    X, Y, rcond=None
)

np.savez(
    "eye_calibration.npz",
    coefficients=coefficients,
    screen_width=SCREEN_W,
    screen_height=SCREEN_H
)

print("\nCALIBRATION COMPLETE!")
print("Saved to eye_calibration.npz")
print("Screen:", SCREEN_W, "x", SCREEN_H)
