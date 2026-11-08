
import cv2
import mediapipe as mp
import numpy as np

MODEL = "face_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL),
    running_mode=RunningMode.VIDEO,
    num_faces=1
)

def get_point(landmarks, index, width, height):
    p = landmarks[index]
    return np.array([p.x * width, p.y * height])

def track_eye(landmarks, iris_index, corner1, corner2,
              upper, lower, width, height):

    iris = get_point(landmarks, iris_index, width, height)
    a = get_point(landmarks, corner1, width, height)
    b = get_point(landmarks, corner2, width, height)
    top = get_point(landmarks, upper, width, height)
    bottom = get_point(landmarks, lower, width, height)

    horizontal = (iris[0] - min(a[0], b[0])) / max(
        abs(b[0] - a[0]), 1
    )

    vertical = (iris[1] - min(top[1], bottom[1])) / max(
        abs(bottom[1] - top[1]), 1
    )

    return iris, horizontal, vertical


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Webcam not detected")

frame_number = 0

with FaceLandmarker.create_from_options(options) as landmarker:

    while True:
        success, frame = cap.read()

        if not success:
            break

        height, width, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb
        )

        timestamp = frame_number * 33
        result = landmarker.detect_for_video(mp_image, timestamp)

        if result.face_landmarks:

            landmarks = result.face_landmarks[0]

            if len(landmarks) >= 478:

                # Right iris: landmark 468
                right, rh, rv = track_eye(
                    landmarks, 468, 33, 133, 159, 145,
                    width, height
                )

                # Left iris: landmark 473
                left, lh, lv = track_eye(
                    landmarks, 473, 362, 263, 386, 374,
                    width, height
                )

                # Draw iris centers
                for point in [right, left]:
                    cv2.circle(
                        frame,
                        tuple(point.astype(int)),
                        5,
                        (0, 0, 255),
                        -1
                    )

                # Display measurements
                cv2.putText(
                    frame,
                    f"Right eye H:{rh:.2f} V:{rv:.2f}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Left eye H:{lh:.2f} V:{lv:.2f}",
                    (20, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 255, 0),
                    2
                )

            else:
                cv2.putText(
                    frame,
                    f"Only {len(landmarks)} landmarks detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (0, 0, 255),
                    2
                )

        cv2.imshow("Iris Tracking System", frame)

        frame_number += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
