#Invoke-WebRequest -Uri "https://googleapis.com" -OutFile "face_landmarker.task"
import cv2
import mediapipe as mp
import time
import numpy as np

# --- CONFIGURATION ---
TEST_DURATION = 1.5      # Shortened time window in seconds
EAR_THRESHOLD = 0.22   # Adjusted starting point. Change this based on your screen debug numbers!
EAR_CONSEC_FRAMES = 1  # Lowered to 1 frame to make it highly sensitive for testing

# Eye Landmark Index mappings for MediaPipe
LEFT_EYE_TOP_BOTTOM = [159, 145]
LEFT_EYE_LEFT_RIGHT = [33, 133]
RIGHT_EYE_TOP_BOTTOM = [386, 374]
RIGHT_EYE_LEFT_RIGHT = [362, 263]

def calculate_ear(landmarks, top_bottom_pairs, left_right_pair):
    # Calculate vertical distances
    v1 = np.linalg.norm(np.array([landmarks[top_bottom_pairs[0]].x, landmarks[top_bottom_pairs[0]].y]) - 
                        np.array([landmarks[top_bottom_pairs[1]].x, landmarks[top_bottom_pairs[1]].y]))
    v2 = np.linalg.norm(np.array([landmarks[left_right_pair[0]].x, landmarks[left_right_pair[0]].y]) - 
                        np.array([landmarks[left_right_pair[1]].x, landmarks[left_right_pair[1]].y]))
    # Vertical to Horizontal Ratio
    return v1 / v2 if v2 > 0 else 0

# Initialize MediaPipe Tasks Face Landmarker
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path='face_landmarker.task'),
    running_mode=VisionRunningMode.VIDEO
)

cap = cv2.VideoCapture(0)
blink_count = 0
frame_counter = 0
start_time = None  
test_active = True
mediapipe_timestamp_ms = 0
avg_ear = 0.30 # Default starting placeholder

print("Looking for face... Check the live EAR value in the top right window corner!")

with FaceLandmarker.create_from_options(options) as landmarker:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        mediapipe_timestamp_ms += 33  
        results = landmarker.detect_for_video(mp_image, mediapipe_timestamp_ms)
        
        current_execution_time = time.time()
        if start_time is not None:
            elapsed_time = current_execution_time - start_time
            time_left = max(0.0, TEST_DURATION - elapsed_time)
            if time_left == 0:
                test_active = False
        else:
            time_left = float(TEST_DURATION)

        if results.face_landmarks and len(results.face_landmarks) > 0 and test_active:
            # MediaPipe Tasks nesting format handles list indexing directly 
            face_landmarks = results.face_landmarks[0] 
            
            l_ear = calculate_ear(face_landmarks, LEFT_EYE_TOP_BOTTOM, LEFT_EYE_LEFT_RIGHT)
            r_ear = calculate_ear(face_landmarks, RIGHT_EYE_TOP_BOTTOM, RIGHT_EYE_LEFT_RIGHT)
            avg_ear = (l_ear + r_ear) / 2.0

            if avg_ear < EAR_THRESHOLD:
                frame_counter += 1
            else:
                if frame_counter >= EAR_CONSEC_FRAMES:
                    blink_count += 1
                    if start_time is None:
                        start_time = time.time()
                frame_counter = 0

        # --- UI DISPLAY ---
        cv2.rectangle(frame, (0, 0), (340, 110), (0, 0, 0), -1)
        
        # LIVE DEBUG OVERLAY (Top Right Corner)
        cv2.rectangle(frame, (w - 180, 0), (w, 50), (50, 50, 50), -1)
        cv2.putText(frame, f"LIVE EAR: {avg_ear:.2f}", (w - 165, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if start_time is None:
            cv2.putText(frame, "STATUS: Awaiting First Blink", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
            cv2.putText(frame, f"Timer: {time_left:.1f}s (Paused)", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            cv2.putText(frame, f"Blinks: {blink_count}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        elif test_active:
            cv2.putText(frame, f"Time Left: {time_left:.1f}s", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Blinks: {blink_count}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "TEST FINISHED!", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Final Count: {blink_count} blinks", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'r' to reset", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

        cv2.imshow("BioHack Save - Dynamic Blink Tracker", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'):
            blink_count, frame_counter, start_time, test_active = 0, 0, None, True

cap.release()
cv2.destroyAllWindows()
