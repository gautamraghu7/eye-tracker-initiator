import cv2
import mediapipe as mp
import time
import numpy as np
import asyncio
from bleak import BleakClient, BleakScanner

# --- CONFIGURATION ---
TEST_DURATION = 1.5      # Shortened time window in seconds
EAR_THRESHOLD = 0.22   # Threshold calibrated from previous debug steps
EAR_CONSEC_FRAMES = 1  
AUTO_RESET_DELAY = 3   # Seconds to wait after a test finishes before starting a new sequence

# --- BLE CONFIGURATION ---
ESP32_BLE_NAME = "ESP32_Blink_Receiver" 
CHARACTERISTIC_UUID = "6E400002-B5A3-F393-E0A9-E50E24DCCA9E" 

# Eye Landmark Index mappings for MediaPipe
LEFT_EYE_TOP_BOTTOM = [159, 145]
LEFT_EYE_LEFT_RIGHT = [33, 133]
RIGHT_EYE_TOP_BOTTOM = [386, 374]
RIGHT_EYE_LEFT_RIGHT = [362, 263]

def calculate_ear(landmarks, top_bottom_pairs, left_right_pair):
    v1 = np.linalg.norm(np.array([landmarks[top_bottom_pairs[0]].x, landmarks[top_bottom_pairs[0]].y]) - 
                        np.array([landmarks[top_bottom_pairs[1]].x, landmarks[top_bottom_pairs[1]].y]))
    v2 = np.linalg.norm(np.array([landmarks[left_right_pair[0]].x, landmarks[left_right_pair[0]].y]) - 
                        np.array([landmarks[left_right_pair[1]].x, landmarks[left_right_pair[1]].y]))
    return v1 / v2 if v2 > 0 else 0

async def send_blink_count_via_ble(count):
    print(f"[BLE] Scanning for target device named '{ESP32_BLE_NAME}'...")
    device = await BleakScanner.find_device_by_name(ESP32_BLE_NAME)
    
    if device is None:
        print(f"[BLE] Error: Could not find any active BLE device named '{ESP32_BLE_NAME}'. Is it powered on?")
        return False
        
    print(f"[BLE] Connected to {device.address}. Transmitting payload...")
    try:
        async with BleakClient(device) as client:
            payload = bytes([count])
            await client.write_gatt_char(CHARACTERISTIC_UUID, payload)
            print(f"[BLE] Transmission Successful! Sent count: {count}")
            return True
    except Exception as e:
        print(f"[BLE] Transmission Error: {e}")
        return False

# NEW: Asynchronous task to handle transmitting and resetting the state machine automatically
async def process_end_of_sequence(count, state_reset_callback):
    # 1. Send the data over BLE
    await send_blink_count_via_ble(count)
    # 2. Wait for the specified delay window
    await asyncio.sleep(AUTO_RESET_DELAY)
    # 3. Trigger the reset callback to loop back to the start
    state_reset_callback()

async def main():
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
    ble_transmission_triggered = False

    # Helper function to reset tracking variables cleanly
    def reset_sequence():
        nonlocal blink_count, frame_counter, start_time, test_active, ble_transmission_triggered
        blink_count = 0
        frame_counter = 0
        start_time = None
        test_active = True
        ble_transmission_triggered = False
        print("[SYSTEM] Sequence auto-reset complete. Awaiting first blink...")

    print("Looking for face... Timer will start automatically upon your first blink!")

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

            # --- UPDATED TRIGGER: RUN TRANSMISSION AND AUTO-RESET TASK ---
            if not test_active and not ble_transmission_triggered:
                ble_transmission_triggered = True
                # Hands control over to our async sequence worker without freezing the camera view loop
                asyncio.create_task(process_end_of_sequence(blink_count, reset_sequence))

            # --- UI DISPLAY PANEL LAYOUT ---
            cv2.rectangle(frame, (0, 0), (340, 110), (0, 0, 0), -1)
            
            if start_time is None:
                cv2.putText(frame, "STATUS: Awaiting First Blink", (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 150, 0), 2)
                cv2.putText(frame, f"Timer: {time_left:.1f}s (Paused)", (15, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
                cv2.putText(frame, f"Blinks: {blink_count}", (15, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            elif test_active:
                cv2.putText(frame, f"Time Left: {time_left:.1f}s", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"Blinks: {blink_count}", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "SENDING & RESETTING...", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                cv2.putText(frame, f"Sent Payload: {blink_count} blinks", (15, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, f"Resetting in {AUTO_RESET_DELAY}s...", (15, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)

            cv2.imshow("BioHack Save - Dynamic BLE Tracker", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): 
                break
            elif key == ord('r'):
                reset_sequence()
            
            await asyncio.sleep(0.01)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
