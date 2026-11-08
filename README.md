# eyeCatcher

A blink-controlled gripper. Eye blinks are detected by a webcam and translated into servo movement, giving people with limited hand function a hands-free way to grip and release objects.

Built at BioHacks 2026 for the Wearable Devices and Quality of Life track.

## How it works

A webcam tracks the user's eyes and counts deliberate blinks within a set time window. The count is sent over Bluetooth to an ESP32, which drives a servo that tightens or loosens a 3D-printed, glove-style gripper.

| Blinks | Action |
|--------|--------|
| 2 | Extend fingers (release) |
| 3 | Contract fingers (grip) |
| 4 | Full release (failsafe) |

Natural blinks are filtered from intentional ones to prevent accidental triggers.

## Softwares Used
- Python: OpenCV, MediaPipe, Bleak
- ESP32: Servo Library

## Components

- **Software:** webcam eye tracking and blink detection that sends commands over Bluetooth
- **Firmware:** ESP32 receives commands and drives the servo
- **Hardware:** 3D-printed gripper with elastic bands that return the fingers to their open or relaxed position

## Limitations and next steps

- Servo angles are currently fixed in code. Next: a per-user calibration step to set grip and release positions.
- Add a built-in failsafe that loosens the grip automatically if it is too tight or uncomfortable.
- Replace the fixed camera with one that follows the user.

## Team
1. Gautam Raghu
2. Prisha Gupta
3. Tarnagon Saysana
4. Ezekiel Serrano
