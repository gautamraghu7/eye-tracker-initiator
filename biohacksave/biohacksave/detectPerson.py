import cv2

# 1. Open a connection to your default webcam (0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

print("Webcam feed active. Press 'q' to quit.")

while True:
    # 2. Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to grab a frame.")
        break

    # 3. Display the raw webcam video feed
    cv2.imshow('Webcam Feed', frame)

    # 4. Break the loop immediately if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 5. Clean up and close everything safely when done
cap.release()
cv2.destroyAllWindows()
