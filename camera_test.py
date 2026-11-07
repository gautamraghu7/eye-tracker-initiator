
import cv2

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Webcam not detected!")
    exit()

print("Webcam started! Press Q to quit.")

while True:
    ret, frame = cap.read()

    if not ret:
        print("Couldn't read webcam.")
        break

    cv2.imshow("Eye Tracking - Camera Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
