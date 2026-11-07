import cv2
import numpy as np

img = np.zeros((600, 800, 3), dtype=np.uint8)

cv2.putText(
    img,
    "CALIBRATION TEST",
    (180, 300),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (0, 255, 0),
    2
)

cv2.imshow("Test Window", img)

print("Window opened. Press Q to close.")

while True:
    key = cv2.waitKey(30) & 0xFF

    if key == ord('q'):
        break

cv2.destroyAllWindows()