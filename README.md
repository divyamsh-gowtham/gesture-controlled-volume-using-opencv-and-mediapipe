# gesture-controlled-volume-using-opencv-and-mediapipe
import cv2
import mediapipe as mp
import math
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from comtypes import CLSCTX_ALL
import numpy as np

# Initialize Mediapipe Hand Tracking
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Initialize Pycaw for system volume control (Windows)
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = interface.QueryInterface(IAudioEndpointVolume)

# Get current volume range
vol_range = volume.GetVolumeRange()
min_vol, max_vol = vol_range[0], vol_range[1]

cap = cv2.VideoCapture(0)

prev_angle = None

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)  # Mirror image
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get index fingertip and wrist
            index_tip = hand_landmarks.landmark[8]
            wrist = hand_landmarks.landmark[0]

            h, w, _ = img.shape
            ix, iy = int(index_tip.x * w), int(index_tip.y * h)
            wx, wy = int(wrist.x * w), int(wrist.y * h)

            # Draw points
            cv2.circle(img, (ix, iy), 10, (255, 0, 0), cv2.FILLED)
            cv2.circle(img, (wx, wy), 10, (0, 255, 0), cv2.FILLED)

            # Calculate angle of index finger relative to wrist
            angle = math.degrees(math.atan2(iy - wy, ix - wx))

            if prev_angle is not None:
                diff = angle - prev_angle

                # Normalize sudden jumps
                if diff > 180:
                    diff -= 360
                elif diff < -180:
                    diff += 360

                if diff > 10:  # rotate right → increase volume
                    current_vol = volume.GetMasterVolumeLevel()
                    volume.SetMasterVolumeLevel(min(current_vol + 1.0, max_vol), None)
                    cv2.putText(img, "Volume UP", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                elif diff < -10:  # rotate left → decrease volume
                    current_vol = volume.GetMasterVolumeLevel()
                    volume.SetMasterVolumeLevel(max(current_vol - 1.0, min_vol), None)
                    cv2.putText(img, "Volume DOWN", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            prev_angle = angle

    cv2.imshow("BMW Gesture Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
         break

cap.release()
cv2.destroyAllWindows()
