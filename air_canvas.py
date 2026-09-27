import cv2
import mediapipe as mp

# Setup Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1) # Mirror view
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) # MediaPipe needs RGB
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_lms in results.multi_hand_landmarks:
            # This line draws the DOTS and LINES
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Air Canvas Tracking", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()