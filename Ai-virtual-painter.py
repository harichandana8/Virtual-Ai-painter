import cv2
import mediapipe as mp
import time
import numpy as np

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cv2.namedWindow("AI Painter", cv2.WINDOW_NORMAL)
cv2.setWindowProperty("AI Painter",
                      cv2.WND_PROP_FULLSCREEN,
                      cv2.WINDOW_FULLSCREEN)


mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    model_complexity=0,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

mpdraw = mp.solutions.drawing_utils

drawColor = (0, 0, 255)
brushThickness = 15
eraserThickness = 60
xp, yp = 0, 0
pasttime = 0

canvas = np.zeros((720, 1280, 3), np.uint8)


while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    lmList = []

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:

            # 🔥 DRAW HAND LANDMARKS (Nodes + Lines)
            mpdraw.draw_landmarks(
                frame,
                handLms,
                mpHands.HAND_CONNECTIONS,
                mpdraw.DrawingSpec(color=(0,255,0), thickness=2, circle_radius=4),
                mpdraw.DrawingSpec(color=(255,0,0), thickness=2)
            )

            for id, lm in enumerate(handLms.landmark):
                h, w, c = frame.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lmList.append((id, cx, cy))

    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]

        
        if lmList[8][2] < lmList[6][2] and lmList[12][2] < lmList[10][2]:
            xp, yp = 0, 0

            if y1 < 100:
                if 0 < x1 < 200:
                    drawColor = (255, 0, 0)
                elif 200 < x1 < 400:
                    drawColor = (0, 255, 0)
                elif 400 < x1 < 600:
                    drawColor = (0, 0, 255)
                elif 600 < x1 < 800:
                    drawColor = (0, 255, 255)
                elif 800 < x1 < 1000:
                    drawColor = (255, 0, 255)
                elif 1000 < x1 < 1280:
                    drawColor = (0, 0, 0)

            cv2.rectangle(frame, (x1, y1-25), (x2, y2+25),
                          drawColor, cv2.FILLED)

        elif lmList[8][2] < lmList[6][2]:

            if xp == 0 and yp == 0:
                xp, yp = x1, y1

            distance = ((x1 - xp)**2 + (y1 - yp)**2)**0.5

            if distance < 100:
                thickness = eraserThickness if drawColor == (0, 0, 0) else brushThickness
                cv2.line(frame, (xp, yp), (x1, y1),
                         drawColor, thickness)
                cv2.line(canvas, (xp, yp), (x1, y1),
                         drawColor, thickness)

            xp, yp = x1, y1

    imgGray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 20, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)

    frame = cv2.bitwise_and(frame, imgInv)
    frame = cv2.bitwise_or(frame, canvas)

    cv2.rectangle(frame, (0,0), (200,100), (255,0,0), -1)
    cv2.rectangle(frame, (200,0), (400,100), (0,255,0), -1)
    cv2.rectangle(frame, (400,0), (600,100), (0,0,255), -1)
    cv2.rectangle(frame, (600,0), (800,100), (0,255,255), -1)
    cv2.rectangle(frame, (800,0), (1000,100), (255,0,255), -1)
    cv2.rectangle(frame, (1000,0), (1280,100), (0,0,0), -1)


    cv2.putText(frame, "BLUE", (50,65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, "GREEN", (240,65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, "RED", (450,65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, "YELLOW", (630,65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)

    cv2.putText(frame, "PINK", (860,65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.putText(frame, "ERASER", (1080,65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    ctime = time.time()
    fps = 1 / (ctime - pasttime)
    pasttime = ctime

    cv2.putText(frame, f'FPS: {int(fps)}',
                (1050,150),
                cv2.FONT_HERSHEY_SIMPLEX,
                1, (255,255,0), 3)

    cv2.imshow("AI Painter", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
