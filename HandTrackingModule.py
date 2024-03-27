import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, complexity=1, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands,self.complexity,
                                        self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        hand_labels = []

        if self.results.multi_hand_landmarks:
            for handLms, handType in zip(self.results.multi_hand_landmarks, self.results.multi_handedness):
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

                # Determine if hand is left or right
                label = handType.classification[0].label  # 'Right' or 'Left'
                hand_labels.append(label)  # Append the label to the list

        return hand_labels, img

    def findposition(self,img, handNo=0, draw=True):

        landmarkPositions = []
        if self.results.multi_hand_landmarks:
            handLms = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(handLms.landmark):  # Grabbing relevant information of finger id landmarks
                # print(id, lm)
                h, w, c = img.shape  # Converting ratios to screen size
                cx, cy = int(lm.x * w), int(lm.y * h)
                # print(id, cx, cy)  # Cleaned up numbers
                landmarkPositions.append([id, cx, cy])  # adding to list
                # if id == 4: # Checking id Detection
                if draw:
                    cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

        return landmarkPositions

def main():
    # Fps initialization
    pTime = 0
    cTime = 0

    cap = cv2.VideoCapture(0)
    detector = handDetector()  # define parameters

    while True:
        success, img = cap.read()
        hands, img = detector.findHands(img)
        landmarkPositions = detector.findposition(img)
        if len(landmarkPositions) != 0:
            print(landmarkPositions[4])  # landmark/significant position we want
        # Fps Updating
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3,
                    (255, 0, 255), 3)  # Prints FPS

        cv2.imshow("Image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()


