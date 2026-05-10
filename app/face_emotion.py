from deepface import DeepFace
import cv2

def detect_face_emotion():

    cap = cv2.VideoCapture(0)

    emotion = "neutral"

    frame = None

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        cv2.imshow(
            "Face Emotion Detection - Press SPACE",
            frame
        )

        key = cv2.waitKey(1)

        # SPACE key to capture
        if key == 32:

            try:

                result = DeepFace.analyze(

                    frame,

                    actions=['emotion'],

                    enforce_detection=False
                )

                emotion = result[0][
                    'dominant_emotion'
                ]

            except:

                emotion = "neutral"

            break

        # ESC key to exit
        elif key == 27:
            break

    cap.release()

    cv2.destroyAllWindows()

    return emotion