import cv2
import numpy as np

from utils.logger import save_log

MODEL_PATH = "embeddings/lbph_model.yml"
LABELS_PATH = "embeddings/labels.npy"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def start_recognition():

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read(
        MODEL_PATH
    )

    labels = np.load(
        LABELS_PATH,
        allow_pickle=True
    ).item()

    cap = cv2.VideoCapture(0)

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        gray = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2GRAY
        )

        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5
        )

        for (x, y, w, h) in faces:

            roi = gray[
                y:y+h,
                x:x+w
            ]

            roi = cv2.resize(
                roi,
                (200, 200)
            )

            label, confidence = recognizer.predict(
                roi
            )

            if confidence < 80:

                name = labels[label]

                score = round(
                    100 - confidence,
                    2
                )

                save_log(
                    name,
                    score
                )

            else:

                name = "Unknown"

                score = round(
                    100 - confidence,
                    2
                )

            cv2.rectangle(
                frame,
                (x, y),
                (x+w, y+h),
                (0, 255, 0),
                2
            )

            cv2.putText(
                frame,
                f"{name} {score:.2f}%",
                (x, y-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

        cv2.imshow(
            "Face Recognition",
            frame
        )

        key = cv2.waitKey(1)

        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()