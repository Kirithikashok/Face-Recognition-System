import os
import cv2
import numpy as np

MODEL_PATH = "embeddings/lbph_model.yml"
LABELS_PATH = "embeddings/labels.npy"

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def train_model():

    faces = []
    labels = []

    label_dict = {}

    current_label = 0

    dataset_path = "dataset"

    for person_name in os.listdir(dataset_path):

        person_folder = os.path.join(
            dataset_path,
            person_name
        )

        if not os.path.isdir(person_folder):
            continue

        label_dict[current_label] = person_name

        for image_name in os.listdir(person_folder):

            image_path = os.path.join(
                person_folder,
                image_name
            )

            img = cv2.imread(
                image_path,
                cv2.IMREAD_GRAYSCALE
            )

            if img is None:
                continue

            detected = face_cascade.detectMultiScale(
                img,
                scaleFactor=1.1,
                minNeighbors=5
            )

            for (x, y, w, h) in detected:

                roi = img[y:y+h, x:x+w]

                roi = cv2.resize(
                    roi,
                    (200, 200)
                )

                faces.append(roi)

                labels.append(
                    current_label
                )

        current_label += 1

    if len(faces) == 0:
        return 0

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.train(
        faces,
        np.array(labels)
    )

    os.makedirs(
        "embeddings",
        exist_ok=True
    )

    recognizer.save(
        MODEL_PATH
    )

    np.save(
        LABELS_PATH,
        label_dict
    )

    return len(faces)