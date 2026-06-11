import cv2

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades +
    "haarcascade_frontalface_default.xml"
)


def validate_face(image_path):

    image = cv2.imread(image_path)

    if image is None:
        return False, "Invalid image."

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5
    )

    if len(faces) == 0:
        return False, "No face detected."

    if len(faces) > 1:
        return False, "Multiple faces detected."

    return True, "Face detected."