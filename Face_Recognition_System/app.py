import os
import shutil

from flask import Flask
from flask import render_template
from flask import request
from flask import redirect
from flask import url_for

from utils.face_validation import validate_face
from utils.augment import augment_image
from utils.trainer import train_model
from utils.recognizer import start_recognition
from utils.database import init_db

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
DATASET_FOLDER = "dataset"

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Create required folders automatically
folders = [
    "uploads",
    "dataset",
    "embeddings",
    "unknown_faces"
]

for folder in folders:
    os.makedirs(folder, exist_ok=True)

init_db()


@app.route("/")
def home():

    return render_template(
        "upload.html"
    )


@app.route("/upload", methods=["POST"])
def upload():

    name = request.form["name"].strip()

    file = request.files["image"]

    if name == "":
        return render_template(
            "result.html",
            message="Please enter a name."
        )

    if file.filename == "":
        return render_template(
            "result.html",
            message="Please select an image."
        )

    upload_path = os.path.join(
        "uploads",
        "current_upload.jpg"
    )

    file.save(upload_path)

    valid, msg = validate_face(
        upload_path
    )

    if not valid:

        os.remove(upload_path)

        return render_template(
            "result.html",
            message=msg
        )

    person_folder = os.path.join(
        DATASET_FOLDER,
        name
    )

    os.makedirs(
        person_folder,
        exist_ok=True
    )

    augment_image(
        upload_path,
        person_folder
    )

    os.makedirs(
        "static/registered_users",
        exist_ok=True
    )

    shutil.copy(
        upload_path,
        f"static/registered_users/{name}.jpg"
    )

    total_images = train_model()

    os.remove(upload_path)

    return render_template(
        "result.html",
        message=f"{name} trained successfully.",
        total=total_images
    )


@app.route("/recognition")
def recognition():

    return render_template(
        "recognition.html"
    )


@app.route("/start_recognition")
def start_camera():

    start_recognition()

    return redirect(
        url_for("home")
    )


if __name__ == "__main__":

    app.run(
        debug=True
    )