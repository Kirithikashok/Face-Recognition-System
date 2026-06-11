# Face Recognition System

## Project Overview

This project is a web-based Face Recognition System developed using Flask and OpenCV. The system allows users to upload a facial image along with their name, automatically generates multiple augmented versions of the image, trains a face recognition model, and performs real-time face recognition using a webcam.

---

## Features

- Upload a single image with a user name
- Automatic face validation
- Rejects images with no face detected
- Rejects images with multiple faces detected
- Automatic image augmentation
- Automatic model training
- Real-time face recognition
- Displays recognized name and confidence score
- Supports multiple users
- Recognition logging using SQLite
- Unknown face detection
- Press `Q` to stop recognition

---

## Technologies Used

### Frontend
- HTML
- CSS

### Backend
- Flask

### Computer Vision
- OpenCV
- Haar Cascade Face Detection
- LBPH Face Recognizer

### Database
- SQLite

### Programming Language
- Python

---

## Project Structure

```text
Face_Recognition_System/
│
├── app.py
├── requirements.txt
├── README.md
│
├── templates/
│   ├── upload.html
│   ├── recognition.html
│   └── result.html
│
├── static/
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── script.js
│
└── utils/
    ├── augment.py
    ├── trainer.py
    ├── recognizer.py
    ├── database.py
    ├── logger.py
    └── face_validation.py
```

---

## Workflow

### Step 1: Upload Image

The user enters:
- Name
- Facial image

The uploaded image is validated using Haar Cascade Face Detection.

### Step 2: Face Validation

The system checks:
- Exactly one face is present
- No multiple faces
- No missing face

Invalid images are rejected.

### Step 3: Data Augmentation

The uploaded image is automatically transformed into:

- Original
- Flip
- Rotate Left
- Rotate Right
- Grayscale
- Bright
- Dark

### Step 4: Model Training

The generated images are processed and used to train the LBPH Face Recognizer.

Generated files:

```text
embeddings/
├── lbph_model.yml
└── labels.npy
```

### Step 5: Face Recognition

The webcam captures live video.

The system:
1. Detects faces
2. Recognizes users
3. Displays name and confidence score

Example:

```text
Ashok 92.5%
```

---

## Dataset Structure

Example:

```text
dataset/
│
└── Ashok/
    ├── original.jpg
    ├── flip.jpg
    ├── rotate_left.jpg
    ├── rotate_right.jpg
    ├── gray.jpg
    ├── bright.jpg
    └── dark.jpg
```

---

## Challenges Faced

- Lighting variations
- Face orientation changes
- Camera quality limitations
- Similar facial appearances
- Limited training images

---

## Future Improvements

- FaceNet Integration
- Deep Learning Based Recognition
- Liveness Detection
- Anti-Spoofing Protection
- Attendance Management System
- Cloud Database Integration
- Multi-Camera Support

---
## Usage

1. Enter the user's name.
2. Upload a facial image.
3. Click **Upload & Train**.
4. Repeat for additional users.
5. Click **Face Recognition**.
6. Start webcam recognition.
7. Press **Q** to stop recognition.
