# Face Verification System

## Overview

This project is a Face Verification API built using FastAPI and FaceNet. The API accepts two face images, detects faces in both images, extracts facial embeddings, calculates similarity between them, and determines whether the faces belong to the same person.

The API returns:

* Verification result (`same person` or `different person`)
* Similarity score
* Bounding box coordinates of detected faces

---

## Project Setup

### 1. Create and Activate Virtual Environment

Create a virtual environment to keep project dependencies isolated.

```bash
python -m venv face_env
```

Activate the environment:

**Windows PowerShell**

```bash
.\face_env\Scripts\Activate.ps1
```

After activation, you should see:

```bash
(face_env)
```

at the beginning of your terminal.

---

### 2. Install Dependencies

Create a `requirements.txt` file and add the following packages:

```text
fastapi==0.111.0
uvicorn==0.30.1
python-multipart==0.0.9
opencv-python-headless==4.9.0.80
keras-facenet==0.3.2
tensorflow-cpu==2.15.0
numpy==1.26.4
scipy==1.12.0
```

Install all dependencies using:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
Face_Verification_System/
│
├── main.py
├── utils.py
├── requirements.txt
└── sample_images/
```

---

## utils.py

This file contains the core logic for:

* Reading uploaded images
* Detecting faces
* Extracting facial embeddings using FaceNet
* Calculating cosine similarity between two face embeddings

### Main Functions

#### `process_image()`

* Converts uploaded image bytes into an OpenCV image
* Detects faces using FaceNet's built-in MTCNN detector
* Extracts face embeddings
* Returns face bounding box coordinates and embedding vectors

#### `calculate_similarity()`

* Computes cosine similarity between two embeddings
* Returns a similarity score between the two faces

---

## main.py

This file creates the FastAPI application and exposes API endpoints.

### Available Endpoints

#### `GET /`

Redirects users to the Swagger UI documentation page.

#### `GET /Home`

Returns a welcome message.

#### `POST /verify`

Accepts two image files and:

1. Detects faces in both images
2. Extracts embeddings
3. Calculates similarity
4. Determines whether both images belong to the same person

Response Example:

```json
{
  "verification_result": "same person",
  "similarity_score": 0.82,
  "faces_image1": [[120, 80, 150, 150]],
  "faces_image2": [[115, 75, 148, 148]]
}
```

---

## Running the Application

Start the FastAPI server using:

```bash
uvicorn main:app
```

By default, the application will run on:

```text
http://127.0.0.1:8000
```

---

## API Documentation

FastAPI automatically generates interactive API documentation.

Open the following URL in your browser:

```text
http://127.0.0.1:8000/docs
```

From the Swagger UI, you can upload two images and test the face verification endpoint directly.

---

## Verification Logic

The system uses cosine similarity to compare facial embeddings.

* Similarity Score ≥ 0.60 → Same Person
* Similarity Score < 0.60 → Different Person

The threshold can be adjusted based on the required verification accuracy.

---

## Technologies Used

* FastAPI
* FaceNet
* TensorFlow
* OpenCV
* NumPy
* Python

---

## Future Improvements

* Support multiple faces in a single image
* Store embeddings in a database
* Add authentication and rate limiting
* Improve accuracy using InsightFace or ArcFace
* Docker containerization for deployment
