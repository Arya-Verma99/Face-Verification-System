from fastapi import FastAPI, UploadFile, File
from fastapi.responses import RedirectResponse
import utils

app = FastAPI(title="Face Authentication API", version="0.1.0")

@app.get("/", include_in_schema=False)
def index():
    return RedirectResponse(url="/docs")

@app.get("/Home")
def home():
    return {"message": "Welcome to the Face Authentication API"}

@app.post("/verify")
def verify_faces(image1: UploadFile = File(...), image2: UploadFile = File(...)):
    # REMOVED 'await' from these lines
    img1_bytes = image1.file.read()
    img2_bytes = image2.file.read()
    
    # Process both images to get bounding boxes and embeddings
    bbox1, embed1 = utils.process_image(img1_bytes)
    bbox2, embed2 = utils.process_image(img2_bytes)
    
    # Handle scenarios where face detection fails
    if not bbox1 or not bbox2:
        return {
            "verification_result": "Face detection failed in one or both images",
            "similarity_score": 0.0,
            "faces_image1": bbox1 if bbox1 else [],
            "faces_image2": bbox2 if bbox2 else []
        }
    
    # Calculate similarity score
    similarity_score = utils.calculate_similarity(embed1, embed2)
    
    # Match outcome threshold
    THRESHOLD = 0.6
    result = "same person" if similarity_score >= THRESHOLD else "different person"
    
    return {
        "verification_result": result,
        "similarity_score": round(similarity_score, 2),
        "faces_image1": bbox1,
        "faces_image2": bbox2
    }
