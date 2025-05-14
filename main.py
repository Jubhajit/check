# main.py

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import shutil
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import cv2
import numpy as np
from embedder import chunk_text, create_faiss_index
from typing import List


# ✅ Explicitly tell pytesseract where tesseract.exe is located
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
print("🚀 main.py is executing NOW")
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ✅ Global FAISS index and chunks storage
faiss_index = None
faiss_chunks: List[str] = []



@app.get("/")
def root():
    return {"message": "Research Chatbot API is running."}


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_location = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_location, "wb") as f:
            shutil.copyfileobj(file.file, f)

        print("📄 PDF uploaded successfully!")
        print("🔍 Starting PDF processing...")

        extracted_text = extract_text_and_ocr(file_location)

        print("🧠 Starting chunking and embedding...")
        chunks = chunk_text(extracted_text)

        # ✅ Print chunks to verify in terminal
        print(f"🧩 Total Chunks Created: {len(chunks)}")
        for i, chunk in enumerate(chunks):
            print(f"\n--- Chunk {i+1} ---\n{chunk}\n")

        global faiss_index, faiss_chunks
        faiss_index, _, faiss_chunks = create_faiss_index(chunks)

        return {"status": "success", "chunks_created": len(chunks)}

    except Exception as e:
        print("❌ Error processing file:", e)
        return JSONResponse(content={"error": str(e)}, status_code=500)


print("✅ /chunks endpoint loaded")

@app.get("/chunks")
def get_chunks():
    global faiss_chunks
    print("📦 Checking /chunks endpoint...")
    if not faiss_chunks:
        print("⚠️ No chunks found.")
        return {"status": "error", "detail": "No chunks available. Please upload a PDF first."}

    print(f"✅ Returning {len(faiss_chunks)} chunks.")
    return {
        "status": "success",
        "total_chunks": len(faiss_chunks),
        "chunks": faiss_chunks
    }



def extract_text_and_ocr(pdf_path):
    doc = fitz.open(pdf_path)
    final_text = ""

    for page_number in range(len(doc)):
        page = doc.load_page(page_number)
        text = page.get_text().strip()

        if text:
            final_text += f"\n\n--- Text from Page {page_number+1} ---\n{text}"
        else:
            # Render page as image and apply OCR
            pix = page.get_pixmap(dpi=300)
            img_data = np.frombuffer(pix.tobytes("png"), np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
            ocr_text = pytesseract.image_to_string(img)
            final_text += f"\n\n--- OCR from Page {page_number+1} ---\n{ocr_text}"

    return final_text
@app.get("/test-endpoint")
def test_endpoint():
    print("✅ Test endpoint hit")
    return {"message": "Test endpoint working"}