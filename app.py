from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from doctr_model import get_text
from PIL import Image
import io

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Doctr OCR API is running."}

@app.post("/ocr/")
async def ocr_endpoint(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    text = get_text(image)
    return JSONResponse(content={"text": text})
