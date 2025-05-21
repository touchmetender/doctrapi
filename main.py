from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from doctr.models.ocr import ocr_predictor  # <--- import changed here
from PIL import Image
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = ocr_predictor(pretrained=True)

@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    result = model([image])
    text = result.export()["pages"][0]["text"]
    return {"text": text}
