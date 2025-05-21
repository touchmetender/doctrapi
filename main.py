from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from doctr.models import ocr_predictor
from PIL import Image
import io

app = FastAPI()

# Allow all origins (or restrict to your domain)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load docTR OCR predictor (TensorFlow backend assumed)
model = ocr_predictor(pretrained=True)

@app.post("/ocr")
async def perform_ocr(file: UploadFile = File(...)):
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    
    # doctr expects a list of PIL images
    result = model([image])
    
    # result.export() returns a dict containing pages and text info
    text = result.export()["pages"][0]["text"]
    return {"text": text}
