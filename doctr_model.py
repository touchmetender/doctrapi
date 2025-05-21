from doctr.models import ocr_predictor
from doctr.io import DocumentFile

model = ocr_predictor(pretrained=True)

def get_text(image):
    doc = DocumentFile.from_images([image])
    result = model(doc)
    return result.render()
