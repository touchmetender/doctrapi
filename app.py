from flask import Flask, request, jsonify
from doctr.models import ocr_predictor
from PIL import Image
import io

app = Flask(__name__)
model = ocr_predictor(pretrained=True)

@app.route('/ocr', methods=['POST'])
def ocr():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    img = Image.open(file.stream).convert('RGB')
    
    result = model(img)
    text = "\n".join([block.text for block in result.pages[0].blocks])
    
    return jsonify({'text': text})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
