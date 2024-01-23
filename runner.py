import os
from flask import Flask, request
from easyocr import Reader
import keras_ocr
import cv2
import numpy as np

app = Flask(__name__)
reader = Reader(['en'])
pipeline = keras_ocr.pipeline.Pipeline()

@app.route('/sailnum', methods=['POST'])
def sailnum():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400
    if file:
        image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        result_easyocr = reader.readtext(image)
        result_kerasocr = pipeline.recognize([image])[0]

        if result_easyocr == result_kerasocr:
            return str(result_easyocr)
        else:
            return 'NumberNotFound'

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
