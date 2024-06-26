from flask import Flask, request, jsonify
from process import setup, set_dest_checkpoint
from flask_cors import CORS
import numpy as np
import cv2
from io import BytesIO
import base64

app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})  # Allow requests from any origin

@app.route('/setdestination', methods=['POST'])
def set_destination():
    data = request.json
    if 'dest_checkpoint' in data:
        dest_checkpoint = data['dest_checkpoint']
        set_dest_checkpoint(dest_checkpoint)

        return jsonify({'message': f'Destination checkpoint set to {dest_checkpoint}'})
    else:
        return jsonify({'error': 'Invalid JSON format or missing destination checkpoint'}), 400


@app.route('/getinstuctions', methods=['POST'])
def get_instructions():
    # Here, you can access the image file sent to the server
    data = request.json
    frame_data = data['frame']

    # Decode base64 image data to bytes
    img_bytes = base64.b64decode(frame_data)

    # Convert bytes to NumPy array
    nparr = np.frombuffer(img_bytes, np.uint8)

    # Decode the image to OpenCV format
    img_np = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    instruction = setup(img_np)
    
    return jsonify({'message': instruction})

if __name__ == '__main__':
    app.run(debug=True, port=5000)
