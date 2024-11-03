from flask import Flask, request, jsonify
from classify_image import classify_image
from PIL import Image
import io

app = Flask(__name__)

@app.route('/')
def home():
    return "Hello, Flask!"

# Define the route to receive and classify the image
@app.route('/classify', methods=['POST'])
def classify_route():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']

    # Check if the file is a .jpg image
    if file.filename == '' or not file.filename.lower().endswith('.jpg'):
        return jsonify({"error": "Invalid file format. Please upload a .jpg file."}), 400

    try:
        # Read the image file as PIL image
        image = Image.open(io.BytesIO(file.read()))

        # Classify the image using the function from classify_image.py
        class_name, confidence = classify_image(image)

        # Return the classification result
        return jsonify({
            "predicted_class": class_name,
            "confidence": f"{confidence:.2f}%"
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)

