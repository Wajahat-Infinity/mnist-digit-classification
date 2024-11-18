from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = load_model('mnist_dense_model.h5')
# Route to handle prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    # Read and process the image
    image = Image.open(io.BytesIO(file.read()))
    image = image.convert('L').resize((28, 28))  # Resize to 28x28 pixels
    image = np.array(image)

    # Flatten the image to match the input shape the model expects
    image = image.reshape(1, 784)  # Flatten the image to (1, 784)
    image = image.astype('float32') / 255.0  # Normalize the image

    # Predict the digit
    prediction = model.predict(image)
    digit = np.argmax(prediction, axis=1)[0]  # Get the predicted digit

    return jsonify({'digit': int(digit)})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
