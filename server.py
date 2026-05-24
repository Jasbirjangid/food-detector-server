from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from PIL import Image
import numpy as np
import io
from food_data import FOOD_DATABASE

app = Flask(__name__)
CORS(app)

# Load model
print("Loading model...")
model = tf.keras.models.load_model('keras_model.h5', compile=False)

# Load labels
with open('labels.txt', 'r') as f:
    labels = [line.strip().split(' ')[1] for line in f.readlines()]

print(f"Model loaded! Classes: {labels}")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        image_data = request.data
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image = image.resize((224, 224))
        
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
        predictions = model.predict(image_array)
        class_idx = np.argmax(predictions[0])
        confidence = float(predictions[0][class_idx])
        food_name = labels[class_idx]
        
        print(f"Detected: {food_name} ({confidence:.2f})")
        
        details = FOOD_DATABASE.get(food_name.lower(), {
            'calories': 'Unknown',
            'harmful': 'Check nutrition label',
            'best_time': 'Varies'
        })
        
        return jsonify({
            'food': food_name,
            'confidence': confidence,
            'calories': details['calories'],
            'harmful': details['harmful'],
            'best_time': details['best_time']
        })
        
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Server starting...")
    app.run(host='0.0.0.0', port=5000)