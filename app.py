from flask import Flask, request, jsonify
import pickle
import numpy as np

# Load the model
with open('dehaze_autoencoder.pkl', 'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    
    data = request.get_json(force=True)
    prediction = model.predict([np.array(data['input'])])
    output = prediction[0]

    return jsonify(output)

if __name__ == '__main__':
    app.run(port=5000, debug=True)