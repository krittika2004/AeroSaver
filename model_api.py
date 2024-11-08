from flask import Flask, request, jsonify
import pickle
import os

model_path = os.path.join('notebook', 'flight.pkl')

model = pickle.load(open(model_path, 'rb'))

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    prediction = model.predict([data['features']]) 
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(port=5001)
