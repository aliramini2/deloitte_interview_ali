from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load('gradient_boosting_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    json_ = request.json
    query_df = pd.DataFrame(json_, index=[0])
    prediction = model.predict(query_df)
    return jsonify({'prediction': list(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
