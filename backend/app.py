import pandas as pd
import joblib
from flask import Flask, request, jsonify
import io

# Initialize Flask application
app = Flask(__name__)

# Load the pre-trained pipeline model
from pathlib import Path

model_path = Path(__file__).resolve().parent / "superkart_model.joblib"
model = joblib.load(model_path)

@app.route("/v1/predict", methods=["POST"])
def predict():
    try:
        # Get JSON payload from request
        data_json = request.get_json(force=True)

        # Convert payload to pandas DataFrame
        df = pd.DataFrame([data_json])

        # Generate prediction using loaded model pipeline
        prediction = model.predict(df)[0]

        # Return response
        return jsonify({"prediction": float(prediction)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/v1/predictbatch", methods=["POST"])
def predict_batch():
    try:
        # Check if file is part of the request
        if 'file' not in request.files:
            return jsonify({"error": "No file part in the request"}), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400

        # Read CSV file directly into DataFrame
        df = pd.read_csv(io.BytesIO(file.read()))

        # Generate batch predictions
        predictions = model.predict(df)

        # Format predictions as a dictionary of index: prediction
        result = {str(i): float(pred) for i, pred in enumerate(predictions)}

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860)
