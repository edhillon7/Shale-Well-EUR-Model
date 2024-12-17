from flask import Flask, request, jsonify, render_template
from flask_cors import CORS  
import joblib  
import os  
from modelfinal import train_model, predict  

app = Flask(__name__)
CORS(app)  

# Define paths for the model and scaler
MODEL_PATH = 'linear_regression_model.pkl'
SCALER_PATH = 'scaler.pkl'


@app.route('/')
def home():
    return render_template('eur_prediction.html')

# Route for training the model
@app.route('/train', methods=['POST'])
def train():
    try:
        metrics = train_model()  # Train the model
        return jsonify({"message": "Model trained successfully!", "metrics": metrics})
    except Exception as e:
        print(f"Error during training: {e}")
        return jsonify({"error": "An error occurred during training"}), 500

# Route for making predictions
@app.route('/predict', methods=['POST'])
def predict_route():
    try:
        data = request.json.get("input_data", [])
        
        if not data or len(data) != 13:  
            return jsonify({"error": "Invalid input, 13 features required"}), 400

        # Check if the model and scaler files exist
        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return jsonify({"error": "Model or scaler file not found. Please train the model first."}), 404

        # Call the predict function
        prediction = predict(data)
        return jsonify({"prediction": prediction})
    except Exception as e:
        print(f"Error during prediction: {e}")
        return jsonify({"error": "An error occurred during prediction"}), 500

if __name__ == "__main__":
    app.run(debug=True)




