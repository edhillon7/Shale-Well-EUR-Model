import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import os 


def train_model():
    try:
        
        df = pd.read_csv('Chapter2_Shale_Gas_Wells_DataSet.csv')
        
        
        X = df.iloc[:, :13]  
        Z = df.iloc[:, -2]   
        
        
        X_train, X_test, Z_train, Z_test = train_test_split(X, Z, test_size=0.3, random_state=42)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Create and train the model
        model = LinearRegression()
        model.fit(X_train_scaled, Z_train)
        
        # Save the model and scaler for later use
        joblib.dump(model, 'linear_regression_model.pkl')
        joblib.dump(scaler, 'scaler.pkl')
        
        # Evaluate the model
        Z_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(Z_test, Z_pred)
        mae = mean_absolute_error(Z_test, Z_pred)
        r2 = r2_score(Z_test, Z_pred)
        
        print(f"Training Complete\nMSE: {mse}, MAE: {mae}, R2: {r2}")
        return {"mse": mse, "mae": mae, "r2": r2}
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return {"error": "Dataset file not found"}
    except Exception as e:
        print(f"An error occurred during training: {e}")
        return {"error": str(e)}


def predict(input_data):
    model_path = 'linear_regression_model.pkl'
    scaler_path = 'scaler.pkl'
    
    try:
        
        if not os.path.exists(model_path) or not os.path.exists(scaler_path):
            return "Model or scaler file not found. Please train the model first."

        
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        
        
        input_data = np.array(input_data).reshape(1, -1)  
        input_data_scaled = scaler.transform(input_data)  
        
        # Make prediction
        prediction = model.predict(input_data_scaled)
        return prediction[0]  
    except Exception as e:
        print(f"An error occurred during prediction: {e}")
        return f"Error: {str(e)}"
