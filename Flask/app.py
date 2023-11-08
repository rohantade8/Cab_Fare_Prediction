# importing the necessary dependencies
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
import os

app = Flask(__name__)  # initializing a Flask app

# Define the file path for the trained model
model_file_path = "C:\\Users\\rohan\\OneDrive\\Desktop\\Vap Project\\Cab_Fare_Prediction\\Training\\model.pkl"

# Define file paths for the CSV data files
cab_rides_file_path = "C:\\Users\\rohan\\OneDrive\\Desktop\\Vap Project\\Cab_Fare_Prediction\\Dataset\\cab_rides.csv"
weather_file_path = "C:\\Users\\rohan\\OneDrive\\Desktop\\Vap Project\\Cab_Fare_Prediction\\Dataset\\weather.csv"

# Load the model from the file using the defined model file path
with open(model_file_path, 'rb') as file:
    loaded_model = pickle.load(file)

# Load the CSV data files
cab_rides_df = pd.read_csv(cab_rides_file_path)
weather_df = pd.read_csv(weather_file_path)


@app.route('/')  # route to display the home page
def home():
    return render_template('index.html')  # rendering the home page

@app.route('/Prediction', methods=['POST', 'GET'])
def prediction():  # route which will take you to the prediction page
    return render_template('index1.html')

@app.route('/Home', methods=['POST', 'GET'])
def my_home():
    return render_template('index.html')

@app.route('/predict', methods=["POST", "GET"])  # route to show the predictions in a web UI
def predict():
    # reading the inputs given by the user
    input_feature = [str(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]
    feature_name = ['cab_type', 'destination', 'source' , 'product_id','name']
    x = pd.DataFrame(features_values, columns=feature_name)

    # predictions using the loaded model file
    prediction = loaded_model.predict(x)
    print("Prediction is ₹:", prediction)
    # showing the prediction results in a UI
    return render_template("result.html", prediction=prediction[0])

if __name__ == "__main__":
#   app.run(host='0.0.0.0', port=8000, debug=True)  # running the app
    port = int(os.environ.get('PORT', 5000))
    app.run(port=port, debug=True, use_reloader=False)
    