from flask import Flask, request, render_template
import joblib
from datetime import datetime
import pandas as pd

# Initialize the Flask app
app = Flask(__name__)

# Load your pre-trained model (adjust the path as needed)
loaded_model = joblib.load('xgb_pipeline_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')  # You can create an HTML form here for input

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Fetch data from form inputs
        origin = request.form['Origin']
        destination = request.form['Destination']
        shipment_date = request.form['Shipment Date']
        planned_delivery_date = request.form['Planned Delivery Date']
        vehicle_type = request.form['Vehicle Type']
        distance = float(request.form['Distance'])
        weather_conditions = request.form['Weather Conditions']
        traffic_conditions = request.form['Traffic Conditions']

        # Convert dates to datetime objects
        shipment_date = datetime.strptime(shipment_date, '%Y-%m-%d')  # Format: YYYY-MM-DD
        planned_delivery_date = datetime.strptime(planned_delivery_date, '%Y-%m-%d')  # Format: YYYY-MM-DD

        # Calculate the difference between the Planned Delivery Date and Shipment Date
        date_diff = (planned_delivery_date - shipment_date).days  # Difference in days
        #Eda
        if(weather_conditions=="Rain" or weather_conditions=="Storm" or weather_conditions=="Fog"):
            weather_conditions="Bad Weather"
        if(traffic_conditions=="Moderate" or traffic_conditions=="Heavy"):
            traffic_conditions="Not Light"    
        # Prepare the input data as a DataFrame (matching the training dataset)
        input_data = pd.DataFrame({
            'Origin': [origin],
            'Destination': [destination],
            'Vehicle Type': [vehicle_type],
            'Distance (km)': [distance],
            'Weather Conditions': [weather_conditions],
            'Traffic Conditions': [traffic_conditions],
            'Expected_Delivery_Time': [date_diff],  # Use date_diff as the feature
        })

        # Make prediction using the pre-trained model
        prediction = loaded_model.predict(input_data)

        # Interpret the prediction
        prediction_result = "Delayed" if prediction[0] == 1 else "Not Delayed"

        # Render the result on the webpage
        return render_template('index.html', prediction_text=f'Predicted Result: {prediction_result}')

if __name__ == '__main__':
    app.run(debug=True)
