from flask import Flask, request, render_template
import pickle
import pandas as pd
import os

app = Flask(__name__)

# Load model with error handling
try:
    model_path = os.path.join(os.path.dirname(__file__), "flight_final.pkl")  # Update the model path
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
except FileNotFoundError:
    print("Error: The model file 'random_forest_model.pkl' was not found.")
    model = None
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    if model is None:
        return render_template("predict.html", prediction_text="Model not available.")
    
    if request.method == "POST":
        try:
            # Date_of_Journey
            date_dep = request.form["departure"]
            journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
            journey_month = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").month)

            # Departure
            dep_hour = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").hour)
            dep_min = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").minute)

            # Arrival
            date_arr = request.form["arrival"]
            arrival_hour = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").hour)
            arrival_min = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").minute)

            # Duration
            Duration_hour = abs(arrival_hour - dep_hour)
            Duration_mins = abs(arrival_min - dep_min)

            # Total Stops
            Total_Stops = int(request.form["stopage"])

            # Airlines
            airline = request.form['airline']
            Airline_Vistara = 1 if airline == 'Vistara' else 0
            Airline_Air_India = 1 if airline == 'Air_India' else 0
            Airline_IndiGo = 1 if airline == 'Indigo' else 0
            Airline_AirAsia = 1 if airline == 'AirAsia' else 0
            Airline_GO_FIRST = 1 if airline == 'GO_FIRST' else 0
            Airline_SpiceJet = 1 if airline == 'SpiceJet' else 0
            Airline_AkasaAir = 1 if airline == 'AkasaAir' else 0
            Airline_AllianceAir = 1 if airline == 'AllianceAir' else 0
            Airline_StarAir = 1 if airline == 'StarAir' else 0

            # Source
            Source = request.form["source"]
            Source_Bangalore = 1 if Source == 'Bangalore' else 0
            Source_Delhi = 1 if Source == 'Delhi' else 0
            Source_Mumbai = 1 if Source == 'Mumbai' else 0
            Source_Chennai = 1 if Source == 'Chennai' else 0

            # Destination
            Destination = request.form["destination"]
            Destination_Mumbai = 1 if Destination == 'Mumbai' else 0
            Destination_Delhi = 1 if Destination == 'Delhi' else 0
            Destination_Kolkata = 1 if Destination == 'Kolkata' else 0
            Destination_Hyderabad = 1 if Destination == 'Hyderabad' else 0
            Destination_Bangalore = 1 if Destination == 'Bangalore' else 0

            # Prediction
            prediction = model.predict([[Total_Stops, journey_day, journey_month, dep_hour, dep_min, 
                                         arrival_hour, arrival_min, Duration_hour, Duration_mins, 
                                         Airline_Vistara, Airline_Air_India, Airline_IndiGo, Airline_AirAsia,
                                         Airline_GO_FIRST, Airline_SpiceJet, Airline_AkasaAir, Airline_AllianceAir, 
                                         Source_Bangalore, Source_Delhi, Source_Mumbai, 
                                         Destination_Mumbai, Destination_Delhi, Destination_Kolkata, Destination_Hyderabad]])

            output = round(prediction[0], 2)
            return render_template("predict.html", prediction_text=f"Your Flight price is Rs. {output}")

        except Exception as e:
            return render_template("predict.html", prediction_text=f"Error in prediction: {e}")

    return render_template("predict.html")

# 404 Error handler
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

if __name__ == "__main__":
    app.run(debug=True)
