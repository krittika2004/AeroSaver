from flask import Flask, request, render_template, jsonify
import pickle
import pandas as pd
import os
import requests

API_KEY = "c464ca384d09a925ab5666aa"
CURRENCY_API_URL = f"https://v6.exchangerate-api.com/v6/{API_KEY}/latest/INR"

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

def get_currency_rates():
    try:
        response = requests.get(CURRENCY_API_URL)
        if response.status_code == 200:
            return response.json()["conversion_rates"]
        else:
            print(f"Error fetching rates: {response.status_code}")
            return None
    except Exception as e:
        print(f"Currency API request error: {e}")
        return None

@app.route("/")
def index():
    return render_template("index.html")

# New route for getting exchange rates via AJAX
@app.route("/get_exchange_rate")
def get_exchange_rate():
    currency = request.args.get("currency", "USD")
    
    rates = get_currency_rates()
    if rates and currency in rates:
        return jsonify({
            "success": True,
            "rate": rates[currency],
            "currency": currency
        })
    
    return jsonify({
        "success": False,
        "message": "Failed to get exchange rate"
    })

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
            Airline_Air_India = 1 if airline == 'Air India' else 0
            Airline_IndiGo = 1 if airline == 'IndiGo' else 0
            Airline_AirAsia = 1 if airline == 'Air Asia' else 0
            Airline_GO_FIRST = 1 if airline == 'GO_FIRST' else 0
            Airline_SpiceJet = 1 if airline == 'SpiceJet' else 0
            Airline_AkasaAir = 1 if airline == 'AkasaAir' else 0
            Airline_AllianceAir = 0  # Default value if not in form
            Airline_StarAir = 0      # Default value if not in form

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
            
            prediction_text = f"Your Flight price is Rs. {output}"
            
            return render_template("predict.html", prediction_text=prediction_text, prediction_value=output)

        except Exception as e:
            return render_template("predict.html", prediction_text=f"Error in prediction: {e}")

    return render_template("predict.html")

# 404 Error handler
@app.errorhandler(404)
def page_not_found(e):
    return render_template("404.html"), 404

if __name__ == "__main__":
    app.run(debug=True)