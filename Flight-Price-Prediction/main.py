from flask import Flask, request, render_template
from flask_cors import cross_origin
import sklearn
import pickle
import pandas as pd
import numpy as np

app = Flask(__name__)
model = pickle.load(open("flight_rf.pkl", "rb"))

@app.route("/")
@cross_origin()
def home():
    return render_template("home.html")

def get_airline_features(airline):
    airlines = ['Jet Airways', 'IndiGo', 'Air India', 'Multiple carriers', 'SpiceJet', 'Vistara', 'GoAir',
                'Multiple carriers Premium economy', 'Jet Airways Business', 'Vistara Premium economy', 'Trujet']
    features = [1 if airline == a else 0 for a in airlines]
    return features

def get_location_features(location, loc_type='Source'):
    locations = ['Delhi', 'Kolkata', 'Mumbai', 'Chennai'] if loc_type == 'Source' else ['Cochin', 'Delhi', 'New Delhi', 'Hyderabad', 'Kolkata']
    features = [1 if location == loc else 0 for loc in locations]
    return features

@app.route("/predict", methods=["GET", "POST"])
@cross_origin()
def predict():
    if request.method == "POST":
        try:
            # Date_of_Journey
            date_dep = request.form["Dep_Time"]
            Journey_day = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").day)
            Journey_month = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").month)

            # Departure
            Dep_hour = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").hour)
            Dep_min = int(pd.to_datetime(date_dep, format="%Y-%m-%dT%H:%M").minute)

            # Arrival
            date_arr = request.form["Arrival_Time"]
            Arrival_hour = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").hour)
            Arrival_min = int(pd.to_datetime(date_arr, format="%Y-%m-%dT%H:%M").minute)

            # Duration
            dur_hour = abs(Arrival_hour - Dep_hour)
            dur_min = abs(Arrival_min - Dep_min)

            # Total Stops
            Total_stops = int(request.form["stops"])

            # Airline
            airline = request.form['airline']
            airline_features = get_airline_features(airline)

            # Source
            Source = request.form["Source"]
            source_features = get_location_features(Source, 'Source')

            # Destination
            Destination = request.form["Destination"]
            destination_features = get_location_features(Destination, 'Destination')

            # Combine all features
            features = [
                Total_stops,
                Journey_day,
                Journey_month,
                Dep_hour,
                Dep_min,
                Arrival_hour,
                Arrival_min,
                dur_hour,
                dur_min,
            ] + airline_features + source_features + destination_features

            prediction = model.predict([features])
            output = round(prediction[0], 2)

            return render_template('home.html', prediction_text="Your Flight price is Rs. {}".format(output))

        except Exception as e:
            return render_template('home.html', prediction_text="Error: {}".format(e))

    return render_template("home.html")

if __name__ == "__main__":
    app.run(debug=True)
