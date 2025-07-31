"""
pm10_forecaster.py

This script reads a JSON input file containing a list of cases and, for each case,
generates a PM10 forecast using CatBoost model at the case's target location.

Input JSON schema:
{
  "cases": [
    {
      "case_id":        string,
      "stations": [     # list of station objects
        {
          "station_code": string,
          "longitude":    float,
          "latitude":     float,
          "history": [    # list of hourly observations
            {
              "timestamp": str (ISO8601, e.g. "2019-01-01T00:00:00"),
              "pm10":       float
            },
            ...
          ]
        },
        ...
      ],
      "target": {
        "longitude":               float,
        "latitude":                float,
        "prediction_start_time":   str (ISO8601)
      },
      "weather": [ ... ]  # optional array of METAR‐style records
    },
    ...
  ]
}

Usage:
    python pm10_forecaster.py --data-file data.json --output-file output.json
"""

import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from catboost import CatBoostRegressor
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


class PM10Forecaster:
    """CatBoost-based PM10 forecasting model"""
    
    def __init__(self):
        self.model = CatBoostRegressor(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            l2_leaf_reg=3,
            random_state=42,
            verbose=False
        )
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_columns = []

    def prepare_features(self, history_data, weather_data=None, target_time=None):
        """Prepare features from historical PM10 and weather data"""
        if not history_data:
            return pd.DataFrame()
        
        # Convert history to DataFrame
        df = pd.DataFrame(history_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values('timestamp')
        
        # Basic statistical features
        recent_24h = df.tail(24)
        recent_7d = df.tail(168) if len(df) >= 168 else df
        
        features = {
            'pm10_mean_24h': recent_24h['pm10'].mean() if len(recent_24h) > 0 else 0,
            'pm10_max_24h': recent_24h['pm10'].max() if len(recent_24h) > 0 else 0,
            'pm10_min_24h': recent_24h['pm10'].min() if len(recent_24h) > 0 else 0,
            'pm10_std_24h': recent_24h['pm10'].std() if len(recent_24h) > 1 else 0,
            'pm10_mean_7d': recent_7d['pm10'].mean() if len(recent_7d) > 0 else 0,
            'pm10_trend': 0,  # Simple trend calculation
            'data_points': len(df)
        }
        
        # Calculate trend
        if len(recent_24h) >= 2:
            x = np.arange(len(recent_24h))
            y = recent_24h['pm10'].values
            if len(x) > 1:
                features['pm10_trend'] = np.polyfit(x, y, 1)[0]
        
        # Add temporal features
        if target_time:
            features['hour'] = target_time.hour
            features['day_of_week'] = target_time.weekday()
            features['month'] = target_time.month
            features['is_weekend'] = 1 if target_time.weekday() >= 5 else 0
        
        # Add weather features if available
        if weather_data:
            # Simple weather feature extraction
            features.update({
                'has_weather': 1,
                'weather_records': len(weather_data)
            })
        else:
            features.update({
                'has_weather': 0,
                'weather_records': 0
            })
        
        return pd.DataFrame([features])

    def train(self, training_data):
        """Train the CatBoost model"""
        if not training_data:
            return False
        
        X_list = []
        y_list = []
        
        for case in training_data:
            stations = case.get('stations', [])
            for station in stations:
                history = station.get('history', [])
                if len(history) < 24:  # Need minimum history
                    continue
                
                # Use last 80% as training, last 20% as targets
                split_idx = int(len(history) * 0.8)
                train_history = history[:split_idx]
                target_history = history[split_idx:]
                
                if len(train_history) < 12 or len(target_history) < 1:
                    continue
                
                # Create features from training history
                target_time = pd.to_datetime(target_history[0]['timestamp'])
                features_df = self.prepare_features(
                    train_history, 
                    case.get('weather', []), 
                    target_time
                )
                
                if not features_df.empty:
                    X_list.append(features_df)
                    y_list.append(target_history[0]['pm10'])
        
        if not X_list:
            print("[WARNING] No training data available")
            return False
        
        # Combine all features
        X = pd.concat(X_list, ignore_index=True)
        y = np.array(y_list)
        
        # Store feature columns
        self.feature_columns = X.columns.tolist()
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        self.is_trained = True
        
        print(f"[INFO] Model trained with {len(X)} samples and {len(self.feature_columns)} features")
        return True

    def predict_single(self, features_df):
        """Make a single prediction"""
        if not self.is_trained:
            return None
        
        if features_df.empty:
            return None
        
        # Ensure all feature columns are present
        for col in self.feature_columns:
            if col not in features_df.columns:
                features_df[col] = 0
        
        # Reorder columns to match training
        features_df = features_df[self.feature_columns]
        
        # Scale and predict
        X_scaled = self.scaler.transform(features_df)
        prediction = self.model.predict(X_scaled)[0]
        
        return max(0.0, prediction)  # Ensure non-negative


def predict_pm10(base_time, history, landuse_data, hours=24):
    """
    PM10 forecasting function using CatBoost model.
    - base_time: datetime from which to start predictions
    - history: list of historical PM10 records
    - landuse_data: optional landuse info (unused in this implementation)
    - hours: number of hourly predictions to generate
    Returns a list of dictionaries with 'timestamp' and 'pm10_pred'.
    """
    global forecaster
    
    if not hasattr(predict_pm10, 'forecaster') or not predict_pm10.forecaster.is_trained:
        print("[WARNING] Model not trained, using baseline predictions")
        # Fallback to simple baseline
        forecast_list = []
        for h in range(hours):
            ts = (base_time + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%MZ")
            # Use simple average of recent history if available
            if history:
                recent_values = [record['pm10'] for record in history[-24:]]
                pm10_pred = round(np.mean(recent_values), 1)
            else:
                pm10_pred = 25.0  # Default baseline
            
            forecast_list.append({
                "timestamp": ts,
                "pm10_pred": pm10_pred
            })
        return forecast_list
    
    # Use trained model for predictions
    forecast_list = []
    
    for h in range(hours):
        prediction_time = base_time + timedelta(hours=h)
        ts = prediction_time.strftime("%Y-%m-%dT%H:%MZ")
        
        # Prepare features for this prediction time
        features_df = predict_pm10.forecaster.prepare_features(
            history, 
            landuse_data,  # This could contain weather data
            prediction_time
        )
        
        # Make prediction
        pm10_pred = predict_pm10.forecaster.predict_single(features_df)
        
        if pm10_pred is None:
            # Fallback prediction
            if history:
                recent_values = [record['pm10'] for record in history[-24:]]
                pm10_pred = np.mean(recent_values)
            else:
                pm10_pred = 25.0
        
        forecast_list.append({
            "timestamp": ts,
            "pm10_pred": round(pm10_pred, 1)
        })
    
    return forecast_list


def generate_output(data, landuse_data=None, forecast_hours=24):
    """
    Generates PM10 forecasts for each case's target location using CatBoost model.
    """
    predictions = []

    # Initialize and train the forecaster with available data
    forecaster = PM10Forecaster()
    
    # Use all cases for training (this is a simplified approach)
    training_cases = data["cases"][:int(len(data["cases"]) * 0.8)]  # Use 80% for training
    
    if training_cases:
        forecaster.train(training_cases)
        predict_pm10.forecaster = forecaster
    else:
        print("[WARNING] No training data available")

    for case in data["cases"]:
        case_id = case["case_id"]
        target = case.get("target")
        if not target or "prediction_start_time" not in target:
            raise ValueError(f"Case '{case_id}' is missing 'prediction_start_time' in target.")

        # Parse 'prediction_start_time' into a datetime object
        try:
            base_forecast_start = datetime.fromisoformat(target["prediction_start_time"])
        except Exception as e:
            raise ValueError(f"Invalid prediction_start_time for case '{case_id}': {e}")

        # Ensure both longitude and latitude are present
        longitude = target.get("longitude")
        latitude = target.get("latitude")
        if longitude is None or latitude is None:
            raise ValueError(f"Case '{case_id}' target must include both 'longitude' and 'latitude'.")

        stations = case.get("stations", [])
        print(f"[DEBUG] Generating for case: {case_id}, "
              f"target: ({latitude}, {longitude}), "
              f"start: {base_forecast_start.isoformat()}")
        print(f"[DEBUG] Available stations: {len(stations)}")

        # Generate forecasts for each station separately
        station_forecasts = []
        target_lat, target_lon = latitude, longitude
        
        for station in stations:
            station_code = station["station_code"]
            history = station.get("history", [])
            station_lat = station["latitude"]
            station_lon = station["longitude"]
            
            print(f"  [INFO] Station {station_code}: {len(history)} history points")
            
            if len(history) < 12:  # Need minimum history for meaningful forecast
                print(f"  [WARNING] Station {station_code}: Insufficient history, skipping")
                continue
            
            # Calculate distance from station to target (simple Euclidean)
            distance = np.sqrt((station_lat - target_lat)**2 + (station_lon - target_lon)**2)
            
            # Generate forecast for this specific station
            station_forecast = predict_pm10(
                base_time=base_forecast_start,
                history=history,  # Use only this station's history
                landuse_data=case.get('weather', []),
                hours=forecast_hours
            )
            
            station_forecasts.append({
                'station_code': station_code,
                'distance': distance,
                'forecast': station_forecast,
                'history_points': len(history)
            })
            
            print(f"  [INFO] Station {station_code}: Distance={distance:.4f}, Generated {len(station_forecast)} forecasts")
        
        # Choose best station forecast (closest with sufficient data)
        if station_forecasts:
            # Sort by distance, then by data availability
            station_forecasts.sort(key=lambda x: (x['distance'], -x['history_points']))
            best_station = station_forecasts[0]
            forecast_list = best_station['forecast']
            
            print(f"  [INFO] Using forecast from station {best_station['station_code']} (distance={best_station['distance']:.4f})")
        else:
            print(f"  [WARNING] No valid station forecasts available, using baseline")
            # Fallback to simple baseline
            forecast_list = []
            for h in range(forecast_hours):
                ts = (base_forecast_start + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%MZ")
                forecast_list.append({
                    "timestamp": ts,
                    "pm10_pred": 25.0  # Default baseline
                })

        predictions.append({
            "case_id": case_id,
            "forecast": forecast_list
        })

    return {"predictions": predictions}


def main():
    parser = argparse.ArgumentParser(description="Generate PM10 forecasts using CatBoost model.")
    parser.add_argument("--data-file", required=True, help="Path to input data.json")
    parser.add_argument("--output-file", required=True, help="Path to write output.json")
    args = parser.parse_args()

    # Read the input JSON file containing cases, stations, and target definitions
    with open(args.data_file, "r") as f:
        data = json.load(f)

    # Generate forecasts using CatBoost model for each case's target
    output = generate_output(data, landuse_data=None)

    # Write the generated forecasts to the specified output JSON file
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Read input from: {args.data_file}")
    print(f"Wrote forecasts to: {args.output_file}")
    print(f"Generated forecasts for {len(output['predictions'])} cases")


if __name__ == "__main__":
    main()
