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
        self.model_path = "models/pm10_forecast_model.cbm"
        self.scaler_path = "models/pm10_scaler.pkl"
        self.feature_columns_path = "models/feature_columns.json"

    def save_model(self):
        """Save the trained model and scaler"""
        import os
        import joblib
        
        os.makedirs("models", exist_ok=True)
        self.model.save_model(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)
        with open(self.feature_columns_path, 'w') as f:
            json.dump(self.feature_columns, f)
        print(f"[INFO] Model, scaler and features saved to {os.path.abspath('models')}")
        
    def load_model(self):
        """Load the trained model and scaler"""
        import os
        import joblib
        
        if not (os.path.exists(self.model_path) and 
                os.path.exists(self.scaler_path) and 
                os.path.exists(self.feature_columns_path)):
            return False
            
        self.model.load_model(self.model_path)
        self.scaler = joblib.load(self.scaler_path)
        
        # Load feature columns
        with open(self.feature_columns_path, 'r') as f:
            self.feature_columns = json.load(f)

        print(f"[INFO] Loaded model with {len(self.feature_columns)} features: {self.feature_columns}")
        self.is_trained = True
        return True
    
    def prepare_features(self, history_data, weather_data=None, target_time=None):
        """Prepare features from historical PM10 and weather data"""
        if not history_data:
            print("[DEBUG] No history data provided")
            return pd.DataFrame()
        
        try:
            # Convert history to DataFrame
            df = pd.DataFrame(history_data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp').drop_duplicates('timestamp')
            
            # Basic PM10 statistical features (always available)
            recent_24h = df.tail(24)
            recent_7d = df.tail(168) if len(df) >= 168 else df
        
            features = {
                'pm10_mean_24h': recent_24h['pm10'].mean() if len(recent_24h) > 0 else 0,
                'pm10_max_24h': recent_24h['pm10'].max() if len(recent_24h) > 0 else 0,
                'pm10_min_24h': recent_24h['pm10'].min() if len(recent_24h) > 0 else 0,
                'pm10_std_24h': recent_24h['pm10'].std() if len(recent_24h) > 1 else 0,
                'pm10_mean_7d': recent_7d['pm10'].mean() if len(recent_7d) > 0 else 0,
                'data_points': len(df),
                'has_weather': 0,  # Default to no weather
                'weather_records': 0
            }

            # Calculate trend if enough data points
            if len(recent_24h) >= 2:
                x = np.arange(len(recent_24h))
                y = recent_24h['pm10'].values
                features['pm10_trend'] = np.polyfit(x, y, 1)[0]
            else:
                features['pm10_trend'] = 0
            # Add temporal features if target time available
            if target_time:
                features.update({
                    'hour': target_time.hour,
                    'day_of_week': target_time.weekday(),
                    'month': target_time.month,
                    'is_weekend': 1 if target_time.weekday() >= 5 else 0
                })
            else:
                # Use last timestamp from history if target_time not provided
                last_time = df['timestamp'].max()
                features.update({
                    'hour': last_time.hour,
                    'day_of_week': last_time.weekday(),
                    'month': last_time.month,
                    'is_weekend': 1 if last_time.weekday() >= 5 else 0
                })
            
            # Only add weather features if available
            if weather_data:
                weather_df = pd.DataFrame([parse_weather_record(w) for w in weather_data])

                # Replace missing or sparse values (optional)
                if not weather_df.empty:
                    features.update({
                        'temp_c_mean': weather_df['temp_c'].mean() if len(weather_df) > 3 else 0.0,
                        'wind_speed_mean': weather_df['wind_speed_ms'].mean() if len(weather_df) > 3 else 0.0,
                        'wind_dir_mean': weather_df['wind_dir_deg'].mean() if len(weather_df) > 3 else 0.0,
                        'wind_var_mode': weather_df['wind_var_code'].mode()[0] if not weather_df['wind_var_code'].mode().empty else 'X'
                    })
                else:
                    features.update({
                        'temp_c_mean': 0.0,
                        'wind_speed_mean': 0.0,
                        'wind_dir_mean': 0.0,
                        'wind_var_mode': 'X'
                    })

            features_df = pd.DataFrame([features])
            print(f"[DEBUG] Created features DataFrame with shape {features_df.shape}")
            return features_df
            
        except Exception as e:
            print(f"[ERROR] Feature preparation failed: {str(e)}")
            # Return minimal feature set
            return pd.DataFrame([{
                'pm10_mean_24h': np.mean([d['pm10'] for d in history_data[-24:]]) if history_data else 0,
                'data_points': len(history_data),
                'has_weather': 0
            }])

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
                
                # Use 80% as training, last 20% as targets
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
        """Make a single prediction with better error handling"""
        if not self.is_trained or not self.feature_columns:
            print("[WARNING] Model not trained or features not loaded")
            return None
        
        try:
            if features_df is None or features_df.empty:
                print("[WARNING] Empty features DataFrame")
                return None
                
            # Create a new DataFrame with all required columns
            prediction_features = pd.DataFrame(index=[0])
            
            # Update with available features
            for col in self.feature_columns:
                if col in features_df.columns:
                    prediction_features[col] = features_df[col].values[0]
                else:
                    prediction_features[col] = 0  # Default value for missing features
                    
            print(f"[DEBUG] Prediction features shape: {prediction_features.shape}")
            print(f"[DEBUG] Features aligned with training columns")
            
            # Scale features
            X_scaled = self.scaler.transform(prediction_features)
            
            # Make prediction
            prediction = self.model.predict(X_scaled)[0]
            return max(0.0, float(prediction))  # Ensure non-negative
            
        except Exception as e:
            print(f"[ERROR] Prediction failed: {str(e)}")
            print(f"Available columns: {features_df.columns.tolist()}")
            print(f"Required columns: {self.feature_columns}")
            return None

def parse_weather_record(record):
    parsed = {
        'temp_c': 0.0,
        'wind_speed_ms': 0.0,
        'wind_dir_deg': 0.0,
        'wind_var_code': 'X',  # placeholder for missing
    }

    try:
        # Temperature
        if 'tmp' in record:
            temp_raw = record['tmp'].split(',')[0]
            parsed['temp_c'] = int(temp_raw) / 10.0

        # Wind
        if 'wnd' in record:
            wnd_parts = record['wnd'].split(',')
            if len(wnd_parts) >= 5:
                parsed['wind_dir_deg'] = float(wnd_parts[0])
                parsed['wind_var_code'] = wnd_parts[2]
                parsed['wind_speed_ms'] = float(wnd_parts[3]) / 10.0

    except Exception as e:
        print(f"[WARNING] Failed to parse weather record: {record} ({e})")

    return parsed

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


def generate_output(data, landuse_data=None, forecast_hours=24, retrain=False):
    """
    Generates PM10 forecasts for each case's target location using CatBoost model.
    When using pre-trained model: predict for all cases
    When training/retraining: use 80% for training, predict for 20% test cases
    """
    predictions = []
    cases_to_predict = []

    # Initialize the PM10 forecaster
    forecaster = PM10Forecaster()

    # Try to load existing model
    if not retrain and forecaster.load_model():
        predict_pm10.forecaster = forecaster
        print("[INFO] Using loaded model for predictions")
        cases_to_predict = data["cases"] # Use all cases
    else:
        # Train new model if needed
        n_train = int(len(data["cases"]) * 0.8)
        training_cases = data["cases"][:n_train]
        test_cases = data["cases"][n_train:]
        print(f"[INFO] Training new model with {len(training_cases)} cases")

        if training_cases:
            print("[INFO] Training new model...")
            forecaster.train(training_cases)
            forecaster.save_model()
            predict_pm10.forecaster = forecaster
            cases_to_predict = test_cases # Only predict on test cases after training
            print(f"[INFO] Generating predictions for {len(cases_to_predict)} test cases")
        else:
            print("[WARNING] No training data available")

    for case in cases_to_predict:
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
    parser.add_argument("--retrain", action="store_true", help="Force model retraining")
    args = parser.parse_args()

    # Read the input JSON file containing cases, stations, and target definitions
    with open(args.data_file, "r") as f:
        data = json.load(f)
        print(f"Read input from: {args.data_file}")

    # Generate forecasts using CatBoost model for each case's target
    output = generate_output(data, landuse_data=None, retrain=args.retrain)

    # Write the generated forecasts to the specified output JSON file
    with open(args.output_file, "w") as f:
        json.dump(output, f, indent=2)

    print(f"Wrote forecasts to: {args.output_file}")
    print(f"Generated forecasts for {len(output['predictions'])} cases")


if __name__ == "__main__":
    main()