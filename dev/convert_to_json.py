import json
import pandas as pd
import os
import argparse
from datetime import datetime, timedelta
import numpy as np

def parse_weather_data(weather_df):
    """Parse weather data into the required format"""
    weather_records = []
    
    # Ensure datetime is in the correct format
    weather_df['date'] = pd.to_datetime(weather_df['date'])
    
    for _, row in weather_df.iterrows():
        weather_record = {
            "date": row['date'].strftime("%Y-%m-%dT%H:%M:%S"),
            "tmp": row['tmp'],
            "wnd": row['wnd'],
            "slp": row['slp'],
            "vis": row['vis'],
            "dew": row['dew']
        }
        weather_records.append(weather_record)
    
    return weather_records

def parse_pm10_data(pm10_df, stations_df):
    """Parse PM10 data and station information into the required format"""
    # Group by station
    stations_data = []
    
    for station_code, group in pm10_df.groupby('station_code'):
        # Get station information
        station_info = stations_df[stations_df['station_code'] == station_code]
        
        if len(station_info) == 0:
            print(f"Warning: No station information found for {station_code}")
            continue
        
        # Create history records
        history = []
        for _, row in group.iterrows():
            history.append({
                "timestamp": row['timestamp'].strftime("%Y-%m-%dT%H:%M:%S"),
                "pm10": float(row['pm10'])
            })
        
        # Create station record
        station_record = {
            "station_code": station_code,
            "longitude": float(station_info['longitude'].values[0]),
            "latitude": float(station_info['latitude'].values[0]),
            "history": history
        }
        
        stations_data.append(station_record)
    
    return stations_data

def create_case(case_id, stations_data, weather_data, target_station_code, prediction_start_time):
    """Create a case with the given data"""
    # Find target station
    target_station = next((s for s in stations_data if s['station_code'] == target_station_code), None)
    
    if target_station is None:
        print(f"Warning: Target station {target_station_code} not found")
        return None
    
    # Create case
    case = {
        "case_id": case_id,
        "stations": stations_data,
        "target": {
            "longitude": target_station['longitude'],
            "latitude": target_station['latitude'],
            "prediction_start_time": prediction_start_time.strftime("%Y-%m-%dT%H:%M:%S")
        },
        "weather": weather_data
    }
    
    return case

def load_data_for_years(years, data_dir, air_quality_dir, weather_dir):
    """Load PM10 and weather data for multiple years"""
    all_pm10_data = []
    all_weather_data = []
    
    for year in years:
        print(f"Loading data for year {year}...")
        
        # Load PM10 data
        pm10_file = os.path.join(air_quality_dir, f"{year}_PM10_1g.xlsx")
        if os.path.exists(pm10_file):
            pm10_df = pd.read_excel(pm10_file)
            pm10_df['timestamp'] = pd.to_datetime(pm10_df['timestamp'])
            all_pm10_data.append(pm10_df)
            print(f"  Loaded PM10 data: {len(pm10_df)} records")
        else:
            print(f"  Warning: PM10 file not found for year {year}")
        
        # Load weather data
        weather_file = os.path.join(weather_dir, f"{year}.csv")
        if os.path.exists(weather_file):
            weather_df = pd.read_csv(weather_file)
            weather_df['date'] = pd.to_datetime(weather_df['date'])
            all_weather_data.append(weather_df)
            print(f"  Loaded weather data: {len(weather_df)} records")
        else:
            print(f"  Warning: Weather file not found for year {year}")
    
    # Combine data from all years
    if all_pm10_data:
        combined_pm10 = pd.concat(all_pm10_data, ignore_index=True)
    else:
        raise ValueError("No PM10 data loaded for any year")
    
    if all_weather_data:
        combined_weather = pd.concat(all_weather_data, ignore_index=True)
    else:
        raise ValueError("No weather data loaded for any year")
    
    return combined_pm10, combined_weather

def create_cases_from_data(pm10_df, weather_df, stations_df, case_configs):
    """Create multiple cases from the data based on configurations"""
    cases = []
    
    for i, config in enumerate(case_configs):
        case_id = config.get('case_id', f"case_{i+1:04d}")
        target_station = config.get('target_station')
        start_date = pd.to_datetime(config.get('start_date'))
        end_date = pd.to_datetime(config.get('end_date'))
        prediction_date = pd.to_datetime(config.get('prediction_date'))
        
        print(f"Creating case {case_id}...")
        print(f"  Time period: {start_date} to {end_date}")
        print(f"  Target station: {target_station}")
        print(f"  Prediction date: {prediction_date}")
        
        # Filter data for the time period
        pm10_filtered = pm10_df[(pm10_df['timestamp'] >= start_date) & (pm10_df['timestamp'] < end_date)]
        weather_filtered = weather_df[(weather_df['date'] >= start_date) & (weather_df['date'] < end_date)]
        
        print(f"  Filtered PM10 records: {len(pm10_filtered)}")
        print(f"  Filtered weather records: {len(weather_filtered)}")
        
        # Parse data into required format
        stations_data = parse_pm10_data(pm10_filtered, stations_df)
        weather_data = parse_weather_data(weather_filtered)
        
        # Create case
        case = create_case(
            case_id=case_id,
            stations_data=stations_data,
            weather_data=weather_data,
            target_station_code=target_station,
            prediction_start_time=prediction_date
        )
        
        if case:
            cases.append(case)
    
    return cases

def main():
    parser = argparse.ArgumentParser(description="Convert PM10 and weather data to JSON format")
    parser.add_argument("--output", default="data_output.json", help="Output JSON file path")
    parser.add_argument("--years", nargs="+", type=int, default=[2019], help="Years to include (e.g., 2019 2020)")
    parser.add_argument("--data-dir", default="data/raw", help="Base data directory")
    parser.add_argument("--config", help="JSON configuration file for cases")
    args = parser.parse_args()
    
    # Define paths
    data_dir = args.data_dir
    air_quality_dir = os.path.join(data_dir, "AirQuality_Krakow")
    weather_dir = os.path.join(data_dir, "Weather_Krakow")
    
    # Load stations data
    stations_file = os.path.join(air_quality_dir, "Stations.xlsx")
    stations_df = pd.read_excel(stations_file)
    print(f"Loaded stations data: {len(stations_df)} stations")
    
    # Load data for specified years
    pm10_df, weather_df = load_data_for_years(args.years, data_dir, air_quality_dir, weather_dir)
    
    # Define case configurations
    if args.config:
        with open(args.config, 'r') as f:
            case_configs = json.load(f)
    else:
        # Default configuration if no config file provided
        case_configs = [
            {
                "case_id": "case_0001",
                "target_station": "MpKrakAlKras",
                "start_date": "2019-01-01",
                "end_date": "2019-01-03",
                "prediction_date": "2019-01-03"
            }
        ]
    
    # Create cases
    cases = create_cases_from_data(pm10_df, weather_df, stations_df, case_configs)
    
    # Create final JSON structure
    output = {
        "cases": cases
    }
    
    # Save to file
    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    
    print(f"Data successfully converted and saved to {args.output}")
    print(f"Created {len(cases)} cases")

if __name__ == "__main__":
    main() 