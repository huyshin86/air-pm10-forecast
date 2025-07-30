import pandas as pd
import json
import os
import argparse
from datetime import datetime
from collections import defaultdict

def load_processed_data(file_path):
    """
    Load the processed weather_pm10_features.csv file
    
    Args:
        file_path (str): Path to the processed CSV file
        
    Returns:
        pd.DataFrame: Loaded DataFrame
    """
    print(f"Loading processed data from {file_path}...")
    df = pd.read_csv(file_path)
    print(f"Loaded {len(df)} records")
    return df

def organize_data_by_case(df):
    """
    Organize the data by case_id for easier processing
    
    Args:
        df (pd.DataFrame): Input DataFrame
        
    Returns:
        dict: Dictionary with case_id as keys and filtered DataFrames as values
    """
    cases = {}
    for case_id in df['case_id'].unique():
        cases[case_id] = df[df['case_id'] == case_id]
    
    print(f"Found {len(cases)} unique cases")
    return cases

def create_json_structure(cases_dict):
    """
    Create the required JSON structure from the organized data
    
    Args:
        cases_dict (dict): Dictionary with case_id as keys and DataFrames as values
        
    Returns:
        dict: JSON structure with cases list
    """
    json_cases = []
    
    # List of all required weather features
    weather_features = [
        'tmp_qc', 'dew_c', 'rel_hum', 'wind_var_code', 'pressure_hpa', 
        'slp_qc', 'vis_m', 'vis_qc', 'wx_past_3h_1', 'max_surf_temp_c', 
        'month', 'doy', 'hour', 'temp_c', 'wind_dir_deg', 
        'wind_speed_ms', 'year', 'week', 'is_winter_month', 'is_rush_hour', 
        'pm10_lag_1', '3_period_SMA'
    ]
    
    for case_id, case_df in cases_dict.items():
        print(f"Processing case {case_id}...")
        
        # Group by station
        stations_data = []
        for station_code, station_df in case_df.groupby('station_code'):
            # Create history records
            history = []
            for _, row in station_df.iterrows():
                # Skip rows with NaN dates
                if pd.isna(row['date']):
                    continue
                
                # Convert timestamp to ISO format
                try:
                    timestamp = pd.to_datetime(row['date']).strftime("%Y-%m-%dT%H:%M:%S")
                    
                    # Skip rows with NaN pm10 values
                    if pd.isna(row['pm10']):
                        continue
                    
                    history.append({
                        "timestamp": timestamp,
                        "pm10": float(row['pm10'])
                    })
                except (ValueError, AttributeError) as e:
                    print(f"Error processing timestamp for row: {e}")
                    continue
            
            # Skip if no valid history records
            if not history:
                continue
            
            # Get first row for station info
            first_row = station_df.iloc[0]
            
            # Skip if longitude or latitude is NaN
            if pd.isna(first_row['longitude']) or pd.isna(first_row['latitude']):
                continue
            
            station_record = {
                "station_code": station_code,
                "longitude": float(first_row['longitude']),
                "latitude": float(first_row['latitude']),
                "history": history
            }
            
            stations_data.append(station_record)
        
        # Skip if no valid stations
        if not stations_data:
            print(f"Skipping case {case_id} - no valid stations found")
            continue
        
        # Create weather records
        weather_records = []
        for _, row in case_df.iterrows():
            # Skip rows with NaN dates
            if pd.isna(row['date']):
                continue
            
            try:
                timestamp = pd.to_datetime(row['date']).strftime("%Y-%m-%dT%H:%M:%S")
                
                # Create weather record with date
                weather_record = {"date": timestamp}
                
                # Add all specified weather features that have valid values
                for feature in weather_features:
                    if feature in row.index and pd.notna(row[feature]):
                        try:
                            # Convert to appropriate type
                            if feature in ['wind_var_code']:
                                # Keep these as strings
                                weather_record[feature] = str(row[feature])
                            else:
                                # Convert numeric values to float
                                weather_record[feature] = float(row[feature])
                        except:
                            # If conversion fails, skip this feature
                            pass
                
                # Only add record if it has at least some weather data
                if len(weather_record) > 1:  # More than just the date
                    weather_records.append(weather_record)
                    
            except (ValueError, AttributeError) as e:
                print(f"Error processing weather data: {e}")
                continue
        
        # Get valid dates for prediction_start_time calculation
        valid_dates = case_df['date'].dropna()
        if valid_dates.empty:
            print(f"Skipping case {case_id} - no valid dates found")
            continue
        
        # Get target information - use the last timestamp as prediction_start_time
        try:
            last_timestamp = pd.to_datetime(valid_dates.max()) + pd.Timedelta(days=1)
            
            # Use the first station as the target location
            first_station = stations_data[0]
            target = {
                "longitude": first_station["longitude"],
                "latitude": first_station["latitude"],
                "prediction_start_time": last_timestamp.strftime("%Y-%m-%dT%H:%M:%S")
            }
            
            # Create case
            case = {
                "case_id": case_id,
                "stations": stations_data,
                "target": target
            }
            
            # Only include weather if we have records
            if weather_records:
                case["weather"] = weather_records
            
            json_cases.append(case)
        except (ValueError, AttributeError) as e:
            print(f"Error creating target for case {case_id}: {e}")
            continue
    
    return {"cases": json_cases}

def main():
    parser = argparse.ArgumentParser(description="Convert processed weather_pm10 data to JSON format")
    parser.add_argument("--input", default="data/processed/weather_pm10_features.csv", help="Input processed CSV file")
    parser.add_argument("--output", default="input.json", help="Output JSON file path")
    parser.add_argument("--limit", type=int, help="Limit the number of cases to process (for testing)")
    args = parser.parse_args()
    
    # Load the processed data
    df = load_processed_data(args.input)
    
    # Organize data by case
    cases_dict = organize_data_by_case(df)
    
    # Limit cases if specified
    if args.limit and args.limit > 0:
        print(f"Limiting to {args.limit} cases for testing")
        cases_dict = {k: cases_dict[k] for k in list(cases_dict.keys())[:args.limit]}
    
    # Create JSON structure
    json_structure = create_json_structure(cases_dict)
    
    # Save to file
    print(f"Saving output to {args.output}...")
    with open(args.output, "w") as f:
        json.dump(json_structure, f, indent=2)
    
    print(f"Successfully converted data to JSON format with {len(json_structure['cases'])} cases")

if __name__ == "__main__":
    main() 