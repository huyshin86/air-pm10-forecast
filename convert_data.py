#!/usr/bin/env python3
"""
Convert Excel/CSV data to the JSON format required by the PM10 forecasting model.
This script processes Krakow air quality and weather data from 2019-2023.
Modified to set prediction_start_time to 2 days after the case day.
"""

import pandas as pd
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataConverter:
    """Convert Krakow Excel/CSV data to model-compatible JSON format"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.air_quality_dir = self.data_dir / "AirQuality_Krakow"
        self.weather_dir = self.data_dir / "Weather_Krakow"
        
    def load_stations(self):
        """Load station metadata"""
        stations_file = self.air_quality_dir / "Stations.xlsx"
        if stations_file.exists():
            return pd.read_excel(stations_file)
        else:
            logger.warning("Stations.xlsx not found, using default station info")
            return None
    
    def load_air_quality_data(self, year: int):
        """Load PM10 data for a specific year"""
        file_path = self.air_quality_dir / f"{year}_PM10_1g.xlsx"
        
        if not file_path.exists():
            logger.error(f"Air quality file not found: {file_path}")
            return None
            
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Convert DateTime column to proper datetime
        if 'DateTime' in df.columns:
            df['DateTime'] = pd.to_datetime(df['DateTime'])
        elif 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.rename(columns={'DATE': 'DateTime'})
        
        return df
    
    def load_weather_data(self, year: int):
        """Load weather data for a specific year"""
        file_path = self.weather_dir / f"{year}.csv"
        
        if not file_path.exists():
            logger.error(f"Weather file not found: {file_path}")
            return None
            
        df = pd.read_csv(file_path)
        
        # Convert DATE column to proper datetime
        if 'DATE' in df.columns:
            df['DATE'] = pd.to_datetime(df['DATE'])
            df = df.rename(columns={'DATE': 'DateTime'})
        
        return df
    
    def encode_weather_metar_style(self, weather_row):
        """Convert weather data to METAR-style format expected by model"""
        weather_record = {}
        
        # Add date for timestamp tracking
        if 'DateTime' in weather_row and pd.notna(weather_row['DateTime']):
            weather_record['date'] = weather_row['DateTime'].strftime('%Y-%m-%dT%H:%M:%S')
        elif 'DATE' in weather_row and pd.notna(weather_row['DATE']):
            date_str = str(weather_row['DATE'])
            if len(date_str) >= 8:
                try:
                    year = date_str[:4]
                    month = date_str[4:6]
                    day = date_str[6:8]
                    hour = date_str[8:10] if len(date_str) >= 10 else '00'
                    formatted_date = f"{year}-{month}-{day}T{hour}:00:00"
                    weather_record['date'] = formatted_date
                except:
                    pass
        
        # Temperature encoding
        if 'TMP' in weather_row and pd.notna(weather_row['TMP']) and weather_row['TMP'] != '99999':
            weather_record['tmp'] = str(weather_row['TMP'])
        
        # Wind encoding
        if 'WND' in weather_row and pd.notna(weather_row['WND']) and weather_row['WND'] != '999999999':
            weather_record['wnd'] = str(weather_row['WND'])
        
        # Sea level pressure
        if 'SLP' in weather_row and pd.notna(weather_row['SLP']) and weather_row['SLP'] != '99999':
            weather_record['slp'] = str(weather_row['SLP'])
            
        # Visibility
        if 'VIS' in weather_row and pd.notna(weather_row['VIS']) and weather_row['VIS'] != '999999':
            weather_record['vis'] = str(weather_row['VIS'])
            
        # Dew point
        if 'DEW' in weather_row and pd.notna(weather_row['DEW']) and weather_row['DEW'] != '99999':
            weather_record['dew'] = str(weather_row['DEW'])
        
        return weather_record
    
    def create_training_cases(self, years: list = [2019, 2020, 2021, 2022, 2023]):
        """Create training cases with prediction_start_time = case_day + 2 days"""
        cases = []
        case_id = 1
        stations_df = self.load_stations()

        for year in years:
            logger.info(f"Processing year {year}...")

            air_df = self.load_air_quality_data(year)
            weather_df = self.load_weather_data(year)

            if air_df is None or weather_df is None:
                logger.warning(f"Skipping year {year} - missing data files")
                continue
            
            # Start from January 1st
            start_date = pd.Timestamp(f'{year}-01-01')
            # End earlier to ensure we have enough data for 2-day-ahead prediction
            end_date = pd.Timestamp(f'{year}-12-29')  # Stop at Dec 29 to allow 2 days ahead
            
            # Create list of all days in the year (except last 2 days)
            date_range = pd.date_range(start=start_date, end=end_date, freq='D')
            
            for current_date in date_range:
                # Each case uses data from current_date (full day)
                case_start = current_date
                case_end = current_date + pd.Timedelta(hours=23, minutes=59)
                
                # Filter air quality data for this day
                case_air_data = air_df[
                    (air_df['DateTime'] >= case_start) & 
                    (air_df['DateTime'] <= case_end)
                ].copy()
                
                # Filter weather data for this day
                case_weather_data = weather_df[
                    (weather_df['DateTime'] >= case_start) & 
                    (weather_df['DateTime'] <= case_end)
                ].copy()
                
                # Need at least 12 hours of data to create a case
                if len(case_air_data) < 12:  
                    continue
                    
                # Prediction start time = current_date + 2 days at 00:00:00
                prediction_date = current_date + pd.Timedelta(days=2)
                prediction_time = prediction_date.strftime('%Y-%m-%dT00:00:00')
                
                # Create case
                case = {
                    "case_id": f"case_{case_id:04d}",
                    "stations": [],
                    "target": {
                        "longitude": 19.926189,  
                        "latitude": 50.057678,
                        "prediction_start_time": prediction_time
                    },
                    "weather": []
                }
                
                # Get station columns (exclude DateTime)
                station_columns = [col for col in case_air_data.columns if col != 'DateTime']
                
                # Add station data
                for station_col in station_columns:
                    station_lat = 50.057678  
                    station_lon = 19.926189
                    
                    if stations_df is not None:
                        station_code = station_col.replace('PM10_', '')
                        station_info = stations_df[stations_df['Station Code'] == station_code]
                        
                        if station_info.empty and station_col.startswith('Mp'):
                            # Try direct match if column name is the station code
                            station_info = stations_df[stations_df['Station Code'] == station_col]
                    
                        if not station_info.empty:
                            station_lat = float(station_info.iloc[0]['WGS84 φ N'])
                            station_lon = float(station_info.iloc[0]['WGS84 λ E'])
                            logger.debug(f"Found coordinates for {station_col}: {station_lat}, {station_lon}")
                        else:
                            logger.warning(f"Station {station_col} not found in stations file, using default coordinates")
                    
                    # Create station history
                    history = []
                    for _, row in case_air_data.iterrows():
                        if pd.notna(row[station_col]):
                            history.append({
                                "timestamp": row['DateTime'].strftime('%Y-%m-%dT%H:%M:%S'),
                                "pm10": float(row[station_col])
                            })
                    
                    # Only add station if has at least 12 hours of data
                    if len(history) >= 12:  
                        case["stations"].append({
                            "station_code": station_col,
                            "longitude": station_lon,
                            "latitude": station_lat,
                            "history": history
                        })
                
                # Add weather data
                for _, weather_row in case_weather_data.iterrows():
                    weather_entry = self.encode_weather_metar_style(weather_row)
                    if weather_entry:  # Only add if has data
                        case["weather"].append(weather_entry)
                
                # Only add case if has sufficient stations and weather data
                if len(case["stations"]) > 0 and len(case["weather"]) > 0:
                    cases.append(case)
                    case_id += 1
                    
                    if case_id % 100 == 0:  # Progress tracking
                        logger.info(f"Created {case_id} cases...")
        
        logger.info(f"Total cases created: {case_id}")
        return cases

    def convert_and_save(self, output_file: str = "krakow_training_data.json", 
                        years: list = [2019, 2020, 2021, 2022, 2023]):
        """Convert data and save to JSON file"""
        
        logger.info("Starting data conversion...")
        
        # Create training cases
        cases = self.create_training_cases(years)
        
        # Wrap in correct format
        training_data = {
            "cases": cases
        }
        
        # Save to JSON
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Conversion complete! Saved {len(cases)} cases to {output_file}")
        
        return training_data

def main():
    """Convert Krakow data to model format with 2-day-ahead prediction"""
    
    converter = DataConverter()
    
    # Convert all years of data
    training_data = converter.convert_and_save(
        output_file="krakow_training_data.json",
        years=[2019, 2020, 2021, 2022, 2023]
    )
    
    print(f"✅ Successfully converted data!")
    print(f"📊 Created {len(training_data['cases'])} training cases")
    print(f"📁 Saved to: krakow_training_data.json")
    
    # Show sample case structure
    if training_data['cases']:
        sample_case = training_data['cases'][0]
        print(f"\n📋 Sample case structure:")
        print(f"   Case ID: {sample_case['case_id']}")
        print(f"   Stations: {len(sample_case['stations'])}")
        print(f"   Weather records: {len(sample_case['weather'])}")
        print(f"   Prediction start time: {sample_case['target']['prediction_start_time']}")
        if sample_case['stations']:
            first_station = sample_case['stations'][0]
            print(f"   Sample station: {first_station['station_code']}")
            print(f"   History length: {len(first_station['history'])} records")
            if first_station['history']:
                print(f"   First timestamp: {first_station['history'][0]['timestamp']}")
                print(f"   Last timestamp: {first_station['history'][-1]['timestamp']}")

if __name__ == "__main__":
    main()