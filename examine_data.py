#!/usr/bin/env python3
"""
Examine the structure of Krakow Excel/CSV data to understand the format
"""

import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def examine_data_structure():
    """Examine the structure of the Excel and CSV files"""
    
    data_dir = Path("data/raw")
    air_quality_dir = data_dir / "AirQuality_Krakow"
    weather_dir = data_dir / "Weather_Krakow"
    
    print("🔍 EXAMINING DATA STRUCTURE")
    print("=" * 50)
    
    # Check Stations file
    stations_file = air_quality_dir / "Stations.xlsx"
    if stations_file.exists():
        print("\n📍 STATIONS FILE:")
        try:
            stations_df = pd.read_excel(stations_file)
            print(f"   Shape: {stations_df.shape}")
            print(f"   Columns: {list(stations_df.columns)}")
            print(f"   Sample rows:\n{stations_df.head()}")
        except Exception as e:
            print(f"   Error reading stations file: {e}")
    
    # Check Air Quality files
    print(f"\n🌬️  AIR QUALITY FILES:")
    for year in [2019, 2020, 2021, 2022, 2023]:
        air_file = air_quality_dir / f"{year}_PM10_1g.xlsx"
        if air_file.exists():
            try:
                df = pd.read_excel(air_file, nrows=5)  # Read only first 5 rows for structure
                print(f"\n   📅 {year} PM10 Data:")
                print(f"      Shape: {df.shape}")
                print(f"      Columns: {list(df.columns)}")
                print(f"      Sample data:\n{df.head(2)}")
                
                # Check data types
                print(f"      Data types:\n{df.dtypes}")
                
            except Exception as e:
                print(f"   Error reading {year} air quality file: {e}")
        else:
            print(f"   ❌ {year} PM10 file not found")
    
    # Check Weather files
    print(f"\n🌤️  WEATHER FILES:")
    for year in [2019, 2020, 2021, 2022, 2023]:
        weather_file = weather_dir / f"{year}.csv"
        if weather_file.exists():
            try:
                df = pd.read_csv(weather_file, nrows=5)  # Read only first 5 rows
                print(f"\n   📅 {year} Weather Data:")
                print(f"      Shape: {df.shape}")
                print(f"      Columns: {list(df.columns)}")
                print(f"      Sample data:\n{df.head(2)}")
                
                # Check data types
                print(f"      Data types:\n{df.dtypes}")
                
            except Exception as e:
                print(f"   Error reading {year} weather file: {e}")
        else:
            print(f"   ❌ {year} weather file not found")
    
    print("\n" + "=" * 50)
    print("✅ Data structure examination complete!")
    # Examine specific columns AA1 and AA2
    print("\n📊 EXAMINING SPECIFIC COLUMNS (AA1 and AA2):")
    
    # Check in Air Quality files
    print("\n   Air Quality Files:")
    for year in [2019, 2020, 2021, 2022, 2023]:
        air_file = air_quality_dir / f"{year}_PM10_1g.xlsx"
        if air_file.exists():
            try:
                df = pd.read_excel(air_file)
                if 'AA1' in df.columns and 'AA2' in df.columns:
                    print(f"\n   📅 {year} - AA1 and AA2 columns:")
                    print(df[['AA1', 'AA2']].head(5))
                else:
                    available_cols = [col for col in ['AA1', 'AA2'] if col in df.columns]
                    if available_cols:
                        print(f"\n   📅 {year} - Found columns: {available_cols}")
                        print(df[available_cols].head(5))
                    else:
                        print(f"\n   📅 {year} - AA1 and AA2 columns not found")
            except Exception as e:
                print(f"   Error examining AA1/AA2 in {year} air quality file: {e}")
    
    # Check in Weather files
    print("\n   Weather Files:")
    for year in [2019, 2020, 2021, 2022, 2023]:
        weather_file = weather_dir / f"{year}.csv"
        if weather_file.exists():
            try:
                df = pd.read_csv(weather_file)
                if 'AA1' in df.columns and 'AA2' in df.columns:
                    print(f"\n   📅 {year} - AA1 and AA2 columns:")
                    print(df[['AA1', 'AA2']].head(5))
                else:
                    available_cols = [col for col in ['AA1', 'AA2'] if col in df.columns]
                    if available_cols:
                        print(f"\n   📅 {year} - Found columns: {available_cols}")
                        print(df[available_cols].head(5))
                    else:
                        print(f"\n   📅 {year} - AA1 and AA2 columns not found")
            except Exception as e:
                print(f"   Error examining AA1/AA2 in {year} weather file: {e}")

if __name__ == "__main__":
    examine_data_structure()
