"""
Testing and Validation Utilities for PM10 Forecasting System
Author: Thảo Vân (ML Engineer)
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import unittest
from typing import Dict, List
import os
import tempfile

def create_sample_data(num_cases: int = 2, history_hours: int = 168) -> Dict:
    """Create sample data for testing"""
    
    cases = []
    
    for case_idx in range(num_cases):
        # Generate sample PM10 data
        base_time = datetime(2025, 1, 1)
        stations = []
        
        for station_idx in range(2):  # 2 stations per case
            history = []
            
            for hour in range(history_hours):
                timestamp = base_time + timedelta(hours=hour)
                
                # Generate realistic PM10 values with patterns
                base_pm10 = 30 + 15 * np.sin(2 * np.pi * hour / 24)  # Daily cycle
                noise = np.random.normal(0, 5)  # Random noise
                pm10_value = max(0, base_pm10 + noise)
                
                history.append({
                    "timestamp": timestamp.isoformat(),
                    "pm10": round(pm10_value, 1)
                })
            
            stations.append({
                "station_code": f"Station_{case_idx}_{station_idx}",
                "longitude": -74.0060 + station_idx * 0.01,
                "latitude": 40.7128 + station_idx * 0.01,
                "history": history
            })
        
        # Generate sample weather data
        weather = []
        for hour in range(history_hours):
            timestamp = base_time + timedelta(hours=hour)
            
            # Generate sample weather values
            temp = 15 + 10 * np.sin(2 * np.pi * hour / 24) + np.random.normal(0, 2)
            wind_speed = max(0, 10 + np.random.normal(0, 3))
            wind_dir = np.random.randint(0, 360)
            
            weather.append({
                "date": timestamp.isoformat(),
                "tmp": f"+{int(temp*10):04d},1",
                "wnd": f"{wind_dir:03d},1,N,{int(wind_speed*10):04d},1"
            })
        
        # Define target
        prediction_start = base_time + timedelta(hours=history_hours)
        
        cases.append({
            "case_id": f"case_{case_idx:04d}",
            "stations": stations,
            "target": {
                "longitude": -74.0060,
                "latitude": 40.7128,
                "prediction_start_time": prediction_start.isoformat()
            },
            "weather": weather
        })
    
    return {"cases": cases}


def create_expected_output(input_data: Dict) -> Dict:
    """Create expected output format for testing"""
    
    predictions = []
    
    for case in input_data['cases']:
        case_id = case['case_id']
        prediction_start = pd.to_datetime(case['target']['prediction_start_time'])
        
        forecast = []
        for hour in range(24):
            timestamp = prediction_start + pd.Timedelta(hours=hour)
            
            # Generate dummy prediction values
            base_pred = 35 + 10 * np.sin(2 * np.pi * hour / 24)
            
            forecast.append({
                "timestamp": timestamp.strftime('%Y-%m-%dT%H:%M:%SZ'),
                "pm10_pred": round(base_pred, 1)
            })
        
        predictions.append({
            "case_id": case_id,
            "forecast": forecast
        })
    
    return {"predictions": predictions}


class TestPM10ForecastingSystem(unittest.TestCase):
    """Unit tests for PM10 forecasting system"""
    
    def setUp(self):
        """Set up test environment"""
        from ml_core_implementation import PM10ForecastingSystem, DataProcessor
        
        self.system = PM10ForecastingSystem()
        self.data_processor = DataProcessor()
        self.sample_data = create_sample_data(num_cases=2, history_hours=72)
    
    def test_data_loading(self):
        """Test data loading functionality"""
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(self.sample_data, f)
            temp_file = f.name
        
        try:
            # Test loading
            loaded_data = self.data_processor.load_data(temp_file)
            
            self.assertIn('cases', loaded_data)
            self.assertEqual(len(loaded_data['cases']), 2)
            
        finally:
            os.unlink(temp_file)
    
    def test_weather_parsing(self):
        """Test weather data parsing"""
        
        weather_record = {
            "date": "2025-01-01T12:00:00",
            "tmp": "+0150,1",
            "wnd": "260,1,N,0050,1"
        }
        
        parsed = self.data_processor.parse_weather_data(weather_record)
        
        self.assertEqual(parsed['temperature'], 15.0)
        self.assertEqual(parsed['wind_direction'], 260.0)
        self.assertEqual(parsed['wind_speed'], 5.0)
    
    def test_feature_creation(self):
        """Test feature engineering"""
        
        # Create sample PM10 dataframe
        pm10_data = pd.DataFrame([
            {'timestamp': '2025-01-01T00:00:00', 'pm10': 30.0},
            {'timestamp': '2025-01-01T01:00:00', 'pm10': 35.0},
            {'timestamp': '2025-01-01T02:00:00', 'pm10': 32.0}
        ])
        
        features = self.data_processor.create_features(pm10_data)
        
        # Check temporal features
        self.assertIn('hour', features.columns)
        self.assertIn('day_of_week', features.columns)
        self.assertIn('hour_sin', features.columns)
        self.assertIn('hour_cos', features.columns)
        
        # Check lag features
        self.assertIn('pm10_lag_1', features.columns)
        self.assertIn('pm10_rolling_mean_3', features.columns)
    
    def test_case_data_preparation(self):
        """Test case data preparation"""
        
        case = self.sample_data['cases'][0]
        features_df, target_info = self.data_processor.prepare_case_data(case)
        
        self.assertIsInstance(features_df, pd.DataFrame)
        self.assertIn('pm10', features_df.columns)
        self.assertIn('timestamp', features_df.columns)
        
        self.assertIn('prediction_start_time', target_info)
        self.assertIn('longitude', target_info)
        self.assertIn('latitude', target_info)
    
    def test_model_training(self):
        """Test model training process"""
        
        # Use smaller dataset for faster testing
        small_data = create_sample_data(num_cases=1, history_hours=48)
        
        try:
            self.system.train_from_historical_data(small_data['cases'])
            self.assertIsNotNone(self.system.ensemble)
            
        except Exception as e:
            self.fail(f"Model training failed: {e}")
    
    def test_forecasting(self):
        """Test forecasting functionality"""
        
        # Train system first
        self.system.train_from_historical_data(self.sample_data['cases'])
        
        # Test forecasting
        case = self.sample_data['cases'][0]
        forecast = self.system.forecast_case(case)
        
        # Validate forecast
        self.assertEqual(len(forecast), 24)
        
        for pred in forecast:
            self.assertIn('timestamp', pred)
            self.assertIn('pm10_pred', pred)
            self.assertIsInstance(pred['pm10_pred'], float)
            self.assertGreaterEqual(pred['pm10_pred'], 0.0)
    
    def test_output_format_validation(self):
        """Test output format validation"""
        
        from ml_core_implementation import validate_output_format
        
        # Test valid output
        valid_output = create_expected_output(self.sample_data)
        self.assertTrue(validate_output_format(valid_output))
        
        # Test invalid output - missing predictions
        invalid_output = {}
        self.assertFalse(validate_output_format(invalid_output))
        
        # Test invalid output - wrong forecast length
        invalid_output = {
            "predictions": [{
                "case_id": "test",
                "forecast": [{"timestamp": "2025-01-01T00:00:00Z", "pm10_pred": 30.0}]  # Only 1 instead of 24
            }]
        }
        self.assertFalse(validate_output_format(invalid_output))


def run_performance_test():
    """Run performance test to ensure <5 minute runtime"""
    
    import time
    from ml_core_implementation import PM10ForecastingSystem
    
    print("Running performance test...")
    
    # Create larger test dataset
    test_data = create_sample_data(num_cases=10, history_hours=168)  # 1 week of data
    
    system = PM10ForecastingSystem()
    
    start_time = time.time()
    
    try:
        # Train models
        print("Training models...")
        system.train_from_historical_data(test_data['cases'])
        
        # Generate predictions
        print("Generating predictions...")
        predictions = system.process_all_cases(test_data)
        
        end_time = time.time()
        runtime = end_time - start_time
        
        print(f"Performance test completed in {runtime:.2f} seconds")
        
        if runtime > 300:  # 5 minutes
            print("WARNING: Runtime exceeds 5-minute limit!")
            return False
        else:
            print("✓ Runtime requirement met")
            return True
            
    except Exception as e:
        print(f"Performance test failed: {e}")
        return False


def run_memory_test():
    """Test memory usage"""
    
    import psutil
    import os
    
    print("Running memory test...")
    
    process = psutil.Process(os.getpid())
    initial_memory = process.memory_info().rss / 1024 / 1024  # MB
    
    # Create test data and run system
    test_data = create_sample_data(num_cases=5, history_hours=168)
    
    from ml_core_implementation import PM10ForecastingSystem
    system = PM10ForecastingSystem()
    
    try:
        system.train_from_historical_data(test_data['cases'])
        predictions = system.process_all_cases(test_data)
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = peak_memory - initial_memory
        
        print(f"Memory used: {memory_used:.2f} MB")
        
        if memory_used > 2048:  # 2GB limit
            print("WARNING: Memory usage exceeds 2GB limit!")
            return False
        else:
            print("✓ Memory requirement met")
            return True
            
    except Exception as e:
        print(f"Memory test failed: {e}")
        return False


def create_test_files():
    """Create test files for CLI testing"""
    
    # Create test input file
    test_data = create_sample_data(num_cases=3, history_hours=72)
    
    with open('test_input.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print("Created test_input.json")
    
    # Create expected output file
    expected_output = create_expected_output(test_data)
    
    with open('expected_output.json', 'w') as f:
        json.dump(expected_output, f, indent=2)
    
    print("Created expected_output.json")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "unittest":
            # Run unit tests
            unittest.main(argv=[''], exit=False)
        
        elif sys.argv[1] == "performance":
            # Run performance test
            success = run_performance_test()
            sys.exit(0 if success else 1)
        
        elif sys.argv[1] == "memory":
            # Run memory test
            success = run_memory_test()
            sys.exit(0 if success else 1)
        
        elif sys.argv[1] == "create_tests":
            # Create test files
            create_test_files()
        
        else:
            print("Available commands: unittest, performance, memory, create_tests")
    
    else:
        print("Usage: python testing_utilities.py [unittest|performance|memory|create_tests]")
        print("Running all tests...")
        
        # Run unit tests
        print("\n" + "="*50)
        print("RUNNING UNIT TESTS")
        print("="*50)
        unittest.main(argv=[''], exit=False)
        
        # Run performance test
        print("\n" + "="*50)
        print("RUNNING PERFORMANCE TEST")
        print("="*50)
        run_performance_test()
        
        # Run memory test
        print("\n" + "="*50)
        print("RUNNING MEMORY TEST")
        print("="*50)
        run_memory_test()
        
        print("\nAll tests completed!")