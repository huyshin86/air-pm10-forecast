import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
import logging
import argparse
import sys
import traceback
from multiprocessing import Pool, cpu_count
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor

# Gradient boosting libraries
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM not available")

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
    # Fixed GPU availability check
    try:
        # Try to create a temporary CatBoostRegressor with GPU to test availability
        tmp_model = cb.CatBoostRegressor(task_type='GPU', devices='0', verbose=False)
        # Try to fit a small dummy dataset to verify GPU works
        dummy_X = np.random.random((10, 5))
        dummy_y = np.random.random(10)
        tmp_model.fit(dummy_X, dummy_y, verbose=False)
        gpu_available = True
        print("CatBoost GPU support available and working")
        del tmp_model
    except Exception as e:
        gpu_available = False
        print(f"CatBoost GPU not available, using CPU mode. Error: {e}")
except ImportError:
    CATBOOST_AVAILABLE = False
    gpu_available = False
    print("CatBoost not available")

# Neural networks
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    print("PyTorch not available. LSTM models will not work.")

# Time series
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    print("Prophet not available - Prophet models disabled")
    PROPHET_AVAILABLE = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handle data loading and preprocessing for PM10 forecasting with optimizations"""

    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_columns = []
        self.weather_default_values = {
            'rel_hum': 50.0,
            'pressure_hpa': 1013.25,
            'temp_c': 15.0,
            'wind_speed_ms': 5.0,
            'wind_dir_deg': 180.0
        }

    def load_data(self, data_file: str) -> Dict:
        """Load data from JSON file with optimization"""
        start_time = time.time()
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            load_time = time.time() - start_time
            logger.info(f"Loaded data with {len(data.get('cases', []))} cases in {load_time:.2f}s")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise

    def split_cases_temporal(self, cases: List[Dict], train_ratio: float = 0.7,
                             valid_ratio: float = 0.15, test_ratio: float = 0.15) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split cases into train/valid/test based on temporal order"""

        # Sort cases by prediction start time to maintain temporal order
        sorted_cases = sorted(cases, key=lambda x: x['target']['prediction_start_time'])

        n_total = len(sorted_cases)
        n_train = int(n_total * train_ratio)
        n_valid = int(n_total * valid_ratio)

        train_cases = sorted_cases[:n_train]
        valid_cases = sorted_cases[n_train:n_train + n_valid]
        test_cases = sorted_cases[n_train + n_valid:]

        logger.info(f"Temporal split - Train: {len(train_cases)}, Valid: {len(valid_cases)}, Test: {len(test_cases)}")

        return train_cases, valid_cases, test_cases

    def split_cases_random(self, cases: List[Dict], train_ratio: float = 0.7,
                           valid_ratio: float = 0.15, random_state: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split cases randomly into train/valid/test"""

        np.random.seed(random_state)
        shuffled_cases = cases.copy()
        np.random.shuffle(shuffled_cases)

        n_total = len(shuffled_cases)
        n_train = int(n_total * train_ratio)
        n_valid = int(n_total * valid_ratio)

        train_cases = shuffled_cases[:n_train]
        valid_cases = shuffled_cases[n_train:n_train + n_valid]
        test_cases = shuffled_cases[n_train + n_valid:]

        logger.info(f"Random split - Train: {len(train_cases)}, Valid: {len(valid_cases)}, Test: {len(test_cases)}")

        return train_cases, valid_cases, test_cases

    def parse_weather_data(self, weather_record: Dict) -> Dict:
        """Parse pre-engineered weather data with optimization"""
        try:
            # Get timestamp from 'date' field
            date_field = weather_record.get('date')
            if not date_field:
                return None

            timestamp = pd.to_datetime(date_field, errors='coerce')
            if pd.isna(timestamp):
                return None

            # Use pre-engineered weather features directly
            parsed = {
                'timestamp': timestamp,
                'temp_c': weather_record.get('temp_c'),
                'wind_speed_ms': weather_record.get('wind_speed_ms'),
                'wind_dir_deg': weather_record.get('wind_dir_deg'),
                'rel_hum': weather_record.get('rel_hum'),
                'pressure_hpa': weather_record.get('pressure_hpa'),
                # Pre-engineered temporal and categorical features
                'month': weather_record.get('month'),
                'year': weather_record.get('year'),
                'week': weather_record.get('week'),
                'hour': weather_record.get('hour'),
                'doy': weather_record.get('doy'),  # day of year
                'is_winter_month': weather_record.get('is_winter_month'),
                'is_rush_hour': weather_record.get('is_rush_hour'),
                '3_period_SMA': weather_record.get('3_period_SMA')
            }

            return parsed

        except Exception:
            return None

    def create_features_optimized(self, pm10_data: pd.DataFrame, weather_data: pd.DataFrame = None) -> pd.DataFrame:
        """Optimized feature creation using vectorized operations"""

        # Ensure timestamp is datetime
        pm10_data['timestamp'] = pd.to_datetime(pm10_data['timestamp'])
        pm10_data = pm10_data.sort_values('timestamp').reset_index(drop=True)

        # Vectorized temporal features
        dt = pm10_data['timestamp'].dt
        pm10_data['pm10_hour'] = dt.hour
        pm10_data['pm10_day_of_week'] = dt.dayofweek
        pm10_data['pm10_month'] = dt.month
        pm10_data['pm10_day_of_year'] = dt.dayofyear

        # Vectorized cyclical encoding
        hour_rad = 2 * np.pi * pm10_data['pm10_hour'] / 24
        dow_rad = 2 * np.pi * pm10_data['pm10_day_of_week'] / 7
        
        pm10_data['pm10_hour_sin'] = np.sin(hour_rad)
        pm10_data['pm10_hour_cos'] = np.cos(hour_rad)
        pm10_data['pm10_dow_sin'] = np.sin(dow_rad)
        pm10_data['pm10_dow_cos'] = np.cos(dow_rad)

        # Vectorized boolean features
        pm10_data['pm10_is_weekend'] = (pm10_data['pm10_day_of_week'] >= 5).astype(np.int8)
        pm10_data['pm10_is_rush_hour'] = pm10_data['pm10_hour'].isin([7, 8, 9, 17, 18, 19]).astype(np.int8)
        pm10_data['pm10_is_night'] = pm10_data['pm10_hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(np.int8)

        # Season encoding
        pm10_data['pm10_season'] = ((pm10_data['pm10_month'] % 12 + 3) // 3)
        pm10_data['pm10_hour_squared'] = pm10_data['pm10_hour'] ** 2

        # Optimized lag features using shift
        pm10_series = pm10_data['pm10']
        for lag in [1, 2, 3, 6, 12, 24, 48]:
            pm10_data[f'pm10_lag_{lag}'] = pm10_series.shift(lag)

        # Optimized rolling statistics
        for window in [3, 6, 12, 24]:
            rolling = pm10_series.rolling(window=window, min_periods=1)
            pm10_data[f'pm10_rolling_mean_{window}'] = rolling.mean()
            pm10_data[f'pm10_rolling_std_{window}'] = rolling.std()

        # Merge weather data if available
        if weather_data is not None and len(weather_data) > 0:
            weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
            
            # Optimized merge with tolerance
            pm10_data = pd.merge_asof(
                pm10_data.sort_values('timestamp'),
                weather_data.sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('2H')
            )

            # Interaction features
            if 'temp_c' in pm10_data.columns:
                pm10_data['temp_pm10hour_interaction'] = pm10_data['temp_c'] * pm10_data['pm10_hour']
                if 'wind_speed_ms' in pm10_data.columns:
                    pm10_data['wind_temp_interaction'] = pm10_data['wind_speed_ms'] * pm10_data['temp_c']

        # Optimized missing value handling
        pm10_data = pm10_data.fillna(method='ffill').fillna(method='bfill')
        
        # Final fill with defaults
        pm10_data = pm10_data.fillna(0.0)

        return pm10_data

    def create_features(self, pm10_data: pd.DataFrame, weather_data: pd.DataFrame = None) -> pd.DataFrame:
        """Wrapper for optimized feature creation"""
        return self.create_features_optimized(pm10_data, weather_data)

    def prepare_case_data_optimized(self, case: Dict, min_history_hours: int = 1) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
        """Optimized case data preparation with reduced loops"""

        target_info = case['target']
        prediction_start = pd.to_datetime(target_info['prediction_start_time'])

        # Pre-process weather data once
        weather_df = None
        if 'weather' in case and case['weather']:
            weather_records = [
                self.parse_weather_data(record) 
                for record in case['weather']
            ]
            weather_records = [r for r in weather_records if r is not None]
            
            if weather_records:
                weather_df = pd.DataFrame(weather_records)

        # Process all stations efficiently
        all_pm10_records = []
        station_data = []

        for station in case['stations']:
            if not station.get('history'):
                continue

            # Vectorized record processing
            pm10_records = [
                {
                    'timestamp': record['timestamp'],
                    'pm10': record['pm10'],
                    'station_code': station['station_code'],
                    'latitude': station['latitude'],
                    'longitude': station['longitude']
                }
                for record in station['history']
            ]

            if not pm10_records:
                continue

            # Create DataFrame and process
            pm10_df = pd.DataFrame(pm10_records)
            pm10_df['timestamp'] = pd.to_datetime(pm10_df['timestamp'])
            pm10_df = pm10_df.sort_values('timestamp')

            # Create features for this station
            features_df = self.create_features_optimized(
                pm10_df.drop(['station_code', 'latitude', 'longitude'], axis=1), 
                weather_df
            )

            station_info = {
                'station_code': station['station_code'],
                'latitude': station['latitude'],
                'longitude': station['longitude'],
                'features_df': features_df
            }
            station_data.append(station_info)
            all_pm10_records.extend(pm10_records)

        # Strategy selection logic (unchanged)
        if station_data:
            return station_data[0]['features_df'], target_info, station_data

        # Aggregated fallback
        if all_pm10_records:
            all_df = pd.DataFrame(all_pm10_records)
            all_df['timestamp'] = pd.to_datetime(all_df['timestamp'])

            aggregated_df = all_df.groupby('timestamp').agg({
                'pm10': 'mean',
                'latitude': 'mean',
                'longitude': 'mean'
            }).reset_index()
            aggregated_df = aggregated_df.sort_values('timestamp')

            features_df = self.create_features_optimized(
                aggregated_df.drop(['latitude', 'longitude'], axis=1), 
                weather_df
            )

            aggregated_station = {
                'station_code': 'AGGREGATED',
                'latitude': aggregated_df['latitude'].mean(),
                'longitude': aggregated_df['longitude'].mean(),
                'features_df': features_df
            }

            return features_df, target_info, [aggregated_station]

        return pd.DataFrame(), target_info, []

    def prepare_case_data(self, case: Dict, min_history_hours: int = 1) -> Tuple[pd.DataFrame, Dict, List[Dict]]:
        """Wrapper for optimized case data preparation"""
        return self.prepare_case_data_optimized(case, min_history_hours)

    def process_cases_parallel(self, cases: List[Dict], max_workers: int = None) -> List[Tuple]:
        """Process multiple cases in parallel"""
        if max_workers is None:
            max_workers = min(cpu_count(), 8)  # Limit to avoid memory issues

        logger.info(f"Processing {len(cases)} cases with {max_workers} workers")
        
        results = []
        batch_size = 50  # Process in smaller batches to manage memory
        
        for i in range(0, len(cases), batch_size):
            batch = cases[i:i + batch_size]
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                batch_results = list(executor.map(self.prepare_case_data_optimized, batch))
            
            results.extend(batch_results)
            
            if (i // batch_size + 1) % 10 == 0:
                logger.info(f"Processed {min(i + batch_size, len(cases))}/{len(cases)} cases")
        
        return results


class BaseForecaster:
    """Base class for PM10 forecasting models"""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False

    def prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training with optimization"""

        # Select feature columns more efficiently
        numeric_dtypes = ['int8', 'int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        feature_cols = [
            col for col in features_df.columns
            if col not in ['timestamp', 'pm10', 'station_code'] and 
            features_df[col].dtype.name in numeric_dtypes
        ]

        X = features_df[feature_cols].values
        y = features_df['pm10'].values

        return X, y, feature_cols

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train the model"""
        raise NotImplementedError

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        raise NotImplementedError

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate model performance"""
        y_pred = self.predict(X)

        metrics = {
            'mae': mean_absolute_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mape': np.mean(np.abs((y - y_pred) / np.maximum(y, 1e-8))) * 100
        }

        return metrics


class LightGBMForecaster(BaseForecaster):
    """LightGBM-based forecaster with optimizations"""

    def __init__(self, params: Dict = None):
        super().__init__("LightGBM")
        self.params = params or {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 80,
            'max_depth': 8,
            'learning_rate': 0.01,  # Increased for faster training
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'min_data_in_leaf': 10,
            'lambda_l1': 0.1,
            'lambda_l2': 0.1,
            'verbose': -1,  # Suppress verbose output
            'num_threads': min(cpu_count(), 8)  # Use multiple threads
        }

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train LightGBM model with optimizations"""

        if not LIGHTGBM_AVAILABLE:
            logger.error("LightGBM not available for training")
            self.is_trained = False
            return

        train_data = lgb.Dataset(X, label=y)

        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=5000,  # Reduced for faster training
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)]
        )

        self.is_trained = True
        logger.info(f"LightGBM model trained with {self.model.num_trees()} trees")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with LightGBM"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)


class CatBoostForecaster(BaseForecaster):
    """CatBoost-based forecaster with fixed GPU detection"""

    def __init__(self, params: Dict = None):
        super().__init__("CatBoost")
        
        # Use the global gpu_available variable set at module level
        self.params = params or {
            'iterations': 5000,  # Reduced for faster training
            'learning_rate': 0.01,  # Increased for faster convergence
            'depth': 8,
            'loss_function': 'MAE',
            'verbose': 100,  # Reduced verbosity
            'early_stopping_rounds': 100,
            'task_type': 'GPU' if gpu_available else 'CPU',
            'devices': '0' if gpu_available else None,
            'l2_leaf_reg': 1,
            'random_strength': 0.1,
            'bagging_temperature': 0.8,
            'od_wait': 50,
            'max_bin': 254 if gpu_available else 128,
            'thread_count': min(cpu_count(), 8) if not gpu_available else None
        }
        
        if gpu_available:
            logger.info("Using GPU for CatBoost training")
        else:
            logger.info("Using CPU for CatBoost training")

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train CatBoost model with optimizations"""

        if not CATBOOST_AVAILABLE:
            logger.error("CatBoost not available for training")
            self.is_trained = False
            return

        self.model = cb.CatBoostRegressor(**self.params)
        self.model.fit(X, y)

        self.is_trained = True
        logger.info(f"CatBoost model trained with {self.model.tree_count_} trees")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with CatBoost"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)


class RandomForestForecaster(BaseForecaster):
    """Random Forest forecaster with optimizations"""

    def __init__(self, params: Dict = None):
        super().__init__("RandomForest")
        self.params = params or {
            'n_estimators': 1000,  # Reduced for faster training
            'max_depth': 12,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': min(cpu_count(), 8),  # Use multiple cores
            'max_features': 'sqrt'  # Faster feature selection
        }

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train Random Forest model"""
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X, y)
        self.is_trained = True
        logger.info(f"Random Forest model trained with {self.params['n_estimators']} trees")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with Random Forest"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        return self.model.predict(X)


# Keep the LSTM and Prophet forecasters unchanged (they're already optimized)
class LSTMModel(nn.Module):
    """PyTorch LSTM model for PM10 forecasting"""

    def __init__(self, input_size: int, lstm_units: int = 64, dropout_rate: float = 0.3):
        super(LSTMModel, self).__init__()

        self.lstm1 = nn.LSTM(input_size, lstm_units, batch_first=True, dropout=dropout_rate)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.lstm2 = nn.LSTM(lstm_units, lstm_units // 2, batch_first=True, dropout=dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.fc1 = nn.Linear(lstm_units // 2, 32)
        self.relu = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate / 2)

        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm1(x)
        lstm_out = self.dropout1(lstm_out)

        lstm_out, (hidden, _) = self.lstm2(lstm_out)
        lstm_out = self.dropout2(hidden[-1])

        out = self.fc1(lstm_out)
        out = self.relu(out)
        out = self.dropout3(out)

        out = self.fc2(out)
        return out


class LSTMForecaster(BaseForecaster):
    """PyTorch LSTM model for PM10 forecasting"""

    def __init__(self, lstm_units: int = 64, dropout_rate: float = 0.3, sequence_length: int = 24, epochs: int = 200):
        super().__init__("LSTM")
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train PyTorch LSTM model"""
        logger.info(f"Training LSTM with {self.lstm_units} units, dropout={self.dropout_rate}")

        X_scaled = self.scaler.fit_transform(X)
        X_seq, y_seq = self._create_sequences(X_scaled, y)

        if len(X_seq) == 0:
            logger.warning("Not enough data for LSTM sequence creation, skipping LSTM training")
            self.is_trained = False
            return

        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        y_tensor = torch.FloatTensor(y_seq).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=True)  # Increased batch size

        self.model = LSTMModel(X_seq.shape[2], self.lstm_units, self.dropout_rate).to(self.device)

        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=0.001)

        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        patience = 15  # Reduced patience for faster training

        for epoch in range(self.epochs):
            epoch_loss = 0.0
            num_batches = 0

            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs.squeeze(), batch_y)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            avg_loss = epoch_loss / num_batches

            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
            else:
                patience_counter += 1

            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}")
                self.model.load_state_dict(best_model_state)
                break

            if (epoch + 1) % 25 == 0:  # Reduced logging frequency
                logger.info(f"Epoch [{epoch + 1}/{self.epochs}], Loss: {avg_loss:.4f}")

        self.model.eval()
        self.is_trained = True
        logger.info(f"LSTM training completed. Final loss: {best_loss:.4f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained PyTorch LSTM model"""
        if self.model is None or not self.is_trained:
            raise ValueError("Model not trained yet")

        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))

        if len(X_seq) == 0:
            last_seq = X_scaled[-self.sequence_length:] if len(X_scaled) >= self.sequence_length else np.pad(X_scaled, ((self.sequence_length - len(X_scaled), 0), (0, 0)), 'edge')
            X_seq = last_seq.reshape(1, self.sequence_length, X_scaled.shape[1])

        X_tensor = torch.FloatTensor(X_seq).to(self.device)

        self.model.eval()
        with torch.no_grad():
            predictions = self.model(X_tensor)
            predictions = predictions.cpu().numpy().flatten()

        return predictions

    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []

        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])

        return np.array(X_seq), np.array(y_seq)


class ProphetForecaster(BaseForecaster):
    """Facebook Prophet model for PM10 forecasting"""

    def __init__(self, seasonality_mode: str = 'multiplicative', yearly_seasonality: bool = True,
                 weekly_seasonality: bool = True, daily_seasonality: bool = True):
        super().__init__("Prophet")
        self.seasonality_mode = seasonality_mode
        self.yearly_seasonality = yearly_seasonality
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.model = None
        self.feature_columns = []

    def train(self, X: np.ndarray, y: np.ndarray):
        """Train Prophet model"""
        if not PROPHET_AVAILABLE:
            logger.error("Prophet not available for training")
            self.is_trained = False
            return

        logger.info(f"Training Prophet with seasonality_mode={self.seasonality_mode}")

        try:
            timestamps = pd.date_range(start='2020-01-01', periods=len(y), freq='H')
            df = pd.DataFrame({'ds': timestamps, 'y': y})

            self.model = Prophet(
                seasonality_mode=self.seasonality_mode,
                yearly_seasonality=self.yearly_seasonality,
                weekly_seasonality=self.weekly_seasonality,
                daily_seasonality=self.daily_seasonality
            )

            self.model.fit(df)
            self.is_trained = True
            logger.info("Prophet model trained successfully")

        except Exception as e:
            logger.error(f"Prophet training failed: {e}")
            self.is_trained = False

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained Prophet model"""
        if self.model is None or not self.is_trained:
            raise ValueError("Model not trained yet")

        try:
            future_timestamps = pd.date_range(start='2023-01-01', periods=len(X), freq='H')
            future_df = pd.DataFrame({'ds': future_timestamps})
            forecast = self.model.predict(future_df)
            predictions = forecast['yhat'].values
            predictions = np.maximum(predictions, 0.0)
            return predictions

        except Exception as e:
            logger.error(f"Prophet prediction failed: {e}")
            return np.zeros(len(X))


class EnsembleForecaster:
    """Ensemble of multiple forecasting models with optimizations"""

    def __init__(self, models: List[BaseForecaster], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.feature_columns = []
        self.scaler = RobustScaler()

    def train(self, X: np.ndarray, y: np.ndarray, feature_columns: List[str]):
        """Train all models in ensemble with optimizations"""

        self.feature_columns = feature_columns
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)

        failed_models = []
        training_times = {}
        
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.model_name}")
            start_time = time.time()
            
            try:
                model.train(X_scaled, y)
                training_time = time.time() - start_time
                training_times[model.model_name] = training_time
                logger.info(f"✓ {model.model_name} trained successfully in {training_time:.2f}s")
            except Exception as e:
                logger.error(f"✗ Failed to train {model.model_name}: {e}")
                failed_models.append(i)
                self.weights[i] = 0.0

        # Normalize weights
        total_weight = sum(self.weights)
        logger.info(f"Training times: {training_times}")

        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
            
            for i, (model, weight) in enumerate(zip(self.models, self.weights)):
                if weight > 0:
                    logger.info(f"  {model.model_name}: {weight:.3f} ({weight*100:.1f}%)")
                else:
                    logger.info(f"  {model.model_name}: {weight:.3f} (FAILED)")
        else:
            raise ValueError("All models failed to train")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions with optimizations"""

        if X.shape[1] != len(self.feature_columns):
            raise ValueError(f"Input has {X.shape[1]} features, but model was trained with {len(self.feature_columns)} features")

        X_scaled = self.scaler.transform(X)
        predictions = []
        valid_weights = []

        for model, weight in zip(self.models, self.weights):
            if weight > 0 and model.is_trained:
                try:
                    pred = model.predict(X_scaled)
                    predictions.append(pred)
                    valid_weights.append(weight)
                except Exception as e:
                    logger.error(f"Prediction failed for {model.model_name}: {e}")

        if not predictions:
            raise ValueError("No models available for prediction")

        predictions = np.array(predictions)
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()

        ensemble_pred = np.average(predictions, axis=0, weights=valid_weights)
        return ensemble_pred

    def predict_with_individual_outputs(self, X: np.ndarray) -> Dict:
        """Make predictions and return both individual model outputs and ensemble result"""

        if X.shape[1] != len(self.feature_columns):
            raise ValueError(f"Input has {X.shape[1]} features, but model was trained with {len(self.feature_columns)} features")

        X_scaled = self.scaler.transform(X)
        individual_predictions = {}
        ensemble_predictions = []
        valid_weights = []

        for model, weight in zip(self.models, self.weights):
            if weight > 0 and model.is_trained:
                try:
                    pred = model.predict(X_scaled)
                    individual_predictions[model.model_name] = pred
                    ensemble_predictions.append(pred)
                    valid_weights.append(weight)
                except Exception as e:
                    logger.error(f"Prediction failed for {model.model_name}: {e}")
                    individual_predictions[model.model_name] = None

        if not ensemble_predictions:
            raise ValueError("No models available for prediction")

        ensemble_predictions = np.array(ensemble_predictions)
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()

        ensemble_pred = np.average(ensemble_predictions, axis=0, weights=valid_weights)

        return {
            'individual_predictions': individual_predictions,
            'ensemble_prediction': ensemble_pred,
            'weights': {model.model_name: weight for model, weight in zip(self.models, self.weights)}
        }


class PM10ForecastingSystem:
    """Main PM10 forecasting system with optimizations"""

    def __init__(self):
        self.data_processor = DataProcessor()
        self.ensemble = None

    def initialize_models(self) -> EnsembleForecaster:
        """Initialize ensemble of models with optimized parameters"""
        
        models = [CatBoostForecaster()]
        weights = [1.0]

        logger.info(f"Initialized CatBoost model with {'GPU' if gpu_available else 'CPU'} support")
        return EnsembleForecaster(models, weights)

    def train_from_historical_data(self, training_cases: List[Dict]):
        """Train models from historical data with optimizations"""

        logger.info(f"Processing {len(training_cases)} training cases")
        start_time = time.time()

        # Use parallel processing for case preparation
        case_results = self.data_processor.process_cases_parallel(training_cases)
        
        all_features = []
        successful_cases = 0
        failed_cases = 0

        for i, (features_df, _, station_data) in enumerate(case_results):
            if len(station_data) == 0:
                failed_cases += 1
                continue

            # Use the first valid station's data for training
            features_df = station_data[0]['features_df']

            # Optimized missing value filling
            features_df = features_df.fillna(self.data_processor.weather_default_values)

            complete_records = features_df.dropna()
            if len(complete_records) > 0:
                all_features.append(complete_records)
                successful_cases += 1
            else:
                failed_cases += 1

        processing_time = time.time() - start_time
        logger.info(f"Data processing completed in {processing_time:.2f}s: {successful_cases} successful cases, {failed_cases} failed cases")

        if not all_features:
            raise ValueError("No valid training data available")

        # Combine all features efficiently
        logger.info("Combining features...")
        combined_features = pd.concat(all_features, ignore_index=True)

        # Prepare training data
        base_forecaster = BaseForecaster("temp")
        X, y, feature_columns = base_forecaster.prepare_training_data(combined_features)

        logger.info(f"Training data shape: {X.shape}, Features: {len(feature_columns)}")

        # Initialize and train ensemble
        self.ensemble = self.initialize_models()
        train_start = time.time()
        self.ensemble.train(X, y, feature_columns)
        train_time = time.time() - train_start

        logger.info(f"Model training completed in {train_time:.2f}s")

    def train_with_validation(self, training_cases: List[Dict], validation_cases: List[Dict] = None,
                              include_lstm: bool = True, include_prophet: bool = False) -> Dict:
        """Train models with validation and return evaluation metrics"""

        # Process training data with parallel processing
        logger.info(f"Processing {len(training_cases)} training cases")
        train_results = self.data_processor.process_cases_parallel(training_cases)
        
        all_features = []
        for features_df, _, station_data in train_results:
            if len(station_data) == 0:
                continue

            features_df = station_data[0]['features_df']
            features_df = features_df.fillna(self.data_processor.weather_default_values)

            complete_records = features_df.dropna()
            if len(complete_records) > 0:
                all_features.append(complete_records)

        if not all_features:
            raise ValueError("No valid training data available")

        combined_features = pd.concat(all_features, ignore_index=True)
        base_forecaster = BaseForecaster("temp")
        X_train, y_train, feature_columns = base_forecaster.prepare_training_data(combined_features)

        logger.info(f"Training data shape: {X_train.shape}, Features: {len(feature_columns)}")

        # Initialize models with optimized parameters
        models = [
            LightGBMForecaster(),
            CatBoostForecaster(),
            RandomForestForecaster()
        ]

        if include_lstm and TORCH_AVAILABLE:
            models.append(LSTMForecaster(epochs=200))  # Reduced epochs
            logger.info("Added LSTM model with reduced epochs for faster training")

        if include_prophet and PROPHET_AVAILABLE:
            models.append(ProphetForecaster())
            logger.info("Added Prophet model")

        # Set weights
        num_models = len(models)
        if num_models == 3:
            weights = [0.1, 0.3, 0.6]
        elif num_models == 4:
            weights = [0.1, 0.25, 0.5, 0.15]
        elif num_models == 5:
            weights = [0.1, 0.2, 0.4, 0.2, 0.1]
        else:
            weights = [1.0/num_models] * num_models

        logger.info(f"Model names: {[model.model_name for model in models]}")
        self.ensemble = EnsembleForecaster(models, weights)
        self.ensemble.train(X_train, y_train, feature_columns)

        # Evaluate on validation data if provided
        validation_metrics = {}
        if validation_cases:
            logger.info(f"Processing {len(validation_cases)} validation cases")
            val_results = self.data_processor.process_cases_parallel(validation_cases)
            
            val_features = []
            for features_df, _, station_data in val_results:
                if len(station_data) == 0:
                    continue

                features_df = station_data[0]['features_df']
                features_df = features_df.fillna(self.data_processor.weather_default_values)

                complete_records = features_df.dropna()
                if len(complete_records) > 0:
                    val_features.append(complete_records)

            if val_features:
                combined_val_features = pd.concat(val_features, ignore_index=True)
                X_val, y_val, _ = base_forecaster.prepare_training_data(combined_val_features)

                y_pred = self.ensemble.predict(X_val)

                validation_metrics = {
                    'mae': float(mean_absolute_error(y_val, y_pred)),
                    'rmse': float(np.sqrt(mean_squared_error(y_val, y_pred))),
                    'r2': float(r2_score(y_val, y_pred)),
                    'mape': float(np.mean(np.abs((y_val - y_pred) / np.maximum(y_val, 1e-8))) * 100)
                }

                logger.info(f"Validation Metrics - MAE: {validation_metrics['mae']:.2f}, "
                            f"RMSE: {validation_metrics['rmse']:.2f}, "
                            f"R²: {validation_metrics['r2']:.3f}, "
                            f"MAPE: {validation_metrics['mape']:.2f}%")

        logger.info("Model training with validation completed")
        return validation_metrics

    def evaluate_test_set(self, test_cases: List[Dict]) -> Dict:
        """Evaluate trained models on test set with individual model outputs"""

        if self.ensemble is None:
            raise ValueError("Models must be trained before evaluation")

        # Process test data with parallel processing
        logger.info(f"Processing {len(test_cases)} test cases")
        test_results = self.data_processor.process_cases_parallel(test_cases)
        
        test_features = []
        for features_df, _, station_data in test_results:
            if len(station_data) == 0:
                continue

            features_df = station_data[0]['features_df']
            features_df = features_df.fillna(self.data_processor.weather_default_values)

            complete_records = features_df.dropna()
            if len(complete_records) > 0:
                test_features.append(complete_records)

        if not test_features:
            raise ValueError("No valid test data available")

        combined_test_features = pd.concat(test_features, ignore_index=True)
        base_forecaster = BaseForecaster("temp")
        X_test, y_test, _ = base_forecaster.prepare_training_data(combined_test_features)

        # Make predictions with individual model outputs
        prediction_results = self.ensemble.predict_with_individual_outputs(X_test)

        y_pred_ensemble = prediction_results['ensemble_prediction']
        individual_predictions = prediction_results['individual_predictions']
        model_weights = prediction_results['weights']

        # Calculate metrics for ensemble
        ensemble_metrics = {
            'mae_percent': float(mean_absolute_error(y_test, y_pred_ensemble) / np.mean(y_test) * 100),
            'rmse_percent': float(np.sqrt(mean_squared_error(y_test, y_pred_ensemble)) / np.mean(y_test) * 100),
            'r2': float(r2_score(y_test, y_pred_ensemble)),
            'mape': float(np.mean(np.abs((y_test - y_pred_ensemble) / np.maximum(y_test, 1e-8))) * 100),
            'samples': len(y_test),
            'mean_actual': float(np.mean(y_test)),
            'mean_predicted': float(np.mean(y_pred_ensemble)),
            'std_actual': float(np.std(y_test)),
            'std_predicted': float(np.std(y_pred_ensemble))
        }
        
        logger.info(f"Test Metrics:")
        logger.info(f"MAE: {ensemble_metrics['mae_percent']:.2f}%")
        logger.info(f"RMSE: {ensemble_metrics['rmse_percent']:.2f}%")
        logger.info(f"MAPE: {ensemble_metrics['mape']:.2f}%")
        logger.info(f"R²: {ensemble_metrics['r2']:.3f}")

        # Calculate metrics for individual models
        individual_metrics = {}
        for model_name, y_pred_individual in individual_predictions.items():
            if y_pred_individual is not None:
                individual_metrics[model_name] = {
                    'mae': float(mean_absolute_error(y_test, y_pred_individual)),
                    'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred_individual))),
                    'r2': float(r2_score(y_test, y_pred_individual)),
                    'mape': float(np.mean(np.abs((y_test - y_pred_individual) / np.maximum(y_test, 1e-8))) * 100),
                    'mean_predicted': float(np.mean(y_pred_individual)),
                    'std_predicted': float(np.std(y_pred_individual)),
                    'weight': float(model_weights.get(model_name, 0.0))
                }

        # Combine all results
        test_metrics = {
            'ensemble_metrics': ensemble_metrics,
            'individual_model_metrics': individual_metrics,
            'model_weights': model_weights,
            'predictions': {
                'actual': y_test.tolist(),
                'ensemble': y_pred_ensemble.tolist(),
                'individual_models': {
                    model_name: pred.tolist() if pred is not None else None
                    for model_name, pred in individual_predictions.items()
                }
            }
        }

        # Log individual model performance
        for model_name, metrics in individual_metrics.items():
            logger.info(f"{model_name} - MAE: {metrics['mae']:.2f}, "
                        f"RMSE: {metrics['rmse']:.2f}, "
                        f"R²: {metrics['r2']:.3f}, "
                        f"Weight: {metrics['weight']:.3f}")

        return test_metrics

    def forecast_case(self, case: Dict) -> Dict:
        """Generate 24-hour forecast for a single case with optimizations"""

        if self.ensemble is None:
            raise ValueError("Models must be trained before forecasting")

        case_id = case.get('case_id', 'unknown')

        try:
            _, target_info, station_data = self.data_processor.prepare_case_data(case)
        except Exception as e:
            logger.error(f"Error preparing case {case_id}: {e}")
            return {
                'case_id': case_id,
                'status': 'error',
                'error': str(e),
                'forecasts': []
            }

        if not station_data:
            logger.warning(f"Case {case_id} has no valid stations with sufficient historical data")
            return {
                'case_id': case_id,
                'status': 'insufficient_data',
                'forecasts': []
            }

        prediction_start = pd.to_datetime(target_info['prediction_start_time'])
        station_forecasts = []

        for station_info in station_data:
            station_code = station_info['station_code']
            features_df = station_info['features_df']

            try:
                # Fill missing weather values with defaults
                features_df = features_df.fillna(self.data_processor.weather_default_values)

                # Generate features for each of the 24 hours (optimized)
                hourly_forecasts = []
                latest_features = features_df.iloc[-1:].copy()

                for hour in range(24):
                    forecast_time = prediction_start + pd.Timedelta(hours=hour)

                    # Update temporal features for forecast time
                    hour_features = latest_features.copy()
                    hour_features.loc[hour_features.index[0], 'hour'] = forecast_time.hour
                    hour_features.loc[hour_features.index[0], 'day_of_week'] = forecast_time.dayofweek
                    hour_features.loc[hour_features.index[0], 'month'] = forecast_time.month
                    hour_features.loc[hour_features.index[0], 'day_of_year'] = forecast_time.dayofyear

                    # Update cyclical features
                    hour_features.loc[hour_features.index[0], 'hour_sin'] = np.sin(2 * np.pi * forecast_time.hour / 24)
                    hour_features.loc[hour_features.index[0], 'hour_cos'] = np.cos(2 * np.pi * forecast_time.hour / 24)
                    hour_features.loc[hour_features.index[0], 'dow_sin'] = np.sin(2 * np.pi * forecast_time.dayofweek / 7)
                    hour_features.loc[hour_features.index[0], 'dow_cos'] = np.cos(2 * np.pi * forecast_time.dayofweek / 7)

                    # Ensure we have all required columns
                    for col in self.ensemble.feature_columns:
                        if col not in hour_features.columns:
                            hour_features[col] = 0.0

                    # Extract feature data
                    feature_data = hour_features[self.ensemble.feature_columns]
                    X = feature_data.values

                    # Make prediction
                    pm10_pred = self.ensemble.predict(X)[0]
                    pm10_pred = max(0.0, pm10_pred)

                    hourly_forecasts.append({
                        'timestamp': forecast_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                        'pm10_pred': float(pm10_pred)
                    })

                station_forecasts.append({
                    'station_code': station_code,
                    'latitude': station_info['latitude'],
                    'longitude': station_info['longitude'],
                    'forecast': hourly_forecasts
                })

                logger.debug(f"Generated forecast for station {station_code} in case {case_id}")

            except Exception as e:
                logger.error(f"Error forecasting station {station_code} in case {case_id}: {e}")
                station_forecasts.append({
                    'station_code': station_code,
                    'latitude': station_info['latitude'],
                    'longitude': station_info['longitude'],
                    'status': 'error',
                    'error': str(e),
                    'forecast': []
                })

        return {
            'case_id': case_id,
            'status': 'success',
            'forecasts': station_forecasts
        }

    def process_all_cases(self, data: Dict) -> Dict:
        """Process all cases and generate forecasts with optimizations"""

        cases = data.get('cases', [])
        predictions = []
        skipped_cases = []

        logger.info(f"Processing {len(cases)} cases for prediction")
        start_time = time.time()

        # Process in batches for better memory management
        batch_size = 100
        for i in range(0, len(cases), batch_size):
            batch = cases[i:i + batch_size]
            
            for case in batch:
                case_id = case.get('case_id', 'unknown')

                try:
                    forecast_result = self.forecast_case(case)

                    if forecast_result['status'] == 'success' and forecast_result['forecasts']:
                        predictions.append(forecast_result)
                    else:
                        skipped_cases.append({
                            'case_id': case_id,
                            'reason': forecast_result.get('error', 'insufficient_data'),
                            'status': forecast_result['status']
                        })

                except Exception as e:
                    logger.error(f"Error processing case {case_id}: {e}")
                    skipped_cases.append({
                        'case_id': case_id,
                        'reason': str(e),
                        'status': 'error'
                    })

            # Log progress
            if (i // batch_size + 1) % 10 == 0:
                processed = min(i + batch_size, len(cases))
                elapsed = time.time() - start_time
                rate = processed / elapsed
                logger.info(f"Processed {processed}/{len(cases)} cases ({rate:.1f} cases/sec)")

        total_time = time.time() - start_time
        logger.info(f"Processing completed in {total_time:.2f}s")
        logger.info(f"Successfully processed {len(predictions)} cases")
        logger.info(f"Skipped {len(skipped_cases)} cases due to insufficient data or errors")

        return {
            'predictions': predictions,
            'metadata': {
                'total_input_cases': len(cases),
                'successful_predictions': len(predictions),
                'skipped_cases': len(skipped_cases),
                'processing_time_seconds': total_time,
                'skipped_case_details': skipped_cases[:10]  # Limit details to first 10
            }
        }

def main():
    """Main function for CLI interface"""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description='PM10 Forecasting System')
    parser.add_argument('--data-file', required=True, help='Input data JSON file')
    parser.add_argument('--output-file', required=True, help='Output JSON file')
    parser.add_argument('--train', action='store_true', help='Train models from data')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate with train/validation/test splits')
    parser.add_argument('--split-type', choices=['temporal', 'random'], default='temporal',
                        help='Data splitting strategy for evaluation')
    parser.add_argument('--include-lstm', action='store_true', help='Include LSTM model in ensemble')
    parser.add_argument('--include-prophet', action='store_true', help='Include Prophet model in ensemble')

    args = parser.parse_args()

    try:
        # Initialize forecasting system
        system = PM10ForecastingSystem()

        # Load data
        logger.info(f"Loading data from {args.data_file}")
        data = system.data_processor.load_data(args.data_file)
        cases = data.get('cases', [])

        if args.evaluate:
            # Evaluation mode with train/validation/test splits
            logger.info(f"Evaluation mode with {args.split_type} splitting")

            # Split data
            if args.split_type == 'temporal':
                train_cases, valid_cases, test_cases = system.data_processor.split_cases_temporal(
                    cases, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15
                )
            else:
                train_cases, valid_cases, test_cases = system.data_processor.split_cases_random(
                    cases, train_ratio=0.7, valid_ratio=0.15, random_state=42
                )

            with open('train_cases.json', 'w', encoding='utf-8') as f:
                json.dump({'cases': train_cases}, f, indent=2)
            with open('valid_cases.json', 'w', encoding='utf-8') as f:
                json.dump({'cases': valid_cases}, f, indent=2)
            with open('test_cases.json', 'w', encoding='utf-8') as f:
                json.dump({'cases': test_cases}, f, indent=2)
            # Train with validation
            logger.info("Training models with validation...")
            validation_metrics = system.train_with_validation(
                train_cases, valid_cases, include_lstm=args.include_lstm, include_prophet=args.include_prophet
            )

            # Evaluate on test set
            logger.info("Evaluating on test set...")
            test_metrics = system.evaluate_test_set(test_cases)

            # Generate predictions on test set
            logger.info("Generating predictions on test set...")
            test_data = {'cases': test_cases}
            test_predictions = system.process_all_cases(test_data)

            # Save test predictions
            test_output = args.output_file.replace('.json', '_test_predictions.json')
            with open(test_output, 'w') as f:
                json.dump(test_predictions, f, indent=2)
            logger.info(f"Test predictions saved to {test_output}")

            # Save detailed model predictions and metrics
            detailed_output = args.output_file.replace('.json', '_detailed_predictions.json')
            detailed_results = {
                'test_metrics': test_metrics,
                'model_info': {
                    'ensemble_weights': test_metrics['model_weights'],
                    'individual_performance': test_metrics['individual_model_metrics']
                },
                'predictions_summary': {
                    'num_samples': len(test_metrics['predictions']['actual']),
                    'actual_stats': {
                        'mean': float(np.mean(test_metrics['predictions']['actual'])),
                        'std': float(np.std(test_metrics['predictions']['actual'])),
                        'min': float(np.min(test_metrics['predictions']['actual'])),
                        'max': float(np.max(test_metrics['predictions']['actual']))
                    },
                    'ensemble_stats': {
                        'mean': float(np.mean(test_metrics['predictions']['ensemble'])),
                        'std': float(np.std(test_metrics['predictions']['ensemble'])),
                        'min': float(np.min(test_metrics['predictions']['ensemble'])),
                        'max': float(np.max(test_metrics['predictions']['ensemble']))
                    }
                }
            }

            with open(detailed_output, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            logger.info(f"Detailed model predictions and metrics saved to {detailed_output}")

            # Save evaluation results
            eval_results = {
                'evaluation_info': {
                    'timestamp': datetime.now().isoformat(),
                    'split_type': args.split_type,
                    'include_lstm': args.include_lstm,
                    'include_prophet': args.include_prophet,
                    'total_cases': len(cases),
                    'train_cases': len(train_cases),
                    'validation_cases': len(valid_cases),
                    'test_cases': len(test_cases)
                },
                'validation_metrics': validation_metrics,
                'test_metrics': test_metrics
            }

            eval_output = args.output_file.replace('.json', '_evaluation.json')
            with open(eval_output, 'w') as f:
                json.dump(eval_results, f, indent=2)

            logger.info(f"Evaluation results saved to {eval_output}")

            # Print summary
            print(f"\nEvaluation Summary:")
            print(f"=== Ensemble Model Performance ===")
            print(f"Test R²: {test_metrics['ensemble_metrics']['r2']:.3f}")
            print(f"Test MAE: {test_metrics['ensemble_metrics']['mae']:.2f}")
            print(f"Test RMSE: {test_metrics['ensemble_metrics']['rmse']:.2f}")

            print(f"\n=== Individual Model Performance ===")
            for model_name, metrics in test_metrics['individual_model_metrics'].items():
                print(f"{model_name}:")
                print(f"  R²: {metrics['r2']:.3f}, MAE: {metrics['mae']:.2f}, RMSE: {metrics['rmse']:.2f}, Weight: {metrics['weight']:.3f}")

            print(f"\nTest predictions saved to: {test_output}")
            print(f"Detailed evaluation saved to: {eval_output}")
            print(f"Detailed model predictions saved to: {detailed_output}")

        elif args.train:
            # Training mode - use data to train models
            logger.info("Training mode: building models from provided data")
            system.train_from_historical_data(cases)
        else:
            # Inference mode - load pre-trained models or use simple baseline
            logger.info("Inference mode: using baseline models")
            # For hackathon: train on available data quickly
            system.train_from_historical_data(cases)

        # Generate predictions (unless in evaluate-only mode)
        if not args.evaluate:
            logger.info("Generating forecasts...")
            predictions = system.process_all_cases(data)

            # Validate output format
            if not validate_output_format(predictions):
                logger.error("Output format validation failed")
                sys.exit(1)

            # Save predictions
            logger.info(f"Saving predictions to {args.output_file}")
            with open(args.output_file, 'w') as f:
                json.dump(predictions, f, indent=2)

        logger.info("Forecasting completed successfully")
        sys.exit(0)

    except Exception as e:
        logger.error(f"Error in forecasting system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def validate_output_format(predictions: Dict) -> bool:
    """Validate output format matches specification"""

    try:
        # Check top-level structure
        if 'predictions' not in predictions:
            logger.error("Missing 'predictions' key in output")
            return False

        predictions_list = predictions['predictions']
        if not isinstance(predictions_list, list):
            logger.error("'predictions' must be a list")
            return False

        for pred in predictions_list:
            # Check case structure
            if 'case_id' not in pred or 'forecast' not in pred:
                logger.error("Missing 'case_id' or 'forecast' in prediction")
                return False

            forecast = pred['forecast']
            if not isinstance(forecast, list):
                logger.error("'forecast' must be a list")
                return False

            # Check forecast length
            if len(forecast) != 24:
                logger.error(f"Forecast must have exactly 24 predictions, got {len(forecast)}")
                return False

            # Check forecast structure
            for i, f in enumerate(forecast):
                if 'timestamp' not in f or 'pm10_pred' not in f:
                    logger.error(f"Missing 'timestamp' or 'pm10_pred' in forecast hour {i}")
                    return False

                # Validate timestamp format
                try:
                    pd.to_datetime(f['timestamp'])
                except:
                    logger.error(f"Invalid timestamp format: {f['timestamp']}")
                    return False

                # Validate prediction value
                if not isinstance(f['pm10_pred'], (int, float)):
                    logger.error(f"pm10_pred must be numeric, got {type(f['pm10_pred'])}")
                    return False

        logger.info(f"Output validation passed: {len(predictions_list)} cases with 24-hour forecasts each")
        return True

    except Exception as e:
        logger.error(f"Error during output validation: {e}")
        return False


if __name__ == "__main__":
    main()