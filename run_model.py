import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import catboost as cb

# Neural networks
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    tf = None
    print("TensorFlow not available. LSTM models will not work.")
    from tensorflow.keras.optimizers import Adam
except ImportError:
    print("TensorFlow not available - LSTM models disabled")

# Time series
try:
    from prophet import Prophet
except ImportError:
    print("Prophet not available - Prophet models disabled")

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handle data loading and preprocessing for PM10 forecasting"""
    
    def __init__(self):
        self.scaler = RobustScaler()
        self.feature_columns = []
        
    def load_data(self, data_file: str) -> Dict:
        """Load data from JSON file"""
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.info(f"Loaded data with {len(data.get('cases', []))} cases")
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
        """Parse METAR-style weather data"""
        parsed = {
            'timestamp': pd.to_datetime(weather_record.get('date')),
            'temperature': None,
            'wind_speed': None,
            'wind_direction': None,
            'humidity': None,
            'pressure': None
        }
        
        # Parse temperature (e.g., "+0050,1" -> 5.0°C)
        if 'tmp' in weather_record:
            tmp_str = weather_record['tmp']
            if tmp_str and tmp_str != '99999':
                try:
                    # Extract temperature value
                    temp_val = tmp_str.split(',')[0]
                    parsed['temperature'] = float(temp_val) / 10.0
                except:
                    pass
        
        # Parse wind (e.g., "260,1,N,0030,1" -> direction=260, speed=3.0)
        if 'wnd' in weather_record:
            wnd_str = weather_record['wnd']
            if wnd_str:
                try:
                    parts = wnd_str.split(',')
                    if len(parts) >= 4:
                        parsed['wind_direction'] = float(parts[0]) if parts[0] != '999' else None
                        parsed['wind_speed'] = float(parts[3]) / 10.0 if parts[3] != '9999' else None
                except:
                    pass
        
        return parsed
    
    def create_features(self, pm10_data: pd.DataFrame, weather_data: pd.DataFrame = None) -> pd.DataFrame:
        """Create features for modeling"""
        
        # Ensure timestamp is datetime
        pm10_data['timestamp'] = pd.to_datetime(pm10_data['timestamp'])
        pm10_data = pm10_data.sort_values('timestamp').reset_index(drop=True)
        
        # Temporal features
        pm10_data['hour'] = pm10_data['timestamp'].dt.hour
        pm10_data['day_of_week'] = pm10_data['timestamp'].dt.dayofweek
        pm10_data['month'] = pm10_data['timestamp'].dt.month
        pm10_data['day_of_year'] = pm10_data['timestamp'].dt.dayofyear
        
        # Cyclical encoding for temporal features
        pm10_data['hour_sin'] = np.sin(2 * np.pi * pm10_data['hour'] / 24)
        pm10_data['hour_cos'] = np.cos(2 * np.pi * pm10_data['hour'] / 24)
        pm10_data['dow_sin'] = np.sin(2 * np.pi * pm10_data['day_of_week'] / 7)
        pm10_data['dow_cos'] = np.cos(2 * np.pi * pm10_data['day_of_week'] / 7)
        
        # # Lag features
        # for lag in [1, 2, 3, 6, 12, 24, 48]:
        #     pm10_data[f'pm10_lag_{lag}'] = pm10_data['pm10'].shift(lag)
        
        # # Rolling statistics
        # for window in [3, 6, 12, 24]:
        #     pm10_data[f'pm10_rolling_mean_{window}'] = pm10_data['pm10'].rolling(window=window).mean()
        #     pm10_data[f'pm10_rolling_std_{window}'] = pm10_data['pm10'].rolling(window=window).std()
        
        # Weather features integration
        if weather_data is not None and len(weather_data) > 0:
            weather_data['timestamp'] = pd.to_datetime(weather_data['timestamp'])
            
            # Merge with tolerance for time matching
            pm10_data = pd.merge_asof(
                pm10_data.sort_values('timestamp'),
                weather_data.sort_values('timestamp'),
                on='timestamp',
                direction='nearest',
                tolerance=pd.Timedelta('2H')
            )
        
        # Fill missing values
        pm10_data = pm10_data.fillna(method='ffill').fillna(method='bfill')
        
        return pm10_data
    
    def prepare_case_data(self, case: Dict) -> Tuple[pd.DataFrame, Dict]:
        """Process single case data"""
        
        # Extract PM10 data from all stations
        pm10_records = []
        for station in case['stations']:
            for record in station['history']:
                pm10_records.append({
                    'timestamp': record['timestamp'],
                    'pm10': record['pm10'],
                    'station_code': station['station_code'],
                    'latitude': station['latitude'],
                    'longitude': station['longitude']
                })
        
        pm10_df = pd.DataFrame(pm10_records)
        
        # Process weather data
        weather_df = None
        if 'weather' in case and case['weather']:
            weather_records = []
            for weather_record in case['weather']:
                parsed_weather = self.parse_weather_data(weather_record)
                weather_records.append(parsed_weather)
            weather_df = pd.DataFrame(weather_records)
        
        # Create features
        features_df = self.create_features(pm10_df, weather_df)
        
        # Target information
        target_info = case['target']
        
        return features_df, target_info


class BaseForecaster:
    """Base class for PM10 forecasting models"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.is_trained = False
        
    def prepare_training_data(self, features_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for training"""
        
        # Select feature columns (exclude non-numeric and target)
        feature_cols = [col for col in features_df.columns 
                       if col not in ['timestamp', 'pm10', 'station_code'] 
                       and features_df[col].dtype in ['int64', 'float64']]
        
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
            'mape': np.mean(np.abs((y - y_pred) / y)) * 100
        }
        
        return metrics


class LightGBMForecaster(BaseForecaster):
    """LightGBM-based forecaster"""
    
    def __init__(self, params: Dict = None):
        super().__init__("LightGBM")
        self.params = params or {
            'objective': 'regression',
            'metric': 'mae',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': 10
        }
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train LightGBM model"""
        
        # Create dataset
        train_data = lgb.Dataset(X, label=y)
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            num_boost_round=1000,
            valid_sets=[train_data],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
        self.is_trained = True
        logger.info(f"LightGBM model trained with {self.model.num_trees()} trees")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions with LightGBM"""
        if not self.is_trained:
            raise ValueError("Model must be trained before prediction")
        
        return self.model.predict(X)


class CatBoostForecaster(BaseForecaster):
    """CatBoost-based forecaster"""
    
    def __init__(self, params: Dict = None):
        super().__init__("CatBoost")
        self.params = params or {
            'iterations': 1000,
            'learning_rate': 0.1,
            'depth': 6,
            'loss_function': 'MAE',
            'verbose': True,
            'early_stopping_rounds': 50
        }
    
    def train(self, X: np.ndarray, y: np.ndarray):
        """Train CatBoost model"""
        
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
    """Random Forest fallback forecaster"""
    
    def __init__(self, params: Dict = None):
        super().__init__("RandomForest")
        self.params = params or {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': 42,
            'n_jobs': 1  # Single core constraint
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


class LSTMForecaster(BaseForecaster):
    """LSTM model with dropout for PM10 forecasting"""
    
    def __init__(self, lstm_units: int = 64, dropout_rate: float = 0.3, sequence_length: int = 24, epochs: int = 100):
        super().__init__("LSTM")
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.model = None
        self.scaler = StandardScaler()
        
    def fit(self, X: np.ndarray, y: np.ndarray):
        """Train LSTM model with dropout"""
        logger.info(f"Training LSTM with {self.lstm_units} units, dropout={self.dropout_rate}")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create sequences for LSTM
        X_seq, y_seq = self._create_sequences(X_scaled, y)
        
        # Build model
        self.model = tf.keras.Sequential([
            tf.keras.layers.LSTM(self.lstm_units, return_sequences=True, 
                               input_shape=(X_seq.shape[1], X_seq.shape[2])),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.LSTM(self.lstm_units // 2, return_sequences=False),
            tf.keras.layers.Dropout(self.dropout_rate),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dropout(self.dropout_rate / 2),
            tf.keras.layers.Dense(1, activation='linear')
        ])
        
        self.model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Train model
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True
        )
        
        history = self.model.fit(
            X_seq, y_seq,
            epochs=self.epochs,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping],
            verbose=0
        )
        
        logger.info(f"LSTM training completed. Final loss: {history.history['loss'][-1]:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions using trained LSTM model"""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        X_seq, _ = self._create_sequences(X_scaled, np.zeros(len(X_scaled)))
        
        if len(X_seq) == 0:
            # Fallback for small datasets
            logger.warning("Not enough data for sequence creation, using last available sequence")
            # Pad with last known values
            last_seq = X_scaled[-self.sequence_length:] if len(X_scaled) >= self.sequence_length else np.pad(X_scaled, ((self.sequence_length - len(X_scaled), 0), (0, 0)), 'edge')
            X_seq = last_seq.reshape(1, self.sequence_length, X_scaled.shape[1])
            return self.model.predict(X_seq, verbose=0).flatten()
        
        return self.model.predict(X_seq, verbose=0).flatten()
    
    def _create_sequences(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for LSTM training"""
        X_seq, y_seq = [], []
        
        for i in range(self.sequence_length, len(X)):
            X_seq.append(X[i-self.sequence_length:i])
            y_seq.append(y[i])
        
        return np.array(X_seq), np.array(y_seq)


class EnsembleForecaster:
    """Ensemble of multiple forecasting models"""
    
    def __init__(self, models: List[BaseForecaster], weights: List[float] = None):
        self.models = models
        self.weights = weights or [1.0 / len(models)] * len(models)
        self.feature_columns = []
        self.scaler = RobustScaler()
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_columns: List[str]):
        """Train all models in ensemble"""
        
        self.feature_columns = feature_columns
        
        # Fit scaler on training data
        self.scaler.fit(X)
        X_scaled = self.scaler.transform(X)
        
        for i, model in enumerate(self.models):
            logger.info(f"Training model {i+1}/{len(self.models)}: {model.model_name}")
            try:
                model.train(X_scaled, y)
            except Exception as e:
                logger.error(f"Failed to train {model.model_name}: {e}")
                # Set weight to 0 for failed models
                self.weights[i] = 0.0
        
        # Normalize weights
        total_weight = sum(self.weights)
        if total_weight > 0:
            self.weights = [w / total_weight for w in self.weights]
        else:
            raise ValueError("All models failed to train")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make ensemble predictions"""
        
        # Ensure the input has the same features as training
        if X.shape[1] != len(self.feature_columns):
            raise ValueError(f"Input has {X.shape[1]} features, but model was trained with {len(self.feature_columns)} features")
        
        # Scale the input data
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
        
        # Weighted average
        predictions = np.array(predictions)
        valid_weights = np.array(valid_weights)
        valid_weights = valid_weights / valid_weights.sum()  # Normalize
        
        ensemble_pred = np.average(predictions, axis=0, weights=valid_weights)
        
        return ensemble_pred


class PM10ForecastingSystem:
    """Main PM10 forecasting system"""
    
    def __init__(self):
        self.data_processor = DataProcessor()
        self.ensemble = None
        
    def initialize_models(self) -> EnsembleForecaster:
        """Initialize ensemble of models"""
        
        models = [
            LightGBMForecaster(),
            CatBoostForecaster(),
            RandomForestForecaster()
        ]
        
        return EnsembleForecaster(models)
    
    def train_from_historical_data(self, training_cases: List[Dict]):
        """Train models from historical data"""
        
        all_features = []
        all_targets = []
        
        logger.info(f"Processing {len(training_cases)} training cases")
        
        for case in training_cases:
            try:
                features_df, _ = self.data_processor.prepare_case_data(case)
                
                # Fill missing weather values with 0 or reasonable defaults
                features_df = features_df.fillna({
                    'humidity': 50.0,  # Default humidity
                    'pressure': 1013.25,  # Default pressure (sea level)
                    'temperature': 15.0,  # Default temperature
                    'wind_speed': 5.0,  # Default wind speed
                    'wind_direction': 180.0  # Default wind direction
                })
                
                # Only use complete records for training (after filling missing values)
                complete_records = features_df.dropna()
                if len(complete_records) > 0:
                    all_features.append(complete_records)
                    logger.info(f"Added {len(complete_records)} records from case {case.get('case_id', 'unknown')}")
                    
            except Exception as e:
                logger.error(f"Error processing training case: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid training data available")
        
        # Combine all features
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Prepare training data
        base_forecaster = BaseForecaster("temp")
        X, y, feature_columns = base_forecaster.prepare_training_data(combined_features)
        
        logger.info(f"Training data shape: {X.shape}, Features: {len(feature_columns)}")
        
        # Initialize and train ensemble
        self.ensemble = self.initialize_models()
        self.ensemble.train(X, y, feature_columns)
        
        logger.info("Model training completed")
    
    def train_with_validation(self, training_cases: List[Dict], validation_cases: List[Dict] = None, 
                             include_lstm: bool = True) -> Dict:
        """Train models with validation and return evaluation metrics"""
        
        # Process training data
        all_features = []
        all_targets = []
        
        logger.info(f"Processing {len(training_cases)} training cases")
        
        for case in training_cases:
            try:
                features_df, _ = self.data_processor.prepare_case_data(case)
                
                # Fill missing weather values with defaults
                features_df = features_df.fillna({
                    'humidity': 50.0,
                    'pressure': 1013.25,
                    'temperature': 15.0,
                    'wind_speed': 5.0,
                    'wind_direction': 180.0
                })
                
                complete_records = features_df.dropna()
                if len(complete_records) > 0:
                    all_features.append(complete_records)
                    
            except Exception as e:
                logger.error(f"Error processing training case: {e}")
                continue
        
        if not all_features:
            raise ValueError("No valid training data available")
        
        # Combine all features
        combined_features = pd.concat(all_features, ignore_index=True)
        
        # Prepare training data
        base_forecaster = BaseForecaster("temp")
        X_train, y_train, feature_columns = base_forecaster.prepare_training_data(combined_features)
        
        logger.info(f"Training data shape: {X_train.shape}, Features: {len(feature_columns)}")
        
        # Initialize models
        models = [
            LightGBMForecaster(),
            CatBoostForecaster(),
            RandomForestForecaster()
        ]
        
        # Add LSTM if TensorFlow is available and requested
        if include_lstm and TF_AVAILABLE:
            models.append(LSTMForecaster())
            logger.info("Added LSTM model with dropout")
        
        self.ensemble = EnsembleForecaster(models)
        self.ensemble.train(X_train, y_train, feature_columns)
        
        # Evaluate on validation data if provided
        validation_metrics = {}
        if validation_cases:
            val_features = []
            val_targets = []
            
            logger.info(f"Processing {len(validation_cases)} validation cases")
            
            for case in validation_cases:
                try:
                    features_df, _ = self.data_processor.prepare_case_data(case)
                    features_df = features_df.fillna({
                        'humidity': 50.0,
                        'pressure': 1013.25,
                        'temperature': 15.0,
                        'wind_speed': 5.0,
                        'wind_direction': 180.0
                    })
                    
                    complete_records = features_df.dropna()
                    if len(complete_records) > 0:
                        val_features.append(complete_records)
                        
                except Exception as e:
                    logger.error(f"Error processing validation case: {e}")
                    continue
            
            if val_features:
                combined_val_features = pd.concat(val_features, ignore_index=True)
                X_val, y_val, _ = base_forecaster.prepare_training_data(combined_val_features)
                
                # Make predictions
                y_pred = self.ensemble.predict(X_val)
                
                # Calculate metrics
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
        """Evaluate trained models on test set"""
        
        if self.ensemble is None:
            raise ValueError("Models must be trained before evaluation")
        
        # Process test data
        test_features = []
        test_targets = []
        
        logger.info(f"Processing {len(test_cases)} test cases")
        
        for case in test_cases:
            try:
                features_df, _ = self.data_processor.prepare_case_data(case)
                features_df = features_df.fillna({
                    'humidity': 50.0,
                    'pressure': 1013.25,
                    'temperature': 15.0,
                    'wind_speed': 5.0,
                    'wind_direction': 180.0
                })
                
                complete_records = features_df.dropna()
                if len(complete_records) > 0:
                    test_features.append(complete_records)
                    
            except Exception as e:
                logger.error(f"Error processing test case: {e}")
                continue
        
        if not test_features:
            raise ValueError("No valid test data available")
        
        # Combine all features
        combined_test_features = pd.concat(test_features, ignore_index=True)
        
        # Prepare test data
        base_forecaster = BaseForecaster("temp")
        X_test, y_test, _ = base_forecaster.prepare_training_data(combined_test_features)
        
        # Make predictions
        y_pred = self.ensemble.predict(X_test)
        
        # Calculate comprehensive metrics
        test_metrics = {
            'mae': float(mean_absolute_error(y_test, y_pred)),
            'rmse': float(np.sqrt(mean_squared_error(y_test, y_pred))),
            'r2': float(r2_score(y_test, y_pred)),
            'mape': float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-8))) * 100),
            'samples': len(y_test),
            'mean_actual': float(np.mean(y_test)),
            'mean_predicted': float(np.mean(y_pred)),
            'std_actual': float(np.std(y_test)),
            'std_predicted': float(np.std(y_pred))
        }
        
        logger.info(f"Test Metrics - MAE: {test_metrics['mae']:.2f}, "
                   f"RMSE: {test_metrics['rmse']:.2f}, "
                   f"R²: {test_metrics['r2']:.3f}, "
                   f"MAPE: {test_metrics['mape']:.2f}%")
        
        return test_metrics
    
    def forecast_case(self, case: Dict) -> List[Dict]:
        """Generate 24-hour forecast for a single case"""
        
        if self.ensemble is None:
            raise ValueError("Models must be trained before forecasting")
        
        # Process case data
        features_df, target_info = self.data_processor.prepare_case_data(case)
        
        # Fill missing weather values with defaults
        features_df = features_df.fillna({
            'humidity': 50.0,
            'pressure': 1013.25,
            'temperature': 15.0,
            'wind_speed': 5.0,
            'wind_direction': 180.0
        })
        
        # Get prediction start time
        prediction_start = pd.to_datetime(target_info['prediction_start_time'])
        
        # Generate features for each of the 24 hours
        forecasts = []
        
        for hour in range(24):
            forecast_time = prediction_start + pd.Timedelta(hours=hour)
            
            # Create feature vector for this time (using latest available data)
            latest_features = features_df.iloc[-1:].copy()
            
            # Update temporal features for forecast time
            latest_features.loc[latest_features.index[0], 'hour'] = forecast_time.hour
            latest_features.loc[latest_features.index[0], 'day_of_week'] = forecast_time.dayofweek
            latest_features.loc[latest_features.index[0], 'month'] = forecast_time.month
            latest_features.loc[latest_features.index[0], 'day_of_year'] = forecast_time.dayofyear
            
            # Update cyclical features
            latest_features.loc[latest_features.index[0], 'hour_sin'] = np.sin(2 * np.pi * forecast_time.hour / 24)
            latest_features.loc[latest_features.index[0], 'hour_cos'] = np.cos(2 * np.pi * forecast_time.hour / 24)
            latest_features.loc[latest_features.index[0], 'dow_sin'] = np.sin(2 * np.pi * forecast_time.dayofweek / 7)
            latest_features.loc[latest_features.index[0], 'dow_cos'] = np.cos(2 * np.pi * forecast_time.dayofweek / 7)
            
            # Ensure we have all required columns, fill missing ones with defaults
            for col in self.ensemble.feature_columns:
                if col not in latest_features.columns:
                    latest_features[col] = 0.0  # Default value for missing features
            
            # Extract only the feature columns that were used during training
            feature_data = latest_features[self.ensemble.feature_columns]
            X = feature_data.values
            
            # Make prediction
            pm10_pred = self.ensemble.predict(X)[0]
            
            # Ensure non-negative prediction
            pm10_pred = max(0.0, pm10_pred)
            
            forecasts.append({
                'timestamp': forecast_time.strftime('%Y-%m-%dT%H:%M:%SZ'),
                'pm10_pred': float(pm10_pred)
            })
        
        return forecasts
    
    def process_all_cases(self, data: Dict) -> Dict:
        """Process all cases and generate forecasts"""
        
        cases = data.get('cases', [])
        predictions = []
        
        logger.info(f"Processing {len(cases)} cases for prediction")
        
        for case in cases:
            try:
                case_id = case['case_id']
                logger.info(f"Processing case: {case_id}")
                
                forecast = self.forecast_case(case)
                
                predictions.append({
                    'case_id': case_id,
                    'forecast': forecast
                })
                
            except Exception as e:
                logger.error(f"Error processing case {case.get('case_id', 'unknown')}: {e}")
                # Create empty forecast to maintain format
                predictions.append({
                    'case_id': case.get('case_id', 'unknown'),
                    'forecast': [
                        {
                            'timestamp': (pd.to_datetime(case['target']['prediction_start_time']) + 
                                        pd.Timedelta(hours=h)).strftime('%Y-%m-%dT%H:%M:%SZ'),
                            'pm10_pred': 0.0
                        } for h in range(24)
                    ]
                })
        
        return {'predictions': predictions}


def main():
    """Main function for CLI interface"""
    import argparse
    import sys
    
    parser = argparse.ArgumentParser(description='PM10 Forecasting System')
    parser.add_argument('--data-file', required=True, help='Input data JSON file')
    parser.add_argument('--landuse-pbf', help='Land use PBF file (optional)')
    parser.add_argument('--output-file', required=True, help='Output JSON file')
    parser.add_argument('--train', action='store_true', help='Train models from data')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate with train/validation/test splits')
    parser.add_argument('--split-type', choices=['temporal', 'random'], default='temporal', 
                       help='Data splitting strategy for evaluation')
    parser.add_argument('--include-lstm', action='store_true', help='Include LSTM model in ensemble')
    
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
                train_cases, valid_cases, include_lstm=args.include_lstm
            )
            
            # Evaluate on test set
            logger.info("Evaluating on test set...")
            test_metrics = system.evaluate_test_set(test_cases)
            
            # Save evaluation results
            eval_results = {
                'evaluation_info': {
                    'timestamp': datetime.now().isoformat(),
                    'split_type': args.split_type,
                    'include_lstm': args.include_lstm,
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
            print(f"Test R²: {test_metrics['r2']:.3f}")
            print(f"Test MAE: {test_metrics['mae']:.2f}")
            print(f"Test RMSE: {test_metrics['rmse']:.2f}")
            
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