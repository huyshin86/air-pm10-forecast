#!/usr/bin/env python3
"""
Evaluation script for PM10 forecasting system with train/validation/test splits
"""

import json
import logging
import sys
import os
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from run_model import PM10ForecastingSystem

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_evaluation(data_file: str, split_type: str = 'temporal', include_lstm: bool = True):
    """
    Run complete evaluation pipeline with train/validation/test splits
    
    Args:
        data_file: Path to input JSON data file
        split_type: 'temporal' or 'random' splitting strategy
        include_lstm: Whether to include LSTM model in ensemble
    """
    
    logger.info("Starting PM10 forecasting evaluation")
    logger.info(f"Data file: {data_file}")
    logger.info(f"Split type: {split_type}")
    logger.info(f"Include LSTM: {include_lstm}")
    
    try:
        # Initialize system
        system = PM10ForecastingSystem()
        
        # Load data
        logger.info("Loading data...")
        data = system.data_processor.load_data(data_file)
        cases = data.get('cases', [])
        
        if len(cases) == 0:
            raise ValueError("No cases found in data file")
        
        logger.info(f"Loaded {len(cases)} cases")
        
        # Split data
        logger.info(f"Splitting data using {split_type} strategy...")
        
        if split_type == 'temporal':
            train_cases, valid_cases, test_cases = system.data_processor.split_cases_temporal(
                cases, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15
            )
        else:
            train_cases, valid_cases, test_cases = system.data_processor.split_cases_random(
                cases, train_ratio=0.7, valid_ratio=0.15, random_state=42
            )
        
        # Train models with validation
        logger.info("Training models with validation...")
        validation_metrics = system.train_with_validation(
            train_cases, valid_cases, include_lstm=include_lstm
        )
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_metrics = system.evaluate_test_set(test_cases)
        
        # Prepare results
        results = {
            'evaluation_info': {
                'timestamp': datetime.now().isoformat(),
                'data_file': data_file,
                'split_type': split_type,
                'include_lstm': include_lstm,
                'total_cases': len(cases),
                'train_cases': len(train_cases),
                'validation_cases': len(valid_cases),
                'test_cases': len(test_cases)
            },
            'validation_metrics': validation_metrics,
            'test_metrics': test_metrics
        }
        
        # Save results
        output_file = f"evaluation_results_{split_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {output_file}")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION SUMMARY")
        print("="*60)
        print(f"Split Strategy: {split_type}")
        print(f"LSTM Included: {include_lstm}")
        print(f"Total Cases: {len(cases)}")
        print(f"Train/Valid/Test: {len(train_cases)}/{len(valid_cases)}/{len(test_cases)}")
        
        if validation_metrics:
            print("\nValidation Metrics:")
            print(f"  MAE: {validation_metrics['mae']:.2f}")
            print(f"  RMSE: {validation_metrics['rmse']:.2f}")
            print(f"  R²: {validation_metrics['r2']:.3f}")
            print(f"  MAPE: {validation_metrics['mape']:.2f}%")
        
        print("\nTest Metrics:")
        print(f"  MAE: {test_metrics['mae']:.2f}")
        print(f"  RMSE: {test_metrics['rmse']:.2f}")
        print(f"  R²: {test_metrics['r2']:.3f}")
        print(f"  MAPE: {test_metrics['mape']:.2f}%")
        print(f"  Test Samples: {test_metrics['samples']}")
        
        # Performance assessment
        r2_score = test_metrics['r2']
        if r2_score > 0.8:
            performance = "Excellent"
        elif r2_score > 0.6:
            performance = "Good"
        elif r2_score > 0.4:
            performance = "Fair"
        else:
            performance = "Poor"
        
        print(f"\nOverall Performance: {performance} (R² = {r2_score:.3f})")
        print("="*60)
        
        return results
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise

def compare_splits(data_file: str, include_lstm: bool = True):
    """Compare temporal vs random splitting strategies"""
    
    logger.info("Comparing temporal vs random splits")
    
    temporal_results = run_evaluation(data_file, 'temporal', include_lstm)
    random_results = run_evaluation(data_file, 'random', include_lstm)
    
    # Compare results
    print("\n" + "="*60)
    print("SPLIT COMPARISON")
    print("="*60)
    
    temporal_r2 = temporal_results['test_metrics']['r2']
    random_r2 = random_results['test_metrics']['r2']
    
    print(f"Temporal Split - Test R²: {temporal_r2:.3f}")
    print(f"Random Split   - Test R²: {random_r2:.3f}")
    
    if temporal_r2 > random_r2:
        print("→ Temporal splitting performs better (expected for time series)")
    else:
        print("→ Random splitting performs better (unexpected for time series)")
    
    print("="*60)

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate PM10 forecasting system')
    parser.add_argument('--data-file', required=True, help='Input JSON data file')
    parser.add_argument('--split-type', choices=['temporal', 'random', 'both'], 
                       default='temporal', help='Data splitting strategy')
    parser.add_argument('--no-lstm', action='store_true', 
                       help='Exclude LSTM from ensemble')
    
    args = parser.parse_args()
    
    include_lstm = not args.no_lstm
    
    if args.split_type == 'both':
        compare_splits(args.data_file, include_lstm)
    else:
        run_evaluation(args.data_file, args.split_type, include_lstm)

if __name__ == "__main__":
    main()
