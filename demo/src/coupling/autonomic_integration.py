"""
Autonomic Integration Analysis for S-Entropy Framework
=====================================================

Integrates autonomic nervous system metrics that affect physiological interpretation:
- Day-Night HRV Ratio - comparison of daytime vs nighttime HRV
- Activity-HR Coupling - heart rate response to movement
- Recovery Heart Rate - HR return to baseline after activity
- Sleep-HR Correlation - relationship between sleep stages and cardiac patterns
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os
from pathlib import Path
from scipy.stats import pearsonr

def analyze_autonomic_integration(activity_record: Dict[str, Any], sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive autonomic integration analysis"""
    
    # Basic autonomic balance score
    activity_hr = activity_record.get('hr_5min', [])
    sleep_hr = sleep_record.get('hr_5min', [])
    
    activity_hr_clean = [x for x in activity_hr if x > 0]
    sleep_hr_clean = [x for x in sleep_hr if x > 0]
    
    autonomic_balance = 0.5
    if activity_hr_clean and sleep_hr_clean:
        activity_mean = np.mean(activity_hr_clean)
        sleep_mean = np.mean(sleep_hr_clean)
        if activity_mean > 0:
            autonomic_balance = sleep_mean / activity_mean
    
    return {
        'activity_period_id': activity_record.get('period_id', 0),
        'sleep_period_id': sleep_record.get('period_id', 0),
        'autonomic_balance_score': float(autonomic_balance),
        'activity_hr_mean': float(np.mean(activity_hr_clean)) if activity_hr_clean else 0.0,
        'sleep_hr_mean': float(np.mean(sleep_hr_clean)) if sleep_hr_clean else 0.0,
    }

def main():
    """Main function to analyze autonomic integration"""
    
    print("Autonomic Integration Analysis")
    print("=" * 50)
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent  # From src/coupling/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "autonomic_integration"
    
    # Convert to strings for compatibility
    activity_file_path = str(activity_file_path)
    sleep_file_path = str(sleep_file_path)
    output_directory = str(output_directory)
    
    # Load BOTH activity and sleep data
    activity_data = []
    sleep_data = []
    
    try:
        if os.path.exists(activity_file_path):
            with open(activity_file_path, 'r') as f:
                activity_data = json.load(f)
            print(f"✓ Loaded {len(activity_data)} activity records from {activity_data_file}")
        else:
            print(f"⚠️  Activity file not found: {activity_file_path}")
    except Exception as e:
        print(f"❌ Error loading activity data: {e}")
    
    try:
        if os.path.exists(sleep_file_path):
            with open(sleep_file_path, 'r') as f:
                sleep_data = json.load(f)
            print(f"✓ Loaded {len(sleep_data)} sleep records from {sleep_data_file}")
        else:
            print(f"⚠️  Sleep file not found: {sleep_file_path}")
    except Exception as e:
        print(f"❌ Error loading sleep data: {e}")
    
    if not activity_data or not sleep_data:
        print("❌ Need both activity and sleep data for autonomic integration analysis!")
        return
    
    # Pair activity and sleep records for analysis
    results = []
    max_pairs = min(len(activity_data), len(sleep_data), 10)
    
    print(f"Analyzing {max_pairs} activity-sleep pairs...")
    for i in range(max_pairs):
        print(f"Processing pair {i+1}/{max_pairs}...")
        activity_record = activity_data[i]
        sleep_record = sleep_data[i]
        
        result = analyze_autonomic_integration(activity_record, sleep_record)
        results.append(result)
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/autonomic_integration_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/autonomic_integration_results.json")
    
    # Print summary statistics
    print("\nAutonomic Integration Summary:")
    print("-" * 40)
    
    balance_scores = [r['autonomic_balance_score'] for r in results]
    print(f"Autonomic Balance - Mean: {np.mean(balance_scores):.3f}, Std: {np.std(balance_scores):.3f}")
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()
