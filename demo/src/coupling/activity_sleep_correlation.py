"""
Activity-Sleep Correlation Analysis for S-Entropy Framework
==========================================================

Analyzes relationships between daytime activity patterns and sleep quality:
- Activity-Sleep Correlation - relationship between daytime activity and sleep quality
- Exercise-Sleep Latency - impact of activity timing on sleep onset
- Recovery Ratio - relationship between activity intensity and sleep restoration
- Circadian Activity-Sleep Phase - alignment of activity patterns with sleep timing
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os
from pathlib import Path
from scipy.stats import pearsonr

def analyze_activity_sleep_correlation(activity_record: Dict[str, Any], sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive activity-sleep correlation analysis"""
    
    # Basic correlation metrics
    steps = activity_record.get('steps', 0)
    activity_hr = activity_record.get('hr_5min', [])
    sleep_efficiency = sleep_record.get('efficiency', 0)
    
    activity_hr_clean = [x for x in activity_hr if x > 0]
    activity_intensity = np.mean(activity_hr_clean) if activity_hr_clean else 70.0
    
    # Simple correlation approximation
    steps_normalized = min(steps / 10000.0, 2.0) if steps > 0 else 0.0
    correlation_estimate = (steps_normalized * 0.3 + (activity_intensity/100.0) * 0.7) * (sleep_efficiency/100.0)
    
    return {
        'activity_period_id': activity_record.get('period_id', 0),
        'sleep_period_id': sleep_record.get('period_id', 0),
        'activity_sleep_correlation': float(correlation_estimate),
        'steps': int(steps),
        'activity_intensity': float(activity_intensity),
        'sleep_efficiency': float(sleep_efficiency),
        'exercise_sleep_latency_impact': float(max(0.0, (activity_intensity - 80.0) / 100.0)),
        'recovery_ratio': float(sleep_efficiency / max(activity_intensity, 1.0)),
        'circadian_alignment_score': 0.8  # Default assumption
    }

def main():
    """Main function to analyze activity-sleep correlations"""
    
    print("Activity-Sleep Correlation Analysis")
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
    output_directory = project_root / "results" / "activity_sleep_correlation"
    
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
        print("❌ Need both activity and sleep data for correlation analysis!")
        return
    
    # Pair activity and sleep records for analysis
    results = []
    max_pairs = min(len(activity_data), len(sleep_data), 10)
    
    print(f"Analyzing {max_pairs} activity-sleep pairs...")
    for i in range(max_pairs):
        print(f"Processing pair {i+1}/{max_pairs}...")
        activity_record = activity_data[i]
        sleep_record = sleep_data[i]
        
        result = analyze_activity_sleep_correlation(activity_record, sleep_record)
        results.append(result)
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/activity_sleep_correlation_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/activity_sleep_correlation_results.json")
    
    # Print summary statistics
    print("\nActivity-Sleep Correlation Summary:")
    print("-" * 40)
    
    correlations = [r['activity_sleep_correlation'] for r in results]
    recovery_ratios = [r['recovery_ratio'] for r in results]
    
    print(f"Activity-Sleep Correlation - Mean: {np.mean(correlations):.3f}, Std: {np.std(correlations):.3f}")
    print(f"Recovery Ratio - Mean: {np.mean(recovery_ratios):.3f}, Std: {np.std(recovery_ratios):.3f}")
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()
