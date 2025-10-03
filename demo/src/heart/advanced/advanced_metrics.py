"""
Respiratory Sinus Arrhythmia (RSA) - HR variation with breathing
Baroreflex Sensitivity - blood pressure-HR coupling
QT Variability - ventricular repolarization variability
T-wave Alternans - repolarization pattern changes
Heart Rate Turbulence - post-ectopic beat dynamics
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from pathlib import Path
import os
from scipy import signal

def respiratory_sinus_arrhythmia(sleep_record: Dict[str, Any]) -> float:
    """Calculate HR variation with breathing"""
    hr_sequence = sleep_record.get('hr_5min', [])
    breath_avg = sleep_record.get('breath_average', 16.0)
    
    if not hr_sequence or len(hr_sequence) < 5:
        return 0.0
    
    # RSA is correlated with HR variability during controlled breathing
    hr_array = np.array([hr for hr in hr_sequence if hr > 0])
    
    if len(hr_array) < 3:
        return 0.0
    
    # Estimate RSA from HR variability scaled by breathing rate
    hr_variability = np.std(hr_array)
    breathing_factor = abs(breath_avg - 16) / 16  # Deviation from normal
    
    rsa = hr_variability * (1 + breathing_factor)
    return float(rsa)

def baroreflex_sensitivity(sleep_record: Dict[str, Any]) -> float:
    """Calculate blood pressure-HR coupling"""
    hr_sequence = sleep_record.get('hr_5min', [])
    
    if not hr_sequence or len(hr_sequence) < 3:
        return 0.0
    
    # Estimate BRS from HR sequence analysis
    hr_array = np.array([hr for hr in hr_sequence if hr > 0])
    
    # BRS approximation from consecutive HR differences
    hr_diffs = np.diff(hr_array)
    brs_estimate = np.std(hr_diffs) * 0.5  # Scaling factor
    
    return float(brs_estimate)

def qt_variability(sleep_record: Dict[str, Any]) -> float:
    """Calculate ventricular repolarization variability"""
    hr_sequence = sleep_record.get('hr_5min', [])
    
    if not hr_sequence:
        return 0.0
    
    # QT variability estimation from HR
    hr_array = np.array([hr for hr in hr_sequence if hr > 0])
    
    if len(hr_array) < 2:
        return 0.0
    
    # QT interval inversely related to HR
    qt_intervals = 400 - (hr_array - 60) * 2  # Simplified QT formula
    qt_variability_val = np.std(qt_intervals)
    
    return float(qt_variability_val)

def heart_rate_turbulence(sleep_record: Dict[str, Any]) -> Dict[str, float]:
    """Calculate post-ectopic beat dynamics"""
    hr_sequence = sleep_record.get('hr_5min', [])
    
    if not hr_sequence or len(hr_sequence) < 5:
        return {'turbulence_onset': 0.0, 'turbulence_slope': 0.0}
    
    hr_array = np.array([hr for hr in hr_sequence if hr > 0])
    
    # Simplified HRT calculation
    # Look for sudden HR changes (simulating ectopic beats)
    hr_diffs = np.diff(hr_array)
    large_changes = np.where(np.abs(hr_diffs) > np.std(hr_diffs) * 2)[0]
    
    if len(large_changes) == 0:
        return {'turbulence_onset': 0.0, 'turbulence_slope': 0.0}
    
    # Turbulence Onset (TO) - initial acceleration
    turbulence_onset = np.mean(np.abs(hr_diffs[large_changes])) / np.mean(hr_array)
    
    # Turbulence Slope (TS) - subsequent deceleration
    if len(large_changes) > 1:
        post_changes = hr_diffs[large_changes[1:]]
        turbulence_slope = np.mean(post_changes)
    else:
        turbulence_slope = 0.0
    
    return {
        'turbulence_onset': float(turbulence_onset),
        'turbulence_slope': float(turbulence_slope)
    }

def analyze_advanced_cardiac_metrics(sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze complete advanced cardiac metrics"""
    
    rsa = respiratory_sinus_arrhythmia(sleep_record)
    brs = baroreflex_sensitivity(sleep_record)
    qtv = qt_variability(sleep_record)
    hrt = heart_rate_turbulence(sleep_record)
    
    result = {
        'period_id': sleep_record.get('period_id', 0),
        'timestamp': sleep_record.get('bedtime_start_dt_adjusted', 0),
        'respiratory_sinus_arrhythmia': rsa,
        'baroreflex_sensitivity': brs,
        'qt_variability': qtv,
        'heart_rate_turbulence_onset': hrt['turbulence_onset'],
        'heart_rate_turbulence_slope': hrt['turbulence_slope']
    }
    
    # Add cardiovascular health interpretation
    if rsa > 5 and brs > 3:
        result['cardiovascular_health'] = 'Good'
    elif rsa > 2 and brs > 1:
        result['cardiovascular_health'] = 'Average'
    else:
        result['cardiovascular_health'] = 'Below Average'
    
    return result

def main():
    """Main function to analyze advanced cardiac metrics"""
    print("Advanced Cardiac Metrics Analysis")
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent.parent  # From src/heart/advanced/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "advanced_cardiac"
    
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
    
    # Combine and process data
    all_results = []
    
    # Process sleep data (primary source for advanced cardiac analysis)
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Analyzing sleep record {i+1}/10...")
            result = analyze_advanced_cardiac_metrics(record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    # Process activity data for additional context
    if activity_data:
        print("Processing activity records...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {i+1}/10...")
            result = analyze_advanced_cardiac_metrics(record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/advanced_cardiac_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/advanced_cardiac_results.json")
    
    # Show data source breakdown
    activity_count = sum(1 for r in all_results if r.get('data_source') == 'activity')
    sleep_count = sum(1 for r in all_results if r.get('data_source') == 'sleep')
    print(f"Data sources: {activity_count} activity records, {sleep_count} sleep records")
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()