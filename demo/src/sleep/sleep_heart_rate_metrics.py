"""
Sleep Heart Rate Metrics
Sleep HR Profile - heart rate patterns across sleep stages
Nocturnal HR Variability - HRV specifically during sleep
Sleep-Wake HR Transition - HR changes at sleep boundaries
Autonomic Recovery - parasympathetic activation during sleep
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os
from pathlib import Path

def sleep_hr_profile(sleep_record: Dict[str, Any]) -> Dict[str, float]:
    """Calculate heart rate patterns across sleep stages"""
    hr_sequence = sleep_record.get('hr_5min', [])
    hypnogram = sleep_record.get('hypnogram_5min', '')
    
    stage_hrs = {'A': [], 'L': [], 'D': [], 'R': []}
    
    for i, stage in enumerate(hypnogram):
        if i < len(hr_sequence) and hr_sequence[i] > 0 and stage in stage_hrs:
            stage_hrs[stage].append(hr_sequence[i])
    
    return {
        'awake_hr': float(np.mean(stage_hrs['A']) if stage_hrs['A'] else 0.0),
        'light_hr': float(np.mean(stage_hrs['L']) if stage_hrs['L'] else 0.0),
        'deep_hr': float(np.mean(stage_hrs['D']) if stage_hrs['D'] else 0.0),
        'rem_hr': float(np.mean(stage_hrs['R']) if stage_hrs['R'] else 0.0)
    }

def nocturnal_hr_variability(sleep_record: Dict[str, Any]) -> Dict[str, float]:
    """Calculate HRV specifically during sleep"""
    hr_sequence = sleep_record.get('hr_5min', [])
    sleep_hrs = [hr for hr in hr_sequence if hr > 0]
    
    if len(sleep_hrs) < 2:
        return {'sleep_rmssd': 0.0, 'sleep_sdnn': 0.0}
    
    hr_array = np.array(sleep_hrs)
    successive_diffs = np.diff(hr_array)
    
    return {
        'sleep_rmssd': float(np.sqrt(np.mean(successive_diffs**2))),
        'sleep_sdnn': float(np.std(hr_array, ddof=1))
    }

def autonomic_recovery(sleep_record: Dict[str, Any]) -> Dict[str, float]:
    """Calculate parasympathetic activation during sleep"""
    rmssd_sequence = sleep_record.get('rmssd_5min', [])
    avg_rmssd = np.mean([r for r in rmssd_sequence if r > 0]) if rmssd_sequence else 0.0
    
    recovery_score = min(100, avg_rmssd * 2)
    return {
        'autonomic_recovery_score': float(recovery_score),
        'parasympathetic_activation': float(min(100, avg_rmssd * 1.5))
    }

def analyze_sleep_heart_rate_metrics(sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze complete sleep heart rate metrics"""
    hr_profile = sleep_hr_profile(sleep_record)
    hrv = nocturnal_hr_variability(sleep_record)
    recovery = autonomic_recovery(sleep_record)
    
    result = {
        'period_id': sleep_record.get('period_id', 0),
        **hr_profile,
        **hrv,
        **recovery
    }
    
    if recovery['autonomic_recovery_score'] >= 80:
        result['recovery_quality'] = 'Excellent'
    elif recovery['autonomic_recovery_score'] >= 65:
        result['recovery_quality'] = 'Good'
    else:
        result['recovery_quality'] = 'Average'
    
    return result

def main():
    """Main function to analyze sleep heart rate metrics"""
    print("Sleep Heart Rate Metrics Analysis")
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent  # From src/sleep/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "sleep_heart_rate"
    
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
    
    # Process sleep data (primary source for sleep HR analysis)
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Analyzing sleep record {i+1}/10...")
            result = analyze_sleep_heart_rate_metrics(record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    # Process activity data for context
    if activity_data:
        print("Processing activity records for context...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {i+1}/10...")
            result = analyze_sleep_heart_rate_metrics(record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/sleep_heart_rate_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/sleep_heart_rate_results.json")
    


if __name__ == "__main__":
    main()
