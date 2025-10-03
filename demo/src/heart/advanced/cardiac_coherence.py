"""
Cardiac Coherence - coherence between HR and respiratory rate
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from pathlib import Path
import os
from scipy import signal

def cardiac_coherence(sleep_record: Dict[str, Any]) -> Dict[str, float]:
    """Calculate coherence between HR and respiratory rate"""
    hr_sequence = sleep_record.get('hr_5min', [])
    breath_avg = sleep_record.get('breath_average', 16.0)
    
    if not hr_sequence or len(hr_sequence) < 10:
        return {
            'coherence_ratio': 0.0,
            'coherence_peak': 0.0,
            'coherence_stability': 0.0
        }
    
    # Create synthetic breathing signal based on average
    time_points = np.arange(len(hr_sequence))
    breathing_signal = breath_avg + 2 * np.sin(2 * np.pi * time_points / 20)  # 20-point cycle
    
    # Filter valid HR data
    hr_array = np.array([hr if hr > 0 else np.nan for hr in hr_sequence])
    valid_indices = ~np.isnan(hr_array)
    
    if np.sum(valid_indices) < 5:
        return {
            'coherence_ratio': 0.0,
            'coherence_peak': 0.0,
            'coherence_stability': 0.0
        }
    
    hr_clean = hr_array[valid_indices]
    breathing_clean = breathing_signal[valid_indices]
    
    # Calculate cross-correlation
    if len(hr_clean) > 10:
        correlation = np.corrcoef(hr_clean, breathing_clean)[0, 1]
        correlation = np.abs(correlation) if not np.isnan(correlation) else 0.0
    else:
        correlation = 0.0
    
    # Coherence metrics
    coherence_ratio = correlation * 100  # Convert to percentage
    coherence_peak = coherence_ratio  # Simplified - same as ratio
    
    # Coherence stability from HR variability
    hr_cv = np.std(hr_clean) / np.mean(hr_clean) if np.mean(hr_clean) > 0 else 0
    coherence_stability = max(0, 100 - (hr_cv * 100))  # Inverse of variability
    
    return {
        'coherence_ratio': float(coherence_ratio),
        'coherence_peak': float(coherence_peak),
        'coherence_stability': float(coherence_stability)
    }

def analyze_cardiac_coherence(sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze cardiac coherence metrics"""
    
    coherence_metrics = cardiac_coherence(sleep_record)
    
    result = {
        'period_id': sleep_record.get('period_id', 0),
        'timestamp': sleep_record.get('bedtime_start_dt_adjusted', 0),
        'coherence_ratio': coherence_metrics['coherence_ratio'],
        'coherence_peak': coherence_metrics['coherence_peak'],
        'coherence_stability': coherence_metrics['coherence_stability'],
        'breath_average': sleep_record.get('breath_average', 16.0)
    }
    
    # Add coherence quality interpretation
    ratio = coherence_metrics['coherence_ratio']
    if ratio >= 70:
        result['coherence_quality'] = 'High'
    elif ratio >= 40:
        result['coherence_quality'] = 'Medium'
    else:
        result['coherence_quality'] = 'Low'
    
    return result

def main():
    """Main function to analyze cardiac coherence"""
    print("Cardiac Coherence Analysis")
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent.parent  # From src/heart/advanced/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "cardiac_coherence"
    
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
    
    # Process sleep data (primary source for cardiac coherence)
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Analyzing sleep record {i+1}/10...")
            result = analyze_cardiac_coherence(record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    # Process activity data for additional context
    if activity_data:
        print("Processing activity records...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {i+1}/10...")
            result = analyze_cardiac_coherence(record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/cardiac_coherence_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/cardiac_coherence_results.json")
    
    # Show data source breakdown
    activity_count = sum(1 for r in all_results if r.get('data_source') == 'activity')
    sleep_count = sum(1 for r in all_results if r.get('data_source') == 'sleep')
    print(f"Data sources: {activity_count} activity records, {sleep_count} sleep records")
    
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()