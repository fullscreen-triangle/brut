"""
Chronotropic Response - HR response to exercise stress
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os
from pathlib import Path

def chronotropic_response(sleep_record: Dict[str, Any], age: int = 35) -> Dict[str, float]:
    """Calculate HR response to exercise stress"""
    hr_sequence = sleep_record.get('hr_5min', [])
    
    if not hr_sequence:
        return {
            'max_hr_achieved': 0.0,
            'predicted_max_hr': 0.0,
            'chronotropic_index': 0.0,
            'hr_reserve_used': 0.0
        }
    
    # Calculate metrics
    max_hr_achieved = max(hr_sequence) if hr_sequence else 0.0
    predicted_max_hr = 208 - (0.7 * age)  # Tanaka formula
    resting_hr = min(hr_sequence) if hr_sequence else 60.0
    
    # Chronotropic index
    hr_reserve = predicted_max_hr - resting_hr
    if hr_reserve > 0:
        chronotropic_index = (max_hr_achieved - resting_hr) / hr_reserve
        hr_reserve_used = chronotropic_index * 100
    else:
        chronotropic_index = 0.0
        hr_reserve_used = 0.0
    
    return {
        'max_hr_achieved': float(max_hr_achieved),
        'predicted_max_hr': float(predicted_max_hr),
        'chronotropic_index': float(chronotropic_index),
        'hr_reserve_used': float(hr_reserve_used)
    }

def analyze_chronotropic_response(sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze chronotropic response metrics"""
    response = chronotropic_response(sleep_record)
    
    result = {
        'period_id': sleep_record.get('period_id', 0),
        **response
    }
    
    # Add fitness interpretation
    ci = response['chronotropic_index']
    if ci >= 0.8:
        result['chronotropic_fitness'] = 'Excellent'
    elif ci >= 0.6:
        result['chronotropic_fitness'] = 'Good'
    else:
        result['chronotropic_fitness'] = 'Average'
    
    return result

def main():
    """Main function to analyze chronotropic response"""
    print("Chronotropic Response Analysis")
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent.parent  # From src/heart/cardiac/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "chronotropic_response"
    
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
    
    # Process activity data (primary source for chronotropic response)
    if activity_data:
        print("Processing activity records...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {i+1}/10...")
            result = analyze_chronotropic_response(record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    # Process sleep data for additional context
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Analyzing sleep record {i+1}/10...")
            result = analyze_chronotropic_response(record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/chronotropic_response_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/chronotropic_response_results.json")
    
    # Show data source breakdown
    activity_count = sum(1 for r in all_results if r.get('data_source') == 'activity')
    sleep_count = sum(1 for r in all_results if r.get('data_source') == 'sleep')
    print(f"Data sources: {activity_count} activity records, {sleep_count} sleep records")
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()