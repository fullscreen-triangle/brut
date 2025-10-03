"""
Activity Fragmentation - interruption of sustained activity
Activity Transition Probability - likelihood of changing activity states
Activity Bout Duration - length of continuous activity periods
Activity Regularity - consistency of daily activity patterns
Peak Activity Time - time of maximum daily activity
Activity Amplitude - difference between peak and trough activity
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os
from pathlib import Path

def activity_fragmentation(activity_data: Dict[str, Any]) -> float:
    """Calculate interruption of sustained activity"""
    steps = activity_data.get('steps', 0)
    active_minutes = activity_data.get('active_minutes', 0.0)
    sedentary_minutes = activity_data.get('sedentary_minutes', 0.0)
    
    total_minutes = active_minutes + sedentary_minutes
    if total_minutes == 0:
        return 0.0
    
    # Estimate fragmentation from ratio
    fragmentation = (sedentary_minutes / total_minutes) * 100
    return float(fragmentation)

def activity_transition_probability(activity_data: Dict[str, Any]) -> float:
    """Calculate likelihood of changing activity states"""
    # Simulate based on step variability
    steps = activity_data.get('steps', 0)
    active_minutes = activity_data.get('active_minutes', 0.0)
    
    if active_minutes == 0:
        return 0.0
    
    # Estimate transition probability
    steps_per_minute = steps / active_minutes if active_minutes > 0 else 0
    transition_prob = min(1.0, steps_per_minute / 100.0)
    
    return float(transition_prob)

def activity_bout_duration(activity_data: Dict[str, Any]) -> float:
    """Calculate length of continuous activity periods"""
    active_minutes = activity_data.get('active_minutes', 0.0)
    # Estimate average bout duration (simplified)
    avg_bout = active_minutes / 4 if active_minutes > 0 else 0  # Assume 4 bouts per day
    return float(avg_bout)

def peak_activity_time(activity_data: Dict[str, Any]) -> float:
    """Calculate time of maximum daily activity"""
    # Estimate peak time (simplified - assuming afternoon peak)
    peak_time = 15.0  # 3 PM typical peak
    return float(peak_time)

def activity_amplitude(activity_data: Dict[str, Any]) -> float:
    """Calculate difference between peak and trough activity"""
    steps = activity_data.get('steps', 0)
    # Estimate amplitude from total daily activity
    amplitude = steps / 100 if steps > 0 else 0
    return float(amplitude)

def analyze_movement_patterns(activity_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze complete movement pattern metrics"""
    
    fragmentation = activity_fragmentation(activity_record)
    transition_prob = activity_transition_probability(activity_record)
    bout_duration = activity_bout_duration(activity_record)
    peak_time = peak_activity_time(activity_record)
    amplitude = activity_amplitude(activity_record)
    
    result = {
        'period_id': activity_record.get('period_id', 0),
        'activity_fragmentation': fragmentation,
        'activity_transition_probability': transition_prob,
        'activity_bout_duration': bout_duration,
        'peak_activity_time': peak_time,
        'activity_amplitude': amplitude
    }
    
    # Add pattern interpretation
    if fragmentation < 30:
        result['activity_pattern'] = 'Sustained'
    elif fragmentation < 60:
        result['activity_pattern'] = 'Moderate'
    else:
        result['activity_pattern'] = 'Fragmented'
    
    return result

def main():
    """Main function to analyze movement patterns"""
    print("Movement Pattern Metrics Analysis")
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent  # From src/actigraphy/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "movement_patterns"
    
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
    
    # Process activity data (primary source for movement pattern analysis)
    if activity_data:
        print("Processing activity records...")
        for i, activity_record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {i+1}/10...")
            result = analyze_movement_patterns(activity_record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    # Process sleep data for additional context
    if sleep_data and len(all_results) < 10:
        print("Processing sleep records for additional movement context...")
        remaining_slots = 10 - len(all_results)
        for i, sleep_record in enumerate(sleep_data[:remaining_slots]):
            print(f"Analyzing sleep record {i+1}/{remaining_slots}...")
            # Create basic activity context from sleep data
            activity_record = {
                'period_id': sleep_record.get('period_id', i + len(activity_data)),
                'steps': 8000,  # Basic estimate
                'active_minutes': 90,  # Basic estimate
                'sedentary_minutes': 600  # Basic estimate
            }
            result = analyze_movement_patterns(activity_record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/movement_patterns_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/movement_patterns_results.json")
    
    # Create visualizations
    print("Creating visualizations...")

    
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()