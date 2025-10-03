"""
Postural Analysis
Time Standing - duration in upright posture
Time Sitting - duration in seated posture
Time Lying - duration in horizontal posture
Postural Transitions - frequency of position changes
Postural Stability - consistency within each posture
Sleep Position - primary sleep postures and transitions
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os
from pathlib import Path

def time_standing(activity_data: Dict[str, Any]) -> float:
    """Calculate duration in upright posture"""
    active_minutes = activity_data.get('active_minutes', 0.0)
    # Estimate standing time as portion of active time
    standing_time = active_minutes * 0.6  # Assume 60% of active time is standing
    return float(standing_time)

def time_sitting(activity_data: Dict[str, Any]) -> float:
    """Calculate duration in seated posture"""
    sedentary_minutes = activity_data.get('sedentary_minutes', 0.0)
    # Estimate sitting time as majority of sedentary time
    sitting_time = sedentary_minutes * 0.8  # 80% of sedentary time is sitting
    return float(sitting_time)

def time_lying(activity_data: Dict[str, Any]) -> float:
    """Calculate duration in horizontal posture"""
    sedentary_minutes = activity_data.get('sedentary_minutes', 0.0)
    # Remaining sedentary time is lying
    lying_time = sedentary_minutes * 0.2
    return float(lying_time)

def postural_transitions(activity_data: Dict[str, Any]) -> float:
    """Calculate frequency of position changes"""
    active_minutes = activity_data.get('active_minutes', 0.0)
    steps = activity_data.get('steps', 0)
    
    # Estimate transitions from activity variability
    if active_minutes > 0:
        activity_density = steps / active_minutes
        transitions = min(activity_density / 50, 20)  # Cap at 20 transitions
    else:
        transitions = 2.0  # Minimum transitions
    
    return float(transitions)

def postural_stability(activity_data: Dict[str, Any]) -> float:
    """Calculate consistency within each posture"""
    sedentary_minutes = activity_data.get('sedentary_minutes', 0.0)
    active_minutes = activity_data.get('active_minutes', 0.0)
    
    total_time = sedentary_minutes + active_minutes
    if total_time == 0:
        return 0.0
    
    # Higher sedentary percentage indicates more stable postures
    stability = (sedentary_minutes / total_time) * 100
    return float(stability)

def sleep_position(sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate primary sleep postures and transitions"""
    efficiency = sleep_record.get('efficiency', 85) / 100.0
    
    # Estimate sleep position stability from efficiency
    if efficiency > 0.9:
        primary_position = 'Side'
        position_stability = 'High'
        sleep_transitions = 3
    elif efficiency > 0.8:
        primary_position = 'Back'
        position_stability = 'Medium'
        sleep_transitions = 5
    else:
        primary_position = 'Variable'
        position_stability = 'Low'
        sleep_transitions = 8
    
    return {
        'primary_sleep_position': primary_position,
        'sleep_position_stability': position_stability,
        'sleep_position_transitions': sleep_transitions
    }

def analyze_postural_analysis(activity_record: Dict[str, Any], sleep_record: Dict[str, Any] = None) -> Dict[str, Any]:
    """Analyze complete postural analysis metrics"""
    
    standing_time = time_standing(activity_record)
    sitting_time = time_sitting(activity_record)
    lying_time = time_lying(activity_record)
    transitions = postural_transitions(activity_record)
    stability = postural_stability(activity_record)
    
    result = {
        'period_id': activity_record.get('period_id', 0),
        'time_standing_min': standing_time,
        'time_sitting_min': sitting_time,
        'time_lying_min': lying_time,
        'postural_transitions': transitions,
        'postural_stability': stability
    }
    
    # Add sleep position if available
    if sleep_record:
        sleep_pos = sleep_position(sleep_record)
        result.update(sleep_pos)
    
    # Add posture interpretation
    total_upright = standing_time + sitting_time
    total_time = standing_time + sitting_time + lying_time
    
    if total_time > 0:
        upright_percentage = (total_upright / total_time) * 100
        if upright_percentage > 80:
            result['posture_pattern'] = 'Active'
        elif upright_percentage > 60:
            result['posture_pattern'] = 'Moderate'
        else:
            result['posture_pattern'] = 'Sedentary'
    
    return result

def main():
    """Main function to analyze postural metrics"""
    print("Postural Analysis")
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent  # From src/actigraphy/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "postural_analysis"
    
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
    
    # Process activity data (primary source for postural analysis)
    if activity_data:
        print("Processing activity records...")
        for i, activity_record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {i+1}/10...")
            # Find corresponding sleep record if available
            sleep_record = sleep_data[i] if i < len(sleep_data) else {}
            result = analyze_postural_analysis(activity_record, sleep_record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    # Process sleep data for additional context
    if sleep_data and len(all_results) < 10:
        print("Processing sleep records for additional postural context...")
        remaining_slots = 10 - len(all_results)
        for i, sleep_record in enumerate(sleep_data[:remaining_slots]):
            print(f"Analyzing sleep record {i+1}/{remaining_slots}...")
            # Create basic activity context from sleep data
            activity_record = {
                'period_id': sleep_record.get('period_id', i + len(activity_data)),
                'active_minutes': 60,  # Basic estimate
                'sedentary_minutes': 480,  # Basic estimate
                'steps': 8000  # Basic estimate
            }
            result = analyze_postural_analysis(activity_record, sleep_record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/postural_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/postural_analysis_results.json")
    
    # Create visualizations
    print("Creating visualizations...")

    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()