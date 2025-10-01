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
    
    try:
        with open('../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    except:
        with open('../../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    
    results = []
    for i, record in enumerate(sleep_data[:10]):
        # Create mock activity data
        activity_record = {
            'period_id': record.get('period_id', i),
            'steps': np.random.randint(2000, 15000),
            'active_minutes': np.random.uniform(30, 120),
            'sedentary_minutes': np.random.uniform(400, 800)
        }
        
        result = analyze_movement_patterns(activity_record)
        results.append(result)
    
    output_dir = '../results/movement_patterns'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/movement_patterns_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()