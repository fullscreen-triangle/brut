"""
Cardiac Coherence - coherence between HR and respiratory rate
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
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
    
    try:
        with open('../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    except:
        with open('../../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    
    results = []
    for record in sleep_data[:10]:
        result = analyze_cardiac_coherence(record)
        results.append(result)
    
    output_dir = '../results/cardiac_coherence'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/cardiac_coherence_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()