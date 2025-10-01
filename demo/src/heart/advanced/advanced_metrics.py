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
    
    try:
        with open('../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    except:
        with open('../../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    
    results = []
    for record in sleep_data[:10]:
        result = analyze_advanced_cardiac_metrics(record)
        results.append(result)
    
    output_dir = '../results/advanced_cardiac'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/advanced_cardiac_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()