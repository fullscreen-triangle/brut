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
    
    try:
        with open('../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    except:
        with open('../../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    
    results = []
    for record in sleep_data[:10]:
        result = analyze_sleep_heart_rate_metrics(record)
        results.append(result)
    
    output_dir = '../results/sleep_heart_rate'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/sleep_heart_rate_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved. Analysis complete!")

if __name__ == "__main__":
    main()
