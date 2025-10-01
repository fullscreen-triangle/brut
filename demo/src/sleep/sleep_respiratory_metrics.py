"""
Sleep Respiratory Metrics
Apnea-Hypopnea Index (AHI) - respiratory events per hour
Oxygen Desaturation Index (ODI) - desaturation events per hour
Respiratory Effort Related Arousals (RERA) - breathing-related wake events
Respiratory Rate Variability - breathing pattern irregularity
Central vs Obstructive Events - classification of respiratory disruptions
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os

def apnea_hypopnea_index(sleep_record: Dict[str, Any]) -> float:
    """Calculate respiratory events per hour"""
    breath_avg = sleep_record.get('breath_average', 16.0)
    total_sleep_hrs = sleep_record.get('total_in_hrs', 8.0)
    
    # Estimate AHI from breath rate irregularity
    if breath_avg < 12 or breath_avg > 20:
        estimated_events = abs(breath_avg - 16) * 2
    else:
        estimated_events = 1.0
    
    ahi = estimated_events / total_sleep_hrs if total_sleep_hrs > 0 else 0.0
    return float(ahi)

def oxygen_desaturation_index(sleep_record: Dict[str, Any]) -> float:
    """Calculate desaturation events per hour"""
    ahi = apnea_hypopnea_index(sleep_record)
    # ODI is typically 70-80% of AHI
    odi = ahi * 0.75
    return float(odi)

def respiratory_effort_related_arousals(sleep_record: Dict[str, Any]) -> float:
    """Calculate breathing-related wake events"""
    efficiency = sleep_record.get('efficiency', 85) / 100.0
    # RERA increases with poor sleep efficiency
    rera_rate = max(0, (1.0 - efficiency) * 10)
    return float(rera_rate)

def respiratory_rate_variability(sleep_record: Dict[str, Any]) -> float:
    """Calculate breathing pattern irregularity"""
    breath_avg = sleep_record.get('breath_average', 16.0)
    # Estimate variability from deviation from normal
    normal_breath_rate = 16.0
    variability = abs(breath_avg - normal_breath_rate) / normal_breath_rate
    return float(variability)

def central_vs_obstructive_events(sleep_record: Dict[str, Any]) -> Dict[str, float]:
    """Classify respiratory disruptions"""
    ahi = apnea_hypopnea_index(sleep_record)
    # Simple estimation: most events are obstructive in typical cases
    obstructive_ratio = 0.8
    central_ratio = 0.2
    
    return {
        'obstructive_events': float(ahi * obstructive_ratio),
        'central_events': float(ahi * central_ratio)
    }

def analyze_sleep_respiratory_metrics(sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze complete sleep respiratory metrics"""
    ahi = apnea_hypopnea_index(sleep_record)
    odi = oxygen_desaturation_index(sleep_record)
    rera = respiratory_effort_related_arousals(sleep_record)
    rrv = respiratory_rate_variability(sleep_record)
    events = central_vs_obstructive_events(sleep_record)
    
    result = {
        'period_id': sleep_record.get('period_id', 0),
        'apnea_hypopnea_index': ahi,
        'oxygen_desaturation_index': odi,
        'respiratory_effort_arousals': rera,
        'respiratory_rate_variability': rrv,
        'obstructive_events': events['obstructive_events'],
        'central_events': events['central_events'],
        'breath_average': sleep_record.get('breath_average', 16.0)
    }
    
    # Add severity classification
    if ahi < 5:
        result['sleep_apnea_severity'] = 'Normal'
    elif ahi < 15:
        result['sleep_apnea_severity'] = 'Mild'
    elif ahi < 30:
        result['sleep_apnea_severity'] = 'Moderate'
    else:
        result['sleep_apnea_severity'] = 'Severe'
    
    return result

def main():
    """Main function to analyze sleep respiratory metrics"""
    print("Sleep Respiratory Metrics Analysis")
    
    try:
        with open('../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    except:
        with open('../../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    
    results = []
    for record in sleep_data[:10]:
        result = analyze_sleep_respiratory_metrics(record)
        results.append(result)
    
    output_dir = '../results/sleep_respiratory'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/sleep_respiratory_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()