"""
Chronotropic Response - HR response to exercise stress
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os

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
    
    try:
        with open('../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    except:
        with open('../../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    
    results = []
    for record in sleep_data[:10]:
        result = analyze_chronotropic_response(record)
        results.append(result)
    
    output_dir = '../results/chronotropic_response'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/chronotropic_response_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()