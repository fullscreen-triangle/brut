"""
Circadian Rhythm Metrics
Circadian Phase - timing of internal biological clock
Phase Shift - deviation from expected circadian timing
Amplitude - strength of circadian rhythm
Acrophase - time of peak circadian activity
Mesor - circadian rhythm-adjusted mean
Interdaily Stability (IS) - consistency of circadian patterns
Intradaily Variability (IV) - fragmentation of circadian rhythm
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import os
from datetime import datetime

def circadian_phase(sleep_record: Dict[str, Any]) -> float:
    """Calculate timing of internal biological clock"""
    bedtime_start = sleep_record.get('bedtime_start_dt_adjusted', 0)
    
    if bedtime_start == 0:
        return 22.0  # Default 10 PM
    
    try:
        if isinstance(bedtime_start, (int, float)):
            dt = datetime.fromtimestamp(bedtime_start / 1000 if bedtime_start > 1e10 else bedtime_start)
        else:
            dt = datetime.fromisoformat(str(bedtime_start).replace('Z', '+00:00'))
        
        circadian_phase_hour = dt.hour + dt.minute / 60.0
        return float(circadian_phase_hour)
    except:
        return 22.0

def phase_shift(sleep_record: Dict[str, Any], reference_bedtime: float = 22.0) -> float:
    """Calculate deviation from expected circadian timing"""
    actual_phase = circadian_phase(sleep_record)
    shift = actual_phase - reference_bedtime
    
    if shift > 12:
        shift -= 24
    elif shift < -12:
        shift += 24
    
    return float(shift)

def amplitude(sleep_record: Dict[str, Any]) -> float:
    """Calculate strength of circadian rhythm"""
    sleep_efficiency = sleep_record.get('efficiency', 85) / 100.0
    total_sleep_hrs = sleep_record.get('total_in_hrs', 7.0)
    
    optimal_sleep_hrs = 8.0
    duration_factor = 1.0 - abs(total_sleep_hrs - optimal_sleep_hrs) / optimal_sleep_hrs
    
    amplitude_estimate = (sleep_efficiency + duration_factor) / 2.0
    return float(max(0.1, min(1.0, amplitude_estimate)))

def acrophase(sleep_record: Dict[str, Any]) -> float:
    """Calculate time of peak circadian activity"""
    bedtime_phase = circadian_phase(sleep_record)
    acrophase_estimate = (bedtime_phase - 8.0) % 24
    return float(acrophase_estimate)

def mesor(sleep_record: Dict[str, Any]) -> float:
    """Calculate circadian rhythm-adjusted mean"""
    hr_sequence = sleep_record.get('hr_5min', [])
    if hr_sequence:
        valid_hrs = [hr for hr in hr_sequence if hr > 0]
        if valid_hrs:
            mesor_hr = np.median(valid_hrs)
            return float(mesor_hr)
    
    return float(sleep_record.get('hr_average', 65.0))

def interdaily_stability(sleep_records: List[Dict[str, Any]]) -> float:
    """Calculate consistency of circadian patterns across days"""
    if len(sleep_records) < 3:
        return 0.5
    
    bedtimes = []
    for record in sleep_records:
        phase = circadian_phase(record)
        bedtimes.append(phase)
    
    if not bedtimes:
        return 0.5
    
    bedtime_variance = np.var(bedtimes)
    max_variance = 16.0  # 4 hours squared
    stability = max(0.0, 1.0 - (bedtime_variance / max_variance))
    
    return float(stability)

def intradaily_variability(sleep_record: Dict[str, Any]) -> float:
    """Calculate fragmentation of circadian rhythm"""
    hr_sequence = sleep_record.get('hr_5min', [])
    if len(hr_sequence) < 5:
        return 0.3
    
    valid_hrs = [hr for hr in hr_sequence if hr > 0]
    if len(valid_hrs) < 5:
        return 0.3
    
    hr_array = np.array(valid_hrs)
    consecutive_diffs = np.diff(hr_array)
    
    mean_hr = np.mean(hr_array)
    if mean_hr == 0:
        return 0.3
    
    iv = np.sum(np.abs(consecutive_diffs)) / (len(consecutive_diffs) * mean_hr)
    iv_normalized = min(1.0, iv / 2.0)
    
    return float(iv_normalized)

def analyze_circadian_rhythm_metrics(sleep_records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Analyze circadian rhythm metrics for multiple sleep records"""
    
    if not sleep_records:
        return []
    
    is_value = interdaily_stability(sleep_records)
    
    results = []
    for record in sleep_records:
        circ_phase = circadian_phase(record)
        phase_shift_val = phase_shift(record)
        amplitude_val = amplitude(record)
        acrophase_val = acrophase(record)
        mesor_val = mesor(record)
        iv_val = intradaily_variability(record)
        
        result = {
            'period_id': record.get('period_id', 0),
            'timestamp': record.get('bedtime_start_dt_adjusted', 0),
            'circadian_phase_hrs': circ_phase,
            'phase_shift_hrs': phase_shift_val,
            'amplitude': amplitude_val,
            'acrophase_hrs': acrophase_val,
            'mesor': mesor_val,
            'interdaily_stability': is_value,
            'intradaily_variability': iv_val,
            'circadian_strength': amplitude_val * (1.0 - iv_val)
        }
        
        # Add circadian rhythm quality interpretation
        if amplitude_val > 0.8 and iv_val < 0.3:
            result['circadian_quality'] = 'Excellent'
        elif amplitude_val > 0.6 and iv_val < 0.5:
            result['circadian_quality'] = 'Good'
        elif amplitude_val > 0.4 and iv_val < 0.7:
            result['circadian_quality'] = 'Average'
        else:
            result['circadian_quality'] = 'Poor'
        
        # Add chronotype estimation
        if circ_phase <= 21.0:
            result['chronotype'] = 'Morning Type'
        elif circ_phase <= 23.0:
            result['chronotype'] = 'Neither Type'
        else:
            result['chronotype'] = 'Evening Type'
        
        results.append(result)
    
    return results

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualizations of circadian rhythm metrics"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("No results available for visualization")
        return
    
    # Extract data for visualization
    circadian_data = []
    for result in results:
        circadian_data.append({
            'period_id': result['period_id'],
            'circadian_phase': result['circadian_phase_hrs'],
            'phase_shift': result['phase_shift_hrs'],
            'amplitude': result['amplitude'],
            'acrophase': result['acrophase_hrs'],
            'mesor': result['mesor'],
            'iv_variability': result['intradaily_variability'],
            'circadian_quality': result['circadian_quality'],
            'chronotype': result['chronotype']
        })
    
    df = pd.DataFrame(circadian_data)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Circadian phase over time
    axes[0,0].plot(df['period_id'], df['circadian_phase'], 'o-', alpha=0.7, color='darkblue')
    axes[0,0].axhline(y=22, color='green', linestyle='--', alpha=0.5, label='Typical Bedtime (10 PM)')
    axes[0,0].set_xlabel('Period ID')
    axes[0,0].set_ylabel('Circadian Phase (hours)')
    axes[0,0].set_title('Circadian Phase (Sleep Onset Time)')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Phase shift distribution
    axes[0,1].hist(df['phase_shift'], bins=8, alpha=0.7, color='orange')
    axes[0,1].axvline(x=0, color='red', linestyle='-', alpha=0.7, label='No Shift')
    axes[0,1].set_xlabel('Phase Shift (hours)')
    axes[0,1].set_ylabel('Frequency')
    axes[0,1].set_title('Phase Shift Distribution')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Circadian quality categories
    if 'circadian_quality' in df.columns:
        quality_counts = df['circadian_quality'].value_counts()
        axes[1,0].pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
        axes[1,0].set_title('Circadian Rhythm Quality')
    
    # Chronotype distribution
    if 'chronotype' in df.columns:
        chronotype_counts = df['chronotype'].value_counts()
        axes[1,1].pie(chronotype_counts.values, labels=chronotype_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Chronotype Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/circadian_rhythm_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Circadian rhythm metrics visualizations saved to {output_dir}/")

def main():
    """Main function to analyze circadian rhythm metrics"""
    
    print("Circadian Rhythm Metrics Analysis")
    print("=" * 50)
    
    # Load sleep data
    try:
        with open('../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    except:
        with open('../../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    
    print(f"Loaded {len(sleep_data)} sleep records")
    
    # Analyze first 20 records
    selected_records = sleep_data[:20]
    results = analyze_circadian_rhythm_metrics(selected_records)
    
    # Save results
    output_dir = '../results/circadian_rhythm'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/circadian_rhythm_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}/circadian_rhythm_results.json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(results, output_dir)
    
    # Print summary statistics
    print("\nCircadian Rhythm Metrics Summary:")
    print("-" * 40)
    
    if results:
        phases = [r['circadian_phase_hrs'] for r in results]
        shifts = [r['phase_shift_hrs'] for r in results]
        amplitudes = [r['amplitude'] for r in results]
        
        print(f"Circadian Phase - Mean: {np.mean(phases):.1f} hrs, Range: {np.min(phases):.1f}-{np.max(phases):.1f} hrs")
        print(f"Phase Shift - Mean: {np.mean(shifts):.1f} hrs, Range: {np.min(shifts):.1f}-{np.max(shifts):.1f} hrs")
        print(f"Amplitude - Mean: {np.mean(amplitudes):.2f}, Range: {np.min(amplitudes):.2f}-{np.max(amplitudes):.2f}")
        
        # Quality and chronotype distributions
        quality_categories = [r['circadian_quality'] for r in results]
        chronotypes = [r['chronotype'] for r in results]
        
        quality_counts = {cat: quality_categories.count(cat) for cat in set(quality_categories)}
        chronotype_counts = {ct: chronotypes.count(ct) for ct in set(chronotypes)}
        
        print(f"\nCircadian Quality Distribution:")
        for cat, count in quality_counts.items():
            print(f"  {cat}: {count} records ({count/len(results)*100:.1f}%)")
        
        print(f"\nChronotype Distribution:")
        for ct, count in chronotype_counts.items():
            print(f"  {ct}: {count} records ({count/len(results)*100:.1f}%)")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
