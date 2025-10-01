"""
Resting Heart Rate (RHR) - baseline cardiac rate during inactivity
Maximum Heart Rate (MHR) - peak sustainable cardiac rate
Heart Rate Recovery (HRR) - rate of HR decline post-exercise
Heart Rate Reserve (HRR) - difference between max and resting HR
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import os

def resting_heart_rate(sleep_record: Dict[str, Any]) -> float:
    """Calculate baseline cardiac rate during inactivity"""
    # Extract heart rate data during sleep (most restful state)
    hr_sequence = sleep_record.get('hr_5min', [])
    if not hr_sequence:
        # Use recorded resting HR if available
        return float(sleep_record.get('hr_lowest', 60.0))
    
    # Filter valid HR values and find lowest 25% (most restful periods)
    valid_hrs = [hr for hr in hr_sequence if 40 <= hr <= 120]
    if not valid_hrs:
        return float(sleep_record.get('hr_lowest', 60.0))
    
    # Take the lowest quartile as resting HR
    sorted_hrs = sorted(valid_hrs)
    quartile_index = len(sorted_hrs) // 4
    resting_hrs = sorted_hrs[:quartile_index] if quartile_index > 0 else sorted_hrs[:1]
    
    return float(np.mean(resting_hrs))

def maximum_heart_rate(sleep_record: Dict[str, Any], age: int = 30) -> float:
    """Calculate peak sustainable cardiac rate"""
    # Use recorded maximum HR if available
    recorded_max = sleep_record.get('hr_highest', 0)
    if recorded_max > 0:
        return float(recorded_max)
    
    # Otherwise estimate using age-based formulas
    # Tanaka formula: 208 - (0.7 × age) - more accurate than 220-age
    estimated_max = 208 - (0.7 * age)
    return float(estimated_max)

def heart_rate_recovery(sleep_record: Dict[str, Any], recovery_minutes: int = 5) -> float:
    """Calculate rate of HR decline post-exercise"""
    hr_sequence = sleep_record.get('hr_5min', [])
    if len(hr_sequence) < recovery_minutes:
        return 0.0
    
    # Find periods of elevated HR followed by decline (simulating post-exercise)
    max_recovery = 0.0
    
    for i in range(len(hr_sequence) - recovery_minutes):
        # Look for HR peaks followed by decline
        if hr_sequence[i] > 0:
            peak_hr = hr_sequence[i]
            recovery_hr = hr_sequence[i + recovery_minutes] if i + recovery_minutes < len(hr_sequence) else hr_sequence[-1]
            
            if recovery_hr > 0 and peak_hr > recovery_hr:
                recovery_rate = peak_hr - recovery_hr
                max_recovery = max(max_recovery, recovery_rate)
    
    return float(max_recovery)

def heart_rate_reserve(sleep_record: Dict[str, Any], age: int = 30) -> float:
    """Calculate difference between max and resting HR"""
    rhr = resting_heart_rate(sleep_record)
    mhr = maximum_heart_rate(sleep_record, age)
    
    reserve = mhr - rhr
    return float(reserve)

def analyze_cardiac_basic_metrics(sleep_record: Dict[str, Any], user_age: int = 30) -> Dict[str, Any]:
    """Analyze complete basic cardiac metrics"""
    
    rhr = resting_heart_rate(sleep_record)
    mhr = maximum_heart_rate(sleep_record, user_age)
    hrr = heart_rate_recovery(sleep_record)
    hr_reserve = heart_rate_reserve(sleep_record, user_age)
    
    # Extract additional heart rate statistics
    hr_sequence = sleep_record.get('hr_5min', [])
    valid_hrs = [hr for hr in hr_sequence if hr > 0]
    
    results = {
        'period_id': sleep_record.get('period_id', 0),
        'timestamp': sleep_record.get('bedtime_start_dt_adjusted', 0),
        'resting_heart_rate': rhr,
        'maximum_heart_rate': mhr,
        'heart_rate_recovery': hrr,
        'heart_rate_reserve': hr_reserve,
        'average_hr_during_sleep': np.mean(valid_hrs) if valid_hrs else 0.0,
        'hr_variability_range': (np.max(valid_hrs) - np.min(valid_hrs)) if valid_hrs else 0.0,
        'hr_data_points': len(valid_hrs)
    }
    
    # Add cardiac fitness interpretation
    if user_age > 0:
        # Age-adjusted interpretations
        if rhr < 60:
            results['rhr_category'] = 'Excellent (Athletic)'
        elif rhr < 70:
            results['rhr_category'] = 'Good'
        elif rhr < 80:
            results['rhr_category'] = 'Average'
        else:
            results['rhr_category'] = 'Below Average'
        
        # Heart Rate Reserve interpretation (fitness indicator)
        if hr_reserve > 160:
            results['fitness_level'] = 'Excellent'
        elif hr_reserve > 140:
            results['fitness_level'] = 'Good'
        elif hr_reserve > 120:
            results['fitness_level'] = 'Average'
        else:
            results['fitness_level'] = 'Below Average'
    
    return results

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualizations of basic cardiac metrics"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("No results available for visualization")
        return
    
    # Extract data for visualization
    cardiac_data = []
    for result in results:
        cardiac_data.append({
            'period_id': result['period_id'],
            'resting_hr': result['resting_heart_rate'],
            'maximum_hr': result['maximum_heart_rate'],
            'hr_recovery': result['heart_rate_recovery'],
            'hr_reserve': result['heart_rate_reserve'],
            'average_sleep_hr': result['average_hr_during_sleep'],
            'rhr_category': result.get('rhr_category', 'Unknown'),
            'fitness_level': result.get('fitness_level', 'Unknown')
        })
    
    df = pd.DataFrame(cardiac_data)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Resting HR over time
    axes[0,0].plot(df['period_id'], df['resting_hr'], 'o-', alpha=0.7, color='blue')
    axes[0,0].axhline(y=60, color='green', linestyle='--', alpha=0.5, label='Athletic (<60)')
    axes[0,0].axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Good (<70)')
    axes[0,0].set_xlabel('Period ID')
    axes[0,0].set_ylabel('Resting Heart Rate (BPM)')
    axes[0,0].set_title('Resting Heart Rate Over Time')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Heart Rate Reserve
    axes[0,1].plot(df['period_id'], df['hr_reserve'], 's-', alpha=0.7, color='red')
    axes[0,1].set_xlabel('Period ID')
    axes[0,1].set_ylabel('Heart Rate Reserve (BPM)')
    axes[0,1].set_title('Heart Rate Reserve (Fitness Indicator)')
    axes[0,1].grid(True, alpha=0.3)
    
    # RHR Categories
    if 'rhr_category' in df.columns:
        rhr_categories = df['rhr_category'].value_counts()
        axes[1,0].pie(rhr_categories.values, labels=rhr_categories.index, autopct='%1.1f%%')
        axes[1,0].set_title('Resting Heart Rate Categories')
    
    # Fitness Level Distribution
    if 'fitness_level' in df.columns:
        fitness_levels = df['fitness_level'].value_counts()
        axes[1,1].pie(fitness_levels.values, labels=fitness_levels.index, autopct='%1.1f%%')
        axes[1,1].set_title('Fitness Level Distribution')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cardiac_basic_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Cardiac basic metrics visualizations saved to {output_dir}/")

def main():
    """Main function to analyze basic cardiac metrics"""
    
    print("Basic Cardiac Metrics Analysis")
    print("=" * 50)
    
    # Load sleep data with heart rate information
    try:
        with open('../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    except:
        with open('../../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    
    print(f"Loaded {len(sleep_data)} sleep records")
    
    # User age for calculations (could be parameter)
    user_age = 35
    
    # Analyze first 10 records
    results = []
    for i, record in enumerate(sleep_data[:10]):
        print(f"Analyzing record {i+1}/10...")
        result = analyze_cardiac_basic_metrics(record, user_age)
        results.append(result)
    
    # Save results
    output_dir = '../results/cardiac_basic'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/cardiac_basic_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}/cardiac_basic_results.json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(results, output_dir)
    
    # Print summary statistics
    print("\nBasic Cardiac Metrics Summary:")
    print("-" * 40)
    
    rhr_vals = [r['resting_heart_rate'] for r in results]
    mhr_vals = [r['maximum_heart_rate'] for r in results]
    hrr_vals = [r['heart_rate_recovery'] for r in results if r['heart_rate_recovery'] > 0]
    reserve_vals = [r['heart_rate_reserve'] for r in results]
    
    print(f"Resting HR - Mean: {np.mean(rhr_vals):.1f} BPM, Range: {np.min(rhr_vals):.1f}-{np.max(rhr_vals):.1f} BPM")
    print(f"Maximum HR - Mean: {np.mean(mhr_vals):.1f} BPM, Range: {np.min(mhr_vals):.1f}-{np.max(mhr_vals):.1f} BPM")
    if hrr_vals:
        print(f"HR Recovery - Mean: {np.mean(hrr_vals):.1f} BPM, Range: {np.min(hrr_vals):.1f}-{np.max(hrr_vals):.1f} BPM")
    print(f"HR Reserve - Mean: {np.mean(reserve_vals):.1f} BPM, Range: {np.min(reserve_vals):.1f}-{np.max(reserve_vals):.1f} BPM")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
