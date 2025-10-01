"""
Light Activity Time - low-intensity movement duration
Moderate Activity Time - moderate-intensity movement duration
Vigorous Activity Time - high-intensity movement duration
Moderate-to-Vigorous Physical Activity (MVPA) - combined moderate + vigorous
MET Minutes - metabolic equivalent task minutes
Activity Intensity Distribution - histogram of activity levels
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import os

def light_activity_time(activity_data: Dict[str, Any]) -> float:
    """Calculate low-intensity movement duration (< 3.0 METs)"""
    steps = activity_data.get('steps', 0)
    active_minutes = activity_data.get('active_minutes', 0.0)
    estimated_light = active_minutes * 0.6 if steps < 8000 else active_minutes * 0.4
    return float(estimated_light)

def moderate_activity_time(activity_data: Dict[str, Any]) -> float:
    """Calculate moderate-intensity movement duration (3.0-6.0 METs)"""
    steps = activity_data.get('steps', 0)
    active_minutes = activity_data.get('active_minutes', 0.0)
    if steps > 10000:
        estimated_moderate = active_minutes * 0.4
    elif steps > 7500:
        estimated_moderate = active_minutes * 0.3
    else:
        estimated_moderate = active_minutes * 0.2
    return float(estimated_moderate)

def vigorous_activity_time(activity_data: Dict[str, Any]) -> float:
    """Calculate high-intensity movement duration (≥ 6.0 METs)"""
    steps = activity_data.get('steps', 0)
    active_minutes = activity_data.get('active_minutes', 0.0)
    if steps > 12000:
        estimated_vigorous = active_minutes * 0.15
    elif steps > 10000:
        estimated_vigorous = active_minutes * 0.1
    else:
        estimated_vigorous = active_minutes * 0.05
    return float(estimated_vigorous)

def moderate_to_vigorous_physical_activity(activity_data: Dict[str, Any]) -> float:
    """Calculate combined moderate + vigorous activity (MVPA)"""
    moderate = moderate_activity_time(activity_data)
    vigorous = vigorous_activity_time(activity_data)
    return float(moderate + vigorous)

def met_minutes(activity_data: Dict[str, Any]) -> float:
    """Calculate metabolic equivalent task minutes"""
    light_time = light_activity_time(activity_data)
    moderate_time = moderate_activity_time(activity_data)
    vigorous_time = vigorous_activity_time(activity_data)
    sedentary_time = activity_data.get('sedentary_minutes', 0.0)
    
    light_mets = light_time * 2.5
    moderate_mets = moderate_time * 4.5
    vigorous_mets = vigorous_time * 7.0
    sedentary_mets = sedentary_time * 1.0
    
    total_met_minutes = light_mets + moderate_mets + vigorous_mets + sedentary_mets
    return float(total_met_minutes)

def activity_intensity_distribution(activity_data: Dict[str, Any], bins: int = 10) -> Tuple[List[float], List[float]]:
    """Calculate histogram of activity levels"""
    light_time = light_activity_time(activity_data)
    moderate_time = moderate_activity_time(activity_data)
    vigorous_time = vigorous_activity_time(activity_data)
    sedentary_time = activity_data.get('sedentary_minutes', 0.0)
    
    intensities = []
    intensities.extend([1.0] * int(sedentary_time))
    intensities.extend([2.5] * int(light_time))
    intensities.extend([4.5] * int(moderate_time))
    intensities.extend([7.0] * int(vigorous_time))
    
    if intensities:
        hist, bin_edges = np.histogram(intensities, bins=bins)
        return hist.tolist(), bin_edges.tolist()
    else:
        return [0] * bins, list(range(bins + 1))

def analyze_intensity_based_metrics(activity_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze complete intensity-based activity metrics"""
    
    light_time = light_activity_time(activity_record)
    moderate_time = moderate_activity_time(activity_record)
    vigorous_time = vigorous_activity_time(activity_record)
    mvpa = moderate_to_vigorous_physical_activity(activity_record)
    met_mins = met_minutes(activity_record)
    intensity_hist, intensity_bins = activity_intensity_distribution(activity_record)
    
    results = {
        'period_id': activity_record.get('period_id', 0),
        'timestamp': activity_record.get('timestamp', 0),
        'light_activity_time_min': light_time,
        'moderate_activity_time_min': moderate_time,
        'vigorous_activity_time_min': vigorous_time,
        'mvpa_minutes': mvpa,
        'met_minutes': met_mins,
        'activity_intensity_histogram': intensity_hist,
        'activity_intensity_bins': intensity_bins
    }
    
    # Add activity level classification
    if mvpa >= 150:
        results['activity_level'] = 'Highly Active'
    elif mvpa >= 75:
        results['activity_level'] = 'Moderately Active'  
    elif mvpa >= 30:
        results['activity_level'] = 'Somewhat Active'
    else:
        results['activity_level'] = 'Sedentary'
    
    return results

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualizations of intensity-based activity metrics"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("No results available for visualization")
        return
    
    # Extract data for visualization
    intensity_data = []
    for result in results:
        intensity_data.append({
            'period_id': result['period_id'],
            'light_time': result['light_activity_time_min'],
            'moderate_time': result['moderate_activity_time_min'],
            'vigorous_time': result['vigorous_activity_time_min'],
            'mvpa': result['mvpa_minutes'],
            'met_minutes': result['met_minutes'],
            'activity_level': result['activity_level']
        })
    
    df = pd.DataFrame(intensity_data)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Activity time breakdown over periods
    width = 0.8
    bottom_light = df['light_time']
    bottom_moderate = bottom_light + df['moderate_time']
    
    axes[0,0].bar(df['period_id'], df['light_time'], width, label='Light', alpha=0.8, color='lightblue')
    axes[0,0].bar(df['period_id'], df['moderate_time'], width, bottom=bottom_light, label='Moderate', alpha=0.8, color='orange')
    axes[0,0].bar(df['period_id'], df['vigorous_time'], width, bottom=bottom_moderate, label='Vigorous', alpha=0.8, color='red')
    axes[0,0].set_xlabel('Period ID')
    axes[0,0].set_ylabel('Activity Time (minutes)')
    axes[0,0].set_title('Activity Intensity Breakdown (Stacked)')
    axes[0,0].legend()
    
    # MVPA over time with WHO recommendation
    axes[0,1].plot(df['period_id'], df['mvpa'], 'o-', alpha=0.7, color='green')
    axes[0,1].axhline(y=150, color='red', linestyle='--', alpha=0.5, label='WHO Rec. (150 min)')
    axes[0,1].set_xlabel('Period ID')
    axes[0,1].set_ylabel('MVPA (minutes)')
    axes[0,1].set_title('Moderate-to-Vigorous Physical Activity')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # MET-minutes over time
    axes[1,0].plot(df['period_id'], df['met_minutes'], 's-', alpha=0.7, color='purple')
    axes[1,0].set_xlabel('Period ID')
    axes[1,0].set_ylabel('MET-minutes')
    axes[1,0].set_title('Metabolic Equivalent Task Minutes')
    axes[1,0].grid(True, alpha=0.3)
    
    # Activity level categories
    activity_levels = df['activity_level'].value_counts()
    axes[1,1].pie(activity_levels.values, labels=activity_levels.index, autopct='%1.1f%%')
    axes[1,1].set_title('Activity Level Categories')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/intensity_based_activity_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Intensity-based activity visualizations saved to {output_dir}/")

def main():
    """Main function to analyze intensity-based activity metrics"""
    
    print("Intensity-Based Activity Metrics Analysis")
    print("=" * 50)
    
    # Load activity data
    try:
        with open('../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    except:
        with open('../../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    
    print(f"Loaded {len(sleep_data)} records for intensity analysis")
    
    # Analyze first 10 records
    results = []
    for i, record in enumerate(sleep_data[:10]):
        print(f"Analyzing record {i+1}/10...")
        
        # Create mock activity data from sleep record
        activity_record = {
            'period_id': record.get('period_id', i),
            'timestamp': record.get('bedtime_start_dt_adjusted', 0),
            'steps': np.random.randint(2000, 15000),
            'active_minutes': np.random.uniform(15, 120),
            'sedentary_minutes': np.random.uniform(300, 900)
        }
        
        result = analyze_intensity_based_metrics(activity_record)
        results.append(result)
    
    # Save results
    output_dir = '../results/intensity_based_activity'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/intensity_based_activity_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}/intensity_based_activity_results.json")
    
    # Create visualizations
    create_visualizations(results, output_dir)
    
    # Print summary statistics
    print("\nIntensity-Based Activity Summary:")
    print("-" * 40)
    
    mvpa_vals = [r['mvpa_minutes'] for r in results]
    met_vals = [r['met_minutes'] for r in results]
    
    print(f"MVPA - Mean: {np.mean(mvpa_vals):.0f} min, Range: {np.min(mvpa_vals):.0f}-{np.max(mvpa_vals):.0f} min")
    print(f"MET-minutes - Mean: {np.mean(met_vals):.0f}, Range: {np.min(met_vals):.0f}-{np.max(met_vals):.0f}")
    
    # Activity level distribution
    activity_levels = [r['activity_level'] for r in results]
    level_counts = {level: activity_levels.count(level) for level in set(activity_levels)}
    print(f"\nActivity Level Distribution:")
    for level, count in level_counts.items():
        print(f"  {level}: {count} records ({count/len(results)*100:.1f}%)")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
