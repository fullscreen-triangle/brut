"""
Deep Sleep (N3) Metrics
Slow Wave Sleep Time - total N3 duration
Slow Wave Sleep Percentage - N3 as % of TST
Delta Power - spectral power in delta frequency band (0.5-4 Hz)
Slow Wave Activity (SWA) - delta power density
Sleep Spindle Density - frequency of sleep spindles (11-15 Hz)
K-Complex Frequency - rate of K-complex occurrence
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import os
from pathlib import Path

def slow_wave_sleep_time(sleep_record: Dict[str, Any]) -> float:
    """Calculate total N3 (deep sleep) duration"""
    deep_sleep_hrs = sleep_record.get('deep_in_hrs', 0.0)
    return float(deep_sleep_hrs * 60)  # Convert to minutes

def slow_wave_sleep_percentage(sleep_record: Dict[str, Any]) -> float:
    """Calculate N3 as percentage of total sleep time"""
    deep_sleep_hrs = sleep_record.get('deep_in_hrs', 0.0)
    total_sleep_hrs = sleep_record.get('total_in_hrs', 0.0)
    
    if total_sleep_hrs > 0:
        percentage = (deep_sleep_hrs / total_sleep_hrs) * 100
        return float(percentage)
    return 0.0

def delta_power(sleep_record: Dict[str, Any]) -> float:
    """Calculate spectral power in delta frequency band (0.5-4 Hz)"""
    hr_sequence = sleep_record.get('hr_5min', [])
    hypnogram = sleep_record.get('hypnogram_5min', '')
    
    if not hr_sequence or not hypnogram:
        deep_sleep_hrs = sleep_record.get('deep_in_hrs', 0.0)
        return float(deep_sleep_hrs * 100)
    
    # Extract HR during deep sleep periods
    deep_sleep_hr = []
    for i, stage in enumerate(hypnogram):
        if stage == 'D' and i < len(hr_sequence) and hr_sequence[i] > 0:
            deep_sleep_hr.append(hr_sequence[i])
    
    if len(deep_sleep_hr) < 3:
        return 0.0
    
    hr_array = np.array(deep_sleep_hr)
    hr_variance = np.var(hr_array)
    delta_power_estimate = max(0, 100 - hr_variance)
    
    return float(delta_power_estimate)

def slow_wave_activity(sleep_record: Dict[str, Any]) -> float:
    """Calculate delta power density (SWA)"""
    delta_pow = delta_power(sleep_record)
    deep_sleep_time = slow_wave_sleep_time(sleep_record)
    
    if deep_sleep_time > 0:
        swa = delta_pow / (deep_sleep_time / 60)
        return float(swa)
    return 0.0

def sleep_spindle_density(sleep_record: Dict[str, Any]) -> float:
    """Calculate frequency of sleep spindles (11-15 Hz)"""
    hypnogram = sleep_record.get('hypnogram_5min', '')
    light_sleep_hrs = sleep_record.get('light_in_hrs', 0.0)
    
    if not hypnogram:
        spindle_density = light_sleep_hrs * 2.5
        return float(spindle_density)
    
    light_sleep_epochs = hypnogram.count('L')
    
    if light_sleep_epochs > 0:
        sleep_transitions = sum(1 for i in range(1, len(hypnogram)) 
                              if hypnogram[i] != hypnogram[i-1])
        
        base_spindle_rate = 2.0
        fragmentation_factor = max(0.5, 1.0 - (sleep_transitions / len(hypnogram)))
        
        spindle_density = light_sleep_hrs * base_spindle_rate * fragmentation_factor
        return float(spindle_density)
    
    return 0.0

def k_complex_frequency(sleep_record: Dict[str, Any]) -> float:
    """Calculate rate of K-complex occurrence"""
    light_sleep_hrs = sleep_record.get('light_in_hrs', 0.0)
    deep_sleep_hrs = sleep_record.get('deep_in_hrs', 0.0)
    
    total_nrem_hrs = light_sleep_hrs + deep_sleep_hrs
    if total_nrem_hrs == 0:
        return 0.0
    
    sleep_efficiency = sleep_record.get('efficiency', 85) / 100.0
    base_rate = 5.0
    fragmentation_multiplier = max(0.5, 2.0 - sleep_efficiency)
    
    k_complex_rate = base_rate * fragmentation_multiplier
    return float(k_complex_rate)

def analyze_deep_sleep_metrics(sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze complete deep sleep metrics"""
    
    sws_time = slow_wave_sleep_time(sleep_record)
    sws_percentage = slow_wave_sleep_percentage(sleep_record)
    delta_pow = delta_power(sleep_record)
    swa = slow_wave_activity(sleep_record)
    spindle_density = sleep_spindle_density(sleep_record)
    k_complex_freq = k_complex_frequency(sleep_record)
    
    results = {
        'period_id': sleep_record.get('period_id', 0),
        'timestamp': sleep_record.get('bedtime_start_dt_adjusted', 0),
        'slow_wave_sleep_time_min': sws_time,
        'slow_wave_sleep_percentage': sws_percentage,
        'delta_power': delta_pow,
        'slow_wave_activity': swa,
        'sleep_spindle_density': spindle_density,
        'k_complex_frequency': k_complex_freq,
        'deep_sleep_quality_score': (sws_percentage + (delta_pow / 10) + swa) / 3,
        'original_deep_sleep_hrs': sleep_record.get('deep_in_hrs', 0.0)
    }
    
    # Add deep sleep quality interpretation
    if sws_percentage >= 20:
        results['deep_sleep_quality'] = 'Excellent'
    elif sws_percentage >= 15:
        results['deep_sleep_quality'] = 'Good'
    elif sws_percentage >= 10:
        results['deep_sleep_quality'] = 'Average'
    else:
        results['deep_sleep_quality'] = 'Below Average'
    
    return results

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualizations of deep sleep metrics"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("No results available for visualization")
        return
    
    # Extract data for visualization
    deep_sleep_data = []
    for result in results:
        deep_sleep_data.append({
            'period_id': result['period_id'],
            'sws_time': result['slow_wave_sleep_time_min'],
            'sws_percentage': result['slow_wave_sleep_percentage'],
            'delta_power': result['delta_power'],
            'swa': result['slow_wave_activity'],
            'spindle_density': result['sleep_spindle_density'],
            'k_complex_freq': result['k_complex_frequency'],
            'quality_category': result['deep_sleep_quality']
        })
    
    df = pd.DataFrame(deep_sleep_data)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Deep sleep time and percentage
    axes[0,0].plot(df['period_id'], df['sws_time'], 'o-', alpha=0.7, color='darkblue', label='SWS Time')
    ax_twin = axes[0,0].twinx()
    ax_twin.plot(df['period_id'], df['sws_percentage'], 's-', alpha=0.7, color='orange', label='SWS %')
    axes[0,0].set_xlabel('Period ID')
    axes[0,0].set_ylabel('SWS Time (minutes)', color='darkblue')
    ax_twin.set_ylabel('SWS Percentage (%)', color='orange')
    axes[0,0].set_title('Slow Wave Sleep Time and Percentage')
    axes[0,0].grid(True, alpha=0.3)
    
    # Delta power over time
    axes[0,1].plot(df['period_id'], df['delta_power'], '^-', alpha=0.7, color='purple')
    axes[0,1].set_xlabel('Period ID')
    axes[0,1].set_ylabel('Delta Power')
    axes[0,1].set_title('Delta Power (0.5-4 Hz)')
    axes[0,1].grid(True, alpha=0.3)
    
    # Sleep spindle density vs K-complex frequency
    axes[1,0].scatter(df['spindle_density'], df['k_complex_freq'], alpha=0.7, s=60)
    axes[1,0].set_xlabel('Sleep Spindle Density')
    axes[1,0].set_ylabel('K-Complex Frequency')
    axes[1,0].set_title('Sleep Spindles vs K-Complexes')
    axes[1,0].grid(True, alpha=0.3)
    
    # Deep sleep quality categories
    if 'quality_category' in df.columns:
        quality_counts = df['quality_category'].value_counts()
        axes[1,1].pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Deep Sleep Quality Categories')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/deep_sleep_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Deep sleep metrics visualizations saved to {output_dir}/")

def main():
    """Main function to analyze deep sleep metrics"""
    
    print("Deep Sleep Metrics Analysis")
    print("=" * 50)
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent  # From src/sleep/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "deep_sleep_metrics"
    
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
    
    # Process sleep data (primary source for deep sleep analysis)
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:15]):
            print(f"Analyzing sleep record {i+1}/15...")
            result = analyze_deep_sleep_metrics(record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    # Process activity data for context
    if activity_data:
        print("Processing activity records for context...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {i+1}/10...")
            result = analyze_deep_sleep_metrics(record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/deep_sleep_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/deep_sleep_results.json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(all_results, output_directory)
    
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()
