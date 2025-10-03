"""
Contextual Modifier Analysis for S-Entropy Framework
===================================================

Implements contextual modifiers that adjust physiological interpretation based on:
- Day-Night HRV Ratio - comparison of daytime vs nighttime HRV
- Activity-HR Coupling - heart rate response to movement
- Recovery Heart Rate - HR return to baseline after activity
- Sleep-HR Correlation - relationship between sleep stages and cardiac patterns
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import os
from pathlib import Path
from datetime import datetime, time
from scipy.stats import pearsonr

def day_night_hrv_ratio(activity_record: Dict[str, Any], sleep_record: Dict[str, Any]) -> Dict[str, float]:
    """Calculate ratio of daytime vs nighttime HRV"""
    
    # Extract HRV data (using hr_5min as proxy for HRV if rmssd not available)
    activity_hrv = activity_record.get('rmssd_5min', activity_record.get('hr_5min', []))
    sleep_hrv = sleep_record.get('rmssd_5min', sleep_record.get('hr_5min', []))
    
    if not activity_hrv or not sleep_hrv:
        return {'day_night_hrv_ratio': 0.0, 'daytime_hrv_mean': 0.0, 'nighttime_hrv_mean': 0.0}
    
    # Clean data
    day_hrv_clean = [x for x in activity_hrv if x > 0]
    night_hrv_clean = [x for x in sleep_hrv if x > 0]
    
    if not day_hrv_clean or not night_hrv_clean:
        return {'day_night_hrv_ratio': 0.0, 'daytime_hrv_mean': 0.0, 'nighttime_hrv_mean': 0.0}
    
    day_mean = np.mean(day_hrv_clean)
    night_mean = np.mean(night_hrv_clean)
    
    ratio = day_mean / night_mean if night_mean > 0 else 0.0
    
    return {
        'day_night_hrv_ratio': float(ratio),
        'daytime_hrv_mean': float(day_mean),
        'nighttime_hrv_mean': float(night_mean)
    }

def activity_hr_coupling(activity_record: Dict[str, Any]) -> Dict[str, float]:
    """Calculate heart rate response to movement patterns"""
    
    hr_sequence = activity_record.get('hr_5min', [])
    steps = activity_record.get('steps', 0)
    
    if not hr_sequence:
        return {'activity_hr_coupling': 0.0, 'hr_steps_correlation': 0.0, 'hr_response_magnitude': 0.0}
    
    hr_clean = [x for x in hr_sequence if x > 0]
    if len(hr_clean) < 2:
        return {'activity_hr_coupling': 0.0, 'hr_steps_correlation': 0.0, 'hr_response_magnitude': 0.0}
    
    # Create mock step sequence if only total steps available
    if isinstance(steps, (int, float)):
        steps_sequence = [steps / len(hr_clean)] * len(hr_clean)
    else:
        steps_sequence = steps if len(steps) == len(hr_clean) else [0] * len(hr_clean)
    
    # Calculate correlation between HR and activity
    if len(steps_sequence) == len(hr_clean) and len(hr_clean) > 1:
        try:
            correlation, _ = pearsonr(hr_clean, steps_sequence)
            correlation = correlation if not np.isnan(correlation) else 0.0
        except:
            correlation = 0.0
    else:
        correlation = 0.0
    
    # HR response magnitude (variability during activity)
    hr_response_magnitude = np.std(hr_clean) if len(hr_clean) > 1 else 0.0
    
    # Activity-HR coupling score
    coupling_score = abs(correlation) * (hr_response_magnitude / 100.0)  # Normalize
    
    return {
        'activity_hr_coupling': float(coupling_score),
        'hr_steps_correlation': float(correlation),
        'hr_response_magnitude': float(hr_response_magnitude)
    }

def recovery_heart_rate(activity_record: Dict[str, Any]) -> Dict[str, float]:
    """Calculate HR return to baseline after activity"""
    
    hr_sequence = activity_record.get('hr_5min', [])
    
    if len(hr_sequence) < 5:
        return {'recovery_hr_slope': 0.0, 'recovery_time_minutes': 0.0, 'recovery_efficiency': 0.0}
    
    hr_clean = [x for x in hr_sequence if x > 0]
    if len(hr_clean) < 5:
        return {'recovery_hr_slope': 0.0, 'recovery_time_minutes': 0.0, 'recovery_efficiency': 0.0}
    
    # Assume last half of sequence is recovery period
    recovery_start = len(hr_clean) // 2
    recovery_hr = hr_clean[recovery_start:]
    
    if len(recovery_hr) < 2:
        return {'recovery_hr_slope': 0.0, 'recovery_time_minutes': 0.0, 'recovery_efficiency': 0.0}
    
    # Calculate recovery slope (negative indicates good recovery)
    time_points = np.arange(len(recovery_hr))
    recovery_slope = np.polyfit(time_points, recovery_hr, 1)[0]
    
    # Recovery time estimate (time to reach 90% of minimum HR)
    max_recovery_hr = max(recovery_hr)
    min_recovery_hr = min(recovery_hr)
    target_hr = min_recovery_hr + 0.1 * (max_recovery_hr - min_recovery_hr)
    
    recovery_time_minutes = 0.0
    for i, hr in enumerate(recovery_hr):
        if hr <= target_hr:
            recovery_time_minutes = i * 5.0  # Assuming 5-minute intervals
            break
    
    # Recovery efficiency (faster negative slope = better)
    recovery_efficiency = abs(recovery_slope) if recovery_slope < 0 else 0.0
    
    return {
        'recovery_hr_slope': float(recovery_slope),
        'recovery_time_minutes': float(recovery_time_minutes),
        'recovery_efficiency': float(recovery_efficiency)
    }

def sleep_hr_correlation(sleep_record: Dict[str, Any]) -> Dict[str, float]:
    """Calculate relationship between sleep stages and cardiac patterns"""
    
    hr_sequence = sleep_record.get('hr_5min', [])
    hypnogram = sleep_record.get('hypnogram_5min', '')
    
    if not hr_sequence or not hypnogram:
        return {'sleep_hr_correlation': 0.0, 'deep_sleep_hr_reduction': 0.0, 'rem_hr_variability': 0.0}
    
    hr_clean = [x for x in hr_sequence if x > 0]
    if len(hr_clean) != len(hypnogram):
        # Align sequences by truncating longer one
        min_len = min(len(hr_clean), len(hypnogram))
        hr_clean = hr_clean[:min_len]
        hypnogram = hypnogram[:min_len]
    
    if len(hr_clean) < 5:
        return {'sleep_hr_correlation': 0.0, 'deep_sleep_hr_reduction': 0.0, 'rem_hr_variability': 0.0}
    
    # Separate HR by sleep stage
    deep_sleep_hr = [hr for hr, stage in zip(hr_clean, hypnogram) if stage == 'D']
    rem_sleep_hr = [hr for hr, stage in zip(hr_clean, hypnogram) if stage == 'R']
    light_sleep_hr = [hr for hr, stage in zip(hr_clean, hypnogram) if stage == 'L']
    awake_hr = [hr for hr, stage in zip(hr_clean, hypnogram) if stage == 'A']
    
    # Calculate stage-specific metrics
    deep_sleep_hr_reduction = 0.0
    if deep_sleep_hr and light_sleep_hr:
        deep_mean = np.mean(deep_sleep_hr)
        light_mean = np.mean(light_sleep_hr)
        deep_sleep_hr_reduction = (light_mean - deep_mean) / light_mean if light_mean > 0 else 0.0
    
    rem_hr_variability = np.std(rem_sleep_hr) if rem_sleep_hr else 0.0
    
    # Overall sleep-HR correlation (stage encoding: A=4, L=3, D=1, R=2)
    stage_encoding = {'A': 4, 'L': 3, 'D': 1, 'R': 2}
    stage_values = [stage_encoding.get(stage, 0) for stage in hypnogram]
    
    try:
        correlation, _ = pearsonr(hr_clean, stage_values)
        correlation = correlation if not np.isnan(correlation) else 0.0
    except:
        correlation = 0.0
    
    return {
        'sleep_hr_correlation': float(correlation),
        'deep_sleep_hr_reduction': float(deep_sleep_hr_reduction),
        'rem_hr_variability': float(rem_hr_variability)
    }

def analyze_contextual_modifiers(activity_record: Dict[str, Any], sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive contextual modifier analysis"""
    
    # Calculate all contextual modifiers
    day_night_hrv = day_night_hrv_ratio(activity_record, sleep_record)
    activity_coupling = activity_hr_coupling(activity_record)
    recovery_metrics = recovery_heart_rate(activity_record)
    sleep_correlation = sleep_hr_correlation(sleep_record)
    
    # Combine results
    result = {
        'activity_period_id': activity_record.get('period_id', 0),
        'sleep_period_id': sleep_record.get('period_id', 0),
        'day_night_analysis': day_night_hrv,
        'activity_coupling_analysis': activity_coupling,
        'recovery_analysis': recovery_metrics,
        'sleep_correlation_analysis': sleep_correlation
    }
    
    # Calculate composite contextual modifier score
    modifier_components = [
        abs(day_night_hrv.get('day_night_hrv_ratio', 0) - 1.0),  # Deviation from 1.0 ratio
        activity_coupling.get('activity_hr_coupling', 0),
        recovery_metrics.get('recovery_efficiency', 0),
        abs(sleep_correlation.get('sleep_hr_correlation', 0))
    ]
    
    contextual_modifier_score = np.mean([x for x in modifier_components if x > 0])
    result['contextual_modifier_score'] = float(contextual_modifier_score)
    
    return result

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualizations of contextual modifier analysis"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("No data for visualization")
        return
    
    # Extract data for visualization
    viz_data = []
    for result in results:
        row = {
            'period_pair': f"{result['activity_period_id']}-{result['sleep_period_id']}",
            'day_night_hrv_ratio': result['day_night_analysis'].get('day_night_hrv_ratio', 0),
            'activity_hr_coupling': result['activity_coupling_analysis'].get('activity_hr_coupling', 0),
            'recovery_efficiency': result['recovery_analysis'].get('recovery_efficiency', 0),
            'sleep_hr_correlation': result['sleep_correlation_analysis'].get('sleep_hr_correlation', 0),
            'contextual_modifier_score': result.get('contextual_modifier_score', 0)
        }
        viz_data.append(row)
    
    df = pd.DataFrame(viz_data)
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Day-Night HRV Ratio
    axes[0,0].bar(range(len(df)), df['day_night_hrv_ratio'], alpha=0.7)
    axes[0,0].axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Ideal Ratio (1.0)')
    axes[0,0].set_title('Day-Night HRV Ratio')
    axes[0,0].set_ylabel('Ratio')
    axes[0,0].legend()
    
    # Activity-HR Coupling
    axes[0,1].bar(range(len(df)), df['activity_hr_coupling'], alpha=0.7, color='orange')
    axes[0,1].set_title('Activity-HR Coupling Score')
    axes[0,1].set_ylabel('Coupling Score')
    
    # Recovery Efficiency
    axes[1,0].bar(range(len(df)), df['recovery_efficiency'], alpha=0.7, color='green')
    axes[1,0].set_title('Recovery Efficiency')
    axes[1,0].set_ylabel('Efficiency Score')
    
    # Sleep-HR Correlation
    axes[1,1].bar(range(len(df)), df['sleep_hr_correlation'], alpha=0.7, color='purple')
    axes[1,1].set_title('Sleep-HR Correlation')
    axes[1,1].set_ylabel('Correlation Coefficient')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/contextual_modifiers.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Composite score visualization
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(df)), df['contextual_modifier_score'], 'o-', alpha=0.7)
    plt.title('Contextual Modifier Composite Score')
    plt.xlabel('Record Pair')
    plt.ylabel('Composite Score')
    plt.grid(True, alpha=0.3)
    plt.savefig(f'{output_dir}/contextual_modifier_scores.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Contextual modifier visualizations saved to {output_dir}/")

def main():
    """Main function to analyze contextual modifiers"""
    
    print("Contextual Modifier Analysis")
    print("=" * 50)
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent  # From src/coupling/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "contextual_modifier"
    
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
    
    if not activity_data or not sleep_data:
        print("❌ Need both activity and sleep data for contextual modifier analysis!")
        return
    
    # Pair activity and sleep records for analysis
    results = []
    max_pairs = min(len(activity_data), len(sleep_data), 10)
    
    print(f"Analyzing {max_pairs} activity-sleep pairs...")
    for i in range(max_pairs):
        print(f"Processing pair {i+1}/{max_pairs}...")
        activity_record = activity_data[i]
        sleep_record = sleep_data[i]
        
        result = analyze_contextual_modifiers(activity_record, sleep_record)
        results.append(result)
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/contextual_modifier_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/contextual_modifier_results.json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(results, output_directory)
    
    # Print summary statistics
    print("\nContextual Modifier Summary:")
    print("-" * 40)
    
    day_night_ratios = [r['day_night_analysis']['day_night_hrv_ratio'] for r in results]
    coupling_scores = [r['activity_coupling_analysis']['activity_hr_coupling'] for r in results]
    recovery_scores = [r['recovery_analysis']['recovery_efficiency'] for r in results]
    sleep_correlations = [r['sleep_correlation_analysis']['sleep_hr_correlation'] for r in results]
    
    print(f"Day-Night HRV Ratio - Mean: {np.mean(day_night_ratios):.3f}, Std: {np.std(day_night_ratios):.3f}")
    print(f"Activity-HR Coupling - Mean: {np.mean(coupling_scores):.3f}, Std: {np.std(coupling_scores):.3f}")
    print(f"Recovery Efficiency - Mean: {np.mean(recovery_scores):.3f}, Std: {np.std(recovery_scores):.3f}")
    print(f"Sleep-HR Correlation - Mean: {np.mean(sleep_correlations):.3f}, Std: {np.std(sleep_correlations):.3f}")
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()
