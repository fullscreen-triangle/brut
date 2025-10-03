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
from pathlib import Path

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
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent.parent  # From src/heart/cardiac/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "basic_cardiac_metrics"
    
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
    
    # User age for calculations (could be parameter)
    user_age = 35
    
    # Combine and process data
    all_results = []
    
    # Process sleep data (primary source for cardiac metrics during rest)
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Analyzing sleep record {i+1}/10...")
            result = analyze_cardiac_basic_metrics(record, user_age)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    # Process activity data for additional cardiac context
    if activity_data:
        print("Processing activity records...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {i+1}/10...")
            result = analyze_cardiac_basic_metrics(record, user_age)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/basic_cardiac_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/basic_cardiac_results.json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(all_results, output_directory)
    
    # Print summary statistics
    print("\nBasic Cardiac Metrics Summary:")
    print("-" * 40)
    
    resting_hrs = [r['resting_heart_rate'] for r in all_results]
    max_hrs = [r['maximum_heart_rate'] for r in all_results]
    hr_reserves = [r['heart_rate_reserve'] for r in all_results]
    
    print(f"Resting HR - Mean: {np.mean(resting_hrs):.1f} bpm, Range: {np.min(resting_hrs):.1f}-{np.max(resting_hrs):.1f} bpm")
    print(f"Max HR - Mean: {np.mean(max_hrs):.1f} bpm, Range: {np.min(max_hrs):.1f}-{np.max(max_hrs):.1f} bpm")
    print(f"HR Reserve - Mean: {np.mean(hr_reserves):.1f} bpm, Range: {np.min(hr_reserves):.1f}-{np.max(hr_reserves):.1f} bpm")
    
    # Show data source breakdown
    activity_count = sum(1 for r in all_results if r.get('data_source') == 'activity')
    sleep_count = sum(1 for r in all_results if r.get('data_source') == 'sleep')
    print(f"Data sources: {activity_count} activity records, {sleep_count} sleep records")
    
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()
