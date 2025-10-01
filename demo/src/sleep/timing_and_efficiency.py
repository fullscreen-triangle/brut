"""
Total Sleep Time (TST) - actual sleep duration
Time in Bed (TIB) - total time between lights out and final wake
Sleep Onset Latency (SOL) - time to fall asleep
Wake After Sleep Onset (WASO) - total wake time during sleep period
Sleep Efficiency (SE) - TST/TIB × 100%
Number of Awakenings - frequency of wake episodes
Awakening Duration - average length of wake episodes
Sleep Fragmentation Index - degree of sleep interruption
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import os
from datetime import datetime

def total_sleep_time(sleep_record: Dict[str, Any]) -> float:
    """Calculate actual sleep duration (TST)"""
    # Extract total sleep time from record (in hours)
    tst = sleep_record.get('total_in_hrs', 0.0)
    return float(tst)

def time_in_bed(sleep_record: Dict[str, Any]) -> float:
    """Calculate total time between lights out and final wake (TIB)"""
    # Extract duration from record (in hours)
    tib = sleep_record.get('duration_in_hrs', 0.0)
    return float(tib)

def sleep_onset_latency(sleep_record: Dict[str, Any]) -> float:
    """Calculate time to fall asleep (SOL) in minutes"""
    # Extract onset latency from record and convert to minutes
    sol_hours = sleep_record.get('onset_latency_in_hrs', 0.0)
    return float(sol_hours * 60)  # Convert to minutes

def wake_after_sleep_onset(sleep_record: Dict[str, Any]) -> float:
    """Calculate total wake time during sleep period (WASO)"""
    # Extract awake time from record (in hours), convert to minutes
    waso_hours = sleep_record.get('awake_in_hrs', 0.0)
    return float(waso_hours * 60)  # Convert to minutes

def sleep_efficiency(sleep_record: Dict[str, Any]) -> float:
    """Calculate Sleep Efficiency (TST/TIB × 100%)"""
    # Use recorded efficiency if available
    efficiency = sleep_record.get('efficiency', None)
    if efficiency is not None:
        return float(efficiency)
    
    # Otherwise calculate from TST and TIB
    tst = total_sleep_time(sleep_record)
    tib = time_in_bed(sleep_record)
    
    if tib > 0:
        return float((tst / tib) * 100)
    return 0.0

def number_of_awakenings(sleep_record: Dict[str, Any]) -> int:
    """Calculate frequency of wake episodes from hypnogram"""
    hypnogram = sleep_record.get('hypnogram_5min', '')
    if not hypnogram:
        return 0
    
    awakenings = 0
    in_wake = False
    
    for stage in hypnogram:
        if stage == 'A' and not in_wake:
            # Start of wake episode
            awakenings += 1
            in_wake = True
        elif stage != 'A' and in_wake:
            # End of wake episode
            in_wake = False
    
    return awakenings

def awakening_duration(sleep_record: Dict[str, Any]) -> float:
    """Calculate average length of wake episodes in minutes"""
    hypnogram = sleep_record.get('hypnogram_5min', '')
    if not hypnogram:
        return 0.0
    
    wake_episodes = []
    current_wake_length = 0
    
    for stage in hypnogram:
        if stage == 'A':
            current_wake_length += 1
        else:
            if current_wake_length > 0:
                wake_episodes.append(current_wake_length)
                current_wake_length = 0
    
    # Handle case where hypnogram ends with wake
    if current_wake_length > 0:
        wake_episodes.append(current_wake_length)
    
    if wake_episodes:
        # Convert from 5-minute epochs to minutes
        average_duration = np.mean(wake_episodes) * 5
        return float(average_duration)
    
    return 0.0

def sleep_fragmentation_index(sleep_record: Dict[str, Any]) -> float:
    """Calculate degree of sleep interruption"""
    hypnogram = sleep_record.get('hypnogram_5min', '')
    if not hypnogram:
        return 0.0
    
    # Count stage transitions
    transitions = 0
    for i in range(1, len(hypnogram)):
        if hypnogram[i] != hypnogram[i-1]:
            transitions += 1
    
    # Calculate fragmentation index as transitions per hour
    duration_hours = len(hypnogram) * 5 / 60  # Convert 5-min epochs to hours
    if duration_hours > 0:
        fragmentation_index = transitions / duration_hours
        return float(fragmentation_index)
    
    return 0.0

def analyze_sleep_timing_efficiency(sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze complete sleep timing and efficiency metrics"""
    
    results = {
        'period_id': sleep_record.get('period_id', 0),
        'timestamp': sleep_record.get('bedtime_start_dt_adjusted', 0),
        'total_sleep_time_hrs': total_sleep_time(sleep_record),
        'time_in_bed_hrs': time_in_bed(sleep_record),
        'sleep_onset_latency_min': sleep_onset_latency(sleep_record),
        'wake_after_sleep_onset_min': wake_after_sleep_onset(sleep_record),
        'sleep_efficiency_pct': sleep_efficiency(sleep_record),
        'number_of_awakenings': number_of_awakenings(sleep_record),
        'awakening_duration_min': awakening_duration(sleep_record),
        'sleep_fragmentation_index': sleep_fragmentation_index(sleep_record)
    }
    
    # Add sleep quality categorization
    efficiency = results['sleep_efficiency_pct']
    if efficiency >= 90:
        results['efficiency_category'] = 'Excellent'
    elif efficiency >= 80:
        results['efficiency_category'] = 'Good'
    elif efficiency >= 70:
        results['efficiency_category'] = 'Fair'
    else:
        results['efficiency_category'] = 'Poor'
    
    return results

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualizations of sleep timing and efficiency metrics"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("No results available for visualization")
        return
    
    # Extract data for visualization
    timing_data = []
    for result in results:
        timing_data.append({
            'period_id': result['period_id'],
            'total_sleep_time': result['total_sleep_time_hrs'],
            'time_in_bed': result['time_in_bed_hrs'],
            'sleep_onset_latency': result['sleep_onset_latency_min'],
            'wake_after_onset': result['wake_after_sleep_onset_min'],
            'sleep_efficiency': result['sleep_efficiency_pct'],
            'num_awakenings': result['number_of_awakenings'],
            'awakening_duration': result['awakening_duration_min'],
            'fragmentation_index': result['sleep_fragmentation_index'],
            'efficiency_category': result['efficiency_category']
        })
    
    df = pd.DataFrame(timing_data)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # Sleep efficiency over time
    axes[0,0].plot(df['period_id'], df['sleep_efficiency'], 'o-', alpha=0.7, color='blue')
    axes[0,0].axhline(y=80, color='green', linestyle='--', alpha=0.5, label='Good (80%)')
    axes[0,0].axhline(y=70, color='orange', linestyle='--', alpha=0.5, label='Fair (70%)')
    axes[0,0].set_xlabel('Period ID')
    axes[0,0].set_ylabel('Sleep Efficiency (%)')
    axes[0,0].set_title('Sleep Efficiency Over Time')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # Total sleep time vs time in bed
    axes[0,1].scatter(df['time_in_bed'], df['total_sleep_time'], alpha=0.7)
    # Add perfect efficiency line
    max_time = max(df['time_in_bed'])
    axes[0,1].plot([0, max_time], [0, max_time], 'r--', alpha=0.5, label='Perfect Efficiency')
    axes[0,1].set_xlabel('Time in Bed (hrs)')
    axes[0,1].set_ylabel('Total Sleep Time (hrs)')
    axes[0,1].set_title('Total Sleep Time vs Time in Bed')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Sleep onset latency distribution
    axes[1,0].hist(df['sleep_onset_latency'], bins=10, alpha=0.7, color='purple')
    axes[1,0].axvline(x=30, color='red', linestyle='--', alpha=0.5, label='Normal (30 min)')
    axes[1,0].set_xlabel('Sleep Onset Latency (min)')
    axes[1,0].set_ylabel('Frequency')
    axes[1,0].set_title('Sleep Onset Latency Distribution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Number of awakenings vs awakening duration
    axes[1,1].scatter(df['num_awakenings'], df['awakening_duration'], alpha=0.7)
    axes[1,1].set_xlabel('Number of Awakenings')
    axes[1,1].set_ylabel('Average Awakening Duration (min)')
    axes[1,1].set_title('Awakenings: Number vs Duration')
    axes[1,1].grid(True, alpha=0.3)
    
    # Sleep fragmentation index over time
    axes[2,0].plot(df['period_id'], df['fragmentation_index'], 's-', alpha=0.7, color='red')
    axes[2,0].set_xlabel('Period ID')
    axes[2,0].set_ylabel('Fragmentation Index (transitions/hr)')
    axes[2,0].set_title('Sleep Fragmentation Over Time')
    axes[2,0].grid(True, alpha=0.3)
    
    # Efficiency category distribution
    category_counts = df['efficiency_category'].value_counts()
    axes[2,1].pie(category_counts.values, labels=category_counts.index, autopct='%1.1f%%')
    axes[2,1].set_title('Sleep Efficiency Categories')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sleep_timing_efficiency_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Additional correlation plot
    plt.figure(figsize=(10, 8))
    corr_cols = ['sleep_efficiency', 'sleep_onset_latency', 'wake_after_onset', 
                'num_awakenings', 'fragmentation_index']
    corr_matrix = df[corr_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True)
    plt.title('Sleep Timing & Efficiency Metrics Correlations')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sleep_timing_correlations.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sleep timing and efficiency visualizations saved to {output_dir}/")

def main():
    """Main function to analyze sleep timing and efficiency metrics"""
    
    print("Sleep Timing and Efficiency Metrics Analysis")
    print("=" * 50)
    
    # Load sleep data
    try:
        with open('../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    except:
        with open('../../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    
    print(f"Loaded {len(sleep_data)} sleep records")
    
    # Analyze first 15 records for better statistics
    results = []
    for i, record in enumerate(sleep_data[:15]):
        print(f"Analyzing record {i+1}/15...")
        result = analyze_sleep_timing_efficiency(record)
        results.append(result)
    
    # Save results
    output_dir = '../results/sleep_timing_efficiency'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/sleep_timing_efficiency_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}/sleep_timing_efficiency_results.json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(results, output_dir)
    
    # Print summary statistics
    print("\nSleep Timing and Efficiency Summary:")
    print("-" * 40)
    
    efficiencies = [r['sleep_efficiency_pct'] for r in results]
    onset_latencies = [r['sleep_onset_latency_min'] for r in results]
    awakenings = [r['number_of_awakenings'] for r in results]
    fragmentation = [r['sleep_fragmentation_index'] for r in results]
    
    print(f"Sleep Efficiency - Mean: {np.mean(efficiencies):.1f}%, Range: {np.min(efficiencies):.1f}-{np.max(efficiencies):.1f}%")
    print(f"Sleep Onset Latency - Mean: {np.mean(onset_latencies):.1f}min, Range: {np.min(onset_latencies):.1f}-{np.max(onset_latencies):.1f}min")
    print(f"Number of Awakenings - Mean: {np.mean(awakenings):.1f}, Range: {np.min(awakenings)}-{np.max(awakenings)}")
    print(f"Fragmentation Index - Mean: {np.mean(fragmentation):.1f} trans/hr")
    
    # Efficiency categories
    categories = [r['efficiency_category'] for r in results]
    category_counts = {cat: categories.count(cat) for cat in set(categories)}
    print(f"\nEfficiency Categories:")
    for cat, count in category_counts.items():
        print(f"  {cat}: {count} records ({count/len(results)*100:.1f}%)")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()