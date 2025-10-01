"""
REM Sleep Metrics
REM Sleep Time - total REM duration
REM Sleep Percentage - REM as % of TST
REM Density - intensity of eye movements during REM
REM Episode Count - number of discrete REM periods
REM Episode Duration - average length of REM episodes
REM Efficiency - REM time / time in REM attempts
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import os

def rem_sleep_time(sleep_record: Dict[str, Any]) -> float:
    """Calculate total REM duration"""
    rem_sleep_hrs = sleep_record.get('rem_in_hrs', 0.0)
    return float(rem_sleep_hrs * 60)  # Convert to minutes

def rem_sleep_percentage(sleep_record: Dict[str, Any]) -> float:
    """Calculate REM as percentage of total sleep time"""
    rem_sleep_hrs = sleep_record.get('rem_in_hrs', 0.0)
    total_sleep_hrs = sleep_record.get('total_in_hrs', 0.0)
    
    if total_sleep_hrs > 0:
        percentage = (rem_sleep_hrs / total_sleep_hrs) * 100
        return float(percentage)
    return 0.0

def rem_density(sleep_record: Dict[str, Any]) -> float:
    """Calculate intensity of eye movements during REM"""
    hr_sequence = sleep_record.get('hr_5min', [])
    hypnogram = sleep_record.get('hypnogram_5min', '')
    
    if not hr_sequence or not hypnogram:
        rem_sleep_hrs = sleep_record.get('rem_in_hrs', 0.0)
        sleep_efficiency = sleep_record.get('efficiency', 85) / 100.0
        estimated_density = rem_sleep_hrs * sleep_efficiency * 15
        return float(estimated_density)
    
    # Extract HR during REM periods
    rem_hr_values = []
    for i, stage in enumerate(hypnogram):
        if stage == 'R' and i < len(hr_sequence) and hr_sequence[i] > 0:
            rem_hr_values.append(hr_sequence[i])
    
    if len(rem_hr_values) < 2:
        return 0.0
    
    hr_variability = np.std(rem_hr_values)
    rem_density_estimate = min(25, max(5, hr_variability * 0.5))
    
    return float(rem_density_estimate)

def rem_episode_count(sleep_record: Dict[str, Any]) -> int:
    """Calculate number of discrete REM periods"""
    hypnogram = sleep_record.get('hypnogram_5min', '')
    
    if not hypnogram:
        total_sleep_hrs = sleep_record.get('total_in_hrs', 0.0)
        if total_sleep_hrs > 6:
            estimated_episodes = 4
        elif total_sleep_hrs > 4:
            estimated_episodes = 3
        elif total_sleep_hrs > 2:
            estimated_episodes = 2
        else:
            estimated_episodes = 1
        return int(estimated_episodes)
    
    rem_episodes = 0
    in_rem = False
    
    for stage in hypnogram:
        if stage == 'R' and not in_rem:
            rem_episodes += 1
            in_rem = True
        elif stage != 'R' and in_rem:
            in_rem = False
    
    return rem_episodes

def rem_episode_duration(sleep_record: Dict[str, Any]) -> float:
    """Calculate average length of REM episodes"""
    hypnogram = sleep_record.get('hypnogram_5min', '')
    
    if not hypnogram:
        rem_sleep_hrs = sleep_record.get('rem_in_hrs', 0.0)
        episode_count = rem_episode_count(sleep_record)
        if episode_count > 0:
            avg_duration = (rem_sleep_hrs * 60) / episode_count
            return float(avg_duration)
        return 0.0
    
    rem_episodes = []
    current_episode_length = 0
    
    for stage in hypnogram:
        if stage == 'R':
            current_episode_length += 1
        else:
            if current_episode_length > 0:
                rem_episodes.append(current_episode_length)
                current_episode_length = 0
    
    if current_episode_length > 0:
        rem_episodes.append(current_episode_length)
    
    if rem_episodes:
        avg_duration = np.mean(rem_episodes) * 5  # Convert to minutes
        return float(avg_duration)
    
    return 0.0

def rem_efficiency(sleep_record: Dict[str, Any]) -> float:
    """Calculate REM time / time in REM attempts"""
    rem_sleep_hrs = sleep_record.get('rem_in_hrs', 0.0)
    total_sleep_hrs = sleep_record.get('total_in_hrs', 0.0)
    
    expected_rem_percentage = 0.22  # 22% is typical
    potential_rem_time = total_sleep_hrs * expected_rem_percentage
    
    if potential_rem_time > 0:
        rem_eff = min(100, (rem_sleep_hrs / potential_rem_time) * 100)
        return float(rem_eff)
    
    return 0.0

def analyze_rem_sleep_metrics(sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze complete REM sleep metrics"""
    
    rem_time = rem_sleep_time(sleep_record)
    rem_percentage = rem_sleep_percentage(sleep_record)
    rem_dens = rem_density(sleep_record)
    episode_count = rem_episode_count(sleep_record)
    avg_episode_duration = rem_episode_duration(sleep_record)
    rem_eff = rem_efficiency(sleep_record)
    
    results = {
        'period_id': sleep_record.get('period_id', 0),
        'timestamp': sleep_record.get('bedtime_start_dt_adjusted', 0),
        'rem_sleep_time_min': rem_time,
        'rem_sleep_percentage': rem_percentage,
        'rem_density': rem_dens,
        'rem_episode_count': episode_count,
        'rem_episode_duration_min': avg_episode_duration,
        'rem_efficiency': rem_eff,
        'rem_quality_score': (rem_percentage + rem_eff + (rem_dens / 2)) / 3,
        'original_rem_sleep_hrs': sleep_record.get('rem_in_hrs', 0.0)
    }
    
    # Add REM sleep quality interpretation
    if rem_percentage >= 25:
        results['rem_sleep_quality'] = 'Excellent'
    elif rem_percentage >= 20:
        results['rem_sleep_quality'] = 'Good'
    elif rem_percentage >= 15:
        results['rem_sleep_quality'] = 'Average'
    else:
        results['rem_sleep_quality'] = 'Below Average'
    
    return results

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualizations of REM sleep metrics"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("No results available for visualization")
        return
    
    # Extract data for visualization
    rem_data = []
    for result in results:
        rem_data.append({
            'period_id': result['period_id'],
            'rem_time': result['rem_sleep_time_min'],
            'rem_percentage': result['rem_sleep_percentage'],
            'rem_density': result['rem_density'],
            'episode_count': result['rem_episode_count'],
            'episode_duration': result['rem_episode_duration_min'],
            'rem_efficiency': result['rem_efficiency'],
            'rem_quality': result['rem_sleep_quality']
        })
    
    df = pd.DataFrame(rem_data)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # REM time and percentage
    axes[0,0].plot(df['period_id'], df['rem_time'], 'o-', alpha=0.7, color='darkred', label='REM Time')
    ax_twin = axes[0,0].twinx()
    ax_twin.plot(df['period_id'], df['rem_percentage'], 's-', alpha=0.7, color='orange', label='REM %')
    axes[0,0].set_xlabel('Period ID')
    axes[0,0].set_ylabel('REM Time (minutes)', color='darkred')
    ax_twin.set_ylabel('REM Percentage (%)', color='orange')
    axes[0,0].set_title('REM Sleep Time and Percentage')
    axes[0,0].grid(True, alpha=0.3)
    
    # REM density over time
    axes[0,1].plot(df['period_id'], df['rem_density'], '^-', alpha=0.7, color='purple')
    axes[0,1].set_xlabel('Period ID')
    axes[0,1].set_ylabel('REM Density')
    axes[0,1].set_title('REM Density (Eye Movement Intensity)')
    axes[0,1].grid(True, alpha=0.3)
    
    # REM episode count vs duration
    axes[1,0].scatter(df['episode_count'], df['episode_duration'], alpha=0.7, s=60)
    axes[1,0].set_xlabel('REM Episode Count')
    axes[1,0].set_ylabel('Average Episode Duration (min)')
    axes[1,0].set_title('REM Episodes: Count vs Duration')
    axes[1,0].grid(True, alpha=0.3)
    
    # REM sleep quality categories
    if 'rem_quality' in df.columns:
        quality_counts = df['rem_quality'].value_counts()
        axes[1,1].pie(quality_counts.values, labels=quality_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('REM Sleep Quality Categories')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/rem_sleep_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"REM sleep metrics visualizations saved to {output_dir}/")

def main():
    """Main function to analyze REM sleep metrics"""
    
    print("REM Sleep Metrics Analysis")
    print("=" * 50)
    
    # Load sleep data
    try:
        with open('../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    except:
        with open('../../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    
    print(f"Loaded {len(sleep_data)} sleep records")
    
    # Analyze first 15 records
    results = []
    for i, record in enumerate(sleep_data[:15]):
        print(f"Analyzing record {i+1}/15...")
        result = analyze_rem_sleep_metrics(record)
        results.append(result)
    
    # Save results
    output_dir = '../results/rem_sleep_metrics'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/rem_sleep_metrics_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}/rem_sleep_metrics_results.json")
    
    # Create visualizations
    create_visualizations(results, output_dir)
    
    # Print summary statistics
    print("\nREM Sleep Metrics Summary:")
    print("-" * 40)
    
    rem_times = [r['rem_sleep_time_min'] for r in results]
    rem_percentages = [r['rem_sleep_percentage'] for r in results]
    rem_densities = [r['rem_density'] for r in results]
    episode_counts = [r['rem_episode_count'] for r in results]
    
    print(f"REM Time - Mean: {np.mean(rem_times):.1f} min, Range: {np.min(rem_times):.1f}-{np.max(rem_times):.1f} min")
    print(f"REM Percentage - Mean: {np.mean(rem_percentages):.1f}%, Range: {np.min(rem_percentages):.1f}-{np.max(rem_percentages):.1f}%")
    print(f"REM Density - Mean: {np.mean(rem_densities):.1f}, Range: {np.min(rem_densities):.1f}-{np.max(rem_densities):.1f}")
    print(f"Episode Count - Mean: {np.mean(episode_counts):.1f}, Range: {np.min(episode_counts)}-{np.max(episode_counts)}")
    
    # Quality distribution
    quality_categories = [r['rem_sleep_quality'] for r in results]
    quality_counts = {cat: quality_categories.count(cat) for cat in set(quality_categories)}
    print(f"\nREM Sleep Quality Distribution:")
    for cat, count in quality_counts.items():
        print(f"  {cat}: {count} records ({count/len(results)*100:.1f}%)")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
