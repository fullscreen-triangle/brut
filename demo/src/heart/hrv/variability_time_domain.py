"""
RMSSD - root mean square of successive RR interval differences
SDNN - standard deviation of NN intervals (overall HRV)
SDANN - standard deviation of averaged NN intervals over time segments
SDNN Index - mean of SDNN values for each time segment
pNN50 - percentage of successive RR intervals differing by >50ms
pNN20 - percentage of successive RR intervals differing by >20ms
TINN - triangular interpolation of NN interval histogram
Geometric Mean - geometric mean of RR intervals
HRV Triangular Index - integral of density of RR interval histogram
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import os
from datetime import datetime
from scipy import stats

def rmssd(rr_intervals: List[float]) -> float:
    """Calculate root mean square of successive RR interval differences"""
    if len(rr_intervals) < 2:
        return 0.0
    
    successive_diffs = np.diff(rr_intervals)
    return float(np.sqrt(np.mean(successive_diffs**2)))

def sdnn(rr_intervals: List[float]) -> float:
    """Calculate standard deviation of NN intervals (overall HRV)"""
    if len(rr_intervals) < 2:
        return 0.0
    
    return float(np.std(rr_intervals, ddof=1))

def sdann(rr_intervals: List[float], segment_length: int = 300) -> float:
    """Calculate standard deviation of averaged NN intervals over time segments"""
    if len(rr_intervals) < segment_length:
        return 0.0
    
    # Split into segments and calculate mean of each
    n_segments = len(rr_intervals) // segment_length
    segment_means = []
    
    for i in range(n_segments):
        start = i * segment_length
        end = start + segment_length
        segment_mean = np.mean(rr_intervals[start:end])
        segment_means.append(segment_mean)
    
    return float(np.std(segment_means, ddof=1)) if len(segment_means) > 1 else 0.0

def sdnn_index(rr_intervals: List[float], segment_length: int = 300) -> float:
    """Calculate mean of SDNN values for each time segment"""
    if len(rr_intervals) < segment_length:
        return 0.0
    
    # Split into segments and calculate SDNN of each
    n_segments = len(rr_intervals) // segment_length
    segment_sdnns = []
    
    for i in range(n_segments):
        start = i * segment_length
        end = start + segment_length
        segment_sdnn = np.std(rr_intervals[start:end], ddof=1)
        segment_sdnns.append(segment_sdnn)
    
    return float(np.mean(segment_sdnns)) if segment_sdnns else 0.0

def pnn50(rr_intervals: List[float]) -> float:
    """Calculate percentage of successive RR intervals differing by >50ms"""
    if len(rr_intervals) < 2:
        return 0.0
    
    successive_diffs = np.abs(np.diff(rr_intervals))
    count_over_50 = np.sum(successive_diffs > 50.0)
    return float((count_over_50 / len(successive_diffs)) * 100)

def pnn20(rr_intervals: List[float]) -> float:
    """Calculate percentage of successive RR intervals differing by >20ms"""
    if len(rr_intervals) < 2:
        return 0.0
    
    successive_diffs = np.abs(np.diff(rr_intervals))
    count_over_20 = np.sum(successive_diffs > 20.0)
    return float((count_over_20 / len(successive_diffs)) * 100)

def tinn(rr_intervals: List[float], bins: int = 128) -> float:
    """Calculate triangular interpolation of NN interval histogram"""
    if len(rr_intervals) < 10:
        return 0.0
    
    # Create histogram
    hist, bin_edges = np.histogram(rr_intervals, bins=bins)
    
    # Find peak of histogram
    peak_idx = np.argmax(hist)
    peak_value = hist[peak_idx]
    
    # Fit triangular interpolation
    # Find baseline points where histogram drops to specified fraction of peak
    baseline_fraction = 0.05  # 5% of peak height
    baseline_threshold = peak_value * baseline_fraction
    
    # Find left baseline
    left_baseline = 0
    for i in range(peak_idx, -1, -1):
        if hist[i] <= baseline_threshold:
            left_baseline = bin_edges[i]
            break
    
    # Find right baseline
    right_baseline = bin_edges[-1]
    for i in range(peak_idx, len(hist)):
        if hist[i] <= baseline_threshold:
            right_baseline = bin_edges[i]
            break
    
    # TINN is the width of the triangular interpolation
    return float(right_baseline - left_baseline)

def geometric_mean(rr_intervals: List[float]) -> float:
    """Calculate geometric mean of RR intervals"""
    if len(rr_intervals) == 0:
        return 0.0
    
    # Remove any zero or negative values
    positive_intervals = [x for x in rr_intervals if x > 0]
    if not positive_intervals:
        return 0.0
    
    return float(stats.gmean(positive_intervals))

def hrv_triangular_index(rr_intervals: List[float], bins: int = 128) -> float:
    """Calculate integral of density of RR interval histogram"""
    if len(rr_intervals) < 10:
        return 0.0
    
    # Create histogram
    hist, bin_edges = np.histogram(rr_intervals, bins=bins)
    
    # Find peak value
    peak_value = np.max(hist)
    
    if peak_value == 0:
        return 0.0
    
    # HRV Triangular Index = Total number of NN intervals / Maximum frequency
    return float(len(rr_intervals) / peak_value)

def convert_hr_to_rr(hr_sequence: List[float]) -> List[float]:
    """Convert heart rate (BPM) to RR intervals (ms)"""
    rr_intervals = []
    for hr in hr_sequence:
        if hr > 0:
            # RR interval (ms) = 60000 / HR (BPM)
            rr = 60000.0 / hr
            rr_intervals.append(rr)
    return rr_intervals

def analyze_hrv_record(sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze HRV time domain metrics from sleep record"""
    
    # Extract heart rate and HRV data
    hr_sequence = sleep_record.get('hr_5min', [])
    rmssd_sequence = sleep_record.get('rmssd_5min', [])
    
    # Convert HR to RR intervals if available
    rr_intervals = convert_hr_to_rr([x for x in hr_sequence if x > 0]) if hr_sequence else []
    
    # Use RMSSD sequence if available, otherwise calculate from RR intervals
    rmssd_values = [x for x in rmssd_sequence if x > 0] if rmssd_sequence else []
    
    results = {
        'period_id': sleep_record.get('period_id', 0),
        'timestamp': sleep_record.get('bedtime_start_dt_adjusted', 0),
        'rmssd': rmssd(rr_intervals) if rr_intervals else (np.mean(rmssd_values) if rmssd_values else 0.0),
        'sdnn': sdnn(rr_intervals) if rr_intervals else 0.0,
        'sdann': sdann(rr_intervals) if rr_intervals else 0.0,
        'sdnn_index': sdnn_index(rr_intervals) if rr_intervals else 0.0,
        'pnn50': pnn50(rr_intervals) if rr_intervals else 0.0,
        'pnn20': pnn20(rr_intervals) if rr_intervals else 0.0,
        'tinn': tinn(rr_intervals) if rr_intervals else 0.0,
        'geometric_mean': geometric_mean(rr_intervals) if rr_intervals else 0.0,
        'hrv_triangular_index': hrv_triangular_index(rr_intervals) if rr_intervals else 0.0,
        'rr_interval_count': len(rr_intervals),
        'original_rmssd_values': len(rmssd_values)
    }
    
    return results

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualizations of HRV time domain metrics"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("No results available for visualization")
        return
    
    # Extract data for visualization
    hrv_data = []
    for result in results:
        hrv_data.append({
            'period_id': result['period_id'],
            'rmssd': result['rmssd'],
            'sdnn': result['sdnn'],
            'pnn50': result['pnn50'],
            'pnn20': result['pnn20'],
            'tinn': result['tinn'],
            'hrv_triangular_index': result['hrv_triangular_index']
        })
    
    df = pd.DataFrame(hrv_data)
    
    # Create comprehensive visualization
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    # RMSSD over time
    axes[0,0].plot(df['period_id'], df['rmssd'], 'o-', alpha=0.7, color='blue')
    axes[0,0].set_xlabel('Period ID')
    axes[0,0].set_ylabel('RMSSD (ms)')
    axes[0,0].set_title('RMSSD Over Time')
    axes[0,0].grid(True, alpha=0.3)
    
    # SDNN over time
    axes[0,1].plot(df['period_id'], df['sdnn'], 's-', alpha=0.7, color='red')
    axes[0,1].set_xlabel('Period ID')
    axes[0,1].set_ylabel('SDNN (ms)')
    axes[0,1].set_title('SDNN Over Time')
    axes[0,1].grid(True, alpha=0.3)
    
    # pNN50 vs pNN20
    axes[1,0].scatter(df['pnn50'], df['pnn20'], alpha=0.7)
    axes[1,0].set_xlabel('pNN50 (%)')
    axes[1,0].set_ylabel('pNN20 (%)')
    axes[1,0].set_title('pNN50 vs pNN20')
    axes[1,0].grid(True, alpha=0.3)
    
    # HRV metrics distribution
    hrv_metrics = ['rmssd', 'sdnn', 'pnn50', 'pnn20']
    df[hrv_metrics].boxplot(ax=axes[1,1])
    axes[1,1].set_title('HRV Time Domain Metrics Distribution')
    axes[1,1].set_ylabel('Value')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    # TINN over time
    axes[2,0].plot(df['period_id'], df['tinn'], '^-', alpha=0.7, color='green')
    axes[2,0].set_xlabel('Period ID')
    axes[2,0].set_ylabel('TINN (ms)')
    axes[2,0].set_title('TINN Over Time')
    axes[2,0].grid(True, alpha=0.3)
    
    # Correlation heatmap
    corr_cols = ['rmssd', 'sdnn', 'pnn50', 'pnn20', 'tinn']
    available_cols = [col for col in corr_cols if col in df.columns and not df[col].isna().all()]
    if len(available_cols) > 1:
        corr_matrix = df[available_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=axes[2,1])
        axes[2,1].set_title('HRV Metrics Correlations')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hrv_time_domain_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"HRV time domain metrics visualization saved to {output_dir}/")

def main():
    """Main function to analyze HRV time domain metrics"""
    
    print("HRV Time Domain Metrics Analysis")
    print("=" * 50)
    
    # Load sleep data with heart rate information
    try:
        with open('../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    except:
        with open('../../public/sleep_ppg_records.json', 'r') as f:
            sleep_data = json.load(f)
    
    print(f"Loaded {len(sleep_data)} sleep records")
    
    # Analyze first 10 records
    results = []
    for i, record in enumerate(sleep_data[:10]):
        print(f"Analyzing record {i+1}/10...")
        result = analyze_hrv_record(record)
        results.append(result)
    
    # Save results
    output_dir = '../results/hrv_time_domain'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/hrv_time_domain_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}/hrv_time_domain_results.json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(results, output_dir)
    
    # Print summary statistics
    print("\nHRV Time Domain Metrics Summary:")
    print("-" * 40)
    
    rmssd_vals = [r['rmssd'] for r in results if r['rmssd'] > 0]
    sdnn_vals = [r['sdnn'] for r in results if r['sdnn'] > 0]
    pnn50_vals = [r['pnn50'] for r in results if r['pnn50'] > 0]
    
    if rmssd_vals:
        print(f"RMSSD - Mean: {np.mean(rmssd_vals):.1f}ms, Range: {np.min(rmssd_vals):.1f}-{np.max(rmssd_vals):.1f}ms")
    if sdnn_vals:
        print(f"SDNN - Mean: {np.mean(sdnn_vals):.1f}ms, Range: {np.min(sdnn_vals):.1f}-{np.max(sdnn_vals):.1f}ms")
    if pnn50_vals:
        print(f"pNN50 - Mean: {np.mean(pnn50_vals):.1f}%, Range: {np.min(pnn50_vals):.1f}-{np.max(pnn50_vals):.1f}%")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()