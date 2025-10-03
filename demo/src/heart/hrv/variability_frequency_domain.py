"""
VLF Power (0.003-0.04 Hz) - very low frequency power
LF Power (0.04-0.15 Hz) - low frequency power (sympathetic/parasympathetic)
HF Power (0.15-0.4 Hz) - high frequency power (parasympathetic)
LF/HF Ratio - sympathovagal balance indicator
Total Power - overall spectral power
Peak Frequency - frequency of maximum spectral density
Normalized LF (LFnu) - LF power in normalized units
Normalized HF (HFnu) - HF power in normalized units
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
import os
from datetime import datetime
from scipy import signal
from pathlib import Path
from scipy.fft import fft, fftfreq

def vlf_power(rr_intervals: List[float], sampling_rate: float = 4.0) -> float:
    """Calculate very low frequency power (0.003-0.04 Hz)"""
    if len(rr_intervals) < 10:
        return 0.0
    
    # Resample to regular intervals if needed
    rr_array = np.array(rr_intervals)
    
    # Calculate power spectral density using Welch's method
    freqs, psd = signal.welch(rr_array, fs=sampling_rate, nperseg=min(len(rr_array), 256))
    
    # Find VLF band indices
    vlf_indices = np.where((freqs >= 0.003) & (freqs < 0.04))
    
    # Calculate power in VLF band
    vlf_power_val = np.trapz(psd[vlf_indices], freqs[vlf_indices])
    return float(vlf_power_val)

def lf_power(rr_intervals: List[float], sampling_rate: float = 4.0) -> float:
    """Calculate low frequency power (0.04-0.15 Hz) - sympathetic/parasympathetic"""
    if len(rr_intervals) < 10:
        return 0.0
    
    rr_array = np.array(rr_intervals)
    freqs, psd = signal.welch(rr_array, fs=sampling_rate, nperseg=min(len(rr_array), 256))
    
    # Find LF band indices
    lf_indices = np.where((freqs >= 0.04) & (freqs < 0.15))
    
    # Calculate power in LF band
    lf_power_val = np.trapz(psd[lf_indices], freqs[lf_indices])
    return float(lf_power_val)

def hf_power(rr_intervals: List[float], sampling_rate: float = 4.0) -> float:
    """Calculate high frequency power (0.15-0.4 Hz) - parasympathetic"""
    if len(rr_intervals) < 10:
        return 0.0
    
    rr_array = np.array(rr_intervals)
    freqs, psd = signal.welch(rr_array, fs=sampling_rate, nperseg=min(len(rr_array), 256))
    
    # Find HF band indices
    hf_indices = np.where((freqs >= 0.15) & (freqs <= 0.4))
    
    # Calculate power in HF band
    hf_power_val = np.trapz(psd[hf_indices], freqs[hf_indices])
    return float(hf_power_val)

def lf_hf_ratio(rr_intervals: List[float], sampling_rate: float = 4.0) -> float:
    """Calculate LF/HF ratio - sympathovagal balance indicator"""
    lf_val = lf_power(rr_intervals, sampling_rate)
    hf_val = hf_power(rr_intervals, sampling_rate)
    
    if hf_val > 0:
        return float(lf_val / hf_val)
    return 0.0

def total_power(rr_intervals: List[float], sampling_rate: float = 4.0) -> float:
    """Calculate overall spectral power (0.003-0.4 Hz)"""
    if len(rr_intervals) < 10:
        return 0.0
    
    rr_array = np.array(rr_intervals)
    freqs, psd = signal.welch(rr_array, fs=sampling_rate, nperseg=min(len(rr_array), 256))
    
    # Find total frequency band of interest
    total_indices = np.where((freqs >= 0.003) & (freqs <= 0.4))
    
    # Calculate total power
    total_power_val = np.trapz(psd[total_indices], freqs[total_indices])
    return float(total_power_val)

def peak_frequency(rr_intervals: List[float], sampling_rate: float = 4.0) -> float:
    """Calculate frequency of maximum spectral density"""
    if len(rr_intervals) < 10:
        return 0.0
    
    rr_array = np.array(rr_intervals)
    freqs, psd = signal.welch(rr_array, fs=sampling_rate, nperseg=min(len(rr_array), 256))
    
    # Find peak frequency within HRV range
    valid_indices = np.where((freqs >= 0.003) & (freqs <= 0.4))
    valid_freqs = freqs[valid_indices]
    valid_psd = psd[valid_indices]
    
    if len(valid_psd) > 0:
        peak_idx = np.argmax(valid_psd)
        return float(valid_freqs[peak_idx])
    return 0.0

def normalized_lf(rr_intervals: List[float], sampling_rate: float = 4.0) -> float:
    """Calculate LF power in normalized units (LFnu)"""
    lf_val = lf_power(rr_intervals, sampling_rate)
    hf_val = hf_power(rr_intervals, sampling_rate)
    
    total_lf_hf = lf_val + hf_val
    if total_lf_hf > 0:
        return float((lf_val / total_lf_hf) * 100)
    return 0.0

def normalized_hf(rr_intervals: List[float], sampling_rate: float = 4.0) -> float:
    """Calculate HF power in normalized units (HFnu)"""
    lf_val = lf_power(rr_intervals, sampling_rate)
    hf_val = hf_power(rr_intervals, sampling_rate)
    
    total_lf_hf = lf_val + hf_val
    if total_lf_hf > 0:
        return float((hf_val / total_lf_hf) * 100)
    return 0.0

def convert_hr_to_rr(hr_sequence: List[float]) -> List[float]:
    """Convert heart rate (BPM) to RR intervals (ms)"""
    rr_intervals = []
    for hr in hr_sequence:
        if hr > 0:
            # RR interval (ms) = 60000 / HR (BPM)
            rr = 60000.0 / hr
            rr_intervals.append(rr)
    return rr_intervals

def analyze_hrv_frequency_domain(sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze HRV frequency domain metrics from sleep record"""
    
    # Extract heart rate and HRV data
    hr_sequence = sleep_record.get('hr_5min', [])
    
    # Convert HR to RR intervals if available
    rr_intervals = convert_hr_to_rr([x for x in hr_sequence if x > 0]) if hr_sequence else []
    
    # Calculate sampling rate (assuming 5-minute intervals)
    sampling_rate = 1.0 / 300.0  # 1 sample per 5 minutes = 0.0033 Hz
    
    # For frequency analysis, we need higher resolution, so we'll interpolate
    if len(rr_intervals) > 5:
        # Interpolate to get better frequency resolution
        t_original = np.arange(len(rr_intervals))
        t_interpolated = np.linspace(0, len(rr_intervals)-1, len(rr_intervals)*4)
        rr_interpolated = np.interp(t_interpolated, t_original, rr_intervals)
        sampling_rate_interp = 4.0 / 300.0  # 4x higher resolution
    else:
        rr_interpolated = rr_intervals
        sampling_rate_interp = sampling_rate
    
    results = {
        'period_id': sleep_record.get('period_id', 0),
        'timestamp': sleep_record.get('bedtime_start_dt_adjusted', 0),
        'vlf_power_ms2': vlf_power(rr_interpolated, sampling_rate_interp),
        'lf_power_ms2': lf_power(rr_interpolated, sampling_rate_interp),
        'hf_power_ms2': hf_power(rr_interpolated, sampling_rate_interp),
        'lf_hf_ratio': lf_hf_ratio(rr_interpolated, sampling_rate_interp),
        'total_power_ms2': total_power(rr_interpolated, sampling_rate_interp),
        'peak_frequency_hz': peak_frequency(rr_interpolated, sampling_rate_interp),
        'normalized_lf_nu': normalized_lf(rr_interpolated, sampling_rate_interp),
        'normalized_hf_nu': normalized_hf(rr_interpolated, sampling_rate_interp),
        'rr_interval_count': len(rr_intervals),
        'interpolated_count': len(rr_interpolated)
    }
    
    # Add autonomic balance interpretation
    lf_hf = results['lf_hf_ratio']
    if lf_hf > 2.0:
        results['autonomic_balance'] = 'Sympathetic Dominant'
    elif lf_hf < 0.5:
        results['autonomic_balance'] = 'Parasympathetic Dominant'
    else:
        results['autonomic_balance'] = 'Balanced'
    
    return results

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualizations of HRV frequency domain metrics"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("No results available for visualization")
        return
    
    # Extract data for visualization
    freq_data = []
    for result in results:
        freq_data.append({
            'period_id': result['period_id'],
            'lf_power': result['lf_power_ms2'],
            'hf_power': result['hf_power_ms2'],
            'lf_hf_ratio': result['lf_hf_ratio'],
            'lf_nu': result['normalized_lf_nu'],
            'hf_nu': result['normalized_hf_nu'],
            'autonomic_balance': result['autonomic_balance']
        })
    
    df = pd.DataFrame(freq_data)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # LF and HF power over time
    axes[0,0].plot(df['period_id'], df['lf_power'], 'o-', alpha=0.7, color='red', label='LF Power')
    axes[0,0].plot(df['period_id'], df['hf_power'], 's-', alpha=0.7, color='blue', label='HF Power')
    axes[0,0].set_xlabel('Period ID')
    axes[0,0].set_ylabel('Power (ms²)')
    axes[0,0].set_title('LF and HF Power Over Time')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3)
    
    # LF/HF Ratio over time
    axes[0,1].plot(df['period_id'], df['lf_hf_ratio'], '^-', alpha=0.7, color='green')
    axes[0,1].axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Sympathetic Threshold')
    axes[0,1].axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='Parasympathetic Threshold')
    axes[0,1].set_xlabel('Period ID')
    axes[0,1].set_ylabel('LF/HF Ratio')
    axes[0,1].set_title('Sympathovagal Balance (LF/HF Ratio)')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Normalized LF vs HF
    axes[1,0].scatter(df['lf_nu'], df['hf_nu'], alpha=0.7)
    axes[1,0].plot([0, 100], [100, 0], 'r--', alpha=0.5, label='LFnu + HFnu = 100')
    axes[1,0].set_xlabel('Normalized LF (%)')
    axes[1,0].set_ylabel('Normalized HF (%)')
    axes[1,0].set_title('Normalized LF vs HF')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Autonomic balance categories
    balance_counts = df['autonomic_balance'].value_counts()
    if len(balance_counts) > 0:
        axes[1,1].pie(balance_counts.values, labels=balance_counts.index, autopct='%1.1f%%')
        axes[1,1].set_title('Autonomic Balance Categories')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/hrv_frequency_domain_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"HRV frequency domain visualizations saved to {output_dir}/")

def main():
    """Main function to analyze HRV frequency domain metrics"""
    
    print("HRV Frequency Domain Metrics Analysis")
    print("=" * 50)
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent.parent  # From src/heart/hrv/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "hrv_frequency_domain"
    
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
    
    # Process sleep data (primary source for HRV frequency analysis during rest)
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Analyzing sleep record {i+1}/10...")
            result = analyze_hrv_frequency_domain(record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    # Process activity data for additional HRV context
    if activity_data:
        print("Processing activity records...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {i+1}/10...")
            result = analyze_hrv_frequency_domain(record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/hrv_frequency_domain_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/hrv_frequency_domain_results.json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(all_results, output_directory)
    
    # Show data source breakdown
    activity_count = sum(1 for r in all_results if r.get('data_source') == 'activity')
    sleep_count = sum(1 for r in all_results if r.get('data_source') == 'sleep')
    print(f"Data sources: {activity_count} activity records, {sleep_count} sleep records")
    
    print("✅ Analysis complete!")


if __name__ == "__main__":
    main()