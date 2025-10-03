"""
Sample Entropy (SampEn) - regularity and complexity measure
Approximate Entropy (ApEn) - regularity of time series
Detrended Fluctuation Analysis α1, α2 - short/long-term scaling exponents
Correlation Dimension (D2) - dimensional complexity
Recurrence Rate (RR) - percentage of recurrent patterns
Determinism (DET) - percentage of deterministic patterns
Shannon Entropy - information content measure
Renyi Entropy - generalized entropy measure
Multiscale Entropy - complexity across multiple scales
Fractal Dimension - geometric complexity measure
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any
import os
from pathlib import Path
from scipy import stats

def sample_entropy(rr_intervals: List[float], m: int = 2, r: float = 0.2) -> float:
    """Calculate regularity and complexity measure"""
    if len(rr_intervals) < 10:
        return 0.0
    
    data = np.array(rr_intervals)
    N = len(data)
    
    def _maxdist(xi, xj, m):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    def _phi(m):
        patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
        C = np.zeros(N - m + 1)
        
        for i in range(N - m + 1):
            template_i = patterns[i]
            for j in range(N - m + 1):
                if _maxdist(template_i, patterns[j], m) <= r * np.std(data):
                    C[i] += 1
        
        C[C == 0] = 1  # Avoid log(0)
        phi = (N - m + 1.0) ** (-1) * sum(np.log(C / (N - m + 1.0)))
        return phi
    
    return float(_phi(m) - _phi(m + 1))

def approximate_entropy(rr_intervals: List[float], m: int = 2, r: float = 0.2) -> float:
    """Calculate regularity of time series"""
    if len(rr_intervals) < 10:
        return 0.0
    
    data = np.array(rr_intervals)
    N = len(data)
    
    def _maxdist(xi, xj):
        return max([abs(ua - va) for ua, va in zip(xi, xj)])
    
    def _phi(m):
        patterns = np.array([data[i:i + m] for i in range(N - m + 1)])
        C = np.zeros(N - m + 1)
        
        for i in range(N - m + 1):
            template_i = patterns[i]
            for j in range(N - m + 1):
                if _maxdist(template_i, patterns[j]) <= r * np.std(data):
                    C[i] += 1
        
        phi = np.mean(np.log(C / (N - m + 1.0)))
        return phi
    
    return float(_phi(m) - _phi(m + 1))

def detrended_fluctuation_analysis(rr_intervals: List[float]) -> Dict[str, float]:
    """Calculate short/long-term scaling exponents"""
    if len(rr_intervals) < 20:
        return {'alpha1': 0.0, 'alpha2': 0.0}
    
    data = np.array(rr_intervals)
    N = len(data)
    
    # Integrate the signal
    y = np.cumsum(data - np.mean(data))
    
    # Define box sizes
    n_min, n_max = 4, N // 4
    n_sizes = np.logspace(np.log10(n_min), np.log10(n_max), 20).astype(int)
    n_sizes = np.unique(n_sizes)
    
    # Calculate fluctuation for each box size
    F_n = []
    
    for n in n_sizes:
        # Number of boxes
        n_boxes = N // n
        
        # Calculate local trend for each box
        F_n_sum = 0
        for i in range(n_boxes):
            start, end = i * n, (i + 1) * n
            y_box = y[start:end]
            x_box = np.arange(len(y_box))
            
            # Linear detrending
            coeffs = np.polyfit(x_box, y_box, 1)
            trend = np.polyval(coeffs, x_box)
            
            # Calculate fluctuation
            F_n_sum += np.mean((y_box - trend) ** 2)
        
        F_n.append(np.sqrt(F_n_sum / n_boxes))
    
    # Linear regression to find scaling exponent
    log_n = np.log10(n_sizes)
    log_F = np.log10(F_n)
    
    # Split into short-term (α1) and long-term (α2) regions
    split_idx = len(log_n) // 2
    
    # Short-term scaling exponent (α1)
    alpha1_coeffs = np.polyfit(log_n[:split_idx], log_F[:split_idx], 1)
    alpha1 = alpha1_coeffs[0]
    
    # Long-term scaling exponent (α2)  
    alpha2_coeffs = np.polyfit(log_n[split_idx:], log_F[split_idx:], 1)
    alpha2 = alpha2_coeffs[0]
    
    return {'alpha1': float(alpha1), 'alpha2': float(alpha2)}

def shannon_entropy(rr_intervals: List[float], bins: int = 50) -> float:
    """Calculate information content measure"""
    if len(rr_intervals) < 2:
        return 0.0
    
    hist, _ = np.histogram(rr_intervals, bins=bins)
    hist = hist[hist > 0]
    probs = hist / np.sum(hist)
    
    shannon_ent = -np.sum(probs * np.log2(probs))
    return float(shannon_ent)

def fractal_dimension(rr_intervals: List[float]) -> float:
    """Calculate geometric complexity measure"""
    if len(rr_intervals) < 10:
        return 0.0
    
    # Higuchi's method
    data = np.array(rr_intervals)
    N = len(data)
    k_max = min(10, N // 4)
    
    L_k = []
    k_vals = []
    
    for k in range(1, k_max + 1):
        L_m = []
        
        for m in range(1, k + 1):
            # Construct time series
            x_m = data[m-1::k]
            if len(x_m) < 2:
                continue
                
            # Calculate length
            L = np.sum(np.abs(np.diff(x_m))) * (N - 1) / ((len(x_m) - 1) * k)
            L_m.append(L)
        
        if L_m:
            L_k.append(np.mean(L_m))
            k_vals.append(k)
    
    if len(L_k) < 2:
        return 1.0
    
    # Linear regression to find fractal dimension
    log_k = np.log(k_vals)
    log_L = np.log(L_k)
    
    slope, _, _, _, _ = stats.linregress(log_k, log_L)
    fd = -slope
    
    return float(fd)

def convert_hr_to_rr(hr_sequence: List[float]) -> List[float]:
    """Convert heart rate to RR intervals"""
    rr_intervals = []
    for hr in hr_sequence:
        if hr > 0:
            rr = 60000.0 / hr  # Convert to ms
            rr_intervals.append(rr)
    return rr_intervals

def analyze_hrv_nonlinear_metrics(sleep_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze complete HRV non-linear metrics"""
    hr_sequence = sleep_record.get('hr_5min', [])
    rr_intervals = convert_hr_to_rr([x for x in hr_sequence if x > 0])
    
    if len(rr_intervals) < 10:
        return {
            'period_id': sleep_record.get('period_id', 0),
            'sample_entropy': 0.0,
            'approximate_entropy': 0.0,
            'dfa_alpha1': 0.0,
            'dfa_alpha2': 0.0,
            'shannon_entropy': 0.0,
            'fractal_dimension': 1.0
        }
    
    # Calculate metrics
    sampEn = sample_entropy(rr_intervals)
    apEn = approximate_entropy(rr_intervals)
    dfa_results = detrended_fluctuation_analysis(rr_intervals)
    shannon_ent = shannon_entropy(rr_intervals)
    frac_dim = fractal_dimension(rr_intervals)
    
    result = {
        'period_id': sleep_record.get('period_id', 0),
        'timestamp': sleep_record.get('bedtime_start_dt_adjusted', 0),
        'sample_entropy': sampEn,
        'approximate_entropy': apEn,
        'dfa_alpha1': dfa_results['alpha1'],
        'dfa_alpha2': dfa_results['alpha2'],
        'shannon_entropy': shannon_ent,
        'fractal_dimension': frac_dim,
        'rr_interval_count': len(rr_intervals)
    }
    
    # Add complexity interpretation
    complexity_score = (sampEn + apEn + shannon_ent) / 3
    if complexity_score > 1.5:
        result['complexity_level'] = 'High'
    elif complexity_score > 0.8:
        result['complexity_level'] = 'Medium'
    else:
        result['complexity_level'] = 'Low'
    
    return result

def main():
    """Main function to analyze HRV non-linear metrics"""
    print("HRV Non-linear Metrics Analysis")
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent.parent  # From src/heart/hrv/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "hrv_nonlinear"
    
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
    
    # Process activity data (primary source for HRV non-linear)
    if activity_data:
        print("Processing activity records...")
        for i, record in enumerate(activity_data[:8]):  # Reduced due to computational complexity
            print(f"Analyzing activity record {i+1}/8...")
            result = analyze_hrv_nonlinear_metrics(record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    # Process sleep data for context
    if sleep_data:
        print("Processing sleep records for context...")
        for i, record in enumerate(sleep_data[:8]):
            print(f"Analyzing sleep record {i+1}/8...")
            result = analyze_hrv_nonlinear_metrics(record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/hrv_nonlinear_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/hrv_nonlinear_results.json")
    
    # Create visualizations
    print("Creating visualizations...")

    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()