"""
St. Stella's Moon Landing Algorithm - Meta-information guided Bayesian inference system
Implements constrained stochastic sampling through tri-dimensional fuzzy window systems
Based on docs/st-stellas/st-stellas-moon-landing.tex
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import os
from pathlib import Path

class MoonLandingAlgorithm:
    """St. Stella's Moon Landing Algorithm - Main implementation"""
    
    def __init__(self, coordinate_dim: int = 3):
        self.coordinate_dim = coordinate_dim
        self.sigma = 0.5  # Sampling covariance
        self.v0 = 1.0  # Base processing velocity
        
    def semantic_gravity_magnitude(self, r: np.ndarray) -> float:
        """Calculate semantic gravity field magnitude |g_s(r)|"""
        # Simplified gravity based on position
        return np.sum(r**2) * 0.1 + 0.1
    
    def fuzzy_window_weight(self, r: np.ndarray) -> float:
        """Calculate combined fuzzy window weight w(r)"""
        if len(r) < 3:
            r = np.pad(r, (0, 3-len(r)), mode='constant')
        # Gaussian windows
        w_t = np.exp(-r[0]**2 / 2)  # Temporal
        w_i = np.exp(-r[1]**2 / 2)  # Informational  
        w_e = np.exp(-r[2]**2 / 2)  # Entropic
        return w_t * w_i * w_e
    
    def constrained_stochastic_sampling(self, n_samples: int = 30) -> List[Dict[str, Any]]:
        """Perform constrained stochastic sampling"""
        samples = []
        r_current = np.random.uniform(-1, 1, self.coordinate_dim)
        
        for n in range(n_samples):
            # Calculate maximum step size
            g_mag = self.semantic_gravity_magnitude(r_current)
            delta_r_max = self.v0 / max(g_mag, 1e-6)
            
            # Sample constrained step
            step = np.random.normal(0, self.sigma, self.coordinate_dim)
            step_size = np.linalg.norm(step)
            if step_size > delta_r_max:
                step = step * (delta_r_max / step_size)
            
            r_next = np.clip(r_current + step, -3, 3)
            
            # Calculate weights and sample information
            weight = self.fuzzy_window_weight(r_next)
            info = np.sin(np.sum(r_next)) + 0.1 * np.random.normal()
            
            samples.append({
                'position': r_next.tolist(),
                'information': float(info),
                'weight': float(weight),
                'step_size': float(step_size)
            })
            
            r_current = r_next
        
        return samples
    
    def bayesian_inference(self, samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply Bayesian inference to samples"""
        if not samples:
            return {}
        
        info_values = [s['information'] for s in samples]
        weights = [s['weight'] for s in samples]
        
        total_weight = sum(weights) if sum(weights) > 0 else 1.0
        weighted_mean = sum(i*w for i,w in zip(info_values, weights)) / total_weight
        
        return {
            'weighted_mean': float(weighted_mean),
            'n_samples': len(samples),
            'total_weight': float(total_weight)
        }
    
    def run_algorithm(self, data: np.ndarray) -> Dict[str, Any]:
        """Run complete Moon Landing Algorithm"""
        # Meta-information extraction (simplified)
        compressed = np.array([np.mean(data), np.std(data), len(data)])
        
        # Stochastic sampling
        samples = self.constrained_stochastic_sampling()
        
        # Bayesian inference
        inference = self.bayesian_inference(samples)
        
        return {
            'compressed_space': compressed.tolist(),
            'compression_ratio': float(len(data) / len(compressed)),
            'samples': samples,
            'inference_results': inference,
            'semantic_navigation_complete': True
        }

def analyze_with_moon_landing(data_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze data using Moon Landing Algorithm"""
    hr_sequence = data_record.get('hr_5min', [60, 62, 58, 65, 63])
    data_array = np.array([x for x in hr_sequence if x > 0], dtype=float)  # Ensure float type
    
    if len(data_array) == 0:
        data_array = np.array([60.0, 62.0, 58.0, 65.0, 63.0])  # Default values
    
    try:
        algorithm = MoonLandingAlgorithm()
        results = algorithm.run_algorithm(data_array)
        
        results['period_id'] = data_record.get('period_id', 0)
        results['semantic_convergence'] = 'Positive' if results['inference_results'].get('weighted_mean', 0) > 0 else 'Negative'
        
        return results
    except Exception as e:
        print(f"Warning: Moon Landing analysis failed: {e}")
        return {
            'period_id': data_record.get('period_id', 0),
            'compressed_space': [60.0, 5.0, 5],
            'compression_ratio': 1.0,
            'samples': [],
            'inference_results': {'weighted_mean': 0.0, 'n_samples': 0, 'total_weight': 0.0},
            'semantic_navigation_complete': False,
            'semantic_convergence': 'Error'
        }

def main():
    """Main function to demonstrate Moon Landing Algorithm"""
    print("St. Stella's Moon Landing Algorithm - Stochastic Sampling")
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent  # From src/linguistic/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "moon_landing"
    
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
    
    # Process sleep data (primary source for Moon Landing analysis)
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:5]):
            print(f"Processing sleep record {i+1}/5...")
            result = analyze_with_moon_landing(record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    # Process activity data for additional context
    if activity_data:
        print("Processing activity records...")
        for i, record in enumerate(activity_data[:5]):
            print(f"Processing activity record {i+1}/5...")
            result = analyze_with_moon_landing(record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/moon_landing_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/moon_landing_results.json")
    
    # Show data source breakdown
    activity_count = sum(1 for r in all_results if r.get('data_source') == 'activity')
    sleep_count = sum(1 for r in all_results if r.get('data_source') == 'sleep')
    print(f"Data sources: {activity_count} activity records, {sleep_count} sleep records")
    
    print("✅ Semantic navigation complete!")

if __name__ == "__main__":
    main()
