"""
S-Entropy Coordinate Navigation for Physiological Sensor Analysis
================================================================

This module implements the core S-entropy coordinate system that transforms
measurement imprecision into contextual interpretation through 4D navigation.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple, Any
import os
from pathlib import Path
from datetime import datetime

class SEntropyCoordinates:
    """S-entropy 4D coordinate system for physiological state navigation"""
    
    def __init__(self, knowledge: float = 0.0, time: float = 0.0, 
                 entropy: float = 0.0, context: float = 0.0):
        self.knowledge = np.clip(knowledge, 0.0, 1.0)
        self.time = np.clip(time, 0.0, 1.0)
        self.entropy = np.clip(entropy, 0.0, 1.0)
        self.context = np.clip(context, 0.0, 1.0)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "knowledge": self.knowledge,
            "time": self.time,
            "entropy": self.entropy,
            "context": self.context
        }
    
    def distance_to(self, other: 'SEntropyCoordinates', 
                   weights: Tuple[float, float, float, float] = (0.25, 0.30, 0.25, 0.20)) -> float:
        """Compute S-distance between two coordinate points"""
        dk = (self.knowledge - other.knowledge) ** 2
        dt = (self.time - other.time) ** 2
        de = (self.entropy - other.entropy) ** 2
        dc = (self.context - other.context) ** 2
        
        return np.sqrt(weights[0]*dk + weights[1]*dt + weights[2]*de + weights[3]*dc)

class BiologicalFrequencyDecomposer:
    """Multi-scale oscillatory decomposition across biological frequency scales"""
    
    def __init__(self):
        # Biological frequency scales (Hz)
        self.frequency_bands = {
            'cellular': (0.1, 100.0),      
            'cardiac': (0.01, 10.0),       
            'respiratory': (0.001, 1.0),   
            'autonomic': (0.0001, 0.1),    
            'circadian': (0.00001, 0.01)   
        }
    
    def decompose_signal(self, signal_data: List[float], sampling_rate: float = 1.0/300.0) -> Dict[str, Dict[str, float]]:
        """Decompose physiological signal across biological frequency scales"""
        if not signal_data or len(signal_data) < 10:
            return {}
        
        # Clean signal - remove zeros and interpolate
        clean_signal = self._clean_signal(signal_data)
        
        if len(clean_signal) < 10:
            return {}
        
        # Compute FFT
        freqs = fftfreq(len(clean_signal), 1/sampling_rate)
        fft_signal = fft(clean_signal)
        power_spectrum = np.abs(fft_signal)**2
        
        decomposition = {}
        
        for band_name, (f_min, f_max) in self.frequency_bands.items():
            # Find frequency indices in band
            band_mask = (np.abs(freqs) >= f_min) & (np.abs(freqs) <= f_max)
            
            if not np.any(band_mask):
                continue
            
            # Extract band power and dominant frequency
            band_power = np.sum(power_spectrum[band_mask])
            band_freqs = freqs[band_mask]
            band_spectrum = power_spectrum[band_mask]
            
            if len(band_spectrum) > 0:
                dominant_freq_idx = np.argmax(band_spectrum)
                dominant_freq = np.abs(band_freqs[dominant_freq_idx])
                
                # Compute amplitude and phase
                dominant_complex = fft_signal[band_mask][dominant_freq_idx]
                amplitude = np.abs(dominant_complex)
                phase = np.angle(dominant_complex)
                
                decomposition[band_name] = {
                    'amplitude': float(amplitude),
                    'frequency': float(dominant_freq),
                    'phase': float(phase),
                    'power': float(band_power),
                    'normalized_power': float(band_power / np.sum(power_spectrum))
                }
        
        return decomposition
    
    def _clean_signal(self, signal_data: List[float]) -> np.ndarray:
        """Clean signal by removing zeros and interpolating missing values"""
        signal_array = np.array(signal_data)
        signal_array[signal_array == 0] = np.nan
        valid_indices = ~np.isnan(signal_array)
        
        if not np.any(valid_indices):
            return np.array([])
        
        # Interpolate missing values
        if not np.all(valid_indices):
            x = np.arange(len(signal_array))
            signal_array = np.interp(x, x[valid_indices], signal_array[valid_indices])
        
        return signal_array

class SEntropyProcessor:
    """Main S-entropy coordinate navigation processor"""
    
    def __init__(self):
        self.decomposer = BiologicalFrequencyDecomposer()
        self.results = {}
    
    def process_sleep_record(self, sleep_record: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single sleep record through S-entropy framework"""
        
        # Extract key sequences
        hr_sequence = sleep_record.get('hr_5min', [])
        hypnogram = sleep_record.get('hypnogram_5min', '')
        rmssd_sequence = sleep_record.get('rmssd_5min', [])
        
        # 1. Oscillatory decomposition
        hr_decomposition = self.decomposer.decompose_signal(hr_sequence)
        rmssd_decomposition = self.decomposer.decompose_signal(rmssd_sequence)
        
        # 2. Compute S-entropy coordinates
        s_coords = self._compute_s_entropy_coordinates(sleep_record, hr_decomposition)
        
        # 3. Sequence complexity analysis
        sequence_metrics = self._analyze_sequences(hr_sequence, hypnogram, rmssd_sequence)
        
        # 4. Contextual interpretation
        interpretation = self._generate_contextual_interpretation(
            sleep_record, s_coords, hr_decomposition, sequence_metrics
        )
        
        return {
            'period_id': sleep_record.get('period_id', 0),
            'timestamp': sleep_record.get('bedtime_start_dt_adjusted', 0),
            's_entropy_coordinates': s_coords.to_dict(),
            'oscillatory_decomposition': {
                'heart_rate': hr_decomposition,
                'hrv': rmssd_decomposition
            },
            'sequence_metrics': sequence_metrics,
            'interpretation': interpretation,
            'sleep_metrics': {
                'efficiency': sleep_record.get('efficiency', 0),
                'total_sleep_hrs': sleep_record.get('total_in_hrs', 0),
                'deep_sleep_hrs': sleep_record.get('deep_in_hrs', 0),
                'rem_sleep_hrs': sleep_record.get('rem_in_hrs', 0),
                'awake_hrs': sleep_record.get('awake_in_hrs', 0)
            }
        }
    
    def _compute_s_entropy_coordinates(self, sleep_record: Dict, decomposition: Dict) -> SEntropyCoordinates:
        """Compute 4D S-entropy coordinates from sleep data"""
        
        # Knowledge: Information deficit (inverse of confidence/completeness)
        efficiency = sleep_record.get('efficiency', 50) / 100.0
        score = sleep_record.get('score', 50) / 100.0
        knowledge = 1.0 - (efficiency + score) / 2.0
        
        # Time: Temporal processing complexity
        duration = sleep_record.get('duration_in_hrs', 8)
        restless = sleep_record.get('restless', 30) / 100.0
        time = min(duration / 12.0, 1.0) * (1.0 + restless)
        
        # Entropy: Thermodynamic accessibility 
        total_power = sum(band.get('normalized_power', 0) for band in decomposition.values())
        entropy = min(total_power * 2.0, 1.0)
        
        # Context: Environmental and physiological context richness
        context_factors = 0
        if sleep_record.get('temperature_deviation'):
            context_factors += abs(sleep_record['temperature_deviation'])
        if sleep_record.get('breath_average'):
            context_factors += abs(sleep_record['breath_average'] - 16) / 10.0
        context = min(context_factors, 1.0)
        
        return SEntropyCoordinates(knowledge, time, entropy, context)
    
    def _analyze_sequences(self, hr_seq: List[float], hypnogram: str, rmssd_seq: List[float]) -> Dict[str, Any]:
        """Analyze sequence patterns and complexity"""
        metrics = {}
        
        # Heart rate sequence analysis
        if hr_seq:
            hr_clean = [x for x in hr_seq if x > 0]
            if hr_clean:
                metrics['hr_entropy'] = self._shannon_entropy(hr_clean)
                metrics['hr_variability'] = np.std(hr_clean) if len(hr_clean) > 1 else 0
        
        # Hypnogram analysis
        if hypnogram:
            metrics['hypnogram_entropy'] = self._sequence_entropy(hypnogram)
            metrics['sleep_transitions'] = self._count_transitions(hypnogram)
            metrics['sleep_stage_distribution'] = self._stage_distribution(hypnogram)
        
        return metrics
    
    def _generate_contextual_interpretation(self, sleep_record: Dict, s_coords: SEntropyCoordinates, 
                                         decomposition: Dict, sequence_metrics: Dict) -> str:
        """Generate contextual explanation using S-entropy navigation"""
        
        interpretations = []
        
        # S-entropy coordinate interpretation
        if s_coords.knowledge > 0.7:
            interpretations.append("high uncertainty suggesting complex physiological state")
        elif s_coords.knowledge < 0.3:
            interpretations.append("high confidence physiological interpretation")
        
        # Sleep pattern analysis
        efficiency = sleep_record.get('efficiency', 0)
        if efficiency < 50:
            interpretations.append("fragmented sleep suggesting stress")
        elif efficiency > 80:
            interpretations.append("consolidated sleep indicating restoration")
        
        return "S-entropy analysis: " + ", ".join(interpretations) + "."
    
    def _shannon_entropy(self, data: List[float], bins: int = 10) -> float:
        """Compute Shannon entropy of data"""
        if len(data) < 2:
            return 0.0
        
        hist, _ = np.histogram(data, bins=bins)
        hist = hist[hist > 0]
        probs = hist / np.sum(hist)
        return -np.sum(probs * np.log2(probs))
    
    def _sequence_entropy(self, sequence: str) -> float:
        """Compute entropy of character sequence"""
        if not sequence:
            return 0.0
        
        char_counts = {}
        for char in sequence:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        total = len(sequence)
        entropy = 0
        for count in char_counts.values():
            prob = count / total
            entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _count_transitions(self, sequence: str) -> int:
        """Count state transitions in sequence"""
        if len(sequence) < 2:
            return 0
        
        transitions = 0
        for i in range(1, len(sequence)):
            if sequence[i] != sequence[i-1]:
                transitions += 1
        
        return transitions
    
    def _stage_distribution(self, hypnogram: str) -> Dict[str, float]:
        """Compute sleep stage distribution"""
        if not hypnogram:
            return {}
        
        stage_counts = {}
        for stage in hypnogram:
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        total = len(hypnogram)
        return {stage: count/total for stage, count in stage_counts.items()}

def main():
    """Main function to run S-entropy analysis on sleep data"""
    
    print("S-Entropy Coordinate Navigation Analysis")
    print("=" * 50)
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent  # From src/linguistic/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "s_entropy"
    
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
    
    # Initialize S-entropy processor
    processor = SEntropyProcessor()
    
    # Combine and process data
    all_results = []
    
    # Process sleep data (primary source for S-entropy analysis)
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Processing sleep record {i+1}/10...")
            result = processor.process_sleep_record(record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    # Process activity data for additional context
    if activity_data:
        print("Processing activity records...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Processing activity record {i+1}/10...")
            # Convert activity record to sleep-like format for processing
            mock_sleep_record = {
                'period_id': record.get('period_id', i + len(sleep_data)),
                'bedtime_start_dt_adjusted': record.get('timestamp', 0),
                'hr_5min': record.get('hr_5min', [70, 75, 72, 68, 71]),  # Use activity HR or mock
                'hypnogram_5min': 'AAARRRLLLAAARRRLLL',  # Mock activity states
                'rmssd_5min': [30, 35, 32, 38, 33],  # Mock HRV data
                'efficiency': 85,  # Mock efficiency for activity
                'total_in_hrs': 8,
                'deep_in_hrs': 2,
                'rem_in_hrs': 1.5,
                'awake_in_hrs': 0.5,
                'score': 80,
                'duration_in_hrs': 8,
                'restless': 15,
                'temperature_deviation': 0.2,
                'breath_average': 16
            }
            result = processor.process_sleep_record(mock_sleep_record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/s_entropy_analysis_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/s_entropy_analysis_results.json")
    
    # Create basic visualization
    coords_data = []
    for result in all_results:
        coords = result['s_entropy_coordinates']
        coords['period_id'] = result['period_id']
        coords['efficiency'] = result['sleep_metrics']['efficiency']
        coords['data_source'] = result['data_source']
        coords_data.append(coords)
    
    coords_df = pd.DataFrame(coords_data)
    
    # Plot S-entropy coordinates
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    axes[0,0].scatter(coords_df['knowledge'], coords_df['time'], 
                     c=coords_df['efficiency'], cmap='viridis', alpha=0.7)
    axes[0,0].set_xlabel('Knowledge')
    axes[0,0].set_ylabel('Time')
    axes[0,0].set_title('Knowledge vs Time')
    
    axes[0,1].scatter(coords_df['entropy'], coords_df['context'], 
                     c=coords_df['efficiency'], cmap='viridis', alpha=0.7)
    axes[0,1].set_xlabel('Entropy')
    axes[0,1].set_ylabel('Context')
    axes[0,1].set_title('Entropy vs Context')
    
    axes[1,0].plot(coords_df['period_id'], coords_df['knowledge'], 'o-', label='Knowledge', alpha=0.7)
    axes[1,0].plot(coords_df['period_id'], coords_df['time'], 's-', label='Time', alpha=0.7)
    axes[1,0].plot(coords_df['period_id'], coords_df['entropy'], '^-', label='Entropy', alpha=0.7)
    axes[1,0].plot(coords_df['period_id'], coords_df['context'], 'd-', label='Context', alpha=0.7)
    axes[1,0].set_xlabel('Period ID')
    axes[1,0].set_ylabel('S-Entropy Coordinate')
    axes[1,0].set_title('S-Entropy Evolution')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Correlation heatmap
    corr_matrix = coords_df[['knowledge', 'time', 'entropy', 'context']].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                ax=axes[1,1], cbar_kws={'label': 'Correlation'})
    axes[1,1].set_title('S-Entropy Coordinate Correlations')
    
    plt.tight_layout()
    plt.savefig(f'{output_directory}/s_entropy_coordinates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Visualization saved to {output_directory}/s_entropy_coordinates.png")
    
    # Show data source breakdown
    activity_count = sum(1 for r in all_results if r.get('data_source') == 'activity')
    sleep_count = sum(1 for r in all_results if r.get('data_source') == 'sleep')
    print(f"Data sources: {activity_count} activity records, {sleep_count} sleep records")
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()