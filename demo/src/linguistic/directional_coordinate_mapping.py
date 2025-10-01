"""
Directional Coordinate Mapping for S-Entropy Framework
=====================================================

Transform physiological states into directional sequences using A-R-D-L mapping:
- A = Activation/Elevated states (above aerobic threshold)  
- R = Steady/Maintenance states (aerobic zone)
- D = Decreased/Recovery states (below resting levels)
- L = Stress/Transition states (anaerobic threshold)
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import os
from datetime import datetime

class DirectionalMapper:
    """Maps physiological measurements to A-R-D-L directional coordinates"""
    
    def __init__(self, 
                 activation_threshold: float = 0.7,
                 steady_threshold: float = 0.3,
                 recovery_threshold: float = -0.3,
                 stress_threshold: float = 0.9):
        """
        Initialize thresholds for directional mapping
        
        Args:
            activation_threshold: Threshold for A (Activation) states
            steady_threshold: Threshold for R (Steady) states  
            recovery_threshold: Threshold for D (Decreased/Recovery) states
            stress_threshold: Threshold for L (Stress/Transition) states
        """
        self.activation_threshold = activation_threshold
        self.steady_threshold = steady_threshold
        self.recovery_threshold = recovery_threshold
        self.stress_threshold = stress_threshold
        
    def map_heart_rate_sequence(self, hr_sequence: List[float], 
                               context: Optional[Dict[str, Any]] = None) -> str:
        """Map heart rate sequence to A-R-D-L directional sequence"""
        
        if not hr_sequence:
            return ""
        
        # Clean and normalize heart rate data
        hr_clean = [x for x in hr_sequence if x > 0]
        if not hr_clean:
            return ""
        
        # Normalize to z-scores for relative mapping
        hr_mean = np.mean(hr_clean)
        hr_std = np.std(hr_clean) if len(hr_clean) > 1 else 1.0
        
        sequence = []
        for hr in hr_sequence:
            if hr <= 0:
                continue
                
            # Normalize heart rate
            z_score = (hr - hr_mean) / hr_std if hr_std > 0 else 0
            
            # Map to directional coordinates
            if z_score >= self.stress_threshold:
                sequence.append('L')  # Stress/Transition
            elif z_score >= self.activation_threshold:
                sequence.append('A')  # Activation/Elevated
            elif z_score >= self.recovery_threshold:
                sequence.append('R')  # Steady/Maintenance
            else:
                sequence.append('D')  # Decreased/Recovery
        
        return ''.join(sequence)
    
    def map_hrv_sequence(self, rmssd_sequence: List[float]) -> str:
        """Map HRV (RMSSD) sequence to directional coordinates"""
        
        if not rmssd_sequence:
            return ""
        
        # Clean HRV data
        rmssd_clean = [x for x in rmssd_sequence if x > 0]
        if not rmssd_clean:
            return ""
        
        # Normalize HRV data
        rmssd_mean = np.mean(rmssd_clean)
        rmssd_std = np.std(rmssd_clean) if len(rmssd_clean) > 1 else 1.0
        
        sequence = []
        for rmssd in rmssd_sequence:
            if rmssd <= 0:
                continue
                
            z_score = (rmssd - rmssd_mean) / rmssd_std if rmssd_std > 0 else 0
            
            # For HRV, higher values typically indicate better autonomic balance
            if z_score >= self.activation_threshold:
                sequence.append('A')  # High HRV - good autonomic balance
            elif z_score >= self.recovery_threshold:
                sequence.append('R')  # Normal HRV - steady state
            elif z_score >= self.stress_threshold * -1:
                sequence.append('D')  # Low HRV - may indicate fatigue
            else:
                sequence.append('L')  # Very low HRV - potential stress
        
        return ''.join(sequence)
    
    def map_sleep_stages(self, hypnogram: str) -> str:
        """Map sleep stage hypnogram to contextual directional sequence"""
        
        if not hypnogram:
            return ""
        
        # Direct mapping of sleep stages to directional coordinates
        stage_mapping = {
            'A': 'L',  # Awake -> Stress/Transition (should be asleep)
            'L': 'R',  # Light sleep -> Steady/Maintenance  
            'D': 'A',  # Deep sleep -> Activation (restorative processes)
            'R': 'D'   # REM sleep -> Decreased (memory consolidation)
        }
        
        return ''.join(stage_mapping.get(stage, 'R') for stage in hypnogram)
    
    def analyze_directional_patterns(self, sequence: str) -> Dict[str, Any]:
        """Analyze patterns in directional sequence"""
        
        if not sequence:
            return {}
        
        analysis = {}
        
        # Character distribution
        char_counts = {'A': 0, 'R': 0, 'D': 0, 'L': 0}
        for char in sequence:
            if char in char_counts:
                char_counts[char] += 1
        
        total = len(sequence)
        analysis['distribution'] = {char: count/total for char, count in char_counts.items()}
        
        # Transition analysis
        analysis['transitions'] = self._analyze_transitions(sequence)
        
        # Pattern complexity
        analysis['entropy'] = self._sequence_entropy(sequence)
        
        # Dominant patterns
        analysis['dominant_patterns'] = self._find_dominant_patterns(sequence)
        
        # Physiological interpretation
        analysis['interpretation'] = self._interpret_pattern(char_counts, total)
        
        return analysis
    
    def _analyze_transitions(self, sequence: str) -> Dict[str, Any]:
        """Analyze state transitions in sequence"""
        
        if len(sequence) < 2:
            return {}
        
        transition_counts = {}
        total_transitions = 0
        
        for i in range(len(sequence) - 1):
            transition = f"{sequence[i]}->{sequence[i+1]}"
            transition_counts[transition] = transition_counts.get(transition, 0) + 1
            total_transitions += 1
        
        # Transition probabilities
        transition_probs = {trans: count/total_transitions 
                          for trans, count in transition_counts.items()}
        
        # Identify most common transitions
        most_common = sorted(transition_probs.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'transition_counts': transition_counts,
            'transition_probabilities': transition_probs,
            'most_common_transitions': most_common,
            'total_transitions': total_transitions
        }
    
    def _sequence_entropy(self, sequence: str) -> float:
        """Compute entropy of directional sequence"""
        
        if not sequence:
            return 0.0
        
        char_counts = {}
        for char in sequence:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        total = len(sequence)
        entropy = 0
        for count in char_counts.values():
            prob = count / total
            if prob > 0:
                entropy -= prob * np.log2(prob)
        
        return entropy
    
    def _find_dominant_patterns(self, sequence: str, min_length: int = 2, max_length: int = 5) -> List[Tuple[str, int]]:
        """Find dominant repeating patterns in sequence"""
        
        pattern_counts = {}
        
        for length in range(min_length, min(max_length + 1, len(sequence) + 1)):
            for i in range(len(sequence) - length + 1):
                pattern = sequence[i:i + length]
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1
        
        # Filter out patterns that appear only once and sort by frequency
        dominant_patterns = [(pattern, count) for pattern, count in pattern_counts.items() 
                           if count > 1]
        dominant_patterns.sort(key=lambda x: x[1], reverse=True)
        
        return dominant_patterns[:10]  # Return top 10 patterns
    
    def _interpret_pattern(self, char_counts: Dict[str, int], total: int) -> str:
        """Generate physiological interpretation of directional pattern"""
        
        interpretations = []
        
        # Analyze dominant states
        percentages = {char: (count/total)*100 for char, count in char_counts.items()}
        
        if percentages['A'] > 40:
            interpretations.append(f"high activation state ({percentages['A']:.1f}%)")
        if percentages['L'] > 30:
            interpretations.append(f"significant stress/transition periods ({percentages['L']:.1f}%)")
        if percentages['R'] > 50:
            interpretations.append(f"predominantly steady maintenance state ({percentages['R']:.1f}%)")
        if percentages['D'] > 35:
            interpretations.append(f"extensive recovery periods ({percentages['D']:.1f}%)")
        
        # Overall pattern assessment
        entropy = self._sequence_entropy(''.join(char_counts.keys()))
        if entropy > 1.5:
            interpretations.append("high variability indicating complex physiological dynamics")
        elif entropy < 0.5:
            interpretations.append("low variability suggesting stable physiological state")
        
        return " | ".join(interpretations) if interpretations else "balanced physiological pattern"

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualizations of directional mapping analysis"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract directional sequence data
    sequence_data = []
    for result in results:
        if 'directional_analysis' in result:
            analysis = result['directional_analysis']
            period_data = {
                'period_id': result['period_id'],
                'hr_sequence': result.get('hr_directional_sequence', ''),
                'hrv_sequence': result.get('hrv_directional_sequence', ''),
                'sleep_sequence': result.get('sleep_directional_sequence', ''),
                'hr_entropy': analysis.get('hr_analysis', {}).get('entropy', 0),
                'hrv_entropy': analysis.get('hrv_analysis', {}).get('entropy', 0),
                'sleep_entropy': analysis.get('sleep_analysis', {}).get('entropy', 0)
            }
            
            # Add distribution data
            for seq_type in ['hr_analysis', 'hrv_analysis', 'sleep_analysis']:
                if seq_type in analysis:
                    dist = analysis[seq_type].get('distribution', {})
                    for char in ['A', 'R', 'D', 'L']:
                        period_data[f"{seq_type}_{char}_pct"] = dist.get(char, 0) * 100
            
            sequence_data.append(period_data)
    
    if not sequence_data:
        print("No directional sequence data found for visualization")
        return
    
    df = pd.DataFrame(sequence_data)
    
    # 1. Distribution of directional states across all sequences
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # HR distribution
    hr_dist_cols = [col for col in df.columns if col.startswith('hr_analysis_') and col.endswith('_pct')]
    if hr_dist_cols:
        hr_dist_data = df[hr_dist_cols].mean()
        hr_dist_data.index = [col.split('_')[-2] for col in hr_dist_data.index]
        axes[0,0].bar(hr_dist_data.index, hr_dist_data.values, alpha=0.7)
        axes[0,0].set_title('Heart Rate Directional Distribution')
        axes[0,0].set_ylabel('Percentage (%)')
    
    # HRV distribution  
    hrv_dist_cols = [col for col in df.columns if col.startswith('hrv_analysis_') and col.endswith('_pct')]
    if hrv_dist_cols:
        hrv_dist_data = df[hrv_dist_cols].mean()
        hrv_dist_data.index = [col.split('_')[-2] for col in hrv_dist_data.index]
        axes[0,1].bar(hrv_dist_data.index, hrv_dist_data.values, alpha=0.7, color='orange')
        axes[0,1].set_title('HRV Directional Distribution')
        axes[0,1].set_ylabel('Percentage (%)')
    
    # Sleep distribution
    sleep_dist_cols = [col for col in df.columns if col.startswith('sleep_analysis_') and col.endswith('_pct')]
    if sleep_dist_cols:
        sleep_dist_data = df[sleep_dist_cols].mean()
        sleep_dist_data.index = [col.split('_')[-2] for col in sleep_dist_data.index]
        axes[1,0].bar(sleep_dist_data.index, sleep_dist_data.values, alpha=0.7, color='green')
        axes[1,0].set_title('Sleep Stage Directional Distribution')
        axes[1,0].set_ylabel('Percentage (%)')
    
    # Entropy comparison
    entropy_cols = ['hr_entropy', 'hrv_entropy', 'sleep_entropy']
    entropy_data = df[entropy_cols].mean()
    axes[1,1].bar(entropy_data.index, entropy_data.values, alpha=0.7, color='purple')
    axes[1,1].set_title('Sequence Entropy Comparison')
    axes[1,1].set_ylabel('Shannon Entropy')
    axes[1,1].set_xticklabels(['HR', 'HRV', 'Sleep'])
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/directional_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Sequence pattern evolution over time
    plt.figure(figsize=(15, 10))
    
    if 'period_id' in df.columns:
        # Plot entropy evolution
        plt.subplot(3, 1, 1)
        plt.plot(df['period_id'], df['hr_entropy'], 'o-', label='HR Entropy', alpha=0.7)
        plt.plot(df['period_id'], df['hrv_entropy'], 's-', label='HRV Entropy', alpha=0.7) 
        plt.plot(df['period_id'], df['sleep_entropy'], '^-', label='Sleep Entropy', alpha=0.7)
        plt.ylabel('Entropy')
        plt.title('Directional Sequence Entropy Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot A state percentage evolution
        plt.subplot(3, 1, 2)
        if 'hr_analysis_A_pct' in df.columns:
            plt.plot(df['period_id'], df['hr_analysis_A_pct'], 'o-', label='HR Activation', alpha=0.7)
        if 'hrv_analysis_A_pct' in df.columns:
            plt.plot(df['period_id'], df['hrv_analysis_A_pct'], 's-', label='HRV Activation', alpha=0.7)
        if 'sleep_analysis_A_pct' in df.columns:
            plt.plot(df['period_id'], df['sleep_analysis_A_pct'], '^-', label='Sleep Activation', alpha=0.7)
        plt.ylabel('Activation State (%)')
        plt.title('Activation State Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot L state percentage evolution
        plt.subplot(3, 1, 3)
        if 'hr_analysis_L_pct' in df.columns:
            plt.plot(df['period_id'], df['hr_analysis_L_pct'], 'o-', label='HR Stress', alpha=0.7)
        if 'hrv_analysis_L_pct' in df.columns:
            plt.plot(df['period_id'], df['hrv_analysis_L_pct'], 's-', label='HRV Stress', alpha=0.7)
        if 'sleep_analysis_L_pct' in df.columns:
            plt.plot(df['period_id'], df['sleep_analysis_L_pct'], '^-', label='Sleep Stress', alpha=0.7)
        plt.ylabel('Stress State (%)')
        plt.xlabel('Period ID')
        plt.title('Stress State Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/directional_evolution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Directional mapping visualizations saved to {output_dir}/")

def main():
    """Main function to demonstrate directional coordinate mapping"""
    
    print("Directional Coordinate Mapping Analysis")
    print("=" * 50)
    
    # Load sleep data
    with open('../public/sleep_ppg_records.json', 'r') as f:
        sleep_data = json.load(f)
    
    print(f"Loaded {len(sleep_data)} sleep records")
    
    # Initialize directional mapper
    mapper = DirectionalMapper()
    
    # Process first 10 records
    results = []
    for i, record in enumerate(sleep_data[:10]):
        print(f"Processing record {i+1}/10...")
        
        # Extract sequences
        hr_sequence = record.get('hr_5min', [])
        rmssd_sequence = record.get('rmssd_5min', [])
        hypnogram = record.get('hypnogram_5min', '')
        
        # Map to directional coordinates
        hr_directional = mapper.map_heart_rate_sequence(hr_sequence)
        hrv_directional = mapper.map_hrv_sequence(rmssd_sequence)
        sleep_directional = mapper.map_sleep_stages(hypnogram)
        
        # Analyze patterns
        hr_analysis = mapper.analyze_directional_patterns(hr_directional)
        hrv_analysis = mapper.analyze_directional_patterns(hrv_directional)
        sleep_analysis = mapper.analyze_directional_patterns(sleep_directional)
        
        result = {
            'period_id': record.get('period_id', i),
            'timestamp': record.get('bedtime_start_dt_adjusted', 0),
            'hr_directional_sequence': hr_directional[:100],  # First 100 chars for readability
            'hrv_directional_sequence': hrv_directional[:100],
            'sleep_directional_sequence': sleep_directional[:100],
            'original_hypnogram': hypnogram[:100],
            'directional_analysis': {
                'hr_analysis': hr_analysis,
                'hrv_analysis': hrv_analysis,
                'sleep_analysis': sleep_analysis
            },
            'sequence_lengths': {
                'hr': len(hr_directional),
                'hrv': len(hrv_directional),
                'sleep': len(sleep_directional)
            }
        }
        
        results.append(result)
    
    # Save results
    output_dir = '../results/directional_mapping'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/directional_mapping_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to {output_dir}/directional_mapping_results.json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(results, output_dir)
    
    # Print sample directional sequences
    print("\nSample Directional Sequences:")
    print("-" * 40)
    for i, result in enumerate(results[:3]):
        print(f"\nRecord {i+1} (Period {result['period_id']}):")
        print(f"  Original Hypnogram: {result['original_hypnogram'][:50]}...")
        print(f"  Sleep Directional:  {result['sleep_directional_sequence'][:50]}...")
        print(f"  HR Directional:     {result['hr_directional_sequence'][:50]}...")
        print(f"  HRV Directional:    {result['hrv_directional_sequence'][:50]}...")
        
        # Print interpretation
        hr_interp = result['directional_analysis']['hr_analysis'].get('interpretation', '')
        print(f"  HR Interpretation: {hr_interp}")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()
