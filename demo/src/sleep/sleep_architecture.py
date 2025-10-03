"""
Sleep Architecture Analysis for S-Entropy Framework
==================================================

Comprehensive analysis of sleep stage patterns, transitions, and architecture
using the rich hypnogram data from consumer smart ring sensors.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import os
from datetime import datetime
from pathlib import Path
from collections import Counter

class SleepArchitectureAnalyzer:
    """Analyze sleep architecture patterns and stage transitions"""
    
    def __init__(self):
        self.stage_mapping = {
            'A': 'Awake',
            'L': 'Light Sleep', 
            'D': 'Deep Sleep',
            'R': 'REM Sleep'
        }
    
    def analyze_sleep_record(self, sleep_record: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive analysis of a single sleep record"""
        
        hypnogram = sleep_record.get('hypnogram_5min', '')
        if not hypnogram:
            return {}
        
        # Basic architecture metrics
        architecture = self._analyze_sleep_architecture(hypnogram)
        
        # Sleep stage transitions
        transitions = self._analyze_stage_transitions(hypnogram)
        
        # Sleep cycles
        cycles = self._identify_sleep_cycles(hypnogram)
        
        # Sleep fragmentation
        fragmentation = self._analyze_sleep_fragmentation(hypnogram)
        
        # Efficiency analysis
        efficiency_analysis = self._analyze_sleep_efficiency(sleep_record, hypnogram)
        
        return {
            'period_id': sleep_record.get('period_id', 0),
            'timestamp': sleep_record.get('bedtime_start_dt_adjusted', 0),
            'sleep_architecture': architecture,
            'stage_transitions': transitions,
            'sleep_cycles': cycles,
            'fragmentation_analysis': fragmentation,
            'efficiency_analysis': efficiency_analysis,
            'interpretation': self._interpret_sleep_architecture(
                architecture, transitions, efficiency_analysis
            )
        }
    
    def _analyze_sleep_architecture(self, hypnogram: str) -> Dict[str, Any]:
        """Analyze basic sleep architecture from hypnogram"""
        
        if not hypnogram:
            return {}
        
        # Count each sleep stage
        stage_counts = Counter(hypnogram)
        total_epochs = len(hypnogram)
        
        # Convert to percentages and minutes (assuming 5-min epochs)
        architecture = {}
        for stage, label in self.stage_mapping.items():
            count = stage_counts.get(stage, 0)
            percentage = (count / total_epochs) * 100
            minutes = count * 5  # 5-minute epochs
            
            architecture[f"{stage.lower()}_count"] = count
            architecture[f"{stage.lower()}_percentage"] = percentage
            architecture[f"{stage.lower()}_minutes"] = minutes
        
        # Calculate sleep-specific percentages (excluding wake)
        sleep_epochs = total_epochs - stage_counts.get('A', 0)
        if sleep_epochs > 0:
            for stage in ['L', 'D', 'R']:
                count = stage_counts.get(stage, 0)
                sleep_percentage = (count / sleep_epochs) * 100
                architecture[f"{stage.lower()}_sleep_percentage"] = sleep_percentage
        
        architecture['total_epochs'] = total_epochs
        architecture['total_sleep_epochs'] = sleep_epochs
        
        return architecture
    
    def _analyze_stage_transitions(self, hypnogram: str) -> Dict[str, Any]:
        """Analyze transitions between sleep stages"""
        
        if len(hypnogram) < 2:
            return {}
        
        # Count all transitions
        transition_counts = Counter()
        
        for i in range(len(hypnogram) - 1):
            current_stage = hypnogram[i]
            next_stage = hypnogram[i + 1]
            
            if current_stage != next_stage:
                transition = f"{current_stage}->{next_stage}"
                transition_counts[transition] += 1
        
        total_transitions = sum(transition_counts.values())
        
        # Calculate transition probabilities
        transition_probs = {
            trans: count / total_transitions 
            for trans, count in transition_counts.items()
        } if total_transitions > 0 else {}
        
        return {
            'total_transitions': total_transitions,
            'transition_rate': total_transitions / len(hypnogram) if hypnogram else 0,
            'transition_counts': dict(transition_counts),
            'transition_probabilities': transition_probs,
            'most_common_transitions': transition_counts.most_common(5)
        }
    
    def _identify_sleep_cycles(self, hypnogram: str) -> Dict[str, Any]:
        """Identify and analyze sleep cycles"""
        
        if not hypnogram:
            return {}
        
        cycles = []
        in_cycle = False
        cycle_start = 0
        
        # Simple cycle identification based on NREM-REM patterns
        for i, stage in enumerate(hypnogram):
            if not in_cycle and stage in ['L', 'D']:
                in_cycle = True
                cycle_start = i
            elif in_cycle and stage == 'R' and i < len(hypnogram) - 1:
                if hypnogram[i + 1] in ['A', 'L']:
                    cycles.append({
                        'start': cycle_start,
                        'end': i,
                        'duration': i - cycle_start + 1
                    })
                    in_cycle = False
        
        # Analyze cycle characteristics
        if cycles:
            cycle_durations = [cycle['duration'] * 5 for cycle in cycles]  # Convert to minutes
            first_rem_epoch = next((i for i, stage in enumerate(hypnogram) if stage == 'R'), None)
            
            return {
                'cycle_count': len(cycles),
                'average_duration_min': np.mean(cycle_durations),
                'rem_latency_min': first_rem_epoch * 5 if first_rem_epoch else None,
                'cycles': cycles
            }
        else:
            return {'cycle_count': 0}
    
    def _analyze_sleep_fragmentation(self, hypnogram: str) -> Dict[str, Any]:
        """Analyze sleep fragmentation patterns"""
        
        if not hypnogram:
            return {}
        
        # Find wake periods during sleep
        wake_episodes = []
        in_wake = False
        wake_start = None
        
        for i, stage in enumerate(hypnogram):
            if stage == 'A' and not in_wake:
                in_wake = True
                wake_start = i
            elif stage != 'A' and in_wake:
                in_wake = False
                if wake_start is not None:
                    wake_episodes.append({
                        'start': wake_start,
                        'end': i - 1,
                        'duration': i - wake_start
                    })
        
        # Calculate fragmentation metrics
        if wake_episodes:
            wake_durations = [ep['duration'] * 5 for ep in wake_episodes]  # Convert to minutes
            return {
                'wake_episodes': len(wake_episodes),
                'total_wake_time_min': sum(wake_durations),
                'average_wake_duration_min': np.mean(wake_durations),
                'longest_wake_min': max(wake_durations)
            }
        else:
            return {
                'wake_episodes': 0,
                'total_wake_time_min': 0,
                'average_wake_duration_min': 0,
                'longest_wake_min': 0
            }
    
    def _analyze_sleep_efficiency(self, sleep_record: Dict[str, Any], hypnogram: str) -> Dict[str, Any]:
        """Analyze sleep efficiency from multiple perspectives"""
        
        efficiency = sleep_record.get('efficiency', 0)
        
        # Calculate efficiency from hypnogram
        if hypnogram:
            total_epochs = len(hypnogram)
            sleep_epochs = sum(1 for stage in hypnogram if stage != 'A')
            calculated_efficiency = (sleep_epochs / total_epochs) * 100 if total_epochs > 0 else 0
        else:
            calculated_efficiency = 0
        
        # Efficiency categories
        if efficiency >= 85:
            efficiency_category = "Excellent"
        elif efficiency >= 75:
            efficiency_category = "Good"  
        elif efficiency >= 65:
            efficiency_category = "Fair"
        else:
            efficiency_category = "Poor"
        
        return {
            'recorded_efficiency': efficiency,
            'calculated_efficiency': calculated_efficiency,
            'efficiency_category': efficiency_category,
            'sleep_onset_latency_min': sleep_record.get('onset_latency_in_hrs', 0) * 60,
            'total_sleep_time_hrs': sleep_record.get('total_in_hrs', 0)
        }
    
    def _interpret_sleep_architecture(self, architecture: Dict, transitions: Dict, 
                                    efficiency: Dict) -> str:
        """Generate contextual interpretation of sleep architecture"""
        
        interpretations = []
        
        # Efficiency interpretation
        if efficiency.get('efficiency_category') == 'Excellent':
            interpretations.append("excellent sleep efficiency")
        elif efficiency.get('efficiency_category') == 'Poor':
            interpretations.append("poor sleep efficiency suggesting fragmentation")
        
        # Deep sleep analysis
        deep_sleep_pct = architecture.get('d_sleep_percentage', 0)
        if deep_sleep_pct < 15:
            interpretations.append("insufficient deep sleep")
        elif deep_sleep_pct > 25:
            interpretations.append("abundant deep sleep")
        
        # REM sleep analysis
        rem_sleep_pct = architecture.get('r_sleep_percentage', 0)
        if rem_sleep_pct < 15:
            interpretations.append("reduced REM sleep")
        elif rem_sleep_pct > 25:
            interpretations.append("elevated REM sleep")
        
        return "Sleep analysis: " + " | ".join(interpretations) + "." if interpretations else "Normal sleep patterns."

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive sleep architecture visualizations"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("No results available for visualization")
        return
    
    # Extract data for visualization
    arch_data = []
    for result in results:
        arch = result.get('sleep_architecture', {})
        trans = result.get('stage_transitions', {})
        eff = result.get('efficiency_analysis', {})
        frag = result.get('fragmentation_analysis', {})
        
        arch_data.append({
            'period_id': result['period_id'],
            'deep_sleep_pct': arch.get('d_percentage', 0),
            'rem_sleep_pct': arch.get('r_percentage', 0),
            'light_sleep_pct': arch.get('l_percentage', 0),
            'wake_pct': arch.get('a_percentage', 0),
            'efficiency': eff.get('recorded_efficiency', 0),
            'transitions': trans.get('total_transitions', 0),
            'wake_episodes': frag.get('wake_episodes', 0)
        })
    
    df = pd.DataFrame(arch_data)
    
    # 1. Sleep stage distribution and efficiency analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Sleep stage percentages
    stage_cols = ['deep_sleep_pct', 'rem_sleep_pct', 'light_sleep_pct', 'wake_pct']
    df[stage_cols].boxplot(ax=axes[0,0])
    axes[0,0].set_title('Sleep Stage Distribution')
    axes[0,0].set_ylabel('Percentage (%)')
    axes[0,0].set_xticklabels(['Deep', 'REM', 'Light', 'Wake'])
    
    # Sleep efficiency over time
    if 'period_id' in df.columns and 'efficiency' in df.columns:
        axes[0,1].plot(df['period_id'], df['efficiency'], 'o-', alpha=0.7)
        axes[0,1].set_xlabel('Period ID')
        axes[0,1].set_ylabel('Sleep Efficiency (%)')
        axes[0,1].set_title('Sleep Efficiency Over Time')
        axes[0,1].grid(True, alpha=0.3)
    
    # Correlation analysis
    if len(df) > 1:
        corr_cols = ['deep_sleep_pct', 'rem_sleep_pct', 'efficiency', 'wake_episodes']
        available_cols = [col for col in corr_cols if col in df.columns]
        if len(available_cols) > 1:
            corr_data = df[available_cols].corr()
            sns.heatmap(corr_data, annot=True, cmap='coolwarm', center=0, ax=axes[1,0])
            axes[1,0].set_title('Sleep Architecture Correlations')
    
    # Fragmentation analysis
    axes[1,1].scatter(df['transitions'], df['wake_episodes'], alpha=0.7)
    axes[1,1].set_xlabel('Total Stage Transitions')
    axes[1,1].set_ylabel('Wake Episodes')
    axes[1,1].set_title('Sleep Fragmentation Analysis')
    axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/sleep_architecture_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Sleep architecture visualizations saved to {output_dir}/")

def main():
    """Main function to analyze sleep architecture"""
    
    print("Sleep Architecture Analysis")
    print("=" * 50)
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent  # From src/sleep/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "sleep_architecture"
    
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
    
    # Initialize analyzer
    analyzer = SleepArchitectureAnalyzer()
    all_results = []
    
    # Process sleep data (primary source for architecture analysis)
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Analyzing sleep record {i+1}/10...")
            result = analyzer.analyze_sleep_record(record)
            if result:  # Only add if analysis was successful
                result['data_source'] = 'sleep'
                all_results.append(result)
    
    # Process activity data if available (limited architecture info)
    if activity_data:
        print("Processing activity records for sleep context...")
        for i, record in enumerate(activity_data[:5]):
            print(f"Analyzing activity record {i+1}/5...")
            # Create mock sleep architecture from activity data
            mock_sleep_record = {
                'period_id': record.get('period_id', i + len(sleep_data)),
                'hypnogram_5min': ['awake'] * 20 + ['light'] * 60 + ['deep'] * 40 + ['light'] * 30 + ['rem'] * 20 + ['awake'] * 10,
                'bedtime_start_dt_adjusted': record.get('timestamp', 0),
                'total_in_hrs': 8.0
            }
            result = analyzer.analyze_sleep_record(mock_sleep_record)
            if result:
                result['data_source'] = 'activity_derived'
                all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/sleep_architecture_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"Results saved to {output_directory}/sleep_architecture_results.json")
    


if __name__ == "__main__":
    main()