"""
Integrated S-Entropy Framework Demo
==================================

This demo integrates all S-entropy framework components to analyze consumer-grade
physiological sensor data, transforming measurement imprecision into contextual insights.

The complete pipeline demonstrates:
1. S-entropy coordinate navigation
2. Directional sequence mapping (A-R-D-L) 
3. Ambiguous compression and meta-information extraction
4. Sleep architecture analysis
5. Multi-modal integration and interpretation
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import os
from datetime import datetime
import sys

# Add src modules to path
sys.path.append('src')

from linguistic.s_entropy import SEntropyProcessor
from linguistic.directional_coordinate_mapping import DirectionalMapper
from linguistic.ambigous_compression import AmbiguousCompressor
from sleep.sleep_architecture import SleepArchitectureAnalyzer

class IntegratedSEntropyAnalyzer:
    """Integrated S-entropy framework for comprehensive physiological analysis"""
    
    def __init__(self):
        self.s_entropy_processor = SEntropyProcessor()
        self.directional_mapper = DirectionalMapper()
        self.compressor = AmbiguousCompressor()
        self.sleep_analyzer = SleepArchitectureAnalyzer()
        
        self.results = []
    
    def analyze_sleep_record(self, sleep_record: Dict[str, Any]) -> Dict[str, Any]:
        """Complete S-entropy analysis of a sleep record"""
        
        print(f"Analyzing Period {sleep_record.get('period_id', 'Unknown')}")
        
        # 1. S-entropy coordinate navigation
        print("  - Computing S-entropy coordinates...")
        s_entropy_result = self.s_entropy_processor.process_sleep_record(sleep_record)
        
        # 2. Directional sequence mapping
        print("  - Mapping to directional coordinates...")
        hr_sequence = sleep_record.get('hr_5min', [])
        rmssd_sequence = sleep_record.get('rmssd_5min', [])
        hypnogram = sleep_record.get('hypnogram_5min', '')
        
        hr_directional = self.directional_mapper.map_heart_rate_sequence(hr_sequence)
        hrv_directional = self.directional_mapper.map_hrv_sequence(rmssd_sequence)
        sleep_directional = self.directional_mapper.map_sleep_stages(hypnogram)
        
        # 3. Ambiguous compression analysis
        print("  - Extracting meta-information via compression...")
        hr_compression = self.compressor.compress_sequence(hr_sequence) if hr_sequence else {}
        hrv_compression = self.compressor.compress_sequence(rmssd_sequence) if rmssd_sequence else {}
        sleep_compression = self.compressor.compress_directional_sequence(hypnogram) if hypnogram else {}
        
        # 4. Sleep architecture analysis
        print("  - Analyzing sleep architecture...")
        sleep_architecture = self.sleep_analyzer.analyze_sleep_record(sleep_record)
        
        # 5. Integrate all analyses
        integrated_result = self._integrate_analyses(
            s_entropy_result, 
            {
                'hr_directional': hr_directional,
                'hrv_directional': hrv_directional,
                'sleep_directional': sleep_directional
            },
            {
                'hr_compression': hr_compression,
                'hrv_compression': hrv_compression,
                'sleep_compression': sleep_compression
            },
            sleep_architecture,
            sleep_record
        )
        
        return integrated_result
    
    def _integrate_analyses(self, s_entropy_result: Dict, directional_results: Dict,
                           compression_results: Dict, sleep_architecture: Dict,
                           original_record: Dict) -> Dict[str, Any]:
        """Integrate all analysis results into comprehensive interpretation"""
        
        # Generate unified contextual interpretation
        unified_interpretation = self._generate_unified_interpretation(
            s_entropy_result, directional_results, compression_results, sleep_architecture
        )
        
        # Calculate integration metrics
        integration_metrics = self._calculate_integration_metrics(
            s_entropy_result, directional_results, compression_results, sleep_architecture
        )
        
        return {
            'period_id': original_record.get('period_id', 0),
            'timestamp': original_record.get('bedtime_start_dt_adjusted', 0),
            
            # Core S-entropy analysis
            's_entropy_analysis': s_entropy_result,
            
            # Directional mapping results
            'directional_analysis': {
                'sequences': directional_results,
                'hr_analysis': self.directional_mapper.analyze_directional_patterns(
                    directional_results.get('hr_directional', '')[:100]
                ),
                'hrv_analysis': self.directional_mapper.analyze_directional_patterns(
                    directional_results.get('hrv_directional', '')[:100]
                ),
                'sleep_analysis': self.directional_mapper.analyze_directional_patterns(
                    directional_results.get('sleep_directional', '')[:100]
                )
            },
            
            # Compression analysis results
            'compression_analysis': compression_results,
            
            # Sleep architecture analysis
            'sleep_architecture_analysis': sleep_architecture,
            
            # Integration results
            'integration_metrics': integration_metrics,
            'unified_interpretation': unified_interpretation,
            
            # Original data context
            'raw_data': {
                'efficiency': original_record.get('efficiency', 0),
                'score': original_record.get('score', 0),
                'total_sleep_hrs': original_record.get('total_in_hrs', 0),
                'deep_sleep_hrs': original_record.get('deep_in_hrs', 0),
                'rem_sleep_hrs': original_record.get('rem_in_hrs', 0)
            }
        }
    
    def _generate_unified_interpretation(self, s_entropy: Dict, directional: Dict, 
                                       compression: Dict, sleep_arch: Dict) -> str:
        """Generate unified contextual interpretation using all analyses"""
        
        interpretations = []
        
        # S-entropy coordinate interpretation
        s_coords = s_entropy.get('s_entropy_coordinates', {})
        knowledge = s_coords.get('knowledge', 0)
        entropy = s_coords.get('entropy', 0)
        
        if knowledge > 0.7:
            interpretations.append("S-entropy navigation reveals high measurement uncertainty")
        if entropy > 0.6:
            interpretations.append("elevated physiological variability detected")
        
        # Directional sequence interpretation
        hr_dist = directional.get('hr_directional', '')
        if hr_dist:
            a_count = hr_dist.count('A')
            l_count = hr_dist.count('L')
            if a_count / len(hr_dist) > 0.3:
                interpretations.append("frequent cardiac activation states")
            if l_count / len(hr_dist) > 0.3:
                interpretations.append("significant stress/transition periods")
        
        # Compression meta-information
        hr_comp = compression.get('hr_compression', {})
        if hr_comp.get('compression_ratio', 1.0) > 0.8:
            interpretations.append("heart rate patterns resistant to compression suggest complex dynamics")
        
        # Sleep architecture integration
        sleep_interp = sleep_arch.get('interpretation', '')
        if 'poor' in sleep_interp.lower():
            interpretations.append("sleep fragmentation confirmed by architecture analysis")
        elif 'excellent' in sleep_interp.lower():
            interpretations.append("consolidated sleep architecture supports restoration")
        
        # Unified S-entropy interpretation
        unified = ("Integrated S-entropy framework analysis: " + 
                  " | ".join(interpretations) + 
                  " | Consumer sensor imprecision transformed into contextual physiological insights.")
        
        return unified
    
    def _calculate_integration_metrics(self, s_entropy: Dict, directional: Dict,
                                     compression: Dict, sleep_arch: Dict) -> Dict[str, float]:
        """Calculate metrics that integrate across all analyses"""
        
        metrics = {}
        
        # S-entropy integration
        s_coords = s_entropy.get('s_entropy_coordinates', {})
        s_distance = np.sqrt(sum(v**2 for v in s_coords.values())) if s_coords else 0
        metrics['s_entropy_magnitude'] = s_distance
        
        # Directional coherence (consistency across modalities)
        sequences = [
            directional.get('hr_directional', ''),
            directional.get('hrv_directional', ''), 
            directional.get('sleep_directional', '')
        ]
        
        if all(seq for seq in sequences):
            # Calculate cross-modal directional coherence
            min_len = min(len(seq) for seq in sequences)
            if min_len > 0:
                coherence_sum = 0
                for i in range(min_len):
                    chars = [seq[i] for seq in sequences]
                    if len(set(chars)) == 1:  # All same direction
                        coherence_sum += 1
                metrics['directional_coherence'] = coherence_sum / min_len
            else:
                metrics['directional_coherence'] = 0
        else:
            metrics['directional_coherence'] = 0
        
        # Compression complexity index
        comp_ratios = []
        for comp_key in ['hr_compression', 'hrv_compression', 'sleep_compression']:
            if comp_key in compression:
                ratio = compression[comp_key].get('compression_ratio', 1.0)
                comp_ratios.append(ratio)
        
        metrics['compression_complexity'] = np.mean(comp_ratios) if comp_ratios else 0
        
        # Sleep quality integration
        sleep_eff = sleep_arch.get('efficiency_analysis', {}).get('recorded_efficiency', 0)
        metrics['integrated_sleep_quality'] = sleep_eff / 100.0
        
        return metrics

def create_comprehensive_visualization(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualization of integrated S-entropy analysis"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("No results available for visualization")
        return
    
    # Extract integration data
    integration_data = []
    for result in results:
        metrics = result.get('integration_metrics', {})
        s_coords = result.get('s_entropy_analysis', {}).get('s_entropy_coordinates', {})
        raw_data = result.get('raw_data', {})
        
        integration_data.append({
            'period_id': result['period_id'],
            's_entropy_magnitude': metrics.get('s_entropy_magnitude', 0),
            'directional_coherence': metrics.get('directional_coherence', 0),
            'compression_complexity': metrics.get('compression_complexity', 0),
            'sleep_quality': metrics.get('integrated_sleep_quality', 0),
            'knowledge': s_coords.get('knowledge', 0),
            'entropy': s_coords.get('entropy', 0),
            'context': s_coords.get('context', 0),
            'efficiency': raw_data.get('efficiency', 0)
        })
    
    df = pd.DataFrame(integration_data)
    
    # Create comprehensive visualization
    fig = plt.figure(figsize=(20, 15))
    
    # 1. S-entropy coordinate evolution
    plt.subplot(3, 3, 1)
    plt.plot(df['period_id'], df['knowledge'], 'o-', label='Knowledge', alpha=0.7)
    plt.plot(df['period_id'], df['entropy'], 's-', label='Entropy', alpha=0.7)
    plt.plot(df['period_id'], df['context'], '^-', label='Context', alpha=0.7)
    plt.xlabel('Period ID')
    plt.ylabel('S-Entropy Coordinate Value')
    plt.title('S-Entropy Coordinate Evolution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. Integration metrics
    plt.subplot(3, 3, 2)
    plt.scatter(df['s_entropy_magnitude'], df['directional_coherence'], 
               c=df['sleep_quality'], cmap='viridis', alpha=0.7, s=60)
    plt.xlabel('S-Entropy Magnitude')
    plt.ylabel('Directional Coherence')
    plt.title('Integration Metrics Relationship')
    plt.colorbar(label='Sleep Quality')
    
    # 3. Compression complexity vs sleep quality
    plt.subplot(3, 3, 3)
    plt.scatter(df['compression_complexity'], df['sleep_quality'], alpha=0.7, s=60)
    plt.xlabel('Compression Complexity')
    plt.ylabel('Sleep Quality')
    plt.title('Complexity vs Sleep Quality')
    plt.grid(True, alpha=0.3)
    
    # 4. Multi-dimensional analysis
    plt.subplot(3, 3, 4)
    bubble_size = df['s_entropy_magnitude'] * 100
    plt.scatter(df['knowledge'], df['entropy'], s=bubble_size, 
               c=df['efficiency'], cmap='plasma', alpha=0.6)
    plt.xlabel('Knowledge Coordinate')
    plt.ylabel('Entropy Coordinate')
    plt.title('S-Entropy Knowledge vs Entropy')
    plt.colorbar(label='Sleep Efficiency')
    
    # 5. Coherence analysis
    plt.subplot(3, 3, 5)
    plt.boxplot([df['directional_coherence'], df['compression_complexity'], df['sleep_quality']], 
                labels=['Directional\nCoherence', 'Compression\nComplexity', 'Sleep\nQuality'])
    plt.title('Integration Metrics Distribution')
    plt.ylabel('Metric Value')
    
    # 6. Time series integration
    plt.subplot(3, 3, 6)
    plt.plot(df['period_id'], df['s_entropy_magnitude'], 'o-', label='S-Entropy Magnitude', alpha=0.7)
    plt.plot(df['period_id'], df['directional_coherence'], 's-', label='Directional Coherence', alpha=0.7)
    plt.xlabel('Period ID')
    plt.ylabel('Metric Value')
    plt.title('Integration Metrics Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 7. Correlation heatmap
    plt.subplot(3, 3, 7)
    corr_cols = ['s_entropy_magnitude', 'directional_coherence', 'compression_complexity', 
                'sleep_quality', 'efficiency']
    available_cols = [col for col in corr_cols if col in df.columns and not df[col].isna().all()]
    if len(available_cols) > 1:
        corr_matrix = df[available_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                   square=True, cbar_kws={'label': 'Correlation'})
        plt.title('Integration Metrics Correlations')
    
    # 8. Efficiency prediction
    plt.subplot(3, 3, 8)
    plt.scatter(df['s_entropy_magnitude'], df['efficiency'], alpha=0.7, s=60, label='Actual')
    if len(df) > 1:
        z = np.polyfit(df['s_entropy_magnitude'], df['efficiency'], 1)
        p = np.poly1d(z)
        plt.plot(sorted(df['s_entropy_magnitude']), p(sorted(df['s_entropy_magnitude'])), 
                "r--", alpha=0.8, label=f'Trend: y={z[0]:.1f}x+{z[1]:.1f}')
        plt.legend()
    plt.xlabel('S-Entropy Magnitude')
    plt.ylabel('Sleep Efficiency (%)')
    plt.title('S-Entropy Predictive Relationship')
    plt.grid(True, alpha=0.3)
    
    # 9. Summary text
    plt.subplot(3, 3, 9)
    plt.axis('off')
    summary_text = f"""
    Integrated S-Entropy Analysis Summary
    
    Records Analyzed: {len(results)}
    
    Average S-Entropy Magnitude: {df['s_entropy_magnitude'].mean():.3f}
    Average Directional Coherence: {df['directional_coherence'].mean():.3f}
    Average Compression Complexity: {df['compression_complexity'].mean():.3f}
    Average Sleep Quality: {df['sleep_quality'].mean():.3f}
    
    Key Insight: Consumer sensor imprecision
    transformed into contextual physiological
    understanding through S-entropy navigation.
    """
    plt.text(0.1, 0.5, summary_text, fontsize=10, verticalalignment='center',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/integrated_s_entropy_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comprehensive visualization saved to {output_dir}/integrated_s_entropy_analysis.png")

def main():
    """Main function for integrated S-entropy framework demonstration"""
    
    print("Integrated S-Entropy Framework Demo")
    print("=" * 60)
    print("Transforming Consumer Sensor Imprecision into Contextual Insights")
    print("=" * 60)
    
    # Load sleep data
    with open('public/sleep_ppg_records.json', 'r') as f:
        sleep_data = json.load(f)
    
    print(f"Loaded {len(sleep_data)} sleep records")
    
    # Initialize integrated analyzer
    analyzer = IntegratedSEntropyAnalyzer()
    
    # Process first 5 records for comprehensive demo
    results = []
    for i, record in enumerate(sleep_data[:5]):
        print(f"\nProcessing record {i+1}/5...")
        result = analyzer.analyze_sleep_record(record)
        results.append(result)
    
    # Save comprehensive results
    output_dir = 'results/integrated_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    with open(f'{output_dir}/integrated_s_entropy_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nResults saved to {output_dir}/integrated_s_entropy_results.json")
    
    # Create comprehensive visualization
    print("Creating comprehensive visualization...")
    create_comprehensive_visualization(results, output_dir)
    
    # Print unified interpretations
    print("\nUnified S-Entropy Interpretations:")
    print("-" * 50)
    for i, result in enumerate(results):
        print(f"\nRecord {i+1} (Period {result['period_id']}):")
        print(f"  {result['unified_interpretation']}")
        
        metrics = result['integration_metrics']
        print(f"  Integration Metrics:")
        print(f"    - S-Entropy Magnitude: {metrics.get('s_entropy_magnitude', 0):.3f}")
        print(f"    - Directional Coherence: {metrics.get('directional_coherence', 0):.3f}")
        print(f"    - Compression Complexity: {metrics.get('compression_complexity', 0):.3f}")
        print(f"    - Sleep Quality: {metrics.get('integrated_sleep_quality', 0):.3f}")
    
    # Final summary
    print(f"\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    print("The S-entropy framework has successfully transformed imprecise consumer")
    print("sensor readings into precise contextual interpretations of physiological")
    print("states. This demonstrates the core principle: instead of making sensors")
    print("more accurate, we make interpretations more intelligent.")
    print("=" * 60)

if __name__ == "__main__":
    main()
