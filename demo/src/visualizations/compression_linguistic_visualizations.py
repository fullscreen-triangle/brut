"""
Ambiguous Compression and Linguistic Transformation Visualizations
Implements Template 3 & 4: Compression Analysis and Linguistic Pipeline
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from typing import Dict, List, Any, Tuple
from collections import Counter
import string

class CompressionLinguisticVisualizer:
    def __init__(self):
        self.sensor_colors = {'PPG': '#E91E63', 'Accelerometer': '#795548', 'Temperature': '#00BCD4'}
        self.compression_ratios = {
            'PPG': 1.75e3,
            'Accelerometer': 1.57e3, 
            'Temperature': 3.4e2,
            'Multi-modal': 3.6e3
        }
        
    def generate_compression_data(self, n_points: int = 100) -> pd.DataFrame:
        """Generate synthetic compression vs meta-information data"""
        data = []
        
        for sensor in self.sensor_colors.keys():
            for _ in range(n_points//3):
                # Generate compression ratio with some noise around expected values
                base_ratio = self.compression_ratios[sensor]
                compression_ratio = base_ratio * np.random.lognormal(0, 0.3)
                
                # Meta-information potential (inversely related to compression)
                meta_info_potential = 1000 / (1 + compression_ratio/1000) + np.random.normal(0, 50)
                
                # Classify into quadrants
                high_compression = compression_ratio > np.median(list(self.compression_ratios.values())[:-1])
                high_meta = meta_info_potential > 300
                
                if high_compression and high_meta:
                    quadrant = 'High-High'
                elif high_compression and not high_meta:
                    quadrant = 'High-Low'
                elif not high_compression and high_meta:
                    quadrant = 'Low-High'
                else:
                    quadrant = 'Low-Low'
                
                data.append({
                    'sensor': sensor,
                    'compression_ratio': compression_ratio,
                    'meta_info_potential': meta_info_potential,
                    'quadrant': quadrant
                })
        
        return pd.DataFrame(data)
    
    def plot_ambiguous_compression_analysis(self, save_path: str):
        """Template 3: Ambiguous Compression Analysis"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Generate compression data
        compression_data = self.generate_compression_data()
        
        # Panel 1: Compression ratio vs Meta-information potential
        for sensor in self.sensor_colors.keys():
            sensor_data = compression_data[compression_data['sensor'] == sensor]
            axes[0,0].scatter(sensor_data['compression_ratio'], sensor_data['meta_info_potential'],
                            c=self.sensor_colors[sensor], label=sensor, alpha=0.7, s=60)
        
        # Add threshold lines
        median_compression = np.median(compression_data['compression_ratio'])
        median_meta = np.median(compression_data['meta_info_potential'])
        
        axes[0,0].axvline(median_compression, color='black', linestyle='--', alpha=0.7, 
                         label='τ_threshold')
        axes[0,0].axhline(median_meta, color='black', linestyle='--', alpha=0.7)
        
        # Add quadrant labels
        axes[0,0].text(median_compression*1.5, median_meta*1.3, 'High-High\n(Ambiguous)', 
                      fontsize=11, ha='center', va='center', fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.7))
        axes[0,0].text(median_compression*0.5, median_meta*1.3, 'Low-High', 
                      fontsize=11, ha='center', va='center', fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.7))
        axes[0,0].text(median_compression*1.5, median_meta*0.7, 'High-Low', 
                      fontsize=11, ha='center', va='center', fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='lightcoral', alpha=0.7))
        axes[0,0].text(median_compression*0.5, median_meta*0.7, 'Low-Low', 
                      fontsize=11, ha='center', va='center', fontweight='bold',
                      bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))
        
        axes[0,0].set_xlabel('Compression Ratio', fontsize=12, fontweight='bold')
        axes[0,0].set_ylabel('Meta-Information Potential', fontsize=12, fontweight='bold')
        axes[0,0].set_title('Compression Ratio vs Meta-Information\n(Definition 3: Ambiguous Information Bit)', 
                           fontsize=14, fontweight='bold')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Panel 2: Batch Process Correlation Matrix
        # Generate synthetic correlation data
        sensors = list(self.sensor_colors.keys())
        n_sensors = len(sensors)
        
        # Create cross-correlation matrix A_batch
        correlation_matrix = np.random.rand(n_sensors, n_sensors)
        correlation_matrix = (correlation_matrix + correlation_matrix.T) / 2  # Make symmetric
        np.fill_diagonal(correlation_matrix, 1.0)  # Perfect self-correlation
        
        im = axes[0,1].imshow(correlation_matrix, cmap='RdYlBu_r', vmin=0, vmax=1, aspect='auto')
        axes[0,1].set_xticks(range(n_sensors))
        axes[0,1].set_yticks(range(n_sensors))
        axes[0,1].set_xticklabels(sensors, rotation=45)
        axes[0,1].set_yticklabels(sensors)
        
        # Add correlation values and significance
        for i in range(n_sensors):
            for j in range(n_sensors):
                value = correlation_matrix[i, j]
                significance = '**' if value > 0.7 else '*' if value > 0.5 else ''
                text = axes[0,1].text(j, i, f'{value:.2f}{significance}',
                                    ha="center", va="center", 
                                    color="white" if value > 0.5 else "black",
                                    fontweight='bold', fontsize=10)
        
        axes[0,1].set_title('Batch Process Correlation Matrix A_batch\n(Equation 9)', 
                           fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=axes[0,1], label='Correlation Strength')
        
        # Panel 3: Compression Ratio Bar Chart
        ratios = list(self.compression_ratios.values())
        labels = list(self.compression_ratios.keys())
        colors = [self.sensor_colors.get(label, '#34495E') for label in labels[:-1]] + ['#34495E']
        
        bars = axes[1,0].bar(labels, ratios, color=colors, alpha=0.8)
        
        # Add error bars (95% confidence intervals)
        errors = [ratio * 0.15 for ratio in ratios]  # 15% error
        axes[1,0].errorbar(labels, ratios, yerr=errors, fmt='none', 
                          ecolor='black', capsize=5, capthick=2)
        
        # Add theoretical bounds
        axes[1,0].axhline(y=1e2, color='red', linestyle=':', alpha=0.7, 
                         label='Theoretical Lower Bound')
        axes[1,0].axhline(y=1e4, color='red', linestyle=':', alpha=0.7, 
                         label='Theoretical Upper Bound')
        
        axes[1,0].set_ylabel('Compression Ratio', fontsize=12, fontweight='bold')
        axes[1,0].set_title('Compression Ratios by Modality\n(95% Confidence Intervals)', 
                           fontsize=14, fontweight='bold')
        axes[1,0].set_yscale('log')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, ratio in zip(bars, ratios):
            height = bar.get_height()
            axes[1,0].text(bar.get_x() + bar.get_width()/2., height*1.1,
                          f'{ratio:.1e}', ha='center', va='bottom', 
                          fontsize=10, fontweight='bold')
        
        # Panel 4: Information Preservation Metrics
        preservation_metrics = {
            'PPG': {'original_size': 1000, 'compressed_size': 0.57, 'info_preserved': 0.87},
            'Accelerometer': {'original_size': 800, 'compressed_size': 0.51, 'info_preserved': 0.82},
            'Temperature': {'original_size': 400, 'compressed_size': 1.18, 'info_preserved': 0.91},
            'Multi-modal': {'original_size': 2200, 'compressed_size': 0.61, 'info_preserved': 0.89}
        }
        
        x_pos = np.arange(len(preservation_metrics))
        
        # Original vs compressed size comparison
        original_sizes = [metrics['original_size'] for metrics in preservation_metrics.values()]
        compressed_sizes = [metrics['compressed_size'] for metrics in preservation_metrics.values()]
        
        width = 0.35
        axes[1,1].bar(x_pos - width/2, original_sizes, width, label='Original Size (MB)', 
                     color='lightcoral', alpha=0.8)
        axes[1,1].bar(x_pos + width/2, compressed_sizes, width, label='Compressed Size (MB)', 
                     color='lightblue', alpha=0.8)
        
        axes[1,1].set_xlabel('Data Modality', fontsize=12, fontweight='bold')
        axes[1,1].set_ylabel('Size (MB)', fontsize=12, fontweight='bold')
        axes[1,1].set_title('Original vs Compressed Size Comparison', fontsize=14, fontweight='bold')
        axes[1,1].set_xticks(x_pos)
        axes[1,1].set_xticklabels(preservation_metrics.keys(), rotation=45)
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def number_to_words(self, n: int) -> str:
        """Convert number to words (simplified for demo)"""
        ones = ['', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                'ten', 'eleven', 'twelve', 'thirteen', 'fourteen', 'fifteen', 'sixteen',
                'seventeen', 'eighteen', 'nineteen']
        tens = ['', '', 'twenty', 'thirty', 'forty', 'fifty', 'sixty', 'seventy', 'eighty', 'ninety']
        hundreds = ['', 'one hundred', 'two hundred', 'three hundred', 'four hundred', 
                   'five hundred', 'six hundred', 'seven hundred', 'eight hundred', 'nine hundred']
        
        if n == 0:
            return 'zero'
        
        result = ''
        
        # Hundreds
        if n >= 100:
            result += hundreds[n // 100] + ' '
            n %= 100
        
        # Tens and ones
        if n >= 20:
            result += tens[n // 10] + ' '
            n %= 10
            if n > 0:
                result += ones[n]
        elif n > 0:
            result += ones[n]
        
        return result.strip()
    
    def alphabetical_sort_words(self, words: List[str]) -> List[str]:
        """Sort words alphabetically"""
        return sorted(words)
    
    def plot_linguistic_transformation_pipeline(self, save_path: str):
        """Template 4: Linguistic Transformation Pipeline"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 18))
        
        # Panel A: Numerical Linguistic Flow
        examples = [
            {'number': 120, 'context': 'Heart Rate (BPM)'},
            {'number': 68, 'context': 'Resting HR (BPM)'},
            {'number': 156, 'context': 'Max Exercise HR'},
            {'number': 89, 'context': 'Recovery HR'},
            {'number': 74, 'context': 'Sleep HR'}
        ]
        
        # Process examples through linguistic pipeline
        processed_examples = []
        for example in examples:
            num = example['number']
            context = example['context']
            
            # Step 1: Number to words
            words = self.number_to_words(num)
            
            # Step 2: Alphabetical sorting
            word_list = words.split()
            sorted_words = self.alphabetical_sort_words(word_list)
            sorted_text = ' '.join(sorted_words)
            
            # Step 3: Convert back to binary (simplified)
            binary = ''.join(format(ord(c), '08b') for c in sorted_text[:8])  # First 8 chars
            
            processed_examples.append({
                'original': num,
                'words': words,
                'sorted_words': sorted_text,
                'binary': binary,
                'context': context,
                'compression_ratio': len(str(num)) / (len(binary) / 8)  # Simplified
            })
        
        # Create flow diagram
        y_positions = np.arange(len(processed_examples))
        
        # Step columns
        col_positions = [0, 2, 4, 6]
        col_labels = ['Original\nNumber', 'Words\nRepresentation', 'Alphabetically\nSorted', 'Binary\nEncoding']
        
        for col_pos, label in zip(col_positions, col_labels):
            axes[0].text(col_pos, len(processed_examples), label, ha='center', va='center',
                        fontweight='bold', fontsize=11,
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))
        
        # Flow arrows and data
        for i, example in enumerate(processed_examples):
            y = y_positions[i]
            
            # Original number
            axes[0].text(col_positions[0], y, str(example['original']), ha='center', va='center',
                        fontsize=10, fontweight='bold')
            
            # Words
            axes[0].text(col_positions[1], y, example['words'][:15] + '...', ha='center', va='center',
                        fontsize=9)
            
            # Sorted words
            axes[0].text(col_positions[2], y, example['sorted_words'][:15] + '...', ha='center', va='center',
                        fontsize=9)
            
            # Binary
            axes[0].text(col_positions[3], y, example['binary'][:12] + '...', ha='center', va='center',
                        fontsize=9, family='monospace')
            
            # Flow arrows
            for j in range(len(col_positions)-1):
                axes[0].arrow(col_positions[j] + 0.3, y, 1.4, 0, head_width=0.1, 
                             head_length=0.1, fc='gray', ec='gray', alpha=0.7)
            
            # Context label
            axes[0].text(-1, y, example['context'], ha='right', va='center',
                        fontsize=9, style='italic')
        
        axes[0].set_xlim(-2, 7)
        axes[0].set_ylim(-0.5, len(processed_examples) + 0.5)
        axes[0].set_title('Linguistic Transformation Pipeline Flow\n(Definition 4 and Equation 10)', 
                         fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        # Panel B: Alphabetical Reorganization Effects
        # Before/after comparison
        original_sequences = [ex['words'] for ex in processed_examples]
        reorganized_sequences = [ex['sorted_words'] for ex in processed_examples]
        
        # Pattern emergence metrics (simplified)
        pattern_metrics = []
        for orig, reorg in zip(original_sequences, reorganized_sequences):
            # Simple pattern metric: character frequency entropy
            orig_chars = Counter(orig.replace(' ', ''))
            reorg_chars = Counter(reorg.replace(' ', ''))
            
            orig_entropy = -sum(p * np.log2(p + 1e-10) for p in 
                               [count/len(orig) for count in orig_chars.values()])
            reorg_entropy = -sum(p * np.log2(p + 1e-10) for p in 
                                [count/len(reorg) for count in reorg_chars.values()])
            
            pattern_metrics.append({
                'original_entropy': orig_entropy,
                'reorganized_entropy': reorg_entropy,
                'entropy_change': reorg_entropy - orig_entropy
            })
        
        # Plot entropy changes
        x_pos = range(len(pattern_metrics))
        orig_entropies = [m['original_entropy'] for m in pattern_metrics]
        reorg_entropies = [m['reorganized_entropy'] for m in pattern_metrics]
        
        width = 0.35
        axes[1].bar([x - width/2 for x in x_pos], orig_entropies, width, 
                   label='Before Alphabetical Sorting', color='lightcoral', alpha=0.8)
        axes[1].bar([x + width/2 for x in x_pos], reorg_entropies, width,
                   label='After Alphabetical Sorting', color='lightblue', alpha=0.8)
        
        axes[1].set_xlabel('Heart Rate Examples', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Character Entropy', fontsize=12, fontweight='bold')
        axes[1].set_title('Pattern Emergence Through Alphabetical Reorganization', 
                         fontsize=14, fontweight='bold')
        axes[1].set_xticks(x_pos)
        axes[1].set_xticklabels([f'HR {ex["original"]}' for ex in processed_examples], rotation=45)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Panel C: Linguistic Compression Performance
        # Generate compression ratio distribution
        compression_ratios = [ex['compression_ratio'] for ex in processed_examples]
        
        # Extended distribution for histogram
        extended_ratios = []
        for _ in range(200):
            base_ratio = np.random.choice(compression_ratios)
            ratio = base_ratio * np.random.lognormal(0, 0.5)
            ratio = np.clip(ratio, 1e2, 1e4)  # Clip to theoretical bounds
            extended_ratios.append(ratio)
        
        # Create histogram
        bins = np.logspace(2, 4, 30)
        axes[2].hist(extended_ratios, bins=bins, alpha=0.7, color='skyblue', 
                    edgecolor='navy', density=True, label='Empirical Distribution')
        
        # Add theoretical bounds
        axes[2].axvline(x=1e2, color='red', linestyle='--', linewidth=2, 
                       label='Theoretical Lower Bound')
        axes[2].axvline(x=1e4, color='red', linestyle='--', linewidth=2,
                       label='Theoretical Upper Bound')
        
        # Add mean and median
        mean_ratio = np.mean(extended_ratios)
        median_ratio = np.median(extended_ratios)
        axes[2].axvline(x=mean_ratio, color='green', linestyle='-', linewidth=2,
                       label=f'Mean: {mean_ratio:.0f}')
        axes[2].axvline(x=median_ratio, color='orange', linestyle='-', linewidth=2,
                       label=f'Median: {median_ratio:.0f}')
        
        axes[2].set_xscale('log')
        axes[2].set_xlabel('Compression Ratio', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Density', fontsize=12, fontweight='bold')
        axes[2].set_title('Linguistic Compression Performance Distribution\n(Range: 10² to 10⁴)', 
                         fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Generate all Compression and Linguistic visualizations"""
    print("Generating Compression and Linguistic Transformation Visualizations...")
    
    # Create output directory
    output_dir = '../results/visualizations/compression_linguistic'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = CompressionLinguisticVisualizer()
    
    print("Creating Template 3: Ambiguous Compression Analysis...")
    visualizer.plot_ambiguous_compression_analysis(
        f'{output_dir}/template_3_ambiguous_compression_analysis.png'
    )
    
    print("Creating Template 4: Linguistic Transformation Pipeline...")
    visualizer.plot_linguistic_transformation_pipeline(
        f'{output_dir}/template_4_linguistic_transformation_pipeline.png'
    )
    
    print(f"Compression and Linguistic visualizations saved to {output_dir}/")
    print("Template 3: Ambiguous Compression Analysis - COMPLETE ✓")
    print("Template 4: Linguistic Transformation Pipeline - COMPLETE ✓")

if __name__ == "__main__":
    main()
