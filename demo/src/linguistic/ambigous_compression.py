"""
Ambiguous Compression for S-Entropy Framework
=============================================

Implements compression-resistant bit identification and meta-information extraction
from physiological sensor measurements. Transform measurement "imprecision" into 
semantic content through empty dictionary synthesis.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any, Optional
import os
from pathlib import Path
from datetime import datetime
from collections import Counter
import zlib
import gzip

class AmbiguousCompressor:
    """Extract meta-information from compression-resistant patterns"""
    
    def __init__(self, compression_threshold: float = 0.7, 
                 ambiguity_threshold: float = 0.3):
        """
        Initialize ambiguous compression parameters
        
        Args:
            compression_threshold: Minimum compression ratio to consider data compressible
            ambiguity_threshold: Minimum ambiguity score to consider bits ambiguous
        """
        self.compression_threshold = compression_threshold
        self.ambiguity_threshold = ambiguity_threshold
        self.dictionary = {}  # Empty dictionary for meta-information synthesis
    
    def compress_sequence(self, sequence: List[float]) -> Dict[str, Any]:
        """Apply ambiguous compression to numerical sequence"""
        
        if not sequence or len(sequence) < 10:
            return {}
        
        # Convert to bit representation
        bit_sequence = self._to_bit_sequence(sequence)
        
        # Apply compression and identify resistant patterns
        compression_analysis = self._analyze_compression_resistance(bit_sequence)
        
        # Extract ambiguous bits
        ambiguous_bits = self._extract_ambiguous_bits(bit_sequence, compression_analysis)
        
        # Synthesize meta-information from ambiguous bits
        meta_information = self._synthesize_meta_information(ambiguous_bits, sequence)
        
        return {
            'original_length': len(sequence),
            'bit_length': len(bit_sequence),
            'compression_ratio': compression_analysis['compression_ratio'],
            'resistant_patterns': compression_analysis['resistant_patterns'],
            'ambiguous_bits': ambiguous_bits,
            'meta_information': meta_information,
            'dictionary_entries': len(self.dictionary)
        }
    
    def compress_directional_sequence(self, directional_seq: str) -> Dict[str, Any]:
        """Apply ambiguous compression to A-R-D-L directional sequence"""
        
        if not directional_seq:
            return {}
        
        # Convert directional sequence to numerical for bit representation
        direction_values = {'A': 3, 'R': 2, 'D': 1, 'L': 0}
        numerical_seq = [direction_values.get(char, 2) for char in directional_seq]
        
        # Apply compression
        compression_result = self.compress_sequence(numerical_seq)
        
        # Add directional-specific analysis
        direction_analysis = self._analyze_directional_ambiguity(directional_seq)
        
        compression_result.update({
            'directional_sequence': directional_seq[:100],  # Truncate for readability
            'directional_ambiguity': direction_analysis
        })
        
        return compression_result
    
    def _to_bit_sequence(self, sequence: List[float]) -> str:
        """Convert numerical sequence to bit representation"""
        
        # Quantize to integers for bit representation
        min_val = min(sequence) if sequence else 0
        max_val = max(sequence) if sequence else 1
        
        if max_val == min_val:
            # Handle constant sequence
            quantized = [128] * len(sequence)  # Use mid-range value
        else:
            # Scale to 8-bit range (0-255)
            quantized = [
                int(((val - min_val) / (max_val - min_val)) * 255)
                for val in sequence
            ]
        
        # Convert to binary representation
        bit_sequence = ''.join(format(val, '08b') for val in quantized)
        
        return bit_sequence
    
    def _analyze_compression_resistance(self, bit_sequence: str) -> Dict[str, Any]:
        """Analyze compression resistance of bit sequence"""
        
        if not bit_sequence:
            return {'compression_ratio': 1.0, 'resistant_patterns': []}
        
        # Apply different compression algorithms
        original_bytes = bit_sequence.encode('utf-8')
        original_size = len(original_bytes)
        
        # Test compression with different algorithms
        compression_results = {}
        
        try:
            zlib_compressed = zlib.compress(original_bytes)
            compression_results['zlib'] = len(zlib_compressed) / original_size
        except:
            compression_results['zlib'] = 1.0
        
        try:
            gzip_compressed = gzip.compress(original_bytes)
            compression_results['gzip'] = len(gzip_compressed) / original_size
        except:
            compression_results['gzip'] = 1.0
        
        # Calculate average compression ratio
        avg_compression_ratio = np.mean(list(compression_results.values()))
        
        # Identify resistant patterns (patterns that don't compress well)
        resistant_patterns = self._find_resistant_patterns(bit_sequence)
        
        return {
            'compression_ratio': avg_compression_ratio,
            'compression_results': compression_results,
            'resistant_patterns': resistant_patterns,
            'is_compressible': avg_compression_ratio < self.compression_threshold
        }
    
    def _find_resistant_patterns(self, bit_sequence: str, pattern_length: int = 8) -> List[Dict[str, Any]]:
        """Find bit patterns that resist compression"""
        
        resistant_patterns = []
        
        # Analyze patterns of different lengths
        for length in range(4, min(pattern_length + 1, len(bit_sequence) // 4)):
            pattern_counts = Counter()
            
            # Count all patterns of this length
            for i in range(len(bit_sequence) - length + 1):
                pattern = bit_sequence[i:i + length]
                pattern_counts[pattern] += 1
            
            # Find patterns with unusual frequency distributions
            for pattern, count in pattern_counts.items():
                frequency = count / (len(bit_sequence) - length + 1)
                
                # Test if pattern resists compression
                pattern_compression_ratio = self._test_pattern_compression(pattern * count)
                
                if pattern_compression_ratio > self.compression_threshold:
                    resistant_patterns.append({
                        'pattern': pattern,
                        'length': length,
                        'count': count,
                        'frequency': frequency,
                        'compression_resistance': pattern_compression_ratio
                    })
        
        # Sort by compression resistance
        resistant_patterns.sort(key=lambda x: x['compression_resistance'], reverse=True)
        
        return resistant_patterns[:10]  # Return top 10 resistant patterns
    
    def _test_pattern_compression(self, pattern: str) -> float:
        """Test compression ratio of a specific pattern"""
        
        if not pattern:
            return 1.0
        
        try:
            original_bytes = pattern.encode('utf-8')
            compressed_bytes = zlib.compress(original_bytes)
            return len(compressed_bytes) / len(original_bytes)
        except:
            return 1.0
    
    def _extract_ambiguous_bits(self, bit_sequence: str, compression_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract ambiguous bits that carry meta-information"""
        
        ambiguous_bits = {}
        
        # Identify bits in resistant patterns
        resistant_patterns = compression_analysis.get('resistant_patterns', [])
        
        ambiguous_positions = set()
        for pattern_info in resistant_patterns:
            pattern = pattern_info['pattern']
            
            # Find all occurrences of resistant pattern
            start = 0
            while True:
                pos = bit_sequence.find(pattern, start)
                if pos == -1:
                    break
                
                # Mark all positions in this pattern as ambiguous
                for i in range(pos, pos + len(pattern)):
                    ambiguous_positions.add(i)
                
                start = pos + 1
        
        # Extract ambiguous bit subsequences
        ambiguous_bits['positions'] = sorted(list(ambiguous_positions))
        ambiguous_bits['count'] = len(ambiguous_positions)
        ambiguous_bits['percentage'] = (len(ambiguous_positions) / len(bit_sequence)) * 100 if bit_sequence else 0
        
        # Extract the actual ambiguous bit values
        ambiguous_bits['values'] = [bit_sequence[pos] for pos in ambiguous_bits['positions']]
        
        return ambiguous_bits
    
    def _synthesize_meta_information(self, ambiguous_bits: Dict[str, Any], 
                                   original_sequence: List[float]) -> Dict[str, Any]:
        """Synthesize meta-information from ambiguous bits using empty dictionary"""
        
        meta_info = {}
        
        if not ambiguous_bits or not ambiguous_bits.get('values'):
            return meta_info
        
        # Analyze ambiguous bit patterns for meta-information
        ambiguous_values = ambiguous_bits['values']
        
        # Pattern entropy of ambiguous bits
        bit_counts = Counter(ambiguous_values)
        total_bits = len(ambiguous_values)
        
        if total_bits > 0:
            entropy = -sum((count/total_bits) * np.log2(count/total_bits) 
                          for count in bit_counts.values())
            meta_info['ambiguous_entropy'] = entropy
        
        # Correlation with original sequence characteristics
        if original_sequence:
            seq_mean = np.mean(original_sequence)
            seq_std = np.std(original_sequence)
            seq_trend = np.polyfit(range(len(original_sequence)), original_sequence, 1)[0]
            
            # Relate ambiguous bit density to sequence characteristics
            bit_density = ambiguous_bits['percentage']
            
            meta_info['meta_correlations'] = {
                'bit_density_vs_variability': bit_density * seq_std / 100,
                'bit_density_vs_mean': bit_density * abs(seq_mean) / 100,
                'bit_density_vs_trend': bit_density * abs(seq_trend) / 100
            }
        
        # Generate empty dictionary entries from ambiguous patterns
        self._update_empty_dictionary(ambiguous_values, meta_info)
        
        return meta_info
    
    def _analyze_directional_ambiguity(self, directional_seq: str) -> Dict[str, Any]:
        """Analyze ambiguity in directional sequences"""
        
        if not directional_seq:
            return {}
        
        # Character transition ambiguity
        transitions = []
        for i in range(len(directional_seq) - 1):
            transitions.append(f"{directional_seq[i]}->{directional_seq[i+1]}")
        
        transition_counts = Counter(transitions)
        transition_entropy = 0
        total_transitions = len(transitions)
        
        if total_transitions > 0:
            for count in transition_counts.values():
                prob = count / total_transitions
                if prob > 0:
                    transition_entropy -= prob * np.log2(prob)
        
        # Find ambiguous transition patterns
        ambiguous_transitions = []
        for transition, count in transition_counts.items():
            frequency = count / total_transitions if total_transitions > 0 else 0
            if 0.1 < frequency < 0.4:  # Neither rare nor very common - ambiguous
                ambiguous_transitions.append({
                    'transition': transition,
                    'count': count,
                    'frequency': frequency,
                    'ambiguity_score': 1 - abs(frequency - 0.25) / 0.25  # Max at 0.25
                })
        
        return {
            'transition_entropy': transition_entropy,
            'total_transitions': total_transitions,
            'unique_transitions': len(transition_counts),
            'ambiguous_transitions': ambiguous_transitions
        }
    
    def _update_empty_dictionary(self, ambiguous_values: List[str], meta_info: Dict[str, Any]):
        """Update empty dictionary with meta-information from ambiguous bits"""
        
        # Create dictionary entries from ambiguous bit patterns
        bit_string = ''.join(ambiguous_values)
        
        # Generate "meanings" from bit patterns
        for i in range(0, len(bit_string), 8):
            byte_pattern = bit_string[i:i+8]
            if len(byte_pattern) == 8:
                # Convert to ASCII character if possible
                try:
                    ascii_val = int(byte_pattern, 2)
                    if 32 <= ascii_val <= 126:  # Printable ASCII
                        char_meaning = chr(ascii_val)
                        self.dictionary[byte_pattern] = {
                            'meaning': char_meaning,
                            'context': 'ambiguous_compression',
                            'frequency': self.dictionary.get(byte_pattern, {}).get('frequency', 0) + 1
                        }
                except:
                    pass
        
        meta_info['new_dictionary_entries'] = len([k for k in self.dictionary.keys() 
                                                  if 'ambiguous_compression' in self.dictionary[k].get('context', '')])

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualizations of ambiguous compression analysis"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract compression data
    compression_data = []
    for result in results:
        if 'compression_analysis' in result:
            analysis = result['compression_analysis']
            compression_data.append({
                'period_id': result['period_id'],
                'hr_compression_ratio': analysis.get('hr_compression', {}).get('compression_ratio', 1.0),
                'hrv_compression_ratio': analysis.get('hrv_compression', {}).get('compression_ratio', 1.0),
                'sleep_compression_ratio': analysis.get('sleep_compression', {}).get('compression_ratio', 1.0),
                'hr_ambiguous_pct': analysis.get('hr_compression', {}).get('ambiguous_bits', {}).get('percentage', 0),
                'hrv_ambiguous_pct': analysis.get('hrv_compression', {}).get('ambiguous_bits', {}).get('percentage', 0),
                'sleep_ambiguous_pct': analysis.get('sleep_compression', {}).get('ambiguous_bits', {}).get('percentage', 0)
            })
    
    if not compression_data:
        print("No compression data found for visualization")
        return
    
    df = pd.DataFrame(compression_data)
    
    # 1. Compression ratio analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Compression ratios
    compression_cols = ['hr_compression_ratio', 'hrv_compression_ratio', 'sleep_compression_ratio']
    df[compression_cols].boxplot(ax=axes[0,0])
    axes[0,0].set_title('Compression Ratios by Data Type')
    axes[0,0].set_ylabel('Compression Ratio')
    axes[0,0].set_xticklabels(['HR', 'HRV', 'Sleep'])
    
    # Ambiguous bit percentages
    ambiguous_cols = ['hr_ambiguous_pct', 'hrv_ambiguous_pct', 'sleep_ambiguous_pct']
    df[ambiguous_cols].boxplot(ax=axes[0,1])
    axes[0,1].set_title('Ambiguous Bit Percentages')
    axes[0,1].set_ylabel('Ambiguous Bits (%)')
    axes[0,1].set_xticklabels(['HR', 'HRV', 'Sleep'])
    
    # Correlation between compression and ambiguity
    axes[1,0].scatter(df['hr_compression_ratio'], df['hr_ambiguous_pct'], alpha=0.7, label='HR')
    axes[1,0].scatter(df['hrv_compression_ratio'], df['hrv_ambiguous_pct'], alpha=0.7, label='HRV')
    axes[1,0].scatter(df['sleep_compression_ratio'], df['sleep_ambiguous_pct'], alpha=0.7, label='Sleep')
    axes[1,0].set_xlabel('Compression Ratio')
    axes[1,0].set_ylabel('Ambiguous Bits (%)')
    axes[1,0].set_title('Compression vs Ambiguity')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Evolution over time
    if 'period_id' in df.columns:
        axes[1,1].plot(df['period_id'], df['hr_compression_ratio'], 'o-', label='HR Compression', alpha=0.7)
        axes[1,1].plot(df['period_id'], df['hrv_compression_ratio'], 's-', label='HRV Compression', alpha=0.7)
        axes[1,1].plot(df['period_id'], df['sleep_compression_ratio'], '^-', label='Sleep Compression', alpha=0.7)
        axes[1,1].set_xlabel('Period ID')
        axes[1,1].set_ylabel('Compression Ratio')
        axes[1,1].set_title('Compression Evolution Over Time')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/compression_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Meta-information extraction visualization
    plt.figure(figsize=(12, 8))
    
    # Extract meta-information data
    meta_data = []
    for result in results:
        analysis = result.get('compression_analysis', {})
        for data_type in ['hr_compression', 'hrv_compression', 'sleep_compression']:
            if data_type in analysis:
                meta_info = analysis[data_type].get('meta_information', {})
                if 'ambiguous_entropy' in meta_info:
                    meta_data.append({
                        'period_id': result['period_id'],
                        'data_type': data_type.split('_')[0].upper(),
                        'ambiguous_entropy': meta_info['ambiguous_entropy']
                    })
    
    if meta_data:
        meta_df = pd.DataFrame(meta_data)
        
        # Box plot of ambiguous entropy by data type
        plt.subplot(2, 1, 1)
        meta_df.boxplot(column='ambiguous_entropy', by='data_type', ax=plt.gca())
        plt.title('Ambiguous Bit Entropy by Data Type')
        plt.suptitle('')  # Remove automatic title
        plt.ylabel('Entropy')
        
        # Evolution over time
        plt.subplot(2, 1, 2)
        for data_type in meta_df['data_type'].unique():
            type_data = meta_df[meta_df['data_type'] == data_type]
            plt.plot(type_data['period_id'], type_data['ambiguous_entropy'], 
                    'o-', label=data_type, alpha=0.7)
        
        plt.xlabel('Period ID')
        plt.ylabel('Ambiguous Entropy')
        plt.title('Meta-Information Entropy Evolution')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/meta_information_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Ambiguous compression visualizations saved to {output_dir}/")

def main():
    """Main function to demonstrate ambiguous compression"""
    
    print("Ambiguous Compression Analysis")
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
    output_directory = project_root / "results" / "ambiguous_compression"
    
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
    
    # Initialize compressor
    compressor = AmbiguousCompressor()
    
    # Combine and process data
    all_results = []
    
    # Process sleep data (primary source for compression analysis)
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Processing sleep record {i+1}/10...")
            
            # Extract sequences
            hr_sequence = record.get('hr_5min', [])
            rmssd_sequence = record.get('rmssd_5min', [])
            hypnogram = record.get('hypnogram_5min', '')
            
            # Apply ambiguous compression
            hr_compression = compressor.compress_sequence(hr_sequence) if hr_sequence else {}
            hrv_compression = compressor.compress_sequence(rmssd_sequence) if rmssd_sequence else {}
            sleep_compression = compressor.compress_directional_sequence(hypnogram) if hypnogram else {}
            
            result = {
                'period_id': record.get('period_id', i),
                'timestamp': record.get('bedtime_start_dt_adjusted', 0),
                'compression_analysis': {
                    'hr_compression': hr_compression,
                    'hrv_compression': hrv_compression,
                    'sleep_compression': sleep_compression
                },
                'dictionary_size': len(compressor.dictionary),
                'data_source': 'sleep'
            }
            
            all_results.append(result)
    
    # Process activity data for additional context
    if activity_data:
        print("Processing activity records...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Processing activity record {i+1}/10...")
            
            # Extract sequences (activity may have different structure)
            hr_sequence = record.get('hr_5min', [])
            steps_sequence = record.get('steps', [])
            
            # Apply ambiguous compression
            hr_compression = compressor.compress_sequence(hr_sequence) if hr_sequence else {}
            steps_compression = compressor.compress_sequence([steps_sequence] if isinstance(steps_sequence, (int, float)) else steps_sequence) if steps_sequence else {}
            
            result = {
                'period_id': record.get('period_id', i + len(sleep_data)),
                'timestamp': record.get('timestamp', 0),
                'compression_analysis': {
                    'hr_compression': hr_compression,
                    'steps_compression': steps_compression,
                    'sleep_compression': {}
                },
                'dictionary_size': len(compressor.dictionary),
                'data_source': 'activity'
            }
            
            all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/compression_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/compression_results.json")
    
    # Save dictionary
    with open(f'{output_directory}/empty_dictionary.json', 'w') as f:
        json.dump(compressor.dictionary, f, indent=2, default=str)
    
    print(f"✓ Empty dictionary saved to {output_directory}/empty_dictionary.json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(all_results, output_directory)
    
    # Print compression statistics
    print("\nCompression Analysis Summary:")
    print("-" * 40)
    
    hr_ratios = [r['compression_analysis']['hr_compression'].get('compression_ratio', 1.0) for r in all_results if 'hr_compression' in r['compression_analysis']]
    hrv_ratios = [r['compression_analysis']['hrv_compression'].get('compression_ratio', 1.0) for r in all_results if 'hrv_compression' in r['compression_analysis']]
    sleep_ratios = [r['compression_analysis']['sleep_compression'].get('compression_ratio', 1.0) for r in all_results if 'sleep_compression' in r['compression_analysis'] and r['compression_analysis']['sleep_compression']]
    
    if hr_ratios:
        print(f"HR Compression    - Mean: {np.mean(hr_ratios):.3f}, Std: {np.std(hr_ratios):.3f}")
    if hrv_ratios:
        print(f"HRV Compression   - Mean: {np.mean(hrv_ratios):.3f}, Std: {np.std(hrv_ratios):.3f}")
    if sleep_ratios:
        print(f"Sleep Compression - Mean: {np.mean(sleep_ratios):.3f}, Std: {np.std(sleep_ratios):.3f}")
    
    print(f"\nEmpty Dictionary Entries: {len(compressor.dictionary)}")
    
    # Show data source breakdown
    activity_count = sum(1 for r in all_results if r.get('data_source') == 'activity')
    sleep_count = sum(1 for r in all_results if r.get('data_source') == 'sleep')
    print(f"Data sources: {activity_count} activity records, {sleep_count} sleep records")
    
    print("✅ Analysis complete!")

if __name__ == "__main__":
    main()
