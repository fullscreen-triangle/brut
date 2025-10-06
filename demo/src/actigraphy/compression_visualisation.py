import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import json
import urllib.request
import os
from datetime import datetime
from pathlib import Path

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2


def load_json_from_url(url):
    """Load JSON data from URL"""
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
    return data


def create_compression_visualization(df):
    """
    Creates comprehensive visualization for compression analysis
    """
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']

    # Extract compression data
    hr_compression = df['compression_analysis'].apply(lambda x: x['hr_compression']['compression_ratio'])
    hrv_compression = df['compression_analysis'].apply(lambda x: x['hrv_compression']['compression_ratio'])
    sleep_compression = df['compression_analysis'].apply(lambda x: x['sleep_compression']['compression_ratio'])

    # Panel A: Compression Ratios Comparison
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(df))
    width = 0.25

    ax1.bar(x - width, hr_compression, width, label='HR Compression',
            color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.bar(x, hrv_compression, width, label='HRV Compression',
            color=colors[1], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.bar(x + width, sleep_compression, width, label='Sleep Compression',
            color=colors[2], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_xlabel('Period ID', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Compression Ratio', fontweight='bold', fontsize=12)
    ax1.set_title('A) Compression Ratios Across Data Types',
                  fontweight='bold', loc='left', pad=15, fontsize=13)
    ax1.legend(frameon=True, fancybox=True, shadow=True, loc='upper right')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Average Compression by Type
    ax2 = fig.add_subplot(gs[0, 2])
    avg_compressions = [hr_compression.mean(), hrv_compression.mean(), sleep_compression.mean()]
    labels = ['HR', 'HRV', 'Sleep']

    bars = ax2.bar(labels, avg_compressions, color=colors[:3],
                   alpha=0.8, edgecolor='black', linewidth=2)

    for i, (bar, val) in enumerate(zip(bars, avg_compressions)):
        ax2.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                 f'{val:.4f}', ha='center', va='bottom',
                 fontweight='bold', fontsize=11)

    ax2.set_ylabel('Average Compression Ratio', fontweight='bold', fontsize=11)
    ax2.set_title('B) Average Compression', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Panel C: Ambiguous Bits Percentage
    ax3 = fig.add_subplot(gs[1, :2])
    hr_ambig = df['compression_analysis'].apply(lambda x: x['hr_compression']['ambiguous_bits']['percentage'])
    hrv_ambig = df['compression_analysis'].apply(lambda x: x['hrv_compression']['ambiguous_bits']['percentage'])
    sleep_ambig = df['compression_analysis'].apply(lambda x: x['sleep_compression']['ambiguous_bits']['percentage'])

    ax3.plot(df['period_id'], hr_ambig, 'o-', linewidth=3, markersize=10,
             label='HR', color=colors[0], markerfacecolor='white',
             markeredgewidth=2.5, markeredgecolor=colors[0])
    ax3.plot(df['period_id'], hrv_ambig, 's-', linewidth=3, markersize=10,
             label='HRV', color=colors[1], markerfacecolor='white',
             markeredgewidth=2.5, markeredgecolor=colors[1])
    ax3.plot(df['period_id'], sleep_ambig, '^-', linewidth=3, markersize=10,
             label='Sleep', color=colors[2], markerfacecolor='white',
             markeredgewidth=2.5, markeredgecolor=colors[2])

    ax3.set_xlabel('Period ID', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Ambiguous Bits (%)', fontweight='bold', fontsize=12)
    ax3.set_title('C) Ambiguous Bits Percentage Over Time',
                  fontweight='bold', loc='left', pad=15, fontsize=13)
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Panel D: Ambiguous Entropy
    ax4 = fig.add_subplot(gs[1, 2])
    hr_entropy = df['compression_analysis'].apply(lambda x: x['hr_compression']['ambiguous_bits']['ambiguous_entropy'])
    hrv_entropy = df['compression_analysis'].apply(
        lambda x: x['hrv_compression']['ambiguous_bits']['ambiguous_entropy'])
    sleep_entropy = df['compression_analysis'].apply(
        lambda x: x['sleep_compression']['ambiguous_bits']['ambiguous_entropy'])

    entropy_data = [hr_entropy.values, hrv_entropy.values, sleep_entropy.values]
    bp = ax4.boxplot(entropy_data, labels=['HR', 'HRV', 'Sleep'],
                     patch_artist=True, widths=0.6,
                     boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=2),
                     medianprops=dict(color='red', linewidth=3),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2))

    for patch, color in zip(bp['boxes'], colors[:3]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax4.set_ylabel('Ambiguous Entropy', fontweight='bold', fontsize=11)
    ax4.set_title('D) Entropy Distribution', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Panel E: Resistant Patterns Count
    ax5 = fig.add_subplot(gs[2, 0])
    hr_patterns = df['compression_analysis'].apply(lambda x: len(x['hr_compression']['resistant_patterns']))
    hrv_patterns = df['compression_analysis'].apply(lambda x: len(x['hrv_compression']['resistant_patterns']))
    sleep_patterns = df['compression_analysis'].apply(lambda x: len(x['sleep_compression']['resistant_patterns']))

    pattern_data = [hr_patterns.mean(), hrv_patterns.mean(), sleep_patterns.mean()]
    bars = ax5.bar(['HR', 'HRV', 'Sleep'], pattern_data, color=colors[:3],
                   alpha=0.8, edgecolor='black', linewidth=2)

    for bar, val in zip(bars, pattern_data):
        ax5.text(bar.get_x() + bar.get_width() / 2, val + 0.2,
                 f'{val:.1f}', ha='center', va='bottom',
                 fontweight='bold', fontsize=11)

    ax5.set_ylabel('Avg Resistant Patterns', fontweight='bold', fontsize=11)
    ax5.set_title('E) Resistant Patterns', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--')

    # Panel F: Bit Length Comparison
    ax6 = fig.add_subplot(gs[2, 1])
    hr_bitlen = df['compression_analysis'].apply(lambda x: x['hr_compression']['bit_length'])
    hrv_bitlen = df['compression_analysis'].apply(lambda x: x['hrv_compression']['bit_length'])
    sleep_bitlen = df['compression_analysis'].apply(lambda x: x['sleep_compression']['bit_length'])

    x = np.arange(len(df))
    ax6.plot(x, hr_bitlen, 'o-', linewidth=2.5, markersize=8, label='HR', color=colors[0])
    ax6.plot(x, hrv_bitlen, 's-', linewidth=2.5, markersize=8, label='HRV', color=colors[1])
    ax6.plot(x, sleep_bitlen, '^-', linewidth=2.5, markersize=8, label='Sleep', color=colors[2])

    ax6.set_xlabel('Period ID', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Bit Length', fontweight='bold', fontsize=11)
    ax6.set_title('F) Bit Length Comparison', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax6.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

    # Panel G: Original Length vs Bit Length
    ax7 = fig.add_subplot(gs[2, 2])
    hr_origlen = df['compression_analysis'].apply(lambda x: x['hr_compression']['original_length'])

    ax7.scatter(hr_origlen, hr_bitlen, s=150, alpha=0.6, color=colors[0],
                edgecolors='black', linewidths=2, label='HR')

    # Add trend line
    z = np.polyfit(hr_origlen, hr_bitlen, 1)
    p = np.poly1d(z)
    ax7.plot(hr_origlen, p(hr_origlen), "r--", linewidth=2.5, alpha=0.7, label='Trend')

    ax7.set_xlabel('Original Length', fontweight='bold', fontsize=11)
    ax7.set_ylabel('Bit Length', fontweight='bold', fontsize=11)
    ax7.set_title('G) Length Correlation', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax7.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax7.grid(True, alpha=0.3, linestyle='--')
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)

    # Panel H: Heatmap of Compression Metrics
    ax8 = fig.add_subplot(gs[3, :])

    heatmap_data = np.array([
        hr_compression.values,
        hrv_compression.values,
        sleep_compression.values,
        hr_ambig.values / 100,  # Normalize to 0-1
        hrv_ambig.values / 100,
        sleep_ambig.values / 100
    ])

    im = ax8.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')

    ax8.set_yticks(np.arange(6))
    ax8.set_yticklabels(['HR Compression', 'HRV Compression', 'Sleep Compression',
                         'HR Ambiguity', 'HRV Ambiguity', 'Sleep Ambiguity'],
                        fontweight='bold')
    ax8.set_xlabel('Period ID', fontweight='bold', fontsize=12)
    ax8.set_title('H) Compression Metrics Heatmap',
                  fontweight='bold', loc='left', pad=15, fontsize=13)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax8, fraction=0.046, pad=0.04)
    cbar.set_label('Value (Normalized)', fontweight='bold', fontsize=11)

    # Add grid
    for i in range(len(heatmap_data) + 1):
        ax8.axhline(i - 0.5, color='black', linewidth=1)

    plt.suptitle('Compression Analysis: Comprehensive Visualization',
                 fontsize=16, fontweight='bold', y=0.995)

    return fig


def main():
    project_root = Path(__file__).parent.parent.parent  # From src/actigraphy/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "compression_results.json"  # Change this for different activity files


    # Construct paths
    activity_file_path = project_root / "public" / activity_data_file


    # Convert to strings for compatibility
    activity_file_path = str(activity_file_path)

    print("Loading data from URL...")


    # Convert to DataFrame
    df = pd.read_json(activity_file_path)



    print(f"Data loaded successfully! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print("Creating compression visualizations...")
    fig = create_compression_visualization(df)

    plt.savefig('compression_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('compression_analysis.pdf', bbox_inches='tight', facecolor='white')
    print("Compression visualizations saved successfully!")

    plt.show()


if __name__ == "__main__":
    main()
