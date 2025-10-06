import os
from datetime import datetime
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import json
import urllib.request
from collections import Counter

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2


def load_json_from_url(url):
    """Load JSON data from URL"""
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
    return data


def create_directional_mapping_visualization(df):
    """
    Creates comprehensive visualization for directional mapping analysis
    """
    fig = plt.figure(figsize=(20, 16))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']
    direction_colors = {'A': '#FF6B6B', 'R': '#4ECDC4', 'D': '#45B7D1', 'L': '#FFA07A'}

    # Extract first record for detailed analysis
    first_record = df.iloc[0]
    hr_seq = first_record['hr_directional_sequence']
    hrv_seq = first_record['hrv_directional_sequence']
    sleep_seq = first_record['sleep_directional_sequence']

    # Panel A: HR Direction Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    hr_dist = first_record['directional_analysis']['hr_analysis']['distribution']

    wedges, texts, autotexts = ax1.pie(hr_dist.values(), labels=hr_dist.keys(),
                                       autopct='%1.1f%%', startangle=90,
                                       colors=[direction_colors[k] for k in hr_dist.keys()],
                                       wedgeprops=dict(edgecolor='black', linewidth=2),
                                       textprops=dict(fontweight='bold', fontsize=11))

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')

    ax1.set_title('A) HR Direction Distribution', fontweight='bold', pad=15, fontsize=13)

    # Panel B: HRV Direction Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    hrv_dist = first_record['directional_analysis']['hrv_analysis']['distribution']

    wedges, texts, autotexts = ax2.pie(hrv_dist.values(), labels=hrv_dist.keys(),
                                       autopct='%1.1f%%', startangle=90,
                                       colors=[direction_colors[k] for k in hrv_dist.keys()],
                                       wedgeprops=dict(edgecolor='black', linewidth=2),
                                       textprops=dict(fontweight='bold', fontsize=11))

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')

    ax2.set_title('B) HRV Direction Distribution', fontweight='bold', pad=15, fontsize=13)

    # Panel C: Sleep Direction Distribution
    ax3 = fig.add_subplot(gs[0, 2])
    sleep_dist = first_record['directional_analysis']['sleep_analysis']['distribution']

    wedges, texts, autotexts = ax3.pie(sleep_dist.values(), labels=sleep_dist.keys(),
                                       autopct='%1.1f%%', startangle=90,
                                       colors=[direction_colors[k] for k in sleep_dist.keys()],
                                       wedgeprops=dict(edgecolor='black', linewidth=2),
                                       textprops=dict(fontweight='bold', fontsize=11))

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')

    ax3.set_title('C) Sleep Direction Distribution', fontweight='bold', pad=15, fontsize=13)

    # Panel D: HR Entropy Comparison
    ax4 = fig.add_subplot(gs[1, :])
    hr_entropy = df['directional_analysis'].apply(lambda x: x['hr_analysis']['entropy'])
    hrv_entropy = df['directional_analysis'].apply(lambda x: x['hrv_analysis']['entropy'])
    sleep_entropy = df['directional_analysis'].apply(lambda x: x['sleep_analysis']['entropy'])

    x = np.arange(len(df))
    width = 0.25

    ax4.bar(x - width, hr_entropy, width, label='HR Entropy',
            color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.bar(x, hrv_entropy, width, label='HRV Entropy',
            color=colors[1], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax4.bar(x + width, sleep_entropy, width, label='Sleep Entropy',
            color=colors[2], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax4.set_xlabel('Period ID', fontweight='bold', fontsize=12)
    ax4.set_ylabel('Entropy', fontweight='bold', fontsize=12)
    ax4.set_title('D) Entropy Comparison Across Data Types',
                  fontweight='bold', loc='left', pad=15, fontsize=13)
    ax4.legend(frameon=True, fancybox=True, shadow=True)
    ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Panel E: HR Transition Matrix Heatmap
    ax5 = fig.add_subplot(gs[2, 0])
    hr_trans_prob = first_record['directional_analysis']['hr_analysis']['transitions']['transition_probabilities']

    # Create transition matrix
    directions = ['A', 'R', 'D', 'L']
    trans_matrix = np.zeros((4, 4))

    for key, prob in hr_trans_prob.items():
        from_dir, to_dir = key.split('->')
        i = directions.index(from_dir)
        j = directions.index(to_dir)
        trans_matrix[i, j] = prob

    im = ax5.imshow(trans_matrix, cmap='YlOrRd', aspect='auto')
    ax5.set_xticks(np.arange(4))
    ax5.set_yticks(np.arange(4))
    ax5.set_xticklabels(directions, fontweight='bold')
    ax5.set_yticklabels(directions, fontweight='bold')
    ax5.set_xlabel('To Direction', fontweight='bold', fontsize=11)
    ax5.set_ylabel('From Direction', fontweight='bold', fontsize=11)
    ax5.set_title('E) HR Transition Matrix', fontweight='bold', loc='left', pad=15, fontsize=13)

    # Add text annotations
    for i in range(4):
        for j in range(4):
            if trans_matrix[i, j] > 0:
                text = ax5.text(j, i, f'{trans_matrix[i, j]:.3f}',
                              ha="center", va="center", color="black" if trans_matrix[i, j] < 0.3 else "white",
                              fontweight='bold', fontsize=9)

    plt.colorbar(im, ax=ax5, fraction=0.046, pad=0.04)

    # Panel F: HRV Transition Matrix Heatmap
    ax6 = fig.add_subplot(gs[2, 1])
    hrv_trans_prob = first_record['directional_analysis']['hrv_analysis']['transitions']['transition_probabilities']

    trans_matrix_hrv = np.zeros((4, 4))
    for key, prob in hrv_trans_prob.items():
        from_dir, to_dir = key.split('->')
        i = directions.index(from_dir)
        j = directions.index(to_dir)
        trans_matrix_hrv[i, j] = prob

    im = ax6.imshow(trans_matrix_hrv, cmap='YlOrRd', aspect='auto')
    ax6.set_xticks(np.arange(4))
    ax6.set_yticks(np.arange(4))
    ax6.set_xticklabels(directions, fontweight='bold')
    ax6.set_yticklabels(directions, fontweight='bold')
    ax6.set_xlabel('To Direction', fontweight='bold', fontsize=11)
    ax6.set_ylabel('From Direction', fontweight='bold', fontsize=11)
    ax6.set_title('F) HRV Transition Matrix', fontweight='bold', loc='left', pad=15, fontsize=13)

    for i in range(4):
        for j in range(4):
            if trans_matrix_hrv[i, j] > 0:
                text = ax6.text(j, i, f'{trans_matrix_hrv[i, j]:.3f}',
                              ha="center", va="center", color="black" if trans_matrix_hrv[i, j] < 0.3 else "white",
                              fontweight='bold', fontsize=9)

    plt.colorbar(im, ax=ax6, fraction=0.046, pad=0.04)

    # Panel G: Sleep Transition Matrix Heatmap
    ax7 = fig.add_subplot(gs[2, 2])
    sleep_trans_prob = first_record['directional_analysis']['sleep_analysis']['transitions']['transition_probabilities']

    trans_matrix_sleep = np.zeros((4, 4))
    for key, prob in sleep_trans_prob.items():
        from_dir, to_dir = key.split('->')
        i = directions.index(from_dir)
        j = directions.index(to_dir)
        trans_matrix_sleep[i, j] = prob

    im = ax7.imshow(trans_matrix_sleep, cmap='YlOrRd', aspect='auto')
    ax7.set_xticks(np.arange(4))
    ax7.set_yticks(np.arange(4))
    ax7.set_xticklabels(directions, fontweight='bold')
    ax7.set_yticklabels(directions, fontweight='bold')
    ax7.set_xlabel('To Direction', fontweight='bold', fontsize=11)
    ax7.set_ylabel('From Direction', fontweight='bold', fontsize=11)
    ax7.set_title('G) Sleep Transition Matrix', fontweight='bold', loc='left', pad=15, fontsize=13)

    for i in range(4):
        for j in range(4):
            if trans_matrix_sleep[i, j] > 0:
                text = ax7.text(j, i, f'{trans_matrix_sleep[i, j]:.3f}',
                              ha="center", va="center", color="black" if trans_matrix_sleep[i, j] < 0.3 else "white",
                              fontweight='bold', fontsize=9)

    plt.colorbar(im, ax=ax7, fraction=0.046, pad=0.04)

    # Panel H: Top Transitions Comparison
    ax8 = fig.add_subplot(gs[3, :2])

    hr_top = first_record['directional_analysis']['hr_analysis']['transitions']['most_common_transitions'][:5]
    hrv_top = first_record['directional_analysis']['hrv_analysis']['transitions']['most_common_transitions'][:5]
    sleep_top = first_record['directional_analysis']['sleep_analysis']['transitions']['most_common_transitions'][:5]

    y_pos = np.arange(5)
    width = 0.25

    hr_vals = [t[1] for t in hr_top]
    hrv_vals = [t[1] for t in hrv_top]
    sleep_vals = [t[1] for t in sleep_top]

    ax8.barh(y_pos - width, hr_vals, width, label='HR',
             color=colors[0], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax8.barh(y_pos, hrv_vals, width, label='HRV',
             color=colors[1], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax8.barh(y_pos + width, sleep_vals, width, label='Sleep',
             color=colors[2], alpha=0.8, edgecolor='black', linewidth=1.5)

    ax8.set_yticks(y_pos)
    ax8.set_yticklabels([f'Top {i+1}' for i in range(5)], fontweight='bold')
    ax8.set_xlabel('Transition Probability', fontweight='bold', fontsize=12)
    ax8.set_title('H) Top 5 Most Common Transitions',
                  fontweight='bold', loc='left', pad=15, fontsize=13)
    ax8.legend(frameon=True, fancybox=True, shadow=True)
    ax8.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)

    # Panel I: Sequence Visualization (First 50 positions)
    ax9 = fig.add_subplot(gs[3, 2])

    seq_length = min(50, len(hr_seq))
    x_seq = np.arange(seq_length)

    # Convert directions to numeric values for plotting
    dir_map = {'A': 3, 'R': 2, 'D': 1, 'L': 0}
    hr_numeric = [dir_map[d] for d in hr_seq[:seq_length]]
    hrv_numeric = [dir_map[d] for d in hrv_seq[:seq_length]]
    sleep_numeric = [dir_map[d] for d in sleep_seq[:seq_length]]

    ax9.plot(x_seq, hr_numeric, 'o-', linewidth=2, markersize=4,
            label='HR', color=colors[0], alpha=0.7)
    ax9.plot(x_seq, hrv_numeric, 's-', linewidth=2, markersize=4,
            label='HRV', color=colors[1], alpha=0.7)
    ax9.plot(x_seq, sleep_numeric, '^-', linewidth=2, markersize=4,
            label='Sleep', color=colors[2], alpha=0.7)

    ax9.set_yticks([0, 1, 2, 3])
    ax9.set_yticklabels(['L', 'D', 'R', 'A'], fontweight='bold')
    ax9.set_xlabel('Position', fontweight='bold', fontsize=11)
    ax9.set_ylabel('Direction', fontweight='bold', fontsize=11)
    ax9.set_title('I) Sequence Pattern (First 50)',
                  fontweight='bold', loc='left', pad=15, fontsize=13)
    ax9.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax9.grid(True, alpha=0.3, linestyle='--')
    ax9.spines['top'].set_visible(False)
    ax9.spines['right'].set_visible(False)

    plt.suptitle('Directional Mapping Analysis: Comprehensive Visualization',
                 fontsize=16, fontweight='bold', y=0.995)

    return fig

def main():
    """Main function to load data and create visualizations"""
    project_root = Path(__file__).parent.parent.parent  # From src/actigraphy/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    data_file = "directional_mapping_results.json"  # Change this for different files

    # Construct paths
    file_path = project_root / "public" / data_file

    # Convert to strings for compatibility
    file_path = str(file_path)

    print("Loading directional mapping data from file...")
    df = pd.read_json(file_path)

    print(f"Data loaded successfully! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print("Creating directional mapping visualizations...")
    fig = create_directional_mapping_visualization(df)

    plt.savefig('directional_mapping_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('directional_mapping_analysis.pdf', bbox_inches='tight', facecolor='white')
    print("Directional mapping visualizations saved successfully!")

    plt.show()

if __name__ == "__main__":
    main()
