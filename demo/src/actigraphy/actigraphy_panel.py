import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import json
from datetime import datetime, timezone
from pathlib import Path
import matplotlib.dates as mdates

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

def directional_encoding(data):
    """Convert numerical data to directional sequences (A=Up, D=Down, L=Left, R=Right)"""
    if len(data) < 2:
        return ""

    sequence = []
    for i in range(1, len(data)):
        diff = data[i] - data[i-1]
        if diff > 0.2:  # Significant increase
            sequence.append('A')  # Up/North
        elif diff < -0.2:  # Significant decrease
            sequence.append('D')  # Down/South
        elif diff > 0:  # Small increase
            sequence.append('R')  # Right/East
        else:  # Small decrease or no change
            sequence.append('L')  # Left/West

    return ''.join(sequence)

def ambiguous_compression_analysis(sequence):
    """Perform basic ambiguous compression analysis on directional sequence"""
    if not sequence:
        return {
            'original_length': 0,
            'compressed_length': 0,
            'compression_ratio': 0,
            'directional_distribution': {},
            'entropy': 0
        }

    # Count directional distributions
    directions = {'A': 0, 'D': 0, 'L': 0, 'R': 0}
    for char in sequence:
        if char in directions:
            directions[char] += 1

    # Calculate entropy
    total = len(sequence)
    entropy = 0
    for count in directions.values():
        if count > 0:
            p = count / total
            entropy -= p * np.log2(p)

    # Simple compression simulation (run-length encoding)
    compressed = []
    current_char = sequence[0]
    count = 1

    for i in range(1, len(sequence)):
        if sequence[i] == current_char:
            count += 1
        else:
            compressed.append(f"{current_char}{count}")
            current_char = sequence[i]
            count = 1
    compressed.append(f"{current_char}{count}")

    compressed_str = ''.join(compressed)
    compression_ratio = len(compressed_str) / len(sequence) if len(sequence) > 0 else 0

    return {
        'original_length': len(sequence),
        'compressed_length': len(compressed_str),
        'compression_ratio': compression_ratio,
        'directional_distribution': directions,
        'entropy': entropy,
        'compressed_sequence': compressed_str[:50] + '...' if len(compressed_str) > 50 else compressed_str
    }

def create_raw_data_panel_1(data):
    """Panel 1: Basic Metrics and Time Series Analysis"""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 4, figure=fig, hspace=0.4, wspace=0.35)

    colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', '#A8DADC']

    # Convert timestamps to datetime
    day_start = datetime.fromtimestamp(data['day_start_dt_adjusted'] / 1000, tz=timezone.utc)
    day_end = datetime.fromtimestamp(data['day_end_dt_adjusted'] / 1000, tz=timezone.utc)

    # Panel A: Daily Overview
    ax1 = fig.add_subplot(gs[0, :2])
    metrics = ['Steps', 'Cal Active', 'Cal Total', 'Daily Movement']
    values = [data['steps'], data['cal_active'], data['cal_total'], data['daily_movement']]

    bars = ax1.bar(metrics, values, color=colors[:4], alpha=0.8, edgecolor='black', linewidth=2)
    for bar, val in zip(bars, values):
        ax1.text(bar.get_x() + bar.get_width()/2, val + max(values) * 0.01,
                 f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=11)

    ax1.set_ylabel('Count/Calories', fontweight='bold', fontsize=12)
    ax1.set_title('A) Daily Activity Overview', fontweight='bold', loc='left', pad=15, fontsize=14)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Activity Intensity Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    intensity_labels = ['Rest', 'Inactive', 'Low', 'Medium', 'High']
    intensity_values = [data['rest'], data['inactive'], data['low'], data['medium'], data['high']]

    wedges, texts, autotexts = ax2.pie(intensity_values, labels=intensity_labels,
                                       autopct='%1.1f%%', startangle=90,
                                       colors=colors[:5],
                                       wedgeprops=dict(edgecolor='black', linewidth=1.5))

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(10)
        autotext.set_fontweight('bold')

    ax2.set_title('B) Activity Intensity\nDistribution (minutes)', fontweight='bold', pad=15, fontsize=14)

    # Panel C: MET Score Analysis
    ax3 = fig.add_subplot(gs[0, 3])
    met_categories = ['Inactive', 'Low', 'Medium', 'High']
    met_values = [data['met_min_inactive'], data['met_min_low'], data['met_min_medium'], data['met_min_high']]

    bars = ax3.bar(met_categories, met_values, color=colors[1:5], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax3.axhline(y=data['average_met'], color='red', linestyle='--', linewidth=2,
                label=f'Avg MET: {data["average_met"]:.2f}')

    for bar, val in zip(bars, met_values):
        if val > 0:
            ax3.text(bar.get_x() + bar.get_width()/2, val + 0.1,
                     f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax3.set_ylabel('MET Minutes', fontweight='bold', fontsize=11)
    ax3.set_title('C) MET Distribution', fontweight='bold', loc='left', pad=15, fontsize=14)
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Panel D: MET 1-minute Time Series
    ax4 = fig.add_subplot(gs[1, :])
    met_data = data['met_1min']
    time_points = np.arange(len(met_data))

    ax4.plot(time_points, met_data, color=colors[0], linewidth=1.5, alpha=0.8)
    ax4.fill_between(time_points, met_data, alpha=0.3, color=colors[0])

    # Highlight non-wear periods (MET = 0.1)
    non_wear_mask = np.array(met_data) <= 0.1
    if np.any(non_wear_mask):
        ax4.fill_between(time_points, 0, max(met_data), where=non_wear_mask,
                         alpha=0.5, color='red', label='Non-wear periods')

    ax4.axhline(y=data['average_met'], color='red', linestyle='--', linewidth=2,
                label=f'Daily Average: {data["average_met"]:.2f}')

    ax4.set_xlabel('Time (minutes from day start)', fontweight='bold', fontsize=12)
    ax4.set_ylabel('MET Value', fontweight='bold', fontsize=12)
    ax4.set_title('D) MET 1-Minute Time Series (Full Day)', fontweight='bold', loc='left', pad=15, fontsize=14)
    ax4.legend(loc='upper right')
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Panel E: Class 5-minute Visualization
    ax5 = fig.add_subplot(gs[2, :])
    class_5min = data['class_5min']
    class_values = [int(c) for c in class_5min]
    time_5min = np.arange(len(class_values)) * 5  # 5-minute intervals

    # Create color map for different activity classes
    class_colors = {0: '#FF0000', 1: '#FFA500', 2: '#FFFF00', 3: '#00FF00', 4: '#0000FF'}
    colors_mapped = [class_colors.get(val, '#808080') for val in class_values]

    ax5.bar(time_5min, class_values, width=4, color=colors_mapped, alpha=0.8, edgecolor='black', linewidth=0.5)

    ax5.set_xlabel('Time (minutes from day start)', fontweight='bold', fontsize=12)
    ax5.set_ylabel('Activity Class', fontweight='bold', fontsize=12)
    ax5.set_title('E) Activity Classification (5-minute intervals)', fontweight='bold', loc='left', pad=15, fontsize=14)
    ax5.set_yticks([0, 1, 2, 3, 4])
    ax5.set_yticklabels(['Sleep', 'Rest', 'Inactive', 'Low', 'Medium+'])
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # Panel F: Score Analysis
    ax6 = fig.add_subplot(gs[3, :2])
    score_categories = ['Daily Targets', 'Move Every Hour', 'Recovery Time', 'Stay Active', 'Training Freq', 'Training Vol']
    score_values = [data['score_meet_daily_targets'], data['score_move_every_hour'],
                   data['score_recovery_time'], data['score_stay_active'],
                   data['score_training_frequency'], data['score_training_volume']]

    bars = ax6.barh(score_categories, score_values, color=colors[:6], alpha=0.8, edgecolor='black', linewidth=1.5)
    ax6.axvline(x=data['score'], color='red', linestyle='--', linewidth=3,
                label=f'Overall Score: {data["score"]}')

    for bar, val in zip(bars, score_values):
        ax6.text(val + 1, bar.get_y() + bar.get_height()/2,
                 f'{val}', ha='left', va='center', fontweight='bold', fontsize=10)

    ax6.set_xlabel('Score (0-100)', fontweight='bold', fontsize=12)
    ax6.set_title('F) Performance Scores', fontweight='bold', loc='left', pad=15, fontsize=14)
    ax6.legend(loc='lower right')
    ax6.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)
    ax6.set_xlim(0, 105)

    # Panel G: Summary Statistics
    ax7 = fig.add_subplot(gs[3, 2:])
    ax7.axis('off')

    summary_text = f"""
    RAW DATA SUMMARY
    {'=' * 40}

    TEMPORAL INFORMATION:
    Timezone Offset:     {data['timezone']} minutes
    Day Duration:        {(data['day_end_dt_adjusted'] - data['day_start_dt_adjusted']) / (1000 * 60 * 60):.1f} hours
    Non-wear Time:       {data['non_wear']} minutes

    ACTIVITY METRICS:
    Total Steps:         {data['steps']:,}
    Daily Movement:      {data['daily_movement']}
    Active Calories:     {data['cal_active']}
    Total Calories:      {data['cal_total']}

    MET ANALYSIS:
    Average MET:         {data['average_met']:.3f}
    MET Data Points:     {len(data['met_1min'])}
    Class Data Points:   {len(data['class_5min'])}

    TARGETS:
    Target Calories:     {data['target_calories']}
    Target Distance:     {data['target_km']:.1f} km
    Distance to Target:  {data['to_target_km']:.1f} km

    OVERALL SCORE:       {data['score']}/100
    """

    ax7.text(0.05, 0.95, summary_text, transform=ax7.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8,
                      edgecolor='black', linewidth=2))

    plt.suptitle('Raw Actigraphy Data: Comprehensive Analysis Panel 1',
                 fontsize=18, fontweight='bold', y=0.98)

    return fig

def create_raw_data_panel_2(data):
    """Panel 2: Directional Encoding and Compression Analysis"""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', '#A8DADC']

    # Generate directional sequences
    met_sequence = directional_encoding(data['met_1min'])
    class_sequence = directional_encoding([int(c) for c in data['class_5min']])

    # Perform compression analysis
    met_compression = ambiguous_compression_analysis(met_sequence)
    class_compression = ambiguous_compression_analysis(class_sequence)

    # Panel A: MET Directional Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    met_dirs = met_compression['directional_distribution']
    directions = ['A (Up)', 'D (Down)', 'L (Left)', 'R (Right)']
    dir_values = [met_dirs['A'], met_dirs['D'], met_dirs['L'], met_dirs['R']]

    bars = ax1.bar(directions, dir_values, color=colors[:4], alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, dir_values):
        if val > 0:
            ax1.text(bar.get_x() + bar.get_width()/2, val + max(dir_values) * 0.01,
                     f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax1.set_ylabel('Count', fontweight='bold', fontsize=11)
    ax1.set_title('A) MET Directional\nDistribution', fontweight='bold', loc='left', pad=15, fontsize=12)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Class Directional Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    class_dirs = class_compression['directional_distribution']
    class_dir_values = [class_dirs['A'], class_dirs['D'], class_dirs['L'], class_dirs['R']]

    bars = ax2.bar(directions, class_dir_values, color=colors[:4], alpha=0.8, edgecolor='black', linewidth=1.5)
    for bar, val in zip(bars, class_dir_values):
        if val > 0:
            ax2.text(bar.get_x() + bar.get_width()/2, val + max(class_dir_values) * 0.01,
                     f'{val}', ha='center', va='bottom', fontweight='bold', fontsize=10)

    ax2.set_ylabel('Count', fontweight='bold', fontsize=11)
    ax2.set_title('B) Class Directional\nDistribution', fontweight='bold', loc='left', pad=15, fontsize=12)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel C: Compression Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    comp_metrics = ['Original Length', 'Compressed Length', 'Compression Ratio', 'Entropy']
    met_values = [met_compression['original_length'], met_compression['compressed_length'],
                  met_compression['compression_ratio'] * 100, met_compression['entropy']]
    class_values = [class_compression['original_length'], class_compression['compressed_length'],
                   class_compression['compression_ratio'] * 100, class_compression['entropy']]

    x = np.arange(len(comp_metrics))
    width = 0.35

    bars1 = ax3.bar(x - width/2, met_values, width, label='MET', color=colors[0], alpha=0.8, edgecolor='black')
    bars2 = ax3.bar(x + width/2, class_values, width, label='Class', color=colors[1], alpha=0.8, edgecolor='black')

    ax3.set_ylabel('Value', fontweight='bold', fontsize=11)
    ax3.set_title('C) Compression\nComparison', fontweight='bold', loc='left', pad=15, fontsize=12)
    ax3.set_xticks(x)
    ax3.set_xticklabels(comp_metrics, rotation=45, ha='right')
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Panel D: MET Sequence Visualization (first 100 points)
    ax4 = fig.add_subplot(gs[1, :])
    sequence_sample = met_sequence[:100] if len(met_sequence) > 100 else met_sequence

    if sequence_sample:
        # Convert sequence to numerical for plotting
        dir_map = {'A': 3, 'D': 1, 'L': 0, 'R': 2}
        seq_values = [dir_map[d] for d in sequence_sample]

        ax4.plot(range(len(seq_values)), seq_values, 'o-', color=colors[0],
                linewidth=2, markersize=6, alpha=0.8)
        ax4.fill_between(range(len(seq_values)), seq_values, alpha=0.3, color=colors[0])

        ax4.set_yticks([0, 1, 2, 3])
        ax4.set_yticklabels(['L (Left)', 'D (Down)', 'R (Right)', 'A (Up)'])
        ax4.set_xlabel('Sequence Position', fontweight='bold', fontsize=12)
        ax4.set_ylabel('Direction', fontweight='bold', fontsize=12)
        ax4.set_title(f'D) MET Directional Sequence (First {len(sequence_sample)} transitions)',
                     fontweight='bold', loc='left', pad=15, fontsize=14)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

    # Panel E: Class Sequence Visualization
    ax5 = fig.add_subplot(gs[2, :])
    class_seq_sample = class_sequence[:50] if len(class_sequence) > 50 else class_sequence

    if class_seq_sample:
        seq_values = [dir_map[d] for d in class_seq_sample]

        ax5.plot(range(len(seq_values)), seq_values, 's-', color=colors[1],
                linewidth=2, markersize=8, alpha=0.8)
        ax5.fill_between(range(len(seq_values)), seq_values, alpha=0.3, color=colors[1])

        ax5.set_yticks([0, 1, 2, 3])
        ax5.set_yticklabels(['L (Left)', 'D (Down)', 'R (Right)', 'A (Up)'])
        ax5.set_xlabel('Sequence Position', fontweight='bold', fontsize=12)
        ax5.set_ylabel('Direction', fontweight='bold', fontsize=12)
        ax5.set_title(f'E) Class Directional Sequence (First {len(class_seq_sample)} transitions)',
                     fontweight='bold', loc='left', pad=15, fontsize=14)
        ax5.grid(True, alpha=0.3, linestyle='--')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)

    # Panel F: Compression Analysis Summary
    ax6 = fig.add_subplot(gs[3, :])
    ax6.axis('off')

    compression_text = f"""
    DIRECTIONAL ENCODING & COMPRESSION ANALYSIS
    {'=' * 60}

    MET SEQUENCE ANALYSIS:
    Original Length:        {met_compression['original_length']} transitions
    Compressed Length:      {met_compression['compressed_length']} characters
    Compression Ratio:      {met_compression['compression_ratio']:.4f}
    Entropy:               {met_compression['entropy']:.4f} bits
    Directional Counts:    A={met_dirs['A']}, D={met_dirs['D']}, L={met_dirs['L']}, R={met_dirs['R']}
    Compressed Sample:     {met_compression['compressed_sequence']}

    CLASS SEQUENCE ANALYSIS:
    Original Length:        {class_compression['original_length']} transitions
    Compressed Length:      {class_compression['compressed_length']} characters
    Compression Ratio:      {class_compression['compression_ratio']:.4f}
    Entropy:               {class_compression['entropy']:.4f} bits
    Directional Counts:    A={class_dirs['A']}, D={class_dirs['D']}, L={class_dirs['L']}, R={class_dirs['R']}
    Compressed Sample:     {class_compression['compressed_sequence']}

    ALIGNMENT POTENTIAL:
    MET-Class Length Ratio: {met_compression['original_length'] / max(class_compression['original_length'], 1):.2f}
    Combined Entropy:       {(met_compression['entropy'] + class_compression['entropy']) / 2:.4f} bits
    """

    ax6.text(0.05, 0.95, compression_text, transform=ax6.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8,
                      edgecolor='black', linewidth=2))

    plt.suptitle('Raw Actigraphy Data: Directional Encoding & Compression Analysis Panel 2',
                 fontsize=18, fontweight='bold', y=0.98)

    return fig

def create_raw_data_panel_3(data):
    """Panel 3: Temporal Analysis and Wear Pattern Detection"""
    fig = plt.figure(figsize=(20, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51', '#A8DADC']

    # Convert timestamps and create time arrays
    day_start = datetime.fromtimestamp(data['day_start_dt_adjusted'] / 1000, tz=timezone.utc)
    day_end = datetime.fromtimestamp(data['day_end_dt_adjusted'] / 1000, tz=timezone.utc)

    met_data = data['met_1min']
    class_data = [int(c) for c in data['class_5min']]

    # Create time arrays
    met_times = [day_start.timestamp() + i * 60 for i in range(len(met_data))]  # 1-minute intervals
    class_times = [day_start.timestamp() + i * 300 for i in range(len(class_data))]  # 5-minute intervals

    # Panel A: Hourly Activity Pattern
    ax1 = fig.add_subplot(gs[0, :2])

    # Group MET data by hour
    hourly_met = []
    for hour in range(24):
        hour_start = hour * 60
        hour_end = (hour + 1) * 60
        if hour_end <= len(met_data):
            hour_data = met_data[hour_start:hour_end]
            hourly_met.append(np.mean([x for x in hour_data if x > 0.1]))  # Exclude non-wear
        else:
            hourly_met.append(0)

    hours = range(24)
    bars = ax1.bar(hours, hourly_met, color=colors[0], alpha=0.8, edgecolor='black', linewidth=1)

    # Highlight peak activity hours
    max_hour = np.argmax(hourly_met)
    bars[max_hour].set_color(colors[4])

    ax1.set_xlabel('Hour of Day', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Average MET', fontweight='bold', fontsize=12)
    ax1.set_title('A) Hourly Activity Pattern', fontweight='bold', loc='left', pad=15, fontsize=14)
    ax1.set_xticks(range(0, 24, 2))
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Non-wear Detection
    ax2 = fig.add_subplot(gs[0, 2])

    wear_status = ['Wear', 'Non-wear']
    wear_minutes = [len(met_data) - data['non_wear'], data['non_wear']]

    wedges, texts, autotexts = ax2.pie(wear_minutes, labels=wear_status,
                                       autopct='%1.1f%%', startangle=90,
                                       colors=[colors[2], colors[4]],
                                       wedgeprops=dict(edgecolor='black', linewidth=2))

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')

    ax2.set_title('B) Device Wear Pattern\n(Total Minutes)', fontweight='bold', pad=15, fontsize=14)

    # Panel C: Activity Intensity Heatmap (24-hour view)
    ax3 = fig.add_subplot(gs[1, :])

    # Create 24-hour heatmap data
    heatmap_data = np.zeros((4, 24))  # 4 intensity levels, 24 hours

    for i, class_val in enumerate(class_data):
        hour = (i * 5) // 60  # Convert 5-minute intervals to hours
        if hour < 24 and class_val <= 3:  # Only plot valid hours and classes
            heatmap_data[class_val, hour] += 1

    im = ax3.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')

    ax3.set_xticks(range(0, 24, 2))
    ax3.set_xticklabels([f'{h:02d}:00' for h in range(0, 24, 2)])
    ax3.set_yticks(range(4))
    ax3.set_yticklabels(['Sleep', 'Rest', 'Inactive', 'Low Activity'])
    ax3.set_xlabel('Hour of Day', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Activity Class', fontweight='bold', fontsize=12)
    ax3.set_title('C) Activity Intensity Heatmap (24-hour view)', fontweight='bold', loc='left', pad=15, fontsize=14)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3, fraction=0.046, pad=0.04)
    cbar.set_label('Frequency (5-min periods)', fontweight='bold', fontsize=11)

    # Panel D: MET Variability Analysis
    ax4 = fig.add_subplot(gs[2, 0])

    # Calculate rolling statistics
    window_size = 60  # 1-hour window
    if len(met_data) >= window_size:
        rolling_mean = pd.Series(met_data).rolling(window=window_size, center=True).mean()
        rolling_std = pd.Series(met_data).rolling(window=window_size, center=True).std()

        valid_indices = ~(rolling_mean.isna() | rolling_std.isna())
        hours_valid = np.array(range(len(met_data)))[valid_indices] / 60

        ax4.plot(hours_valid, rolling_mean[valid_indices], color=colors[0], linewidth=2, label='Mean')
        ax4.fill_between(hours_valid,
                        rolling_mean[valid_indices] - rolling_std[valid_indices],
                        rolling_mean[valid_indices] + rolling_std[valid_indices],
                        alpha=0.3, color=colors[0], label='±1 SD')

    ax4.set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
    ax4.set_ylabel('MET Value', fontweight='bold', fontsize=11)
    ax4.set_title('D) MET Variability\n(1-hour rolling)', fontweight='bold', loc='left', pad=15, fontsize=12)
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Panel E: Activity Transition Analysis
    ax5 = fig.add_subplot(gs[2, 1])

    # Count transitions between activity classes
    transitions = {}
    for i in range(len(class_data) - 1):
        current = class_data[i]
        next_val = class_data[i + 1]
        key = f'{current}→{next_val}'
        transitions[key] = transitions.get(key, 0) + 1

    # Get top 10 transitions
    top_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]

    if top_transitions:
        trans_labels = [t[0] for t in top_transitions]
        trans_counts = [t[1] for t in top_transitions]

        bars = ax5.barh(trans_labels, trans_counts, color=colors[1], alpha=0.8, edgecolor='black')

        for bar, count in zip(bars, trans_counts):
            ax5.text(count + max(trans_counts) * 0.01, bar.get_y() + bar.get_height()/2,
                     f'{count}', ha='left', va='center', fontweight='bold', fontsize=9)

    ax5.set_xlabel('Transition Count', fontweight='bold', fontsize=11)
    ax5.set_title('E) Activity Transitions\n(Top 10)', fontweight='bold', loc='left', pad=15, fontsize=12)
    ax5.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # Panel F: Sleep-Wake Pattern
    ax6 = fig.add_subplot(gs[2, 2])

    # Identify sleep periods (class 0)
    sleep_periods = []
    wake_periods = []

    for i, class_val in enumerate(class_data):
        hour = (i * 5) / 60  # Convert to hours
        if class_val == 0:
            sleep_periods.append(hour)
        else:
            wake_periods.append(hour)

    if sleep_periods and wake_periods:
        ax6.hist([sleep_periods, wake_periods], bins=24, alpha=0.7,
                label=['Sleep', 'Wake'], color=[colors[4], colors[2]], edgecolor='black')
        ax6.set_xlabel('Hour of Day', fontweight='bold', fontsize=11)
        ax6.set_ylabel('Frequency', fontweight='bold', fontsize=11)
        ax6.set_title('F) Sleep-Wake\nDistribution', fontweight='bold', loc='left', pad=15, fontsize=12)
        ax6.legend(fontsize=9)
        ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)

    # Panel G: Comprehensive Summary
    ax7 = fig.add_subplot(gs[3, :])
    ax7.axis('off')

    # Calculate additional statistics
    active_hours = sum(1 for x in hourly_met if x > 1.2)
    peak_activity_hour = np.argmax(hourly_met)
    total_transitions = sum(transitions.values()) if transitions else 0

    temporal_text = f"""
    TEMPORAL ANALYSIS & WEAR PATTERN SUMMARY
    {'=' * 70}

    TEMPORAL CHARACTERISTICS:
    Day Start:              {day_start.strftime('%Y-%m-%d %H:%M:%S UTC')}
    Day End:                {day_end.strftime('%Y-%m-%d %H:%M:%S UTC')}
    Total Duration:         {(day_end - day_start).total_seconds() / 3600:.1f} hours
    Data Points (MET):      {len(met_data)} (1-minute intervals)
    Data Points (Class):    {len(class_data)} (5-minute intervals)

    ACTIVITY PATTERNS:
    Peak Activity Hour:     {peak_activity_hour:02d}:00 (MET = {hourly_met[peak_activity_hour]:.2f})
    Active Hours (MET>1.2): {active_hours}/24 hours
    Total Class Transitions: {total_transitions}
    Most Common Transition: {top_transitions[0][0] if top_transitions else 'N/A'} ({top_transitions[0][1] if top_transitions else 0} times)

    WEAR PATTERN:
    Total Wear Time:        {len(met_data) - data['non_wear']} minutes ({((len(met_data) - data['non_wear']) / len(met_data) * 100):.1f}%)
    Non-wear Periods:       {data['non_wear']} minutes ({(data['non_wear'] / len(met_data) * 100):.1f}%)
    Wear Compliance:        {'Excellent' if data['non_wear'] < 60 else 'Good' if data['non_wear'] < 120 else 'Fair'}

    SLEEP ANALYSIS:
    Sleep Periods Detected: {len(sleep_periods)} (5-minute intervals)
    Wake Periods Detected:  {len(wake_periods)} (5-minute intervals)
    Sleep Percentage:       {(len(sleep_periods) / len(class_data) * 100):.1f}% of classified time
    """

    ax7.text(0.05, 0.95, temporal_text, transform=ax7.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8,
                      edgecolor='black', linewidth=2))

    plt.suptitle('Raw Actigraphy Data: Temporal Analysis & Wear Pattern Detection Panel 3',
                 fontsize=18, fontweight='bold', y=0.98)

    return fig

def main():
    """Main function to load data and create visualizations"""
    # Get project root
    project_root = Path(__file__).parent.parent.parent

    # Load raw activity data
    activity_file_path = project_root / "public" / "activity_ppg_records.json"

    print("Loading raw activity data...")
    with open(activity_file_path, 'r') as f:
        raw_data = json.load(f)

    # Take the first record as sample
    if isinstance(raw_data, list) and len(raw_data) > 0:
        sample_data = raw_data[0]
    else:
        sample_data = raw_data

    print(f"Sample data loaded successfully!")
    print(f"Data keys: {list(sample_data.keys())}")
    print(f"MET data points: {len(sample_data['met_1min'])}")
    print(f"Class data points: {len(sample_data['class_5min'])}")

    # Create all three panels
    print("Creating Panel 1: Basic Metrics and Time Series...")
    fig1 = create_raw_data_panel_1(sample_data)
    plt.savefig('raw_actigraphy_panel_1.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('raw_actigraphy_panel_1.pdf', bbox_inches='tight', facecolor='white')
    print("Panel 1 saved successfully!")

    print("Creating Panel 2: Directional Encoding and Compression...")
    fig2 = create_raw_data_panel_2(sample_data)
    plt.savefig('raw_actigraphy_panel_2.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('raw_actigraphy_panel_2.pdf', bbox_inches='tight', facecolor='white')
    print("Panel 2 saved successfully!")

    print("Creating Panel 3: Temporal Analysis and Wear Patterns...")
    fig3 = create_raw_data_panel_3(sample_data)
    plt.savefig('raw_actigraphy_panel_3.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('raw_actigraphy_panel_3.pdf', bbox_inches='tight', facecolor='white')
    print("Panel 3 saved successfully!")

    plt.show()

if __name__ == "__main__":
    main()
