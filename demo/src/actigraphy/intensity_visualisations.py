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


def create_intensity_activity_visualization(df):
    """
    Creates comprehensive visualization for intensity-based activity analysis
    """
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = ['#90E0EF', '#00B4D8', '#0077B6', '#023E8A', '#FF6B35']

    # Panel A: Activity Time by Intensity Level
    ax1 = fig.add_subplot(gs[0, :2])
    x = np.arange(len(df))
    width = 0.2

    ax1.bar(x - width * 1.5, df['light_activity_time_min'], width,
            label='Light', color=colors[0], alpha=0.8,
            edgecolor='black', linewidth=1.5)
    ax1.bar(x - width / 2, df['moderate_activity_time_min'], width,
            label='Moderate', color=colors[1], alpha=0.8,
            edgecolor='black', linewidth=1.5)
    ax1.bar(x + width / 2, df['vigorous_activity_time_min'], width,
            label='Vigorous', color=colors[2], alpha=0.8,
            edgecolor='black', linewidth=1.5)
    # Create very vigorous data (not in original data, so use zeros)
    very_vigorous_data = pd.Series([0.0] * len(df), name='very_vigorous_activity_time_min')
    ax1.bar(x + width * 1.5, very_vigorous_data, width,
            label='Very Vigorous', color=colors[3], alpha=0.8,
            edgecolor='black', linewidth=1.5)

    ax1.set_xlabel('Period ID', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Activity Time (minutes)', fontweight='bold', fontsize=12)
    ax1.set_title('A) Activity Time by Intensity Level',
                  fontweight='bold', loc='left', pad=15, fontsize=13)
    ax1.legend(frameon=True, fancybox = True, shadow = True, loc = 'upper right')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Average Activity Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    avg_activities = [
        df['light_activity_time_min'].mean(),
        df['moderate_activity_time_min'].mean(),
        df['vigorous_activity_time_min'].mean(),
        very_vigorous_data.mean()
    ]
    labels = ['Light', 'Moderate', 'Vigorous', 'Very\nVigorous']

    wedges, texts, autotexts = ax2.pie(avg_activities, labels=labels,
                                       autopct='%1.1f%%', startangle=90,
                                       colors=colors[:4],
                                       wedgeprops=dict(edgecolor='black', linewidth=2),
                                       textprops=dict(fontweight='bold', fontsize=10))

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')

    ax2.set_title('B) Average Activity Distribution', fontweight='bold', pad=15, fontsize=13)

    # Panel C: Stacked Area Chart
    ax3 = fig.add_subplot(gs[1, :])

    ax3.fill_between(df['period_id'], 0, df['light_activity_time_min'],
                     alpha=0.7, color=colors[0], label='Light')
    ax3.fill_between(df['period_id'], df['light_activity_time_min'],
                     df['light_activity_time_min'] + df['moderate_activity_time_min'],
                     alpha=0.7, color=colors[1], label='Moderate')
    ax3.fill_between(df['period_id'],
                     df['light_activity_time_min'] + df['moderate_activity_time_min'],
                     df['light_activity_time_min'] + df['moderate_activity_time_min'] + df[
                         'vigorous_activity_time_min'],
                     alpha=0.7, color=colors[2], label='Vigorous')
    ax3.fill_between(df['period_id'],
                     df['light_activity_time_min'] + df['moderate_activity_time_min'] + df[
                         'vigorous_activity_time_min'],
                     df['light_activity_time_min'] + df['moderate_activity_time_min'] + df[
                         'vigorous_activity_time_min'] + very_vigorous_data,
                     alpha=0.7, color=colors[3], label='Very Vigorous')

    ax3.set_xlabel('Period ID', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Cumulative Time (minutes)', fontweight='bold', fontsize=12)
    ax3.set_title('C) Stacked Activity Time Distribution',
                  fontweight='bold', loc='left', pad=15, fontsize=13)
    ax3.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
    ax3.grid(True, alpha=0.3, linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Panel D: Light Activity Time Series
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.plot(df['period_id'], df['light_activity_time_min'], 'o-',
             color=colors[0], linewidth=3, markersize=10,
             markerfacecolor='white', markeredgewidth=2.5,
             markeredgecolor=colors[0])
    ax4.fill_between(df['period_id'], df['light_activity_time_min'],
                     alpha=0.3, color=colors[0])

    mean_light = df['light_activity_time_min'].mean()
    ax4.axhline(y=mean_light, color='red', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Mean: {mean_light:.1f}')

    ax4.set_xlabel('Period ID', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Time (min)', fontweight='bold', fontsize=11)
    ax4.set_title('D) Light Activity', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax4.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax4.grid(True, alpha=0.3, linestyle='--')
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Panel E: Moderate Activity Time Series
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(df['period_id'], df['moderate_activity_time_min'], 'o-',
             color=colors[1], linewidth=3, markersize=10,
             markerfacecolor='white', markeredgewidth=2.5,
             markeredgecolor=colors[1])
    ax5.fill_between(df['period_id'], df['moderate_activity_time_min'],
                     alpha=0.3, color=colors[1])

    mean_mod = df['moderate_activity_time_min'].mean()
    ax5.axhline(y=mean_mod, color='red', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Mean: {mean_mod:.1f}')

    ax5.set_xlabel('Period ID', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Time (min)', fontweight='bold', fontsize=11)
    ax5.set_title('E) Moderate Activity', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax5.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax5.grid(True, alpha=0.3, linestyle='--')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # Panel F: Vigorous Activity Time Series
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.plot(df['period_id'], df['vigorous_activity_time_min'], 'o-',
             color=colors[2], linewidth=3, markersize=10,
             markerfacecolor='white', markeredgewidth=2.5,
             markeredgecolor=colors[2])
    ax6.fill_between(df['period_id'], df['vigorous_activity_time_min'],
                     alpha=0.3, color=colors[2])

    mean_vig = df['vigorous_activity_time_min'].mean()
    ax6.axhline(y=mean_vig, color='red', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Mean: {mean_vig:.1f}')

    ax6.set_xlabel('Period ID', fontweight='bold', fontsize=11)
    ax6.set_ylabel('Time (min)', fontweight='bold', fontsize=11)
    ax6.set_title('F) Vigorous Activity', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax6.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

    # Panel G: Box Plot Comparison
    ax7 = fig.add_subplot(gs[3, 0])
    activity_data = [
        df['light_activity_time_min'].values,
        df['moderate_activity_time_min'].values,
        df['vigorous_activity_time_min'].values,
        very_vigorous_data.values
    ]

    bp = ax7.boxplot(activity_data, labels=['Light', 'Mod.', 'Vig.', 'V.Vig.'],
                     patch_artist=True, widths=0.6,
                     boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=2),
                     medianprops=dict(color='red', linewidth=3),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2))

    for patch, color in zip(bp['boxes'], colors[:4]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax7.set_ylabel('Time (minutes)', fontweight='bold', fontsize=11)
    ax7.set_title('G) Activity Distribution', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax7.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)

    # Panel H: Total Activity Time
    ax8 = fig.add_subplot(gs[3, 1])
    total_activity = (df['light_activity_time_min'] +
                      df['moderate_activity_time_min'] +
                      df['vigorous_activity_time_min'] +
                      very_vigorous_data)

    ax8.bar(df['period_id'], total_activity, color=colors[4],
            alpha=0.7, edgecolor='black', linewidth=1.5)

    mean_total = total_activity.mean()
    ax8.axhline(y=mean_total, color='red', linestyle='--',
                linewidth=2.5, alpha=0.7, label=f'Mean: {mean_total:.1f}')

    ax8.set_xlabel('Period ID', fontweight='bold', fontsize=11)
    ax8.set_ylabel('Total Time (min)', fontweight='bold', fontsize=11)
    ax8.set_title('H) Total Activity Time', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax8.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax8.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax8.spines['top'].set_visible(False)
    ax8.spines['right'].set_visible(False)

    # Panel I: Statistics Summary
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.axis('off')

    stats_text = f"""
        ACTIVITY STATISTICS (min)
        {'=' * 35}

        LIGHT ACTIVITY:
        Mean:     {df['light_activity_time_min'].mean():.2f}
        Std:      {df['light_activity_time_min'].std():.2f}
        Total:    {df['light_activity_time_min'].sum():.2f}

        MODERATE ACTIVITY:
        Mean:     {df['moderate_activity_time_min'].mean():.2f}
        Std:      {df['moderate_activity_time_min'].std():.2f}
        Total:    {df['moderate_activity_time_min'].sum():.2f}

        VIGOROUS ACTIVITY:
        Mean:     {df['vigorous_activity_time_min'].mean():.2f}
        Std:      {df['vigorous_activity_time_min'].std():.2f}
        Total:    {df['vigorous_activity_time_min'].sum():.2f}

        VERY VIGOROUS:
        Mean:     {very_vigorous_data.mean():.2f}
        Std:      {very_vigorous_data.std():.2f}
        Total:    {very_vigorous_data.sum():.2f}

        TOTAL ACTIVITY:
        Mean:     {total_activity.mean():.2f}
        Total:    {total_activity.sum():.2f}

        Periods:  {len(df)}
        """

    ax9.text(0.1, 0.95, stats_text, transform=ax9.transAxes,
             fontsize=8.5, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8,
                       edgecolor='black', linewidth=2))
    ax9.set_title('I) Statistics', fontweight='bold', pad=15, fontsize=13)

    plt.suptitle('Intensity-Based Activity Analysis: Comprehensive Visualization',
                 fontsize=16, fontweight='bold', y=0.995)

    return fig


def main():
    project_root = Path(__file__).parent.parent.parent  # From src/actigraphy/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "intensity_based_activity_results.json"  # Change this for different activity files


    # Construct paths
    activity_file_path = project_root / "public" / activity_data_file


    # Convert to strings for compatibility
    activity_file_path = str(activity_file_path)

    print("Loading data from URL...")


    # Convert to DataFrame
    df = pd.read_json(activity_file_path)

    print(f"Data loaded successfully! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print("Creating intensity-based activity visualizations...")
    fig = create_intensity_activity_visualization(df)

    plt.savefig('intensity_activity_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('intensity_activity_analysis.pdf', bbox_inches='tight', facecolor='white')
    print("Intensity-based activity visualizations saved successfully!")

    plt.show()


if __name__ == "__main__":
    main()
