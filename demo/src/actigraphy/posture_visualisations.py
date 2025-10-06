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


def create_postural_analysis_visualization(df):
    """
    Creates comprehensive visualization for postural analysis
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = ['#6A4C93', '#1982C4', '#8AC926', '#FFCA3A', '#FF595E']

    # Panel A: Time Standing Over Periods
    ax1 = fig.add_subplot(gs[0, :2])

    if df['time_standing_min'].sum() > 0:
        ax1.plot(df['period_id'], df['time_standing_min'], 'o-',
                 color=colors[0], linewidth=3, markersize=12,
                 markerfacecolor=colors[1], markeredgewidth=2.5,
                 markeredgecolor=colors[0])

        mean_standing = df['time_standing_min'].mean()
        ax1.axhline(y=mean_standing, color='red', linestyle='--',
                    linewidth=2.5, alpha=0.7, label=f'Mean: {mean_standing:.1f}')
        ax1.fill_between(df['period_id'], df['time_standing_min'],
                         alpha=0.3, color=colors[1])

        ax1.set_xlabel('Period ID', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Time Standing (minutes)', fontweight='bold', fontsize=12)
        ax1.legend(frameon=True, fancybox=True, shadow=True)
        ax1.grid(True, alpha=0.3, linestyle='--')
    else:
        ax1.text(0.5, 0.5, 'No Standing Time Data Available',
                 ha='center', va='center', fontsize=16, fontweight='bold',
                 transform=ax1.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax1.set_xticks([])
        ax1.set_yticks([])

    ax1.set_title('A) Time Standing Over Time',
                  fontweight='bold', loc='left', pad=15, fontsize=13)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Standing Time Distribution
    ax2 = fig.add_subplot(gs[0, 2])

    if df['time_standing_min'].sum() > 0:
        ax2.hist(df['time_standing_min'], bins=8, color=colors[1],
                 alpha=0.7, edgecolor='black', linewidth=1.5)
        mean_standing = df['time_standing_min'].mean()
        ax2.axvline(mean_standing, color='red', linestyle='--', linewidth=2.5,
                    label=f'Mean: {mean_standing:.1f}')
        ax2.set_xlabel('Time Standing (min)', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Frequency', fontweight='bold', fontsize=11)
        ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    else:
        ax2.text(0.5, 0.5, 'No Data',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 transform=ax2.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax2.set_xticks([])
        ax2.set_yticks([])

    ax2.set_title('B) Distribution', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel C: Cumulative Standing Time
    ax3 = fig.add_subplot(gs[1, :2])

    if df['time_standing_min'].sum() > 0:
        cumulative_standing = df['time_standing_min'].cumsum()
        ax3.fill_between(df['period_id'], cumulative_standing,
                         alpha=0.4, color=colors[2])
        ax3.plot(df['period_id'], cumulative_standing, 'o-',
                 color=colors[2], linewidth=3, markersize=10,
                 markerfacecolor='white', markeredgewidth=2.5,
                 markeredgecolor=colors[2])

        ax3.set_xlabel('Period ID', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Cumulative Time (minutes)', fontweight='bold', fontsize=12)
        ax3.grid(True, alpha=0.3, linestyle='--')
    else:
        ax3.text(0.5, 0.5, 'No Cumulative Data Available',
                 ha='center', va='center', fontsize=16, fontweight='bold',
                 transform=ax3.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax3.set_xticks([])
        ax3.set_yticks([])

    ax3.set_title('C) Cumulative Standing Time',
                  fontweight='bold', loc='left', pad=15, fontsize=13)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Panel D: Box Plot
    ax4 = fig.add_subplot(gs[1, 2])

    if df['time_standing_min'].sum() > 0:
        bp = ax4.boxplot([df['time_standing_min'].values],
                         labels=['Standing Time'],
                         patch_artist=True, widths=0.6,
                         boxprops=dict(facecolor=colors[3], alpha=0.7, linewidth=2),
                         medianprops=dict(color='red', linewidth=3),
                         whiskerprops=dict(linewidth=2),
                         capprops=dict(linewidth=2))

        ax4.set_ylabel('Time (minutes)', fontweight='bold', fontsize=11)
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')
    else:
        ax4.text(0.5, 0.5, 'No Data',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 transform=ax4.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax4.set_xticks([])
        ax4.set_yticks([])

    ax4.set_title('D) Box Plot Analysis', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Panel E: Daily Average Bar Chart
    ax5 = fig.add_subplot(gs[2, 0])

    if df['time_standing_min'].sum() > 0:
        bars = ax5.bar(df['period_id'], df['time_standing_min'],
                       color=colors[4], alpha=0.7, edgecolor='black', linewidth=1.5)

        # Color bars based on value
        for i, bar in enumerate(bars):
            if df['time_standing_min'].iloc[i] > df['time_standing_min'].mean():
                bar.set_color(colors[2])
            else:
                bar.set_color(colors[4])

        ax5.set_xlabel('Period ID', fontweight='bold', fontsize=11)
        ax5.set_ylabel('Time (min)', fontweight='bold', fontsize=11)
        ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
    else:
        ax5.text(0.5, 0.5, 'No Data',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 transform=ax5.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax5.set_xticks([])
        ax5.set_yticks([])

    ax5.set_title('E) Daily Standing Time', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # Panel F: Trend Analysis
    ax6 = fig.add_subplot(gs[2, 1])

    if df['time_standing_min'].sum() > 0 and len(df) > 1:
        # Calculate moving average
        window = min(3, len(df))
        moving_avg = df['time_standing_min'].rolling(window=window, center=True).mean()

        ax6.plot(df['period_id'], df['time_standing_min'], 'o',
                 color=colors[0], markersize=8, alpha=0.5, label='Actual')
        ax6.plot(df['period_id'], moving_avg, '-',
                 color=colors[4], linewidth=3, label=f'MA({window})')

        # Add trend line
        if len(df) > 2:
            z = np.polyfit(df['period_id'], df['time_standing_min'], 1)
            p = np.poly1d(z)
            ax6.plot(df['period_id'], p(df['period_id']), "--",
                     linewidth=2.5, alpha=0.7, color='green', label='Trend')

        ax6.set_xlabel('Period ID', fontweight='bold', fontsize=11)
        ax6.set_ylabel('Time (min)', fontweight='bold', fontsize=11)
        ax6.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
        ax6.grid(True, alpha=0.3, linestyle='--')
    else:
        ax6.text(0.5, 0.5, 'Insufficient Data\nfor Trend Analysis',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 transform=ax6.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax6.set_xticks([])
        ax6.set_yticks([])

    ax6.set_title('F) Trend Analysis', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

    # Panel G: Statistics Summary
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    if df['time_standing_min'].sum() > 0:
        stats_text = f"""
        POSTURAL STATISTICS
        {'=' * 35}

        STANDING TIME (minutes):

        Total:        {df['time_standing_min'].sum():.2f}
        Mean:         {df['time_standing_min'].mean():.2f}
        Median:       {df['time_standing_min'].median():.2f}
        Std Dev:      {df['time_standing_min'].std():.2f}

        Min:          {df['time_standing_min'].min():.2f}
        Max:          {df['time_standing_min'].max():.2f}
        Range:        {df['time_standing_min'].max() - df['time_standing_min'].min():.2f}

        Q1 (25%):     {df['time_standing_min'].quantile(0.25):.2f}
        Q3 (75%):     {df['time_standing_min'].quantile(0.75):.2f}
        IQR:          {df['time_standing_min'].quantile(0.75) - df['time_standing_min'].quantile(0.25):.2f}

        Avg/Day:      {df['time_standing_min'].mean():.2f}

        Total Periods: {len(df)}
        Non-zero:      {(df['time_standing_min'] > 0).sum()}
        """
    else:
        stats_text = """
        POSTURAL STATISTICS
        ===================================

        NO STANDING TIME DATA AVAILABLE

        All values are zero or missing.

        This could indicate:
        - Data not collected
        - Sensor not active
        - Processing error

        Please check data source
        and collection methods.
        """

    ax7.text(0.1, 0.95, stats_text, transform=ax7.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8,
                       edgecolor='black', linewidth=2))
    ax7.set_title('G) Statistics', fontweight='bold', pad=15, fontsize=13)

    plt.suptitle('Postural Analysis: Comprehensive Visualization',
                 fontsize=16, fontweight='bold', y=0.995)

    return fig


def main():
    project_root = Path(__file__).parent.parent.parent  # From src/actigraphy/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "postural_analysis_results.json"  # Change this for different activity files


    # Construct paths
    activity_file_path = project_root / "public" / activity_data_file


    # Convert to strings for compatibility
    activity_file_path = str(activity_file_path)

    print("Loading data from URL...")


    # Convert to DataFrame
    df = pd.read_json(activity_file_path)

    print(f"Data loaded successfully! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print("Creating postural analysis visualizations...")
    fig = create_postural_analysis_visualization(df)

    plt.savefig('postural_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('postural_analysis.pdf', bbox_inches='tight', facecolor='white')
    print("Postural analysis visualizations saved successfully!")

    plt.show()


if __name__ == "__main__":
    main()
