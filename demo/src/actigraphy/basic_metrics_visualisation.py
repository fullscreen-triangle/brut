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

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2


def load_json_from_url(url):
    """Load JSON data from URL"""
    with urllib.request.urlopen(url) as response:
        data = json.loads(response.read().decode())
    return data


def create_basic_activity_visualization(df):
    """
    Creates comprehensive visualization for basic activity metrics
    """
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

    color_primary = '#1B4965'
    color_secondary = '#62B6CB'
    color_tertiary = '#FF6B35'
    color_quaternary = '#5FA8D3'

    # Panel A: Step Count Time Series
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df['period_id'], df['step_count'], 'o-',
             color=color_primary, linewidth=3, markersize=12,
             markerfacecolor=color_secondary, markeredgewidth=2.5,
             markeredgecolor=color_primary)

    mean_steps = df['step_count'].mean()
    ax1.axhline(y=mean_steps, color='red', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Mean: {mean_steps:.1f}')
    ax1.fill_between(df['period_id'], df['step_count'], alpha=0.3, color=color_secondary)

    ax1.set_xlabel('Period ID', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Step Count', fontweight='bold', fontsize=12)
    ax1.set_title('A) Step Count Over Time', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Step Count Distribution
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.hist(df['step_count'], bins=8, color=color_secondary,
             alpha=0.7, edgecolor='black', linewidth=1.5)
    ax2.axvline(mean_steps, color='red', linestyle='--', linewidth=2.5,
                label=f'Mean: {mean_steps:.1f}')
    ax2.set_xlabel('Step Count', fontweight='bold', fontsize=11)
    ax2.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax2.set_title('B) Distribution', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)

    # Panel C: Distance Analysis
    ax3 = fig.add_subplot(gs[1, 0])
    if df['distance'].sum() > 0:
        ax3.bar(df['period_id'], df['distance'], color=color_tertiary,
                alpha=0.7, edgecolor='black', linewidth=1.5)
        ax3.set_xlabel('Period ID', fontweight='bold', fontsize=11)
        ax3.set_ylabel('Distance', fontweight='bold', fontsize=11)
    else:
        ax3.text(0.5, 0.5, 'No Distance Data\nAvailable',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 transform=ax3.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax3.set_xticks([])
        ax3.set_yticks([])
    ax3.set_title('C) Distance Traveled', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Panel D: Active Minutes
    ax4 = fig.add_subplot(gs[1, 1])
    if df['active_minutes'].sum() > 0:
        ax4.bar(df['period_id'], df['active_minutes'], color=color_quaternary,
                alpha=0.7, edgecolor='black', linewidth=1.5)
        ax4.set_xlabel('Period ID', fontweight='bold', fontsize=11)
        ax4.set_ylabel('Active Minutes', fontweight='bold', fontsize=11)
    else:
        ax4.text(0.5, 0.5, 'No Active Minutes\nData Available',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 transform=ax4.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax4.set_xticks([])
        ax4.set_yticks([])
    ax4.set_title('D) Active Minutes', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Panel E: Sedentary Time
    ax5 = fig.add_subplot(gs[1, 2])
    if df['sedentary_time'].sum() > 0:
        ax5.bar(df['period_id'], df['sedentary_time'], color='#8B0000',
                alpha=0.7, edgecolor='black', linewidth=1.5)
        ax5.set_xlabel('Period ID', fontweight='bold', fontsize=11)
        ax5.set_ylabel('Sedentary Time', fontweight='bold', fontsize=11)
    else:
        ax5.text(0.5, 0.5, 'No Sedentary Time\nData Available',
                 ha='center', va='center', fontsize=14, fontweight='bold',
                 transform=ax5.transAxes,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax5.set_xticks([])
        ax5.set_yticks([])
    ax5.set_title('E) Sedentary Time', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # Panel F: Cumulative Steps
    ax6 = fig.add_subplot(gs[2, :2])
    cumulative = df['step_count'].cumsum()
    ax6.fill_between(df['period_id'], cumulative, alpha=0.4, color=color_tertiary)
    ax6.plot(df['period_id'], cumulative, 'o-', color=color_tertiary,
             linewidth=3, markersize=10, markerfacecolor='white',
             markeredgewidth=2.5, markeredgecolor=color_tertiary)
    ax6.set_xlabel('Period ID', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Cumulative Steps', fontweight='bold', fontsize=12)
    ax6.set_title('F) Cumulative Step Count', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

    # Panel G: Statistical Summary
    ax7 = fig.add_subplot(gs[2, 2])
    ax7.axis('off')

    stats_text = f"""
    STEP COUNT STATISTICS
    {'=' * 30}

    Total Steps:    {df['step_count'].sum():.0f}
    Mean:           {df['step_count'].mean():.1f}
    Median:         {df['step_count'].median():.1f}
    Std Dev:        {df['step_count'].std():.1f}

    Min:            {df['step_count'].min():.1f}
    Max:            {df['step_count'].max():.1f}
    Range:          {df['step_count'].max() - df['step_count'].min():.1f}

    Q1 (25%):       {df['step_count'].quantile(0.25):.1f}
    Q3 (75%):       {df['step_count'].quantile(0.75):.1f}
    IQR:            {df['step_count'].quantile(0.75) - df['step_count'].quantile(0.25):.1f}

    Data Source:    {df['data_source'].iloc[0]}
    Total Periods:  {len(df)}
    """

    ax7.text(0.1, 0.95, stats_text, transform=ax7.transAxes,
             fontsize=10, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8,
                       edgecolor='black', linewidth=2))
    ax7.set_title('G) Statistics', fontweight='bold', pad=15, fontsize=13)

    plt.suptitle('Basic Activity Metrics: Comprehensive Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    return fig


def main():
    """Main function to load data and create visualizations"""
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent  # From src/actigraphy/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "basic_activity_metrics_results.json"  # Change this for different activity files


    # Construct paths
    activity_file_path = project_root / "public" / activity_data_file


    # Convert to strings for compatibility
    activity_file_path = str(activity_file_path)

    print("Loading data from URL...")


    # Convert to DataFrame
    df = pd.read_json(activity_file_path)

    print(f"Data loaded successfully! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    # Create visualization
    print("Creating visualizations...")
    fig = create_basic_activity_visualization(df)

    # Save figures
    plt.savefig('basic_activity_metrics_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('basic_activity_metrics_analysis.pdf', bbox_inches='tight', facecolor='white')
    print("Visualizations saved successfully!")

    plt.show()


if __name__ == "__main__":
    main()
