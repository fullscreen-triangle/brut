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


def create_energy_expenditure_visualization(df):
    """
    Creates comprehensive visualization for energy expenditure analysis
    """
    fig = plt.figure(figsize=(18, 14))
    gs = GridSpec(4, 3, figure=fig, hspace=0.4, wspace=0.35)

    colors = ['#1B4965', '#62B6CB', '#FF6B35', '#5FA8D3', '#F4A261']

    # Panel A: Total Daily Energy Expenditure
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.plot(df['period_id'], df['total_daily_energy_expenditure'], 'o-',
             color=colors[0], linewidth=3, markersize=12,
             markerfacecolor=colors[1], markeredgewidth=2.5,
             markeredgecolor=colors[0])

    mean_tdee = df['total_daily_energy_expenditure'].mean()
    ax1.axhline(y=mean_tdee, color='red', linestyle='--',
                linewidth=2.5, alpha=0.7, label=f'Mean: {mean_tdee:.1f}')
    ax1.fill_between(df['period_id'], df['total_daily_energy_expenditure'],
                     alpha=0.3, color=colors[1])

    ax1.set_xlabel('Period ID', fontweight='bold', fontsize=12)
    ax1.set_ylabel('TDEE (calories)', fontweight='bold', fontsize=12)
    ax1.set_title('A) Total Daily Energy Expenditure Over Time',
                  fontweight='bold', loc='left', pad=15, fontsize=13)
    ax1.legend(frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

    # Panel B: Energy Components Breakdown
    ax2 = fig.add_subplot(gs[0, 2])
    energy_components = [
        df['active_energy_expenditure'].mean(),
        df['basal_metabolic_rate'].mean(),
        df['thermic_effect_of_food'].mean()
    ]
    labels = ['Active', 'BMR', 'TEF']

    wedges, texts, autotexts = ax2.pie(energy_components, labels=labels,
                                       autopct='%1.1f%%', startangle=90,
                                       colors=colors[:3],
                                       wedgeprops=dict(edgecolor='black', linewidth=2),
                                       textprops=dict(fontweight='bold', fontsize=11))

    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(12)
        autotext.set_fontweight('bold')

    ax2.set_title('B) Energy Components', fontweight='bold', pad=15, fontsize=13)

    # Panel C: Active vs BMR Energy
    ax3 = fig.add_subplot(gs[1, :2])
    x = np.arange(len(df))
    width = 0.35

    ax3.bar(x - width / 2, df['active_energy_expenditure'], width,
            label='Active Energy', color=colors[2], alpha=0.8,
            edgecolor='black', linewidth=1.5)
    ax3.bar(x + width / 2, df['basal_metabolic_rate'], width,
            label='BMR Energy', color=colors[3], alpha=0.8,
            edgecolor='black', linewidth=1.5)

    ax3.set_xlabel('Period ID', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Energy (calories)', fontweight='bold', fontsize=12)
    ax3.set_title('C) Active vs BMR Energy Expenditure',
                  fontweight='bold', loc='left', pad=15, fontsize=13)
    ax3.legend(frameon=True, fancybox=True, shadow=True)
    ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax3.spines['top'].set_visible(False)
    ax3.spines['right'].set_visible(False)

    # Panel D: BMR Distribution
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.hist(df['basal_metabolic_rate'], bins=8, color=colors[4],
             alpha=0.7, edgecolor='black', linewidth=1.5)
    mean_bmr = df['basal_metabolic_rate'].mean()
    ax4.axvline(mean_bmr, color='red', linestyle='--', linewidth=2.5,
                label=f'Mean: {mean_bmr:.1f}')
    ax4.set_xlabel('BMR (calories)', fontweight='bold', fontsize=11)
    ax4.set_ylabel('Frequency', fontweight='bold', fontsize=11)
    ax4.set_title('D) BMR Distribution', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax4.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
    ax4.spines['top'].set_visible(False)
    ax4.spines['right'].set_visible(False)

    # Panel E: Energy Efficiency Ratio
    ax5 = fig.add_subplot(gs[2, 0])
    efficiency_ratio = df['active_energy_expenditure'] / df['total_daily_energy_expenditure']

    ax5.bar(df['period_id'], efficiency_ratio * 100, color=colors[0],
            alpha=0.7, edgecolor='black', linewidth=1.5)
    ax5.axhline(y=efficiency_ratio.mean() * 100, color='red', linestyle='--',
                linewidth=2, alpha=0.7)
    ax5.set_xlabel('Period ID', fontweight='bold', fontsize=11)
    ax5.set_ylabel('Active/TDEE (%)', fontweight='bold', fontsize=11)
    ax5.set_title('E) Energy Efficiency', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax5.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax5.spines['top'].set_visible(False)
    ax5.spines['right'].set_visible(False)

    # Panel F: Cumulative Energy Expenditure
    ax6 = fig.add_subplot(gs[2, 1:])
    cumulative_active = df['active_energy_expenditure'].cumsum()
    cumulative_bmr = df['basal_metabolic_rate'].cumsum()
    cumulative_total = df['total_daily_energy_expenditure'].cumsum()

    ax6.fill_between(df['period_id'], cumulative_total, alpha=0.3,
                     color=colors[0], label='Total')
    ax6.plot(df['period_id'], cumulative_total, 'o-', color=colors[0],
             linewidth=3, markersize=10, markerfacecolor='white',
             markeredgewidth=2.5, markeredgecolor=colors[0])

    ax6.fill_between(df['period_id'], cumulative_active, alpha=0.3,
                     color=colors[2], label='Active')
    ax6.plot(df['period_id'], cumulative_active, 's-', color=colors[2],
             linewidth=2.5, markersize=8)

    ax6.set_xlabel('Period ID', fontweight='bold', fontsize=12)
    ax6.set_ylabel('Cumulative Energy (calories)', fontweight='bold', fontsize=12)
    ax6.set_title('F) Cumulative Energy Expenditure',
                  fontweight='bold', loc='left', pad=15, fontsize=13)
    ax6.legend(frameon=True, fancybox=True, shadow=True)
    ax6.grid(True, alpha=0.3, linestyle='--')
    ax6.spines['top'].set_visible(False)
    ax6.spines['right'].set_visible(False)

    # Panel G: Box Plot Comparison
    ax7 = fig.add_subplot(gs[3, 0])
    energy_data = [
        df['active_energy_expenditure'].values,
        df['basal_metabolic_rate'].values,
        df['thermic_effect_of_food'].values
    ]

    bp = ax7.boxplot(energy_data, labels=['Active', 'BMR', 'TEF'],
                     patch_artist=True, widths=0.6,
                     boxprops=dict(facecolor='lightblue', alpha=0.7, linewidth=2),
                     medianprops=dict(color='red', linewidth=3),
                     whiskerprops=dict(linewidth=2),
                     capprops=dict(linewidth=2))

    for patch, color in zip(bp['boxes'], colors[:3]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax7.set_ylabel('Energy (calories)', fontweight='bold', fontsize=11)
    ax7.set_title('G) Energy Distribution', fontweight='bold', loc='left', pad=15, fontsize=13)
    ax7.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax7.spines['top'].set_visible(False)
    ax7.spines['right'].set_visible(False)

    # Panel H: Correlation Heatmap
    ax8 = fig.add_subplot(gs[3, 1])
    corr_data = df[['total_daily_energy_expenditure', 'active_energy_expenditure',
                    'basal_metabolic_rate', 'thermic_effect_of_food']].corr()

    im = ax8.imshow(corr_data, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)

    labels_short = ['TDEE', 'Active', 'BMR', 'TEF']
    ax8.set_xticks(np.arange(4))
    ax8.set_yticks(np.arange(4))
    ax8.set_xticklabels(labels_short, fontweight='bold', rotation=45, ha='right')
    ax8.set_yticklabels(labels_short, fontweight='bold')

    # Add correlation values
    for i in range(4):
        for j in range(4):
            text = ax8.text(j, i, f'{corr_data.iloc[i, j]:.2f}',
                            ha="center", va="center",
                            color="white" if abs(corr_data.iloc[i, j]) > 0.5 else "black",
                            fontweight='bold', fontsize=10)

    ax8.set_title('H) Correlation Matrix', fontweight='bold', loc='left', pad=15, fontsize=13)
    plt.colorbar(im, ax=ax8, fraction=0.046, pad=0.04)

    # Panel I: Statistics Summary
    ax9 = fig.add_subplot(gs[3, 2])
    ax9.axis('off')

    stats_text = f"""
    ENERGY STATISTICS
    {'=' * 35}

    TOTAL DAILY ENERGY:
    Mean:     {df['total_daily_energy_expenditure'].mean():.1f}
    Std:      {df['total_daily_energy_expenditure'].std():.1f}
    Min:      {df['total_daily_energy_expenditure'].min():.1f}
    Max:      {df['total_daily_energy_expenditure'].max():.1f}

    ACTIVE ENERGY:
    Mean:     {df['active_energy_expenditure'].mean():.1f}
    Std:      {df['active_energy_expenditure'].std():.1f}

    TEF ENERGY:
    Mean:     {df['thermic_effect_of_food'].mean():.1f}
    Std:      {df['thermic_effect_of_food'].std():.1f}

    BMR:
    Mean:     {df['basal_metabolic_rate'].mean():.1f}
    Std:      {df['basal_metabolic_rate'].std():.1f}

    Total Periods: {len(df)}
    """

    ax9.text(0.1, 0.95, stats_text, transform=ax9.transAxes,
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8,
                       edgecolor='black', linewidth=2))
    ax9.set_title('I) Statistics', fontweight='bold', pad=15, fontsize=13)

    plt.suptitle('Energy Expenditure Analysis: Comprehensive Visualization',
                 fontsize=16, fontweight='bold', y=0.995)

    return fig


def main():
    project_root = Path(__file__).parent.parent.parent  # From src/actigraphy/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "energy_expenditure_results.json"  # Change this for different activity files


    # Construct paths
    activity_file_path = project_root / "public" / activity_data_file


    # Convert to strings for compatibility
    activity_file_path = str(activity_file_path)

    print("Loading data from URL...")


    # Convert to DataFrame
    df = pd.read_json(activity_file_path)

    print(f"Data loaded successfully! Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")

    print("Creating energy expenditure visualizations...")
    fig = create_energy_expenditure_visualization(df)

    plt.savefig('energy_expenditure_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('energy_expenditure_analysis.pdf', bbox_inches='tight', facecolor='white')
    print("Energy expenditure visualizations saved successfully!")

    plt.show()


if __name__ == "__main__":
    main()
