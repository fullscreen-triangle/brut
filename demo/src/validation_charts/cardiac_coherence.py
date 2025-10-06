import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime


def main():
    """Generate publication-quality charts for cardiac coherence data."""

    # Load data
    data_path = Path('public/cardiac_coherence_results.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Filter sleep data
    sleep_data = [d for d in data if d['data_source'] == 'sleep']

    # Extract metrics
    timestamps = [datetime.fromtimestamp(d['timestamp'] / 1000) for d in sleep_data]
    coherence_ratio = [d['coherence_ratio'] for d in sleep_data]
    coherence_peak = [d['coherence_peak'] for d in sleep_data]
    coherence_stability = [d['coherence_stability'] for d in sleep_data]
    breath_avg = [d['breath_average'] for d in sleep_data]

    # Set publication style
    plt.style.use('seaborn-v0_8-paper')

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    fig.suptitle('Cardiac Coherence and Heart Rate Variability Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: Coherence Ratio Time Series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timestamps, coherence_ratio, marker='o', linewidth=2.5,
             markersize=9, color='#E63946', markerfacecolor='white',
             markeredgewidth=2.5, label='Coherence Ratio')
    ax1.fill_between(timestamps, coherence_ratio, alpha=0.3, color='#E63946')

    ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Coherence Ratio', fontsize=11, fontweight='bold')
    ax1.set_title('A) Cardiac Coherence Ratio Over Time', fontsize=12, loc='left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)
    ax1.legend(loc='upper right', frameon=True, fancybox=True, shadow=True)

    # Add mean and std bands
    mean_coh = np.mean(coherence_ratio)
    std_coh = np.std(coherence_ratio)
    ax1.axhline(mean_coh, color='blue', linestyle='--', linewidth=2,
                alpha=0.7, label=f'Mean: {mean_coh:.2f}')
    ax1.fill_between(timestamps, mean_coh - std_coh, mean_coh + std_coh,
                     alpha=0.2, color='blue')

    # Plot 2: Coherence Stability
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(timestamps, coherence_stability, marker='s', linewidth=2.5,
             markersize=8, color='#457B9D', markerfacecolor='white',
             markeredgewidth=2)
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Stability (%)', fontsize=11, fontweight='bold')
    ax2.set_title('B) Coherence Stability', fontsize=12, loc='left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim([85, 95])

    # Plot 3: Breathing Rate
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(timestamps, breath_avg, marker='^', linewidth=2.5,
             markersize=8, color='#06FFA5', markerfacecolor='white',
             markeredgewidth=2, markeredgecolor='#06FFA5')
    ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Breaths per Minute', fontsize=11, fontweight='bold')
    ax3.set_title('C) Average Breathing Rate', fontsize=12, loc='left')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # Add optimal breathing zone
    ax3.axhspan(12, 18, alpha=0.2, color='green', label='Optimal Zone')
    ax3.legend(loc='upper right', frameon=True, fancybox=True)

    # Plot 4: Coherence Ratio vs Stability
    ax4 = fig.add_subplot(gs[2, 0])
    scatter = ax4.scatter(coherence_ratio, coherence_stability,
                          s=200, alpha=0.6, c=breath_avg,
                          cmap='plasma', edgecolors='black', linewidth=1.5)

    ax4.set_xlabel('Coherence Ratio', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Coherence Stability (%)', fontsize=11, fontweight='bold')
    ax4.set_title('D) Coherence Ratio vs Stability', fontsize=12, loc='left')
    ax4.grid(True, alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('Breathing Rate (bpm)', fontsize=10, fontweight='bold')

    # Add correlation
    corr = np.corrcoef(coherence_ratio, coherence_stability)[0, 1]
    ax4.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    # Plot 5: Multi-metric Radar Chart
    ax5 = fig.add_subplot(gs[2, 1], projection='polar')

    # Normalize metrics for radar chart
    metrics_norm = {
        'Coherence\nRatio': np.mean(coherence_ratio) / np.max(coherence_ratio),
        'Peak': np.mean(coherence_peak) / np.max(coherence_peak),
        'Stability': np.mean(coherence_stability) / 100,
        'Breathing\nRate': 1 - abs(np.mean(breath_avg) - 15) / 15  # Optimal at 15
    }

    categories = list(metrics_norm.keys())
    values = list(metrics_norm.values())
    values += values[:1]  # Complete the circle

    angles = np.linspace(0, 2 * np.pi, len(categories), endpoint=False).tolist()
    angles += angles[:1]

    ax5.plot(angles, values, 'o-', linewidth=2.5, color='#F72585', markersize=10)
    ax5.fill(angles, values, alpha=0.25, color='#F72585')
    ax5.set_xticks(angles[:-1])
    ax5.set_xticklabels(categories, fontsize=10, fontweight='bold')
    ax5.set_ylim(0, 1)
    ax5.set_title('E) Coherence Metrics Profile', fontsize=12,
                  pad=20, fontweight='bold')
    ax5.grid(True, alpha=0.3)

    # Add reference circle at 0.8
    ax5.plot(angles, [0.8] * len(angles), 'g--', linewidth=1.5, alpha=0.5)

    plt.savefig('cardiac_coherence.png', dpi=300, bbox_inches='tight')
    plt.savefig('cardiac_coherence.pdf', bbox_inches='tight')
    print("✓ Cardiac Coherence charts saved")
    plt.show()


if __name__ == "__main__":
    main()
