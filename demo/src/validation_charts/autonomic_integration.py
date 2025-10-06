import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path


def main():
    """Generate publication-quality charts for autonomic integration data."""

    # Load data
    data_path = Path('public/autonomic_integration_results.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Extract metrics
    balance_scores = [d['autonomic_balance_score'] for d in data]
    sleep_hr = [d['sleep_hr_mean'] for d in data]

    # Set publication style
    plt.style.use('seaborn-v0_8-darkgrid')

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Autonomic Nervous System Integration',
                 fontsize=16, fontweight='bold')

    # Plot 1: Sleep Heart Rate Distribution
    axes[0].hist(sleep_hr, bins=10, alpha=0.7, color='#3498db',
                 edgecolor='black', linewidth=1.5)
    axes[0].axvline(np.mean(sleep_hr), color='red', linestyle='--',
                    linewidth=2.5, label=f'Mean: {np.mean(sleep_hr):.1f} bpm')
    axes[0].axvline(np.median(sleep_hr), color='green', linestyle='--',
                    linewidth=2.5, label=f'Median: {np.median(sleep_hr):.1f} bpm')
    axes[0].set_xlabel('Sleep Heart Rate (bpm)', fontsize=11, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0].set_title('A) Sleep HR Distribution', fontsize=12, loc='left')
    axes[0].legend(frameon=True, fancybox=True, shadow=True)

    # Plot 2: Autonomic Balance Score
    x_pos = np.arange(len(balance_scores))
    bars = axes[1].bar(x_pos, balance_scores, alpha=0.7,
                       color='#2ecc71', edgecolor='black', linewidth=1.5)
    axes[1].axhline(np.mean(balance_scores), color='red', linestyle='--',
                    linewidth=2, label=f'Mean: {np.mean(balance_scores):.2f}')
    axes[1].set_xlabel('Observation', fontsize=11, fontweight='bold')
    axes[1].set_ylabel('Autonomic Balance Score', fontsize=11, fontweight='bold')
    axes[1].set_title('B) Autonomic Balance Scores', fontsize=12, loc='left')
    axes[1].set_ylim([0, 1])
    axes[1].legend(frameon=True, fancybox=True, shadow=True)

    # Color bars based on value
    for i, bar in enumerate(bars):
        if balance_scores[i] > 0.6:
            bar.set_color('#27ae60')
        elif balance_scores[i] > 0.4:
            bar.set_color('#f39c12')
        else:
            bar.set_color('#e74c3c')

    # Plot 3: Sleep HR Variability
    axes[2].plot(range(len(sleep_hr)), sleep_hr, marker='o', linewidth=2,
                 markersize=8, color='#9b59b6', markerfacecolor='white',
                 markeredgewidth=2, label='Sleep HR')

    # Add confidence interval
    mean_hr = np.mean(sleep_hr)
    std_hr = np.std(sleep_hr)
    axes[2].fill_between(range(len(sleep_hr)),
                         mean_hr - std_hr, mean_hr + std_hr,
                         alpha=0.2, color='#9b59b6', label='±1 SD')

    axes[2].set_xlabel('Observation', fontsize=11, fontweight='bold')
    axes[2].set_ylabel('Sleep Heart Rate (bpm)', fontsize=11, fontweight='bold')
    axes[2].set_title('C) Sleep HR Trend', fontsize=12, loc='left')
    axes[2].legend(frameon=True, fancybox=True, shadow=True)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('autonomic_integration.png', dpi=300, bbox_inches='tight')
    plt.savefig('autonomic_integration.pdf', bbox_inches='tight')
    print("✓ Autonomic Integration charts saved")
    plt.show()


if __name__ == "__main__":
    main()
