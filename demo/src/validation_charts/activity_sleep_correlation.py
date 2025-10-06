import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path


def main():
    """Generate publication-quality charts for activity-sleep correlation data."""

    # Load data
    data_path = Path('public/activity_sleep_correlation_results.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Extract metrics
    steps = [d['steps'] for d in data]
    correlation = [d['activity_sleep_correlation'] for d in data]
    sleep_efficiency = [d['sleep_efficiency'] for d in data]
    recovery_ratio = [d['recovery_ratio'] for d in data]

    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    sns.set_palette("husl")

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Activity-Sleep Correlation Analysis', fontsize=16, fontweight='bold')

    # Plot 1: Steps vs Correlation
    axes[0, 0].scatter(steps, correlation, alpha=0.6, s=100, edgecolors='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Daily Steps', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Activity-Sleep Correlation', fontsize=11, fontweight='bold')
    axes[0, 0].set_title('A) Steps vs Sleep Correlation', fontsize=12, loc='left')

    # Add trend line
    z = np.polyfit(steps, correlation, 1)
    p = np.poly1d(z)
    axes[0, 0].plot(steps, p(steps), "r--", alpha=0.8, linewidth=2)

    # Calculate R²
    correlation_coef = np.corrcoef(steps, correlation)[0, 1]
    axes[0, 0].text(0.05, 0.95, f'r = {correlation_coef:.3f}',
                    transform=axes[0, 0].transAxes, fontsize=10,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Plot 2: Sleep Efficiency Distribution
    axes[0, 1].hist(sleep_efficiency, bins=8, alpha=0.7, edgecolor='black', linewidth=1.2)
    axes[0, 1].axvline(np.mean(sleep_efficiency), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(sleep_efficiency):.1f}%')
    axes[0, 1].set_xlabel('Sleep Efficiency (%)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title('B) Sleep Efficiency Distribution', fontsize=12, loc='left')
    axes[0, 1].legend(frameon=True, fancybox=True)

    # Plot 3: Recovery Ratio vs Sleep Efficiency
    scatter = axes[1, 0].scatter(recovery_ratio, sleep_efficiency,
                                 c=steps, cmap='viridis', s=100,
                                 alpha=0.6, edgecolors='black', linewidth=0.5)
    axes[1, 0].set_xlabel('Recovery Ratio', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Sleep Efficiency (%)', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('C) Recovery vs Sleep Efficiency', fontsize=12, loc='left')
    cbar = plt.colorbar(scatter, ax=axes[1, 0])
    cbar.set_label('Steps', fontsize=10, fontweight='bold')

    # Plot 4: Correlation Heatmap
    metrics_data = np.array([steps, correlation, sleep_efficiency, recovery_ratio])
    corr_matrix = np.corrcoef(metrics_data)

    im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
    axes[1, 1].set_xticks(range(4))
    axes[1, 1].set_yticks(range(4))
    labels = ['Steps', 'Correlation', 'Sleep Eff.', 'Recovery']
    axes[1, 1].set_xticklabels(labels, rotation=45, ha='right')
    axes[1, 1].set_yticklabels(labels)
    axes[1, 1].set_title('D) Correlation Matrix', fontsize=12, loc='left')

    # Add correlation values
    for i in range(4):
        for j in range(4):
            text = axes[1, 1].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                   ha="center", va="center", color="black", fontsize=9)

    cbar2 = plt.colorbar(im, ax=axes[1, 1])
    cbar2.set_label('Correlation', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('activity_sleep_correlation.png', dpi=300, bbox_inches='tight')
    plt.savefig('activity_sleep_correlation.pdf', bbox_inches='tight')
    print("✓ Activity-Sleep Correlation charts saved")
    plt.show()


if __name__ == "__main__":
    main()
