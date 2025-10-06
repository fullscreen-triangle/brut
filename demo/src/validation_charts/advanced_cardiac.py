import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime


def main():
    """Generate publication-quality charts for advanced cardiac metrics."""

    # Load data
    data_path = Path('public/advanced_cardiac_results.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Filter sleep data only (exclude zero timestamps)
    sleep_data = [d for d in data if d['timestamp'] > 0]

    # Extract metrics
    timestamps = [datetime.fromtimestamp(d['timestamp'] / 1000) for d in sleep_data]
    rsa = [d['respiratory_sinus_arrhythmia'] for d in sleep_data]
    brs = [d['baroreflex_sensitivity'] for d in sleep_data]
    qt_var = [d['qt_variability'] for d in sleep_data]
    hrt_onset = [d['heart_rate_turbulence_onset'] for d in sleep_data]
    health_status = [d['cardiovascular_health'] for d in sleep_data]

    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

    fig.suptitle('Advanced Cardiac Autonomic Function Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: RSA Time Series
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timestamps, rsa, marker='o', linewidth=2, markersize=8,
             color='#2E86AB', markerfacecolor='white', markeredgewidth=2)
    ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax1.set_ylabel('RSA (ms)', fontsize=11, fontweight='bold')
    ax1.set_title('A) Respiratory Sinus Arrhythmia Over Time', fontsize=12, loc='left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Add mean line
    ax1.axhline(np.mean(rsa), color='red', linestyle='--',
                linewidth=2, alpha=0.7, label=f'Mean: {np.mean(rsa):.2f} ms')
    ax1.legend(loc='upper right', frameon=True, fancybox=True)

    # Plot 2: Baroreflex Sensitivity
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(timestamps, brs, marker='s', linewidth=2, markersize=7,
             color='#A23B72', markerfacecolor='white', markeredgewidth=2)
    ax2.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax2.set_ylabel('BRS (ms/mmHg)', fontsize=11, fontweight='bold')
    ax2.set_title('B) Baroreflex Sensitivity', fontsize=12, loc='left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='x', rotation=45)

    # Plot 3: QT Variability
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(timestamps, qt_var, marker='^', linewidth=2, markersize=7,
             color='#F18F01', markerfacecolor='white', markeredgewidth=2)
    ax3.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax3.set_ylabel('QT Variability (ms)', fontsize=11, fontweight='bold')
    ax3.set_title('C) QT Variability Index', fontsize=12, loc='left')
    ax3.grid(True, alpha=0.3)
    ax3.tick_params(axis='x', rotation=45)

    # Plot 4: Cardiovascular Health Status
    ax4 = fig.add_subplot(gs[2, 0])
    health_counts = {}
    for status in health_status:
        health_counts[status] = health_counts.get(status, 0) + 1

    colors = {'Average': '#4CAF50', 'Below Average': '#FF9800', 'Poor': '#F44336'}
    bars = ax4.bar(health_counts.keys(), health_counts.values(),
                   color=[colors.get(k, '#999999') for k in health_counts.keys()],
                   edgecolor='black', linewidth=1.5, alpha=0.8)
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('D) Cardiovascular Health Distribution', fontsize=12, loc='left')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2., height,
                 f'{int(height)}', ha='center', va='bottom', fontweight='bold')

    # Plot 5: Multi-metric Box Plot
    ax5 = fig.add_subplot(gs[2, 1])

    # Normalize metrics for comparison
    metrics_normalized = {
        'RSA': np.array(rsa) / np.max(rsa),
        'BRS': np.array(brs) / np.max(brs),
        'QT Var': np.array(qt_var) / np.max(qt_var),
        'HRT Onset': np.array(hrt_onset) / np.max(hrt_onset)
    }

    bp = ax5.boxplot(metrics_normalized.values(), labels=metrics_normalized.keys(),
                     patch_artist=True, showmeans=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     meanprops=dict(marker='D', markerfacecolor='green', markersize=8))

    ax5.set_ylabel('Normalized Value', fontsize=11, fontweight='bold')
    ax5.set_title('E) Normalized Cardiac Metrics Comparison', fontsize=12, loc='left')
    ax5.grid(True, alpha=0.3, axis='y')

    plt.savefig('advanced_cardiac_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig('advanced_cardiac_metrics.pdf', bbox_inches='tight')
    print("✓ Advanced Cardiac Metrics charts saved")
    plt.show()


if __name__ == "__main__":
    main()
