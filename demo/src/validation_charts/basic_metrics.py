import json
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path
from datetime import datetime


def main():
    """Generate publication-quality charts for basic activity metrics."""

    # Load data
    data_path = Path('public/basic_activity_metrics_results.json')
    with open(data_path, 'r') as f:
        data = json.load(f)

    # Separate activity and sleep-derived data
    activity_data = [d for d in data if d['data_source'] == 'activity']
    sleep_derived = [d for d in data if d['data_source'] == 'sleep_derived']

    # Extract metrics from sleep-derived data
    steps = [d['step_count'] for d in sleep_derived]
    distance = [d['distance'] for d in sleep_derived]
    active_min = [d['active_minutes'] for d in sleep_derived]
    sedentary = [d['sedentary_time'] for d in sleep_derived]
    timestamps = [datetime.fromtimestamp(d['timestamp'] / 1000) for d in sleep_derived]

    # Set publication style
    plt.style.use('seaborn-v0_8-whitegrid')

    # Create figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.35, wspace=0.3)

    fig.suptitle('Physical Activity Metrics Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    # Plot 1: Daily Steps Over Time
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(timestamps, steps, marker='o', linewidth=2.5, markersize=9,
             color='#FF6B6B', markerfacecolor='white', markeredgewidth=2.5)
    ax1.fill_between(timestamps, steps, alpha=0.3, color='#FF6B6B')
    ax1.set_xlabel('Date', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Step Count', fontsize=11, fontweight='bold')
    ax1.set_title('A) Daily Step Count Progression', fontsize=12, loc='left')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='x', rotation=45)

    # Add WHO recommendation line
    ax1.axhline(10000, color='green', linestyle='--', linewidth=2,
                alpha=0.7, label='WHO Recommendation (10,000 steps)')
    ax1.legend(loc='upper left', frameon=True, fancybox=True)

    # Plot 2: Active vs Sedentary Time
    ax2 = fig.add_subplot(gs[1, 0])
    x_pos = np.arange(len(active_min))
    width = 0.35

    bars1 = ax2.bar(x_pos - width / 2, active_min, width, label='Active',
                    color='#4ECDC4', edgecolor='black', linewidth=1.2, alpha=0.8)
    bars2 = ax2.bar(x_pos + width / 2, sedentary, width, label='Sedentary',
                    color='#FFE66D', edgecolor='black', linewidth=1.2, alpha=0.8)

    ax2.set_xlabel('Observation', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Time (minutes)', fontsize=11, fontweight='bold')
    ax2.set_title('B) Active vs Sedentary Time', fontsize=12, loc='left')
    ax2.legend(frameon=True, fancybox=True, shadow=True)
    ax2.grid(True, alpha=0.3, axis='y')

    # Plot 3: Distance Covered
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.bar(range(len(distance)), distance, color='#95E1D3',
            edgecolor='black', linewidth=1.5, alpha=0.8)
    ax3.set_xlabel('Observation', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Distance (km)', fontsize=11, fontweight='bold')
    ax3.set_title('C) Distance Covered', fontsize=12, loc='left')
    ax3.grid(True, alpha=0.3, axis='y')

    # Add mean line
    ax3.axhline(np.mean(distance), color='red', linestyle='--',
                linewidth=2, label=f'Mean: {np.mean(distance):.2f} km')
    ax3.legend(frameon=True, fancybox=True)

    # Plot 4: Steps vs Distance Correlation
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.scatter(steps, distance, s=150, alpha=0.6, c=active_min,
                cmap='viridis', edgecolors='black', linewidth=1.5)

    # Add trend line
    z = np.polyfit(steps, distance, 1)
    p = np.poly1d(z)
    ax4.plot(steps, p(steps), "r--", linewidth=2.5, alpha=0.8)

    ax4.set_xlabel('Step Count', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Distance (km)', fontsize=11, fontweight='bold')
    ax4.set_title('D) Steps vs Distance Relationship', fontsize=12, loc='left')
    ax4.grid(True, alpha=0.3)

    # Add correlation coefficient
    corr = np.corrcoef(steps, distance)[0, 1]
    ax4.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax4.transAxes,
             fontsize=11, verticalalignment='top', fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))

    cbar = plt.colorbar(ax4.collections[0], ax=ax4)
    cbar.set_label('Active Minutes', fontsize=10, fontweight='bold')

    # Plot 5: Activity Summary Statistics
    ax5 = fig.add_subplot(gs[2, 1])

    metrics = {
        'Steps': steps,
        'Distance\n(km)': distance,
        'Active\n(min)': active_min,
        'Sedentary\n(min)': sedentary
    }

    bp = ax5.boxplot(metrics.values(), labels=metrics.keys(),
                     patch_artist=True, showmeans=True, showfliers=True,
                     boxprops=dict(facecolor='lightcoral', alpha=0.7, linewidth=1.5),
                     medianprops=dict(color='darkred', linewidth=2.5),
                     meanprops=dict(marker='D', markerfacecolor='yellow',
                                    markeredgecolor='black', markersize=8),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

    ax5.set_ylabel('Normalized Value', fontsize=11, fontweight='bold')
    ax5.set_title('E) Activity Metrics Distribution', fontsize=12, loc='left')
    ax5.grid(True, alpha=0.3, axis='y')

    # Normalize for better visualization
    for i, (key, values) in enumerate(metrics.items()):
        normalized = np.array(values) / np.max(values) * 100
        bp['boxes'][i].set_facecolor(plt.cm.Set3(i))

    plt.savefig('basic_activity_metrics.png', dpi=300, bbox_inches='tight')
    plt.savefig('basic_activity_metrics.pdf', bbox_inches='tight')
    print("✓ Basic Activity Metrics charts saved")
    plt.show()


if __name__ == "__main__":
    main()
