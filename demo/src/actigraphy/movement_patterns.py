import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2


if __name__ == "__main__":
    from pathlib import Path
    import json

    # Load data from JSON file
    project_root = Path(__file__).parent.parent.parent  # From src/actigraphy/script to project root
    data_file = "movement_patterns_results.json"  # Change this for different files
    file_path = project_root / "public" / data_file

    # Load the data
    with open(file_path, 'r') as f:
        data = json.load(f)

    print(f"Loaded {len(data)} records from {data_file}")

    # If you want to use the hardcoded data instead, uncomment below:
    # data = [
    #   {
    #     "period_id": 0,
    #     "activity_fragmentation": 0.0,
    #     "activity_transition_probability": 0.0,
    #     "activity_bout_duration": 0.0,
    #     "peak_activity_time": 15.0,
    #     "activity_amplitude": 4.68,
    #     "activity_pattern": "Sustained",
    #     "data_source": "activity"
    #       #   },
    #   {
    #     "period_id": 0,
    #     "activity_fragmentation": 0.0,
    #     "activity_transition_probability": 0.0,
    #     "activity_bout_duration": 0.0,
    #     "peak_activity_time": 15.0,
    #     "activity_amplitude": 30.06,
    #     "activity_pattern": "Sustained",
    #     "data_source": "activity"
    # ... (rest of hardcoded data commented out)
    # ]

    # Convert to DataFrame
    df_activity = pd.DataFrame(data)
    df_activity['day'] = range(1, len(df_activity) + 1)

    print("Activity data loaded successfully!")
    print(f"Shape: {df_activity.shape}")
    print(df_activity.head())


    def create_activity_amplitude_analysis(df):
        """
        Comprehensive activity amplitude analysis with multiple visualizations
        """
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35)

        color_primary = '#1B4965'
        color_secondary = '#62B6CB'
        color_tertiary = '#FF6B35'
        color_quaternary = '#004E89'

        # Panel A: Activity Amplitude Time Series
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.plot(df['day'], df['activity_amplitude'], 'o-',
                 color=color_primary, linewidth=3, markersize=12,
                 markerfacecolor=color_secondary, markeredgewidth=2.5,
                 markeredgecolor=color_primary, label='Activity Amplitude')

        # Add confidence band
        mean_amp = df['activity_amplitude'].mean()
        std_amp = df['activity_amplitude'].std()
        ax1.axhline(y=mean_amp, color='red', linestyle='--',
                    linewidth=2, alpha=0.7, label=f'Mean: {mean_amp:.2f}')
        ax1.fill_between(df['day'], mean_amp - std_amp, mean_amp + std_amp,
                         alpha=0.2, color='red', label=f'±1 SD')

        ax1.set_xlabel('Day', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Activity Amplitude', fontweight='bold', fontsize=12)
        ax1.set_title('A) Activity Amplitude Over Time',
                      fontweight='bold', loc='left', pad=15, fontsize=13)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Panel B: Distribution of Activity Amplitude
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.hist(df['activity_amplitude'], bins=8, color=color_secondary,
                 alpha=0.7, edgecolor='black', linewidth=1.5)
        ax2.axvline(mean_amp, color='red', linestyle='--', linewidth=2.5,
                    label=f'Mean: {mean_amp:.2f}')
        ax2.axvline(df['activity_amplitude'].median(), color='orange',
                    linestyle='--', linewidth=2.5,
                    label=f'Median: {df["activity_amplitude"].median():.2f}')
        ax2.set_xlabel('Activity Amplitude', fontweight='bold', fontsize=11)
        ax2.set_ylabel('Frequency', fontweight='bold', fontsize=11)
        ax2.set_title('B) Distribution', fontweight='bold', loc='left', pad=15, fontsize=13)
        ax2.legend(frameon=True, fancybox=True, shadow=True, fontsize=9)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Panel C: Cumulative Activity Amplitude
        ax3 = fig.add_subplot(gs[1, :2])
        cumulative = df['activity_amplitude'].cumsum()
        ax3.fill_between(df['day'], cumulative, alpha=0.4, color=color_tertiary)
        ax3.plot(df['day'], cumulative, 'o-', color=color_tertiary,
                 linewidth=3, markersize=10, markerfacecolor='white',
                 markeredgewidth=2.5, markeredgecolor=color_tertiary)
        ax3.set_xlabel('Day', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Cumulative Activity Amplitude', fontweight='bold', fontsize=12)
        ax3.set_title('C) Cumulative Activity Amplitude',
                      fontweight='bold', loc='left', pad=15, fontsize=13)
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # Panel D: Box Plot with Quartiles
        ax4 = fig.add_subplot(gs[1, 2])
        bp = ax4.boxplot([df['activity_amplitude']], widths=0.6, patch_artist=True,
                         boxprops=dict(facecolor=color_secondary, alpha=0.7, linewidth=2),
                         medianprops=dict(color='red', linewidth=3),
                         whiskerprops=dict(linewidth=2),
                         capprops=dict(linewidth=2),
                         flierprops=dict(marker='o', markerfacecolor='red',
                                         markersize=10, alpha=0.7))

        # Add scatter of actual points
        y = df['activity_amplitude']
        x = np.random.normal(1, 0.04, size=len(y))
        ax4.scatter(x, y, alpha=0.6, s=100, color='darkblue', zorder=3)

        ax4.set_ylabel('Activity Amplitude', fontweight='bold', fontsize=11)
        ax4.set_title('D) Box Plot', fontweight='bold', loc='left', pad=15, fontsize=13)
        ax4.set_xticks([])
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        ax4.grid(True, alpha=0.3, axis='y', linestyle='--')

        # Panel E: Activity Amplitude Categories
        ax5 = fig.add_subplot(gs[2, 0])
        # Categorize amplitude
        bins = [0, 50, 100, 250]
        labels = ['Low\n(0-50)', 'Medium\n(50-100)', 'High\n(100+)']
        df['amplitude_category'] = pd.cut(df['activity_amplitude'], bins=bins, labels=labels)
        category_counts = df['amplitude_category'].value_counts()

        colors_cat = ['#90E0EF', '#00B4D8', '#0077B6']
        wedges, texts, autotexts = ax5.pie(category_counts, labels=category_counts.index,
                                           autopct='%1.1f%%', startangle=90,
                                           colors=colors_cat,
                                           wedgeprops=dict(edgecolor='black', linewidth=2),
                                           textprops=dict(fontweight='bold', fontsize=11))
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontsize(12)
            autotext.set_fontweight('bold')

        ax5.set_title('E) Amplitude Categories', fontweight='bold', pad=15, fontsize=13)

        # Panel F: Daily Change in Amplitude
        ax6 = fig.add_subplot(gs[2, 1])
        daily_change = df['activity_amplitude'].diff()
        colors_bar = ['green' if x >= 0 else 'red' for x in daily_change[1:]]
        ax6.bar(df['day'][1:], daily_change[1:], color=colors_bar,
                alpha=0.7, edgecolor='black', linewidth=1.5)
        ax6.axhline(y=0, color='black', linestyle='-', linewidth=1.5)
        ax6.set_xlabel('Day', fontweight='bold', fontsize=11)
        ax6.set_ylabel('Daily Change', fontweight='bold', fontsize=11)
        ax6.set_title('F) Day-to-Day Change', fontweight='bold', loc='left', pad=15, fontsize=13)
        ax6.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax6.spines['top'].set_visible(False)
        ax6.spines['right'].set_visible(False)

        # Panel G: Statistical Summary
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.axis('off')

        stats_text = f"""
        STATISTICAL SUMMARY
        {'=' * 30}

        Mean:           {df['activity_amplitude'].mean():.2f}
        Median:         {df['activity_amplitude'].median():.2f}
        Std Dev:        {df['activity_amplitude'].std():.2f}

        Min:            {df['activity_amplitude'].min():.2f}
        Max:            {df['activity_amplitude'].max():.2f}
        Range:          {df['activity_amplitude'].max() - df['activity_amplitude'].min():.2f}

        Q1 (25%):       {df['activity_amplitude'].quantile(0.25):.2f}
        Q3 (75%):       {df['activity_amplitude'].quantile(0.75):.2f}
        IQR:            {df['activity_amplitude'].quantile(0.75) - df['activity_amplitude'].quantile(0.25):.2f}

        CV:             {(df['activity_amplitude'].std() / df['activity_amplitude'].mean() * 100):.2f}%
        """

        ax7.text(0.1, 0.95, stats_text, transform=ax7.transAxes,
                 fontsize=10, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8,
                           edgecolor='black', linewidth=2))
        ax7.set_title('G) Statistics', fontweight='bold', pad=15, fontsize=13)

        plt.suptitle('Activity Amplitude: Comprehensive Analysis',
                     fontsize=16, fontweight='bold', y=0.995)

        return fig


    fig1 = create_activity_amplitude_analysis(df_activity)
    plt.savefig('activity_amplitude_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('activity_amplitude_analysis.pdf', bbox_inches='tight', facecolor='white')
    print("Activity amplitude analysis saved!")
    plt.show()


    def create_peak_activity_clock(df):
        """
        Creates a circular clock-style visualization of peak activity times
        """
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(111, projection='polar')

        # Convert peak activity time to radians (15:00 = 3 PM)
        peak_time = df['peak_activity_time'].iloc[0]  # All are 15.0
        theta = (peak_time / 24) * 2 * np.pi - np.pi / 2  # Adjust so 12 is at top

        # Create clock face
        hours = np.arange(0, 24)
        hour_angles = (hours / 24) * 2 * np.pi - np.pi / 2

        # Plot activity amplitude at peak time for each day
        amplitudes = df['activity_amplitude'].values
        days = df['day'].values

        # Normalize amplitudes for radius
        max_amp = df['activity_amplitude'].max()
        radii = amplitudes / max_amp

        # Create color map based on amplitude
        colors = plt.cm.viridis(amplitudes / max_amp)

        # Plot each day as a wedge
        width = 2 * np.pi / len(df)
        for i, (day, amp, radius, color) in enumerate(zip(days, amplitudes, radii, colors)):
            angle = theta + (i - len(df) / 2) * width / 2
            ax.bar(angle, radius, width=width * 0.8, bottom=0.0,
                   alpha=0.8, color=color, edgecolor='black', linewidth=1.5,
                   label=f'Day {day}' if i < 3 else '')

            # Add hour labels
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_xticks(hour_angles)
        ax.set_xticklabels([f'{h:02d}:00' for h in hours], fontsize=10, fontweight='bold')

        # Customize radial axis
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(['25%', '50%', '75%', '100%'], fontsize=9)
        ax.set_rlabel_position(45)

        # Add grid
        ax.grid(True, linestyle='--', alpha=0.5)

        # Highlight peak activity time
        ax.plot([theta, theta], [0, 1.1], 'r-', linewidth=4, alpha=0.7, label='Peak Time (15:00)')
        ax.scatter([theta], [1.15], s=500, c='red', marker='v', edgecolors='black',
                   linewidths=2, zorder=5)

        # Add title and legend
        plt.title('Activity Amplitude Distribution by Peak Time\n(Radial Clock Visualization)',
                  fontsize=14, fontweight='bold', pad=30)

        # Create custom legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='red', linewidth=4, label='Peak Activity Time'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple',
                   markersize=10, label='Low Amplitude', markeredgecolor='black'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow',
                   markersize=10, label='High Amplitude', markeredgecolor='black')
        ]
        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.3, 1.1),
                  frameon=True, fancybox=True, shadow=True, fontsize=11)

        return fig


    fig2 = create_peak_activity_clock(df_activity)
    plt.savefig('peak_activity_clock.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('peak_activity_clock.pdf', bbox_inches='tight', facecolor='white')
    print("Peak activity clock saved!")
    plt.show()


    def create_activity_pattern_panel(df):
        """
        Creates a comprehensive panel showing activity patterns
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        color_palette = ['#264653', '#2A9D8F', '#E9C46A', '#F4A261', '#E76F51']

        # Panel A: Activity Amplitude Bar Chart with Gradient
        ax1 = axes[0, 0]
        bars = ax1.bar(df['day'], df['activity_amplitude'],
                       color=plt.cm.viridis(df['activity_amplitude'] / df['activity_amplitude'].max()),
                       edgecolor='black', linewidth=1.5, alpha=0.8)

        # Add value labels on bars
        for i, (day, amp) in enumerate(zip(df['day'], df['activity_amplitude'])):
            ax1.text(day, amp + 5, f'{amp:.1f}', ha='center', va='bottom',
                     fontweight='bold', fontsize=9)

        ax1.set_xlabel('Day', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Activity Amplitude', fontweight='bold', fontsize=12)
        ax1.set_title('A) Daily Activity Amplitude', fontweight='bold',
                      loc='left', pad=15, fontsize=13)
        ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Panel B: Cumulative Distribution Function
        ax2 = axes[0, 1]
        sorted_amp = np.sort(df['activity_amplitude'])
        cumulative_prob = np.arange(1, len(sorted_amp) + 1) / len(sorted_amp)

        ax2.plot(sorted_amp, cumulative_prob, 'o-', color='#2A9D8F',
                 linewidth=3, markersize=10, markerfacecolor='white',
                 markeredgewidth=2.5, markeredgecolor='#2A9D8F')
        ax2.fill_between(sorted_amp, cumulative_prob, alpha=0.3, color='#2A9D8F')

        # Add percentile lines
        percentiles = [25, 50, 75]
        for p in percentiles:
            val = np.percentile(df['activity_amplitude'], p)
            ax2.axvline(val, color='red', linestyle='--', alpha=0.6, linewidth=1.5)
            ax2.text(val, 0.05, f'P{p}', fontsize=9, fontweight='bold',
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax2.set_xlabel('Activity Amplitude', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Cumulative Probability', fontweight='bold', fontsize=12)
        ax2.set_title('B) Cumulative Distribution Function', fontweight='bold',
                      loc='left', pad=15, fontsize=13)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Panel C: Moving Average and Volatility
        ax3 = axes[1, 0]
        window = 3
        if len(df) >= window:
            moving_avg = df['activity_amplitude'].rolling(window=window, center=True).mean()
            moving_std = df['activity_amplitude'].rolling(window=window, center=True).std()

            ax3.plot(df['day'], df['activity_amplitude'], 'o', color='lightgray',
                     markersize=8, label='Actual', alpha=0.6)
            ax3.plot(df['day'], moving_avg, '-', color='#E76F51',
                     linewidth=3, label=f'{window}-Day Moving Avg')
            ax3.fill_between(df['day'],
                             moving_avg - moving_std,
                             moving_avg + moving_std,
                             alpha=0.3, color='#E76F51', label='±1 SD')

        ax3.set_xlabel('Day', fontweight='bold', fontsize=12)
        ax3.set_ylabel('Activity Amplitude', fontweight='bold', fontsize=12)
        ax3.set_title('C) Moving Average & Volatility', fontweight='bold',
                      loc='left', pad=15, fontsize=13)
        ax3.legend(frameon=True, fancybox=True, shadow=True, loc='upper left')
        ax3.grid(True, alpha=0.3, linestyle='--')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # Panel D: Percentile Ranges
        ax4 = axes[1, 1]

        percentile_data = {
            'Min-P25': [df['activity_amplitude'].min(),
                        df['activity_amplitude'].quantile(0.25)],
            'P25-P50': [df['activity_amplitude'].quantile(0.25),
                        df['activity_amplitude'].quantile(0.50)],
            'P50-P75': [df['activity_amplitude'].quantile(0.50),
                        df['activity_amplitude'].quantile(0.75)],
            'P75-Max': [df['activity_amplitude'].quantile(0.75),
                        df['activity_amplitude'].max()]
        }

        colors_range = ['#90E0EF', '#00B4D8', '#0077B6', '#023E8A']
        y_pos = np.arange(len(percentile_data))

        for i, (label, (low, high)) in enumerate(percentile_data.items()):
            ax4.barh(i, high - low, left=low, height=0.6,
                     color=colors_range[i], edgecolor='black', linewidth=1.5,
                     label=label, alpha=0.8)
            # Add range text
            ax4.text(low + (high - low) / 2, i, f'{high - low:.1f}',
                     ha='center', va='center', fontweight='bold', fontsize=10,
                     color='white')

        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(percentile_data.keys(), fontweight='bold')
        ax4.set_xlabel('Activity Amplitude', fontweight='bold', fontsize=12)
        ax4.set_title('D) Percentile Range Distribution', fontweight='bold',
                      loc='left', pad=15, fontsize=13)
        ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        plt.suptitle('Activity Pattern Analysis: Multi-Panel View',
                     fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        return fig


    fig3 = create_activity_pattern_panel(df_activity)
    plt.savefig('activity_pattern_panel.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('activity_pattern_panel.pdf', bbox_inches='tight', facecolor='white')
    print("Activity pattern panel saved!")
    plt.show()


    def create_activity_heatmap_calendar(df):
        """
        Creates a calendar-style heatmap of activity amplitude
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Reshape data for calendar view (assuming 10 days, create 2 rows of 5)
        n_cols = 5
        n_rows = int(np.ceil(len(df) / n_cols))

        # Create matrix
        matrix = np.full((n_rows, n_cols), np.nan)
        for i, amp in enumerate(df['activity_amplitude']):
            row = i // n_cols
            col = i % n_cols
            matrix[row, col] = amp

        # Create heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto',
                       vmin=df['activity_amplitude'].min(),
                       vmax=df['activity_amplitude'].max())

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activity Amplitude', fontweight='bold', fontsize=12)

        # Add grid
        for i in range(n_rows + 1):
            ax.axhline(i - 0.5, color='black', linewidth=2)
        for j in range(n_cols + 1):
            ax.axvline(j - 0.5, color='black', linewidth=2)

        # Add text annotations
        for i in range(n_rows):
            for j in range(n_cols):
                day_num = i * n_cols + j + 1
                if day_num <= len(df):
                    value = matrix[i, j]
                    text_color = 'white' if value > df['activity_amplitude'].mean() else 'black'
                    ax.text(j, i, f'Day {day_num}\n{value:.1f}',
                            ha='center', va='center', fontweight='bold',
                            fontsize=11, color=text_color)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title('Activity Amplitude: Calendar Heatmap View',
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig


    fig4 = create_activity_heatmap_calendar(df_activity)
    plt.savefig('activity_heatmap_calendar.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('activity_heatmap_calendar.pdf', bbox_inches='tight', facecolor='white')
    print("Activity heatmap calendar saved!")
    plt.show()


    def create_activity_heatmap_calendar(df):
        """
        Creates a calendar-style heatmap of activity amplitude
        """
        fig, ax = plt.subplots(figsize=(14, 6))

        # Reshape data for calendar view (assuming 10 days, create 2 rows of 5)
        n_cols = 5
        n_rows = int(np.ceil(len(df) / n_cols))

        # Create matrix
        matrix = np.full((n_rows, n_cols), np.nan)
        for i, amp in enumerate(df['activity_amplitude']):
            row = i // n_cols
            col = i % n_cols
            matrix[row, col] = amp

        # Create heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto',
                       vmin=df['activity_amplitude'].min(),
                       vmax=df['activity_amplitude'].max())

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Activity Amplitude', fontweight='bold', fontsize=12)

        # Add grid
        for i in range(n_rows + 1):
            ax.axhline(i - 0.5, color='black', linewidth=2)
        for j in range(n_cols + 1):
            ax.axvline(j - 0.5, color='black', linewidth=2)

        # Add text annotations
        for i in range(n_rows):
            for j in range(n_cols):
                day_num = i * n_cols + j + 1
                if day_num <= len(df):
                    value = matrix[i, j]
                    text_color = 'white' if value > df['activity_amplitude'].mean() else 'black'
                    ax.text(j, i, f'Day {day_num}\n{value:.1f}',
                            ha='center', va='center', fontweight='bold',
                            fontsize=11, color=text_color)

        # Remove ticks
        ax.set_xticks([])
        ax.set_yticks([])

        ax.set_title('Activity Amplitude: Calendar Heatmap View',
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig


    fig4 = create_activity_heatmap_calendar(df_activity)
    plt.savefig('activity_heatmap_calendar.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('activity_heatmap_calendar.pdf', bbox_inches='tight', facecolor='white')
    print("Activity heatmap calendar saved!")
    plt.show()


    def create_trend_forecast_analysis(df):
        """
        Creates trend analysis with simple forecasting
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        data = df['activity_amplitude'].values
        days = df['day'].values

        # Panel A: Polynomial Fits
        ax1 = axes[0]

        # Plot actual data
        ax1.scatter(days, data, s=200, color='black', zorder=5,
                    edgecolors='white', linewidths=2, label='Actual Data')

        # Fit different polynomials
        colors_poly = ['#2E86AB', '#A23B72', '#F18F01']
        degrees = [1, 2, 3]

        x_smooth = np.linspace(days.min(), days.max(), 100)

        for degree, color in zip(degrees, colors_poly):
            z = np.polyfit(days, data, degree)
            p = np.poly1d(z)
            ax1.plot(x_smooth, p(x_smooth), '-', linewidth=3,
                     color=color, alpha=0.7, label=f'Poly Degree {degree}')

        ax1.set_xlabel('Day', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Activity Amplitude', fontweight='bold', fontsize=12)
        ax1.set_title('A) Polynomial Trend Fitting', fontweight='bold',
                      loc='left', pad=15, fontsize=13)
        ax1.legend(frameon=True, fancybox=True, shadow=True, loc='upper left', fontsize=11)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Panel B: Forecast with Confidence Interval
        ax2 = axes[1]

        # Linear regression for forecast
        z = np.polyfit(days, data, 1)
        p = np.poly1d(z)

        # Calculate residuals and prediction interval
        y_pred = p(days)
        residuals = data - y_pred
        std_residuals = np.std(residuals)

        # Forecast future days
        future_days = np.arange(days.max() + 1, days.max() + 4)
        all_days = np.concatenate([days, future_days])
        forecast = p(all_days)

        # Plot historical data
        ax2.plot(days, data, 'o-', color='#2E86AB', linewidth=3,
                 markersize=12, markerfacecolor='white', markeredgewidth=2.5,
                 markeredgecolor='#2E86AB', label='Historical Data')

        # Plot forecast
        ax2.plot(future_days, p(future_days), 's--', color='red',
                 linewidth=3, markersize=12, markerfacecolor='white',
                 markeredgewidth=2.5, markeredgecolor='red', label='Forecast')

        # Add confidence interval
        confidence = 1.96 * std_residuals  # 95% CI
        ax2.fill_between(all_days, forecast - confidence, forecast + confidence,
                         alpha=0.3, color='gray', label='95% Confidence Interval')

        # Add vertical line at forecast start
        ax2.axvline(days.max() + 0.5, color='black', linestyle=':', linewidth=2)
        ax2.text(days.max() + 0.5, ax2.get_ylim()[1] * 0.95, 'Forecast Start',
                 ha='center', fontweight='bold', fontsize=10,
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        ax2.set_xlabel('Day', fontweight='bold', fontsize=12)
        ax2.set_ylabel('Activity Amplitude', fontweight='bold', fontsize=12)
        ax2.set_title('B) Trend Forecast with Confidence Interval', fontweight='bold',
                      loc='left', pad=15, fontsize=13)
        ax2.legend(frameon=True, fancybox=True, shadow=True, loc='upper left', fontsize=11)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        plt.suptitle('Trend Analysis and Forecasting', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()

        return fig


    fig6 = create_trend_forecast_analysis(df_activity)
    plt.savefig('trend_forecast_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('trend_forecast_analysis.pdf', bbox_inches='tight', facecolor='white')
    print("Trend forecast analysis saved!")
    plt.show()
