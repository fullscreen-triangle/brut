import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches

# Set publication-quality style
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.linewidth'] = 1.2
plt.rcParams['xtick.major.width'] = 1.2
plt.rcParams['ytick.major.width'] = 1.2

if __name__ == "__main__":
    # Your data
    data = [
      {
        "activity_period_id": 0,
        "sleep_period_id": 0,
        "activity_sleep_correlation": 0.3427472,
        "steps": 468,
        "activity_intensity": 70.0,
        "sleep_efficiency": 68.0,
        "exercise_sleep_latency_impact": 0.0,
        "recovery_ratio": 0.9714285714285714,
        "circadian_alignment_score": 0.8
      },
      {
        "activity_period_id": 0,
        "sleep_period_id": 2,
        "activity_sleep_correlation": 0.36551339999999993,
        "steps": 3006,
        "activity_intensity": 70.0,
        "sleep_efficiency": 63.0,
        "exercise_sleep_latency_impact": 0.0,
        "recovery_ratio": 0.9,
        "circadian_alignment_score": 0.8
      },
      {
        "activity_period_id": 0,
        "sleep_period_id": 0,
        "activity_sleep_correlation": 0.3069866,
        "steps": 2974,
        "activity_intensity": 70.0,
        "sleep_efficiency": 53.0,
        "exercise_sleep_latency_impact": 0.0,
        "recovery_ratio": 0.7571428571428571,
        "circadian_alignment_score": 0.8
      },
      {
        "activity_period_id": 0,
        "sleep_period_id": 0,
        "activity_sleep_correlation": 0.36551199999999995,
        "steps": 3640,
        "activity_intensity": 70.0,
        "sleep_efficiency": 61.0,
        "exercise_sleep_latency_impact": 0.0,
        "recovery_ratio": 0.8714285714285714,
        "circadian_alignment_score": 0.8
      },
      {
        "activity_period_id": 0,
        "sleep_period_id": 0,
        "activity_sleep_correlation": 0.42457520000000004,
        "steps": 8939,
        "activity_intensity": 70.0,
        "sleep_efficiency": 56.0,
        "exercise_sleep_latency_impact": 0.0,
        "recovery_ratio": 0.8,
        "circadian_alignment_score": 0.8
      },
      {
        "activity_period_id": 0,
        "sleep_period_id": 1,
        "activity_sleep_correlation": 0.4184264,
        "steps": 8573,
        "activity_intensity": 70.0,
        "sleep_efficiency": 56.0,
        "exercise_sleep_latency_impact": 0.0,
        "recovery_ratio": 0.8,
        "circadian_alignment_score": 0.8
      },
      {
        "activity_period_id": 0,
        "sleep_period_id": 2,
        "activity_sleep_correlation": 0.4897825999999999,
        "steps": 11338,
        "activity_intensity": 70.0,
        "sleep_efficiency": 59.0,
        "exercise_sleep_latency_impact": 0.0,
        "recovery_ratio": 0.8428571428571429,
        "circadian_alignment_score": 0.8
      },
      {
        "activity_period_id": 0,
        "sleep_period_id": 0,
        "activity_sleep_correlation": 0.38421979999999994,
        "steps": 5374,
        "activity_intensity": 70.0,
        "sleep_efficiency": 59.0,
        "exercise_sleep_latency_impact": 0.0,
        "recovery_ratio": 0.8428571428571429,
        "circadian_alignment_score": 0.8
      },
      {
        "activity_period_id": 0,
        "sleep_period_id": 0,
        "activity_sleep_correlation": 0.265486,
        "steps": 685,
        "activity_intensity": 70.0,
        "sleep_efficiency": 52.0,
        "exercise_sleep_latency_impact": 0.0,
        "recovery_ratio": 0.7428571428571429,
        "circadian_alignment_score": 0.8
      },
      {
        "activity_period_id": 0,
        "sleep_period_id": 3,
        "activity_sleep_correlation": 0.6975999999999999,
        "steps": 21553,
        "activity_intensity": 70.0,
        "sleep_efficiency": 64.0,
        "exercise_sleep_latency_impact": 0.0,
        "recovery_ratio": 0.9142857142857143,
        "circadian_alignment_score": 0.8
      }
    ]

    # Convert to DataFrame
    df = pd.DataFrame(data)
    df['day'] = range(1, len(df) + 1)

    print("Data loaded successfully!")
    print(f"Shape: {df.shape}")
    print(df.head())


    def create_multipanel_figure(df):
        """
        Creates a professional multi-panel figure with 6 subplots
        """
        fig = plt.figure(figsize=(14, 10))
        gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

        # Color palette
        color_primary = '#2E86AB'
        color_secondary = '#A23B72'
        color_tertiary = '#F18F01'
        color_quaternary = '#C73E1D'

        # Panel A: Steps over time with trend line
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(df['day'], df['steps'], 'o-', color=color_primary,
                 linewidth=2, markersize=8, markerfacecolor='white',
                 markeredgewidth=2, markeredgecolor=color_primary)
        z = np.polyfit(df['day'], df['steps'], 1)
        p = np.poly1d(z)
        ax1.plot(df['day'], p(df['day']), "--", color=color_quaternary,
                 alpha=0.7, linewidth=2, label=f'Trend')
        ax1.set_xlabel('Day', fontweight='bold')
        ax1.set_ylabel('Steps', fontweight='bold')
        ax1.set_title('A) Daily Step Count', fontweight='bold', loc='left', pad=10)
        ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.legend(frameon=False)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)

        # Panel B: Sleep Efficiency over time
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(df['day'], df['sleep_efficiency'], 's-', color=color_secondary,
                 linewidth=2, markersize=8, markerfacecolor='white',
                 markeredgewidth=2, markeredgecolor=color_secondary)
        ax2.axhline(y=df['sleep_efficiency'].mean(), color='gray',
                    linestyle='--', alpha=0.7, linewidth=1.5, label='Mean')
        ax2.set_xlabel('Day', fontweight='bold')
        ax2.set_ylabel('Sleep Efficiency (%)', fontweight='bold')
        ax2.set_title('B) Sleep Efficiency', fontweight='bold', loc='left', pad=10)
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.legend(frameon=False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)

        # Panel C: Activity-Sleep Correlation
        ax3 = fig.add_subplot(gs[1, 0])
        bars = ax3.bar(df['day'], df['activity_sleep_correlation'],
                       color=color_tertiary, alpha=0.8, edgecolor='black', linewidth=1.2)
        ax3.axhline(y=df['activity_sleep_correlation'].mean(), color='red',
                    linestyle='--', alpha=0.7, linewidth=2, label='Mean')
        ax3.set_xlabel('Day', fontweight='bold')
        ax3.set_ylabel('Correlation Coefficient', fontweight='bold')
        ax3.set_title('C) Activity-Sleep Correlation', fontweight='bold', loc='left', pad=10)
        ax3.legend(frameon=False)
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)

        # Panel D: Recovery Ratio
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(df['day'], df['recovery_ratio'], '^-', color=color_quaternary,
                 linewidth=2, markersize=8, markerfacecolor='white',
                 markeredgewidth=2, markeredgecolor=color_quaternary)
        ax4.fill_between(df['day'], df['recovery_ratio'], alpha=0.3, color=color_quaternary)
        ax4.set_xlabel('Day', fontweight='bold')
        ax4.set_ylabel('Recovery Ratio', fontweight='bold')
        ax4.set_title('D) Recovery Ratio', fontweight='bold', loc='left', pad=10)
        ax4.grid(True, alpha=0.3, linestyle='--')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)

        # Panel E: Scatter - Steps vs Sleep Efficiency
        ax5 = fig.add_subplot(gs[2, 0])
        scatter = ax5.scatter(df['steps'], df['sleep_efficiency'],
                              c=df['activity_sleep_correlation'],
                              s=200, cmap='viridis', alpha=0.7,
                              edgecolors='black', linewidth=1.5)
        # Add regression line
        z = np.polyfit(df['steps'], df['sleep_efficiency'], 1)
        p = np.poly1d(z)
        ax5.plot(df['steps'], p(df['steps']), "--", color='red',
                 alpha=0.7, linewidth=2)
        ax5.set_xlabel('Steps', fontweight='bold')
        ax5.set_ylabel('Sleep Efficiency (%)', fontweight='bold')
        ax5.set_title('E) Steps vs Sleep Efficiency', fontweight='bold', loc='left', pad=10)
        cbar = plt.colorbar(scatter, ax=ax5)
        cbar.set_label('Correlation', fontweight='bold')
        ax5.spines['top'].set_visible(False)
        ax5.spines['right'].set_visible(False)

        # Panel F: Heatmap of key metrics
        ax6 = fig.add_subplot(gs[2, 1])
        metrics_df = df[['activity_sleep_correlation', 'sleep_efficiency',
                         'recovery_ratio', 'circadian_alignment_score']].T
        metrics_df.columns = [f'D{i}' for i in range(1, len(df) + 1)]

        # Normalize for better visualization
        metrics_normalized = (metrics_df - metrics_df.min(axis=1).values.reshape(-1, 1)) / \
                             (metrics_df.max(axis=1) - metrics_df.min(axis=1)).values.reshape(-1, 1)

        im = ax6.imshow(metrics_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax6.set_yticks(range(len(metrics_df.index)))
        ax6.set_yticklabels(['Act-Sleep Corr', 'Sleep Eff', 'Recovery', 'Circadian'],
                            fontsize=9)
        ax6.set_xticks(range(len(df)))
        ax6.set_xticklabels([f'D{i}' for i in range(1, len(df) + 1)], fontsize=8)
        ax6.set_title('F) Normalized Metrics Heatmap', fontweight='bold', loc='left', pad=10)
        cbar2 = plt.colorbar(im, ax=ax6)
        cbar2.set_label('Normalized Value', fontweight='bold')

        plt.suptitle('Activity and Sleep Analysis: Multi-Panel Overview',
                     fontsize=16, fontweight='bold', y=0.995)

        return fig


    # Create and save the figure
    fig1 = create_multipanel_figure(df)
    plt.savefig('multipanel_analysis.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('multipanel_analysis.pdf', bbox_inches='tight', facecolor='white')
    print("Multi-panel figure saved!")
    plt.show()


    def create_correlation_matrix(df):
        """
        Creates a professional correlation matrix with annotations
        """
        fig, ax = plt.subplots(figsize=(10, 8))

        # Select numeric columns
        numeric_cols = ['activity_sleep_correlation', 'steps', 'sleep_efficiency',
                        'recovery_ratio', 'circadian_alignment_score']
        corr_matrix = df[numeric_cols].corr()

        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)

        # Custom colormap
        cmap = sns.diverging_palette(250, 10, as_cmap=True)

        # Create heatmap
        sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.3f',
                    cmap=cmap, center=0, square=True, linewidths=2,
                    cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
                    annot_kws={"fontsize": 11, "fontweight": "bold"},
                    vmin=-1, vmax=1, ax=ax)

        # Customize labels
        labels = ['Act-Sleep\nCorr', 'Steps', 'Sleep\nEfficiency',
                  'Recovery\nRatio', 'Circadian\nScore']
        ax.set_xticklabels(labels, rotation=45, ha='right', fontweight='bold')
        ax.set_yticklabels(labels, rotation=0, fontweight='bold')

        ax.set_title('Correlation Matrix of Activity and Sleep Metrics',
                     fontsize=14, fontweight='bold', pad=20)

        plt.tight_layout()
        return fig


    fig2 = create_correlation_matrix(df)
    plt.savefig('correlation_matrix.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('correlation_matrix.pdf', bbox_inches='tight', facecolor='white')
    print("Correlation matrix saved!")
    plt.show()


    def create_scatter_with_marginals(df):
        """
        Creates scatter plot with marginal distributions (joint plot)
        """
        # Set style
        sns.set_style("whitegrid")

        # Create joint plot
        g = sns.JointGrid(data=df, x="steps", y="sleep_efficiency", height=8, ratio=5)

        # Main scatter plot
        g.plot_joint(sns.scatterplot, s=200, alpha=0.7, edgecolor='black',
                     linewidth=1.5, color='#2E86AB')

        # Add regression line
        g.plot_joint(sns.regplot, scatter=False, color='red',
                     line_kws={'linewidth': 2, 'linestyle': '--'})

        # Marginal distributions
        g.plot_marginals(sns.histplot, kde=True, color='#2E86AB', alpha=0.6)

        # Customize
        g.set_axis_labels('Steps', 'Sleep Efficiency (%)', fontweight='bold', fontsize=12)
        g.fig.suptitle('Steps vs Sleep Efficiency with Marginal Distributions',
                       fontsize=14, fontweight='bold', y=1.00)

        # Add correlation coefficient
        corr = df['steps'].corr(df['sleep_efficiency'])
        g.ax_joint.text(0.05, 0.95, f'r = {corr:.3f}',
                        transform=g.ax_joint.transAxes,
                        fontsize=12, fontweight='bold',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        return g.fig


    fig3 = create_scatter_with_marginals(df)
    plt.savefig('scatter_marginals.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('scatter_marginals.pdf', bbox_inches='tight', facecolor='white')
    print("Scatter with marginals saved!")
    plt.show()


    def create_dual_axis_timeseries(df):
        """
        Creates a time series plot with dual y-axes
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))

        color1 = '#2E86AB'
        color2 = '#A23B72'

        # First axis - Steps
        ax1.set_xlabel('Day', fontweight='bold', fontsize=12)
        ax1.set_ylabel('Steps', color=color1, fontweight='bold', fontsize=12)
        line1 = ax1.plot(df['day'], df['steps'], 'o-', color=color1,
                         linewidth=2.5, markersize=10, markerfacecolor='white',
                         markeredgewidth=2.5, label='Steps')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.fill_between(df['day'], df['steps'], alpha=0.2, color=color1)
        ax1.grid(True, alpha=0.3, linestyle='--')

        # Second axis - Sleep Efficiency
        ax2 = ax1.twinx()
        ax2.set_ylabel('Sleep Efficiency (%)', color=color2, fontweight='bold', fontsize=12)
        line2 = ax2.plot(df['day'], df['sleep_efficiency'], 's-', color=color2,
                         linewidth=2.5, markersize=10, markerfacecolor='white',
                         markeredgewidth=2.5, label='Sleep Efficiency')
        ax2.tick_params(axis='y', labelcolor=color2)

        # Title
        ax1.set_title('Daily Steps and Sleep Efficiency Over Time',
                      fontsize=14, fontweight='bold', pad=20)

        # Combined legend
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left', frameon=True,
                   fancybox=True, shadow=True, fontsize=11)

        # Remove top spines
        ax1.spines['top'].set_visible(False)
        ax2.spines['top'].set_visible(False)

        plt.tight_layout()
        return fig


    fig4 = create_dual_axis_timeseries(df)
    plt.savefig('dual_axis_timeseries.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('dual_axis_timeseries.pdf', bbox_inches='tight', facecolor='white')
    print("Dual axis time series saved!")
    plt.show()


    def create_violin_comparison(df):
        """
        Creates violin plots for key metrics
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # Prepare data
        metrics = {
            'Sleep Efficiency (%)': df['sleep_efficiency'],
            'Recovery Ratio': df['recovery_ratio'],
            'Activity-Sleep\nCorrelation': df['activity_sleep_correlation']
        }

        colors = ['#2E86AB', '#A23B72', '#F18F01']

        for idx, (metric_name, metric_data) in enumerate(metrics.items()):
            ax = axes[idx]

            # Create violin plot
            parts = ax.violinplot([metric_data], positions=[0], widths=0.7,
                                  showmeans=True, showextrema=True, showmedians=True)

            # Customize violin colors
            for pc in parts['bodies']:
                pc.set_facecolor(colors[idx])
                pc.set_alpha(0.7)
                pc.set_edgecolor('black')
                pc.set_linewidth(1.5)

            # Customize other elements
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                if partname in parts:
                    vp = parts[partname]
                    vp.set_edgecolor('black')
                    vp.set_linewidth(2)

            # Add individual points
            ax.scatter(np.zeros(len(metric_data)), metric_data,
                       alpha=0.6, s=80, color='black', zorder=3)

            # Statistics
            mean_val = metric_data.mean()
            median_val = metric_data.median()
            std_val = metric_data.std()

            # Add text box with statistics
            textstr = f'Mean: {mean_val:.3f}\nMedian: {median_val:.3f}\nSD: {std_val:.3f}'
            ax.text(0.5, 0.95, textstr, transform=ax.transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

            ax.set_ylabel(metric_name, fontweight='bold', fontsize=11)
            ax.set_xticks([])
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['bottom'].set_visible(False)
            ax.grid(True, alpha=0.3, axis='y', linestyle='--')

        plt.suptitle('Distribution of Key Metrics', fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        return fig


    fig5 = create_violin_comparison(df)
    plt.savefig('violin_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('violin_comparison.pdf', bbox_inches='tight', facecolor='white')
    print("Violin comparison saved!")
    plt.show()


    def create_radar_chart(df):
        """
        Creates radar charts comparing different days
        """
        from math import pi

        # Select days to compare (first, middle, last, and best)
        best_day_idx = df['activity_sleep_correlation'].idxmax()
        days_to_compare = [0, len(df) // 2, len(df) - 1, best_day_idx]

        # Metrics for radar chart
        categories = ['Sleep\nEfficiency', 'Recovery\nRatio',
                      'Act-Sleep\nCorr', 'Circadian\nScore']

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

        # Number of variables
        num_vars = len(categories)
        angles = [n / float(num_vars) * 2 * pi for n in range(num_vars)]
        angles += angles[:1]

        # Colors for different days
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']

        for idx, day_idx in enumerate(days_to_compare):
            # Normalize values to 0-1 scale
            values = [
                df.loc[day_idx, 'sleep_efficiency'] / 100,
                df.loc[day_idx, 'recovery_ratio'],
                df.loc[day_idx, 'activity_sleep_correlation'],
                df.loc[day_idx, 'circadian_alignment_score']
            ]
            values += values[:1]

            # Plot
            ax.plot(angles, values, 'o-', linewidth=2.5,
                    label=f'Day {day_idx + 1}', color=colors[idx])
            ax.fill(angles, values, alpha=0.15, color=colors[idx])

        # Fix axis to go in the right order
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontweight='bold', fontsize=11)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.7)

        # Add legend
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
                  frameon=True, fancybox=True, shadow=True, fontsize=11)

        plt.title('Multi-Metric Comparison Across Days',
                  fontsize=14, fontweight='bold', pad=30)

        return fig


    fig6 = create_radar_chart(df)
    plt.savefig('radar_comparison.png', dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig('radar_comparison.pdf', bbox_inches='tight', facecolor='white')
    print("Radar chart saved!")
    plt.show()





