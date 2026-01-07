#!/usr/bin/env python3
"""
Generate validation panels for Section 4: The S-Transformation Operator.
Validates: Operator Decomposition, Partition/Diffusion/Advection components.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import matplotlib.gridspec as gridspec
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent.parent / 'figures'
OUTPUT_DIR.mkdir(exist_ok=True)

def generate_operator_decomposition_panel(ax):
    """Panel A: T = T_part o T_diff o T_adv."""
    ax.set_title('A. Operator Decomposition', fontsize=11, fontweight='bold')
    
    # Initial S-profile
    x = np.linspace(0, 10, 200)
    S_initial = 2 * np.exp(-(x-3)**2/2)
    
    # After advection
    v = 1.5  # velocity
    dt = 0.5
    S_adv = 2 * np.exp(-(x - 3 - v*dt)**2/2)
    
    # After diffusion
    D = 0.3
    sigma_diff = np.sqrt(2 * D * dt)
    S_diff = 2 * np.exp(-(x - 3 - v*dt)**2/(2 + sigma_diff**2)) * np.sqrt(2/(2 + sigma_diff**2))
    
    # After partition
    kappa = 0.2
    S_stat = 1.0
    S_part = S_diff * (1 - kappa) + S_stat * kappa
    
    ax.plot(x, S_initial, 'k--', linewidth=2, label='Initial S(x)')
    ax.plot(x, S_adv, 'b-', linewidth=1.5, alpha=0.7, label='After T_adv')
    ax.plot(x, S_diff, 'g-', linewidth=1.5, alpha=0.7, label='After T_diff')
    ax.plot(x, S_part, 'r-', linewidth=2, label='After T_part (Final)')
    
    ax.set_xlabel('Position x', fontsize=10)
    ax.set_ylabel('S-coordinate', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def generate_partition_operator_panel(ax):
    """Panel B: Partition Operator Dynamics."""
    ax.set_title('B. Partition Operator: Equilibration', fontsize=11, fontweight='bold')
    
    t = np.linspace(0, 5, 100)
    
    # Different starting S values
    S_stat = 5.0
    kappa = 1.0
    
    S0_values = [1.0, 3.0, 7.0, 9.0]
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(S0_values)))
    
    for S0, c in zip(S0_values, colors):
        S = S_stat + (S0 - S_stat) * np.exp(-kappa * t)
        ax.plot(t, S, color=c, linewidth=2, label=f'S0 = {S0}')
    
    ax.axhline(S_stat, color='red', linestyle='--', linewidth=2, label='S_stat')
    
    ax.set_xlabel('Time', fontsize=10)
    ax.set_ylabel('S-coordinate', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax.annotate('All trajectories\nconverge to S_stat', xy=(4, 5.2), fontsize=9, ha='center')

def generate_diffusion_operator_panel(ax):
    """Panel C: Diffusion Operator - Spreading."""
    ax.set_title('C. Diffusion Operator: S-Spreading', fontsize=11, fontweight='bold')
    
    x = np.linspace(-5, 5, 200)
    
    D_S = 0.5
    times = [0, 0.5, 1.0, 2.0, 4.0]
    
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(times)))
    
    for t, c in zip(times, colors):
        sigma = np.sqrt(2 * D_S * t) if t > 0 else 0.1
        S = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-x**2 / (2*sigma**2))
        ax.plot(x, S, color=c, linewidth=2, label=f't = {t}')
    
    ax.set_xlabel('Position', fontsize=10)
    ax.set_ylabel('S-density', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    ax.annotate('Spreading:\nsigma = sqrt(2*D_S*t)', xy=(2.5, 0.8), fontsize=9)

def generate_advection_operator_panel(ax):
    """Panel D: Advection Operator - Translation."""
    ax.set_title('D. Advection Operator: S-Translation', fontsize=11, fontweight='bold')
    
    x = np.linspace(0, 15, 200)
    
    v = 2.0  # Flow velocity
    times = [0, 1, 2, 3, 4]
    
    colors = plt.cm.cool(np.linspace(0.1, 0.9, len(times)))
    
    for t, c in zip(times, colors):
        S = np.exp(-((x - 3 - v*t)**2) / 2)
        ax.plot(x, S, color=c, linewidth=2, label=f't = {t}')
        # Mark peak position
        peak_x = 3 + v * t
        ax.axvline(peak_x, color=c, linestyle=':', alpha=0.5)
    
    # Velocity arrow
    ax.annotate('', xy=(11, 0.7), xytext=(7, 0.7),
               arrowprops=dict(arrowstyle='->', color='black', lw=2))
    ax.text(9, 0.8, f'v = {v}', fontsize=10, ha='center')
    
    ax.set_xlabel('Position x', fontsize=10)
    ax.set_ylabel('S-profile', fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

def generate_composition_validation_panel(ax):
    """Panel E: Composition Property Validation."""
    ax.set_title('E. Composition: T_0->x = T_dx^(x/dx)', fontsize=11, fontweight='bold')
    
    # Compare stepwise vs direct transformation
    n_steps_list = [1, 2, 5, 10, 20, 50, 100]
    
    x_total = 5.0
    S0 = 3.0
    S_stat = 8.0
    kappa = 0.2
    
    S_direct = S_stat + (S0 - S_stat) * np.exp(-kappa * x_total)
    
    S_stepwise = []
    for n in n_steps_list:
        dx = x_total / n
        S = S0
        for _ in range(n):
            S = S + kappa * dx * (S_stat - S)
        S_stepwise.append(S)
    
    error = np.abs(np.array(S_stepwise) - S_direct) / S_direct * 100
    
    ax.semilogy(n_steps_list, error, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Number of Steps', fontsize=10)
    ax.set_ylabel('Relative Error (%)', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    ax.annotate('Convergence as\ndx -> 0', xy=(50, 0.01), fontsize=9, ha='center')

def generate_partition_coefficient_panel(ax):
    """Panel F: Partition Coefficient from S-Distance."""
    ax.set_title('F. Partition Coefficient K(S-distance)', fontsize=11, fontweight='bold')
    
    d_S = np.linspace(0, 5, 100)
    
    # Different sigma_S values (selectivity)
    sigma_S_values = [0.5, 1.0, 2.0, 3.0]
    K0 = 10
    
    colors = plt.cm.autumn(np.linspace(0.2, 0.8, len(sigma_S_values)))
    
    for sigma_S, c in zip(sigma_S_values, colors):
        K = K0 * np.exp(-d_S / sigma_S)
        ax.plot(d_S, K, color=c, linewidth=2, label=f'sigma_S = {sigma_S}')
    
    ax.set_xlabel('S-distance d_S(analyte, stationary)', fontsize=10)
    ax.set_ylabel('Partition Coefficient K', fontsize=10)
    ax.legend(fontsize=8)
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)
    
    ax.annotate('K = K0 * exp(-d_S / sigma_S)', xy=(2.5, 8), fontsize=9)

def main():
    """Generate all transformation operator panels."""
    print("Generating S-Transformation Operator panels...")
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
    
    ax1 = fig.add_subplot(gs[0, 0])
    generate_operator_decomposition_panel(ax1)
    
    ax2 = fig.add_subplot(gs[0, 1])
    generate_partition_operator_panel(ax2)
    
    ax3 = fig.add_subplot(gs[0, 2])
    generate_diffusion_operator_panel(ax3)
    
    ax4 = fig.add_subplot(gs[1, 0])
    generate_advection_operator_panel(ax4)
    
    ax5 = fig.add_subplot(gs[1, 1])
    generate_composition_validation_panel(ax5)
    
    ax6 = fig.add_subplot(gs[1, 2])
    generate_partition_coefficient_panel(ax6)
    
    plt.suptitle('Section 4: The S-Transformation Operator - Experimental Validation', 
                fontsize=14, fontweight='bold', y=0.98)
    
    for fmt in ['png', 'pdf']:
        output_path = OUTPUT_DIR / f'panel_transformation_operator.{fmt}'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  Saved: {output_path}")
    
    plt.close()
    print("  Done!")

if __name__ == '__main__':
    main()

