#!/usr/bin/env python3
"""
Generate panels for Partition Lag section of Fluid Dynamics paper.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def main():
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Panel A: Partition Lag Distribution in Fluids
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Different fluids have different lag distributions
    tau = np.linspace(0, 20, 200)
    
    # Light gas (fast)
    tau_gas = 2
    dist_gas = (tau / tau_gas**2) * np.exp(-tau / tau_gas)
    
    # Liquid (medium)
    tau_liquid = 5
    dist_liquid = (tau / tau_liquid**2) * np.exp(-tau / tau_liquid)
    
    # Viscous (slow)
    tau_viscous = 10
    dist_viscous = (tau / tau_viscous**2) * np.exp(-tau / tau_viscous)
    
    ax1.plot(tau, dist_gas, 'b-', linewidth=2.5, label='Gas (He)')
    ax1.plot(tau, dist_liquid, 'g-', linewidth=2.5, label='Liquid (H₂O)')
    ax1.plot(tau, dist_viscous, 'r-', linewidth=2.5, label='Viscous (Glycerol)')
    
    ax1.set_xlabel(r'Partition lag $\tau_p$ (ps)', fontsize=11)
    ax1.set_ylabel('Probability density', fontsize=11)
    ax1.set_title('(A) Partition Lag Distributions: Faster in Gases', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Panel B: Temperature Dependence
    ax2 = fig.add_subplot(gs[0, 1])
    
    T = np.linspace(200, 500, 100)  # Kelvin
    k_B = 1.38e-23
    m = 3e-26  # kg (water-like)
    
    # tau_p ~ sqrt(m/(k_B T))
    tau_p = 1e12 * np.sqrt(m / (k_B * T))  # in ps
    
    ax2.plot(T, tau_p, 'b-', linewidth=2.5)
    ax2.fill_between(T, 0, tau_p, alpha=0.2, color='blue')
    
    # Reference points
    ax2.axvline(x=273, color='cyan', linestyle='--', alpha=0.7, label='0°C')
    ax2.axvline(x=373, color='orange', linestyle='--', alpha=0.7, label='100°C')
    
    ax2.set_xlabel('Temperature T (K)', fontsize=11)
    ax2.set_ylabel(r'Mean partition lag $\langle\tau_p\rangle$ (ps)', fontsize=11)
    ax2.set_title('(B) Temperature Dependence: Higher T → Faster Partitioning', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    ax2.text(350, max(tau_p)*0.8, r'$\tau_p \propto \sqrt{\frac{m}{k_B T}}$',
             fontsize=12, 
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel C: Collision Time vs Partition Lag
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Collision time and partition lag
    density = np.linspace(0.1, 2.0, 100)  # relative to standard
    
    tau_collision = 1 / density  # inversely proportional to density
    tau_partition = 0.5 / density + 0.3  # includes minimum lag
    
    ax3.plot(density, tau_collision, 'b-', linewidth=2.5, label='Collision time τ_c')
    ax3.plot(density, tau_partition, 'r-', linewidth=2.5, label='Partition lag τ_p')
    
    ax3.axhline(y=0.3, color='gray', linestyle=':', alpha=0.7, label='Minimum τ_p (uncertainty)')
    
    ax3.set_xlabel('Relative density ρ/ρ₀', fontsize=11)
    ax3.set_ylabel('Time (normalized)', fontsize=11)
    ax3.set_title('(C) Collision Time vs Partition Lag', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax3.text(1.0, 1.5, r'$\tau_p = \max\left(\frac{1}{\nu_c}, \frac{\hbar}{\Delta E}\right)$',
             fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel D: Pressure Dependence
    ax4 = fig.add_subplot(gs[1, 1])
    
    P = np.logspace(-1, 2, 100)  # atm
    
    # Higher pressure -> more collisions -> shorter lag
    tau_p_P = 5 / np.sqrt(P)  # ps
    
    ax4.loglog(P, tau_p_P, 'g-', linewidth=2.5)
    
    ax4.axvline(x=1, color='gray', linestyle='--', alpha=0.7, label='1 atm')
    
    ax4.set_xlabel('Pressure P (atm)', fontsize=11)
    ax4.set_ylabel(r'Partition lag $\tau_p$ (ps)', fontsize=11)
    ax4.set_title('(D) Pressure Dependence: Higher P → Faster Partitioning', fontweight='bold')
    ax4.legend(loc='upper right', fontsize=9)
    ax4.grid(True, alpha=0.3, which='both')
    
    ax4.text(3, 3, r'$\tau_p \propto P^{-1/2}$',
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    plt.suptitle('Panel F-A: Molecular Partition Lag in Fluids', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_partition_lag.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_partition_lag.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_partition_lag.png'}")
    print(f"Saved: {output_dir / 'panel_partition_lag.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

