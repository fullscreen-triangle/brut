#!/usr/bin/env python3
"""
Generate panels for Transport Coefficients in Fluid Dynamics.
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
    
    # Panel A: Viscosity from tau_p and g
    ax1 = fig.add_subplot(gs[0, 0])
    
    # Temperature range
    T = np.linspace(250, 400, 100)  # K
    
    # Viscosity model: mu = sum(tau_p * g)
    # tau_p decreases with T, g depends on interaction
    k_B = 1.38e-23
    
    # Water-like viscosity
    mu_water = 0.001 * np.exp(1800 / T)  # Pa·s (empirical fit)
    
    # Decompose into tau_p and g
    tau_p = 1e-12 * np.exp(500 / T)  # ps, increases at low T
    g_coupling = mu_water / tau_p  # Effective coupling
    
    ax1.plot(T, mu_water * 1000, 'b-', linewidth=2.5, label='Viscosity μ (mPa·s)')
    ax1.plot(T, tau_p * 1e12, 'r--', linewidth=2, label=r'Partition lag $\tau_p$ (ps)')
    ax1.plot(T, g_coupling * 1e3, 'g:', linewidth=2, label='Coupling g (scaled)')
    
    ax1.set_xlabel('Temperature T (K)', fontsize=11)
    ax1.set_ylabel('Value (scaled)', fontsize=11)
    ax1.set_title('(A) Viscosity = Partition Lag × Coupling', fontweight='bold')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    ax1.text(320, 3, r'$\mu = \sum_{i,j} \tau_{p,ij} \cdot g_{ij}$',
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel B: Thermal Conductivity
    ax2 = fig.add_subplot(gs[0, 1])
    
    # Different materials
    materials = ['Air', 'Water', 'Oil', 'Glycerol', 'Metal']
    tau_p_vals = [0.1, 1, 5, 20, 0.01]  # ps (relative)
    g_vals = [0.1, 5, 3, 10, 100]  # Coupling (relative)
    kappa_vals = [g / tau_p for g, tau_p in zip(g_vals, tau_p_vals)]
    
    x_pos = np.arange(len(materials))
    
    bars = ax2.bar(x_pos, kappa_vals, color=['lightblue', 'blue', 'orange', 'brown', 'gray'],
                   edgecolor='black', linewidth=1.5)
    
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(materials, fontsize=10)
    ax2.set_ylabel('Thermal Conductivity κ (relative)', fontsize=11)
    ax2.set_title('(B) Thermal Conductivity from g/τ_p', fontweight='bold')
    ax2.set_yscale('log')
    
    # Add value labels
    for bar, val in zip(bars, kappa_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2, 
                 f'{val:.1f}', ha='center', fontsize=9)
    
    ax2.text(2, 0.5, r'$\kappa \propto \frac{g}{\tau_p}$',
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel C: Diffusivity
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Particle size (radius)
    r = np.linspace(0.1, 5, 100)  # nm
    
    # Stokes-Einstein
    T_ref = 300  # K
    mu_ref = 0.001  # Pa·s (water)
    k_B = 1.38e-23
    
    D = k_B * T_ref / (6 * np.pi * mu_ref * r * 1e-9)  # m^2/s
    
    ax3.loglog(r, D * 1e9, 'b-', linewidth=2.5)
    
    ax3.set_xlabel('Particle radius r (nm)', fontsize=11)
    ax3.set_ylabel('Diffusivity D (nm²/ns)', fontsize=11)
    ax3.set_title('(C) Diffusivity from Stokes-Einstein', fontweight='bold')
    ax3.grid(True, alpha=0.3, which='both')
    
    # Mark key sizes
    sizes = {'H₂O': 0.15, 'Glucose': 0.5, 'Protein': 3}
    for name, size in sizes.items():
        D_val = k_B * T_ref / (6 * np.pi * mu_ref * size * 1e-9) * 1e9
        ax3.scatter([size], [D_val], s=80, zorder=5, edgecolors='black')
        ax3.annotate(name, xy=(size, D_val), xytext=(size*1.3, D_val*0.6),
                    fontsize=9)
    
    ax3.text(1, 50, r'$D = \frac{k_B T}{6\pi\mu r}$',
             fontsize=12,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel D: All Three Transport Coefficients
    ax4 = fig.add_subplot(gs[1, 1])
    
    # Unified formula visualization
    properties = ['Viscosity\n(μ)', 'Thermal Cond.\n(κ)', 'Diffusivity\n(D)']
    formulas = [r'$\tau_p \cdot g$', r'$g / \tau_p$', r'$k_B T / (6\pi\mu r)$']
    colors = ['royalblue', 'orangered', 'forestgreen']
    
    for i, (prop, formula, color) in enumerate(zip(properties, formulas, colors)):
        # Box
        rect = plt.Rectangle((0.1, 2.2 - i*0.8), 0.8, 0.6, 
                             facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
        ax4.add_patch(rect)
        ax4.text(0.5, 2.5 - i*0.8, prop, ha='center', va='center', 
                fontsize=11, fontweight='bold')
        
        # Formula
        ax4.text(1.5, 2.5 - i*0.8, formula, ha='center', va='center', fontsize=12)
        
        # Arrow
        ax4.annotate('', xy=(0.95, 2.5 - i*0.8), xytext=(1.05, 2.5 - i*0.8),
                    arrowprops=dict(arrowstyle='-', color='black', lw=1))
    
    # Common origin
    ax4.text(2.5, 1.7, 'All from\npartition lag\nand coupling!',
             ha='center', fontsize=11, fontweight='bold',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax4.set_xlim(0, 3.5)
    ax4.set_ylim(0, 3)
    ax4.set_title('(D) Unified Transport Coefficients', fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle('Panel F-D: Transport Coefficients from Partition Dynamics', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_transport_coefficients.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_transport_coefficients.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_transport_coefficients.png'}")
    print(f"Saved: {output_dir / 'panel_transport_coefficients.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

