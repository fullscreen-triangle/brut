#!/usr/bin/env python3
"""
Generate panels for Molecular Apertures in Fluid Dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle, Wedge, FancyBboxPatch
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def main():
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Panel A: Bond as Aperture
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_xlim(-0.5, 3.5)
    ax1.set_ylim(-0.5, 2)
    ax1.set_aspect('equal')
    
    # Two molecules with bond
    mol1 = Circle((1.0, 1.0), 0.25, facecolor='lightblue', edgecolor='blue', linewidth=2)
    mol2 = Circle((2.0, 1.0), 0.25, facecolor='lightcoral', edgecolor='red', linewidth=2)
    ax1.add_patch(mol1)
    ax1.add_patch(mol2)
    
    ax1.text(1.0, 1.0, 'A', ha='center', va='center', fontsize=14, fontweight='bold')
    ax1.text(2.0, 1.0, 'B', ha='center', va='center', fontsize=14, fontweight='bold')
    
    # Bond line
    ax1.plot([1.25, 1.75], [1.0, 1.0], 'k-', linewidth=4)
    
    # Aperture region (constraint)
    aperture = FancyBboxPatch((1.3, 0.8), 0.4, 0.4, boxstyle="round,pad=0.02",
                              facecolor='yellow', edgecolor='black', linewidth=1, alpha=0.5)
    ax1.add_patch(aperture)
    ax1.text(1.5, 0.6, 'Aperture\n(bond constraints)', ha='center', fontsize=9)
    
    # Allowed configurations (inside aperture)
    ax1.scatter([1.35, 1.5, 1.65], [0.9, 1.05, 0.95], c='green', s=30, marker='o', 
                label='Allowed configs')
    
    # Blocked configurations (outside aperture)
    ax1.scatter([0.8, 2.2, 1.5], [1.5, 0.5, 1.6], c='red', s=30, marker='x', 
                label='Blocked configs', linewidth=2)
    
    ax1.text(1.5, 1.7, 'Bond creates selective aperture:\nonly specific configs allowed',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax1.legend(loc='lower right', fontsize=8)
    ax1.set_title('(A) Chemical Bond as Categorical Aperture', fontweight='bold')
    ax1.axis('off')
    
    # Panel B: Selectivity Table
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis('off')
    
    # Create table data
    table_data = [
        ['Interaction Type', 'Selectivity (s)', 'Potential (Φ/kT)', 'Energy (kJ/mol)'],
        ['Covalent Bond', '~10⁻⁴', '~9.2', '200-400'],
        ['Ionic Bond', '~10⁻³', '~6.9', '100-300'],
        ['Hydrogen Bond', '~0.1', '~2.3', '10-40'],
        ['Dipole-Dipole', '~0.3', '~1.2', '5-25'],
        ['Van der Waals', '~0.5', '~0.7', '0.5-5']
    ]
    
    colors = [['lightgray']*4] + [
        ['mistyrose', 'mistyrose', 'mistyrose', 'mistyrose'],
        ['lightyellow', 'lightyellow', 'lightyellow', 'lightyellow'],
        ['lightgreen', 'lightgreen', 'lightgreen', 'lightgreen'],
        ['lightcyan', 'lightcyan', 'lightcyan', 'lightcyan'],
        ['white', 'white', 'white', 'white']
    ]
    
    table = ax2.table(cellText=table_data, cellColours=colors,
                      loc='center', cellLoc='center',
                      colWidths=[0.3, 0.2, 0.25, 0.25])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    
    ax2.set_title('(B) Aperture Selectivity by Interaction Type', fontweight='bold', y=0.95)
    
    # Panel C: Transport Rate via Aperture Product
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Number of apertures in path
    n_apertures = np.arange(1, 11)
    
    # Different selectivities
    s_weak = 0.9  # Weak interactions
    s_medium = 0.5  # Medium
    s_strong = 0.1  # Strong interactions
    
    rate_weak = s_weak ** n_apertures
    rate_medium = s_medium ** n_apertures
    rate_strong = s_strong ** n_apertures
    
    ax3.semilogy(n_apertures, rate_weak, 'g-o', linewidth=2, markersize=8, 
                 label=f's = {s_weak} (weak)')
    ax3.semilogy(n_apertures, rate_medium, 'b-s', linewidth=2, markersize=8, 
                 label=f's = {s_medium} (medium)')
    ax3.semilogy(n_apertures, rate_strong, 'r-^', linewidth=2, markersize=8, 
                 label=f's = {s_strong} (strong)')
    
    ax3.set_xlabel('Number of apertures in path', fontsize=11)
    ax3.set_ylabel('Relative transport rate', fontsize=11)
    ax3.set_title('(C) Transport Rate = Product of Selectivities', fontweight='bold')
    ax3.legend(loc='upper right', fontsize=9)
    ax3.grid(True, alpha=0.3, which='both')
    
    ax3.text(5, 0.5, r'Rate $\propto \prod_{a \in path} s_a$',
             fontsize=12, ha='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    # Panel D: Phase Transitions as Aperture Changes
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(-0.5, 4)
    ax4.set_ylim(-0.5, 2.5)
    ax4.set_aspect('equal')
    
    # Solid (many apertures, rigid)
    for i in range(3):
        for j in range(3):
            x, y = 0.2 + i*0.3, 1.5 + j*0.3
            circle = Circle((x, y), 0.1, facecolor='lightblue', edgecolor='blue', linewidth=1)
            ax4.add_patch(circle)
            # Strong bonds (thick lines)
            if i < 2:
                ax4.plot([x+0.1, x+0.2], [y, y], 'b-', linewidth=3)
            if j < 2:
                ax4.plot([x, x], [y+0.1, y+0.2], 'b-', linewidth=3)
    ax4.text(0.5, 1.1, 'Solid: Strong\napertures', ha='center', fontsize=9)
    
    # Liquid (fewer apertures, flexible)
    np.random.seed(42)
    for _ in range(9):
        x = np.random.uniform(1.3, 2.3)
        y = np.random.uniform(1.3, 2.3)
        circle = Circle((x, y), 0.1, facecolor='lightgreen', edgecolor='green', linewidth=1)
        ax4.add_patch(circle)
    # Weaker bonds (thinner lines)
    ax4.plot([1.5, 1.8], [1.8, 1.6], 'g-', linewidth=1.5, alpha=0.5)
    ax4.plot([1.9, 2.1], [2.0, 1.8], 'g-', linewidth=1.5, alpha=0.5)
    ax4.text(1.8, 1.1, 'Liquid: Weak\napertures', ha='center', fontsize=9)
    
    # Gas (no apertures)
    for _ in range(9):
        x = np.random.uniform(2.7, 3.7)
        y = np.random.uniform(1.3, 2.3)
        circle = Circle((x, y), 0.1, facecolor='lightyellow', edgecolor='orange', linewidth=1)
        ax4.add_patch(circle)
    ax4.text(3.2, 1.1, 'Gas: No\napertures', ha='center', fontsize=9)
    
    # Arrows for transitions
    ax4.annotate('', xy=(1.2, 1.9), xytext=(1.0, 1.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax4.text(1.1, 2.0, '+ΔH\n(melt)', fontsize=8, ha='center')
    
    ax4.annotate('', xy=(2.6, 1.9), xytext=(2.4, 1.9),
                arrowprops=dict(arrowstyle='->', color='red', lw=2))
    ax4.text(2.5, 2.0, '+ΔH\n(boil)', fontsize=8, ha='center')
    
    ax4.text(2.0, 0.3, 'Phase transition = Aperture destruction\nLatent heat = Sum of aperture potentials',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax4.set_title('(D) Phase Transitions: Aperture Reconfiguration', fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle('Panel F-C: Molecular Bonds as Categorical Apertures', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_molecular_apertures.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_molecular_apertures.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_molecular_apertures.png'}")
    print(f"Saved: {output_dir / 'panel_molecular_apertures.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

