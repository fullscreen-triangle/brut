#!/usr/bin/env python3
"""
Generate panels for Coupling and Phase-Lock Networks in Fluid Dynamics.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx
from matplotlib.patches import Circle, FancyArrowPatch
from pathlib import Path

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

def create_phase_lock_network(n_molecules=20, coupling_prob=0.3, seed=42):
    """Create a phase-lock network for molecules."""
    np.random.seed(seed)
    G = nx.Graph()
    
    # Add molecules with positions
    for i in range(n_molecules):
        G.add_node(i, pos=(np.random.uniform(0, 3), np.random.uniform(0, 2)))
    
    # Add edges based on proximity and coupling probability
    positions = nx.get_node_attributes(G, 'pos')
    for i in range(n_molecules):
        for j in range(i+1, n_molecules):
            dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                          (positions[i][1] - positions[j][1])**2)
            # Coupling probability decreases with distance
            p_coupling = coupling_prob * np.exp(-dist / 0.5)
            if np.random.random() < p_coupling:
                # Assign coupling type
                coupling_type = np.random.choice(['vdw', 'dipole', 'hbond'], 
                                                  p=[0.6, 0.3, 0.1])
                G.add_edge(i, j, type=coupling_type, weight=1/dist)
    
    return G

def main():
    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Panel A: Phase-Lock Network Structure
    ax1 = fig.add_subplot(gs[0, 0])
    
    G = create_phase_lock_network(n_molecules=25, coupling_prob=0.4, seed=42)
    pos = nx.get_node_attributes(G, 'pos')
    
    # Color nodes by degree
    degrees = dict(G.degree())
    node_colors = [degrees[n] for n in G.nodes()]
    
    # Color edges by type
    edge_colors = {'vdw': 'gray', 'dipole': 'blue', 'hbond': 'red'}
    edge_styles = {'vdw': '-', 'dipole': '--', 'hbond': '-'}
    edge_widths = {'vdw': 1, 'dipole': 1.5, 'hbond': 2}
    
    for edge_type, color in edge_colors.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d.get('type') == edge_type]
        nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax1, 
                               edge_color=color, width=edge_widths[edge_type],
                               style=edge_styles[edge_type], alpha=0.6)
    
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=node_colors, 
                                    node_size=200, cmap=plt.cm.YlOrRd,
                                    edgecolors='black', linewidths=0.5)
    
    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='gray', linewidth=1, label='Van der Waals'),
        Line2D([0], [0], color='blue', linewidth=1.5, linestyle='--', label='Dipole-Dipole'),
        Line2D([0], [0], color='red', linewidth=2, label='H-Bond')
    ]
    ax1.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    ax1.set_title('(A) Phase-Lock Network: Molecular Coupling Types', fontweight='bold')
    ax1.set_xlim(-0.3, 3.3)
    ax1.set_ylim(-0.3, 2.3)
    ax1.axis('off')
    
    # Panel B: Coupling Strength vs Distance
    ax2 = fig.add_subplot(gs[0, 1])
    
    r = np.linspace(0.3, 2.0, 100)  # nm
    
    # Different coupling types
    g_vdw = 1 / r**6  # Van der Waals
    g_dipole = 1 / r**3  # Dipole-dipole
    g_hbond = 5 * np.exp(-3 * (r - 0.3))  # H-bond (short range)
    
    ax2.semilogy(r, g_vdw / max(g_vdw), 'gray', linewidth=2.5, label='Van der Waals (~r⁻⁶)')
    ax2.semilogy(r, g_dipole / max(g_dipole), 'b--', linewidth=2.5, label='Dipole-Dipole (~r⁻³)')
    ax2.semilogy(r, g_hbond / max(g_hbond), 'r-', linewidth=2.5, label='H-Bond (exp)')
    
    ax2.set_xlabel('Intermolecular distance r (nm)', fontsize=11)
    ax2.set_ylabel('Coupling strength g(r) (normalized)', fontsize=11)
    ax2.set_title('(B) Coupling Strength vs Distance', fontweight='bold')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3, which='both')
    ax2.set_ylim(1e-4, 2)
    
    # Panel C: Network Density and Cohesive Energy
    ax3 = fig.add_subplot(gs[1, 0])
    
    # Simulate different network densities
    densities = np.linspace(0.1, 0.8, 50)
    
    # Cohesive energy proportional to edge count
    cohesive_energy = densities**2 * 10  # kJ/mol
    
    # Different phases
    gas_mask = densities < 0.2
    liquid_mask = (densities >= 0.2) & (densities < 0.6)
    solid_mask = densities >= 0.6
    
    ax3.fill_between(densities[gas_mask], 0, cohesive_energy[gas_mask], 
                     alpha=0.3, color='lightblue', label='Gas phase')
    ax3.fill_between(densities[liquid_mask], 0, cohesive_energy[liquid_mask], 
                     alpha=0.3, color='lightgreen', label='Liquid phase')
    ax3.fill_between(densities[solid_mask], 0, cohesive_energy[solid_mask], 
                     alpha=0.3, color='lightyellow', label='Solid phase')
    
    ax3.plot(densities, cohesive_energy, 'k-', linewidth=2.5)
    
    ax3.set_xlabel(r'Network density $\rho_G = 2|E|/(N(N-1))$', fontsize=11)
    ax3.set_ylabel('Cohesive energy (kJ/mol)', fontsize=11)
    ax3.set_title('(C) Cohesive Energy from Network Density', fontweight='bold')
    ax3.legend(loc='upper left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    ax3.text(0.5, 5, r'$E_{cohesive} = \sum_{(i,j) \in E} g_{ij}$',
             fontsize=11, ha='center',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
    
    # Panel D: Transport via Phase-Lock Navigation
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_xlim(-0.5, 4)
    ax4.set_ylim(-0.5, 2)
    ax4.set_aspect('equal')
    
    # Draw molecules as circles
    np.random.seed(44)
    n_mol = 12
    positions = [(np.random.uniform(0.2, 3.8), np.random.uniform(0.2, 1.8)) for _ in range(n_mol)]
    
    for i, (x, y) in enumerate(positions):
        circle = Circle((x, y), 0.15, facecolor='lightblue', edgecolor='black', linewidth=1)
        ax4.add_patch(circle)
    
    # Draw coupling network
    for i in range(n_mol):
        for j in range(i+1, n_mol):
            dist = np.sqrt((positions[i][0] - positions[j][0])**2 + 
                          (positions[i][1] - positions[j][1])**2)
            if dist < 0.8:
                ax4.plot([positions[i][0], positions[j][0]], 
                        [positions[i][1], positions[j][1]], 
                        'gray', linewidth=0.5, alpha=0.5)
    
    # Highlight transport path
    path_indices = [0, 3, 7, 10]
    path_positions = [positions[i] for i in path_indices]
    
    for i in range(len(path_positions) - 1):
        ax4.annotate('', xy=path_positions[i+1], xytext=path_positions[i],
                    arrowprops=dict(arrowstyle='->', color='red', lw=2))
    
    # Highlight path molecules
    for idx in path_indices:
        circle = Circle(positions[idx], 0.15, facecolor='red', edgecolor='black', linewidth=2, alpha=0.7)
        ax4.add_patch(circle)
    
    ax4.text(2, -0.3, 'Transport = Navigation through phase-lock network',
             ha='center', fontsize=10,
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    
    ax4.set_title('(D) Transport as Network Navigation', fontweight='bold')
    ax4.axis('off')
    
    plt.suptitle('Panel F-B: Intermolecular Coupling and Phase-Lock Networks', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    # Save
    output_dir = Path(__file__).parent.parent / 'figures'
    output_dir.mkdir(exist_ok=True)
    
    plt.savefig(output_dir / 'panel_coupling_networks.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig(output_dir / 'panel_coupling_networks.pdf', bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print(f"Saved: {output_dir / 'panel_coupling_networks.png'}")
    print(f"Saved: {output_dir / 'panel_coupling_networks.pdf'}")
    
    plt.close()

if __name__ == "__main__":
    main()

