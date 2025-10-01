"""
Heart Rate Variability as Coupling Signature Visualizations
Implements Template 2: HRV Coupling Analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import hilbert, butter, filtfilt
from scipy.stats import pearsonr
import pandas as pd
import os
from typing import Dict, List, Any, Tuple

class HRVCouplingVisualizer:
    def __init__(self):
        self.scale_pairs = [
            ('Cellular', 'Cardiac'),
            ('Cardiac', 'Respiratory'), 
            ('Respiratory', 'Autonomic'),
            ('Autonomic', 'Circadian'),
            ('Cellular', 'Respiratory')
        ]
        self.colors = ['#E74C3C', '#3498DB', '#2ECC71', '#F39C12', '#8E44AD']
        
    def generate_coupled_hrv_signal(self, duration: int = 300, fs: int = 4) -> Dict[str, np.ndarray]:
        """Generate synthetic HRV signal with multi-scale coupling"""
        t = np.arange(0, duration, 1/fs)
        
        # Scale oscillations with different frequencies
        oscillations = {
            'Cellular': 2.0 * np.sin(2 * np.pi * 2.5 * t),  # 2.5 Hz
            'Cardiac': 1.5 * np.sin(2 * np.pi * 1.0 * t),   # 1.0 Hz  
            'Respiratory': 1.0 * np.sin(2 * np.pi * 0.25 * t),  # 0.25 Hz
            'Autonomic': 0.8 * np.sin(2 * np.pi * 0.08 * t),    # 0.08 Hz
            'Circadian': 0.5 * np.sin(2 * np.pi * 0.0001 * t)   # Very low freq
        }
        
        # Coupling coefficients (matching healthy state)
        C_matrix = {
            ('Cellular', 'Cardiac'): 0.78,
            ('Cardiac', 'Respiratory'): 0.85,
            ('Respiratory', 'Autonomic'): 0.72,
            ('Autonomic', 'Circadian'): 0.69,
            ('Cellular', 'Respiratory'): 0.45
        }
        
        # Generate traditional HRV (sum of oscillations)
        traditional_hrv = sum(oscillations.values())
        traditional_hrv += 0.2 * np.random.normal(0, 1, len(t))  # Add noise
        
        # Generate coupling-based HRV using Equation 4
        coupling_hrv = np.zeros_like(t)
        coupling_components = {}
        
        for (scale1, scale2), C_ij in C_matrix.items():
            # Phase difference (random but consistent)
            phi_diff = np.random.uniform(0, 2*np.pi)
            
            # Coupling term: C_ij * cos(φ_i - φ_j) 
            coupling_term = C_ij * oscillations[scale1] * oscillations[scale2] * np.cos(phi_diff)
            coupling_components[(scale1, scale2)] = coupling_term
            coupling_hrv += coupling_term
        
        # Uncoupled residual component ε(t)
        residual = traditional_hrv - coupling_hrv
        
        return {
            'time': t,
            'traditional_hrv': traditional_hrv,
            'coupling_hrv': coupling_hrv, 
            'residual': residual,
            'oscillations': oscillations,
            'coupling_components': coupling_components
        }
    
    def plot_traditional_vs_coupling_hrv(self, data: Dict[str, np.ndarray], save_path: str):
        """Panel A: Traditional vs. Coupling-Based HRV"""
        fig, axes = plt.subplots(3, 1, figsize=(16, 12))
        
        t = data['time']
        traditional = data['traditional_hrv']
        coupling = data['coupling_hrv']
        residual = data['residual']
        
        # Top: Traditional HRV time series
        axes[0].plot(t, traditional, 'b-', linewidth=1.5, alpha=0.8, label='Traditional HRV')
        axes[0].set_ylabel('HRV Amplitude', fontsize=12, fontweight='bold')
        axes[0].set_title('Traditional HRV Time Series', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Bottom: Coupling-reconstructed HRV
        axes[1].plot(t, coupling, 'r-', linewidth=1.5, alpha=0.8, label='Coupling-Reconstructed HRV')
        axes[1].set_ylabel('HRV Amplitude', fontsize=12, fontweight='bold')
        axes[1].set_title('Coupling-Reconstructed HRV (Equation 4)', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Residual plot showing uncoupled components ε(t)
        axes[2].plot(t, residual, 'g-', linewidth=1.5, alpha=0.8, label='Uncoupled Components ε(t)')
        axes[2].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Residual Amplitude', fontsize=12, fontweight='bold')
        axes[2].set_title('Uncoupled Components ε(t) = Traditional - Coupling', fontsize=14, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Calculate and display R² correlation
        r2 = pearsonr(traditional, coupling)[0]**2
        
        # Add correlation text box
        correlation_text = f'R² = {r2:.3f}\nCorrelation between\nTraditional and Coupling-based HRV'
        fig.text(0.02, 0.5, correlation_text, fontsize=12, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.8),
                verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_coupling_component_decomposition(self, data: Dict[str, np.ndarray], save_path: str):
        """Panel B: Coupling Component Decomposition"""
        fig, ax = plt.subplots(1, 1, figsize=(16, 10))
        
        t = data['time']
        coupling_components = data['coupling_components']
        
        # Prepare data for stacked area chart
        component_data = []
        labels = []
        colors = []
        
        y_bottom = np.zeros_like(t)
        
        for i, ((scale1, scale2), component) in enumerate(coupling_components.items()):
            # Ensure all components are positive for stacking
            component_positive = component - np.min(component) + 0.1
            
            ax.fill_between(t, y_bottom, y_bottom + component_positive, 
                           alpha=0.7, color=self.colors[i % len(self.colors)],
                           label=f'C_{{{scale1},{scale2}}} cos(φ_i - φ_j)')
            
            y_bottom += component_positive
        
        ax.set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Coupling Amplitude', fontsize=12, fontweight='bold')
        ax.set_title('Coupling Component Decomposition\nStacked Contributions of Each Scale Pair', 
                    fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_phase_relationship_dynamics(self, data: Dict[str, np.ndarray], save_path: str):
        """Panel C: Phase Relationship Dynamics"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12), subplot_kw=dict(projection='polar'))
        axes = axes.flatten()
        
        oscillations = data['oscillations']
        t = data['time']
        
        # Calculate phase relationships for each significant pair
        for idx, (scale1, scale2) in enumerate(self.scale_pairs):
            if idx >= 6:  # We have 6 subplots
                break
                
            # Extract oscillations for this pair
            osc1 = oscillations[scale1]
            osc2 = oscillations[scale2]
            
            # Calculate instantaneous phase using Hilbert transform
            analytic1 = hilbert(osc1)
            analytic2 = hilbert(osc2)
            
            phase1 = np.angle(analytic1)
            phase2 = np.angle(analytic2)
            phase_diff = phase1 - phase2
            
            # Calculate coupling strength (simplified)
            coupling_strength = np.abs(np.mean(np.exp(1j * phase_diff)))
            
            # Create time evolution as color gradient
            colors = plt.cm.viridis(np.linspace(0, 1, len(t[::20])))  # Subsample for clarity
            
            # Plot phase differences
            for i, (phi, color) in enumerate(zip(phase_diff[::20], colors)):
                # Radial distance represents coupling strength
                r = coupling_strength * (1 + 0.3 * np.sin(0.1 * i))  # Some variation
                
                axes[idx].scatter(phi, r, c=[color], s=20, alpha=0.7)
            
            # Add coupling strength as reference circle
            theta_circle = np.linspace(0, 2*np.pi, 100)
            r_circle = np.full_like(theta_circle, coupling_strength)
            axes[idx].plot(theta_circle, r_circle, 'k--', alpha=0.5, linewidth=2)
            
            axes[idx].set_ylim(0, 1)
            axes[idx].set_title(f'{scale1} ↔ {scale2}\nCoupling: {coupling_strength:.3f}', 
                              fontsize=12, fontweight='bold', pad=20)
            axes[idx].set_theta_zero_location('N')
            axes[idx].set_theta_direction(1)
            
            # Add radial grid labels
            axes[idx].set_rticks([0.2, 0.4, 0.6, 0.8])
            axes[idx].set_rlabel_position(45)
        
        # Use last subplot for legend
        if len(self.scale_pairs) < 6:
            axes[-1].axis('off')
            legend_text = """Phase Relationship Dynamics
            
            • Angle: φᵢ - φⱼ (phase difference)
            • Radius: Coupling strength
            • Color: Time evolution
            • Dashed circle: Mean coupling strength
            
            Strong coupling → tight phase locking
            Weak coupling → scattered phase differences"""
            
            axes[-1].text(0.1, 0.5, legend_text, fontsize=11,
                         verticalalignment='center', transform=axes[-1].transAxes,
                         bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Generate all HRV Coupling visualizations"""
    print("Generating HRV as Coupling Signature Visualizations...")
    
    # Create output directory
    output_dir = '../results/visualizations/hrv_coupling'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = HRVCouplingVisualizer()
    
    # Generate coupled HRV data
    print("Generating synthetic coupled HRV signals...")
    hrv_data = visualizer.generate_coupled_hrv_signal()
    
    print("Creating Panel A: Traditional vs Coupling-Based HRV...")
    visualizer.plot_traditional_vs_coupling_hrv(
        hrv_data,
        f'{output_dir}/panel_a_traditional_vs_coupling_hrv.png'
    )
    
    print("Creating Panel B: Coupling Component Decomposition...")
    visualizer.plot_coupling_component_decomposition(
        hrv_data,
        f'{output_dir}/panel_b_coupling_component_decomposition.png'
    )
    
    print("Creating Panel C: Phase Relationship Dynamics...")
    visualizer.plot_phase_relationship_dynamics(
        hrv_data,
        f'{output_dir}/panel_c_phase_relationship_dynamics.png'
    )
    
    print(f"HRV Coupling visualizations saved to {output_dir}/")
    print("Template 2: Heart Rate Variability as Coupling Signature - COMPLETE ✓")

if __name__ == "__main__":
    main()
