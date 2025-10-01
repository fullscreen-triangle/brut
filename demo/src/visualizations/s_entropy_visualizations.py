"""
S-Entropy Framework Comprehensive Visualizations
Implements Template 1: S-Entropy Coordinate System Overview
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd
import os
from typing import Dict, List, Any, Tuple

class SEntropyVisualizer:
    def __init__(self):
        self.states = ['Rest', 'Exercise', 'Sleep', 'Stress', 'Recovery']
        self.colors = ['#2E86C1', '#28B463', '#8E44AD', '#E74C3C', '#F39C12']
        
    def generate_s_space_data(self, n_points: int = 100) -> Dict[str, np.ndarray]:
        """Generate 4D S-space data points"""
        np.random.seed(42)
        
        # Generate points for each physiological state
        data = {}
        for i, state in enumerate(self.states):
            # Each state has characteristic S-entropy coordinates
            base_coords = np.array([
                [0.2, 0.8, 0.3, 0.6],  # Rest
                [0.9, 0.4, 0.7, 0.2],  # Exercise  
                [0.1, 0.9, 0.2, 0.8],  # Sleep
                [0.8, 0.2, 0.9, 0.1],  # Stress
                [0.4, 0.6, 0.4, 0.7]   # Recovery
            ])[i]
            
            # Add noise around base coordinates
            points = np.random.normal(base_coords, 0.15, (n_points//5, 4))
            points = np.clip(points, 0, 1)  # Keep in [0,1] bounds
            
            data[state] = points
            
        return data
    
    def plot_multidimensional_s_space(self, data: Dict[str, np.ndarray], save_path: str):
        """Panel A: Multi-Dimensional S-Space Visualization"""
        fig = plt.figure(figsize=(20, 15))
        
        # 3D scatter plot with 4th dimension as color/size
        ax = fig.add_subplot(221, projection='3d')
        
        for i, (state, points) in enumerate(data.items()):
            # Use 4th dimension for color intensity and size
            colors_4d = points[:, 3]  # S_context as color
            sizes = 50 + points[:, 3] * 100  # Size based on 4th dimension
            
            scatter = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                               c=colors_4d, s=sizes, alpha=0.6, 
                               cmap='viridis', label=state)
        
        ax.set_xlabel('S_knowledge')
        ax.set_ylabel('S_time') 
        ax.set_zlabel('S_entropy')
        ax.set_title('4D S-Space Visualization\n(4th dimension: color/size)', fontsize=14)
        ax.legend()
        
        # Navigation trajectories between states
        ax2 = fig.add_subplot(222, projection='3d')
        
        # Create trajectory between states
        state_centers = {}
        for state, points in data.items():
            state_centers[state] = np.mean(points[:, :3], axis=0)
        
        # Draw trajectories between adjacent states
        trajectory_pairs = [('Rest', 'Exercise'), ('Exercise', 'Recovery'), 
                          ('Recovery', 'Sleep'), ('Sleep', 'Stress'), ('Stress', 'Rest')]
        
        for start, end in trajectory_pairs:
            start_pos = state_centers[start]
            end_pos = state_centers[end]
            
            # Create smooth trajectory
            t = np.linspace(0, 1, 50)
            trajectory = np.array([start_pos + t_val * (end_pos - start_pos) for t_val in t])
            
            ax2.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                    'o-', alpha=0.7, markersize=3, linewidth=2)
        
        # Plot state centers
        for i, (state, center) in enumerate(state_centers.items()):
            ax2.scatter(*center, s=200, c=self.colors[i], marker='*', 
                       edgecolors='black', linewidth=2, label=state)
        
        ax2.set_xlabel('S_knowledge')
        ax2.set_ylabel('S_time')
        ax2.set_zlabel('S_entropy') 
        ax2.set_title('Navigation Trajectories in S-Space', fontsize=14)
        ax2.legend()
        
        # S-distance metric visualization
        ax3 = fig.add_subplot(223)
        
        # Calculate S-distances between state centers
        centers_array = np.array(list(state_centers.values()))
        
        # S-distance metric: weighted Euclidean distance
        weights = np.array([0.3, 0.25, 0.25, 0.2])  # w_i weights
        
        s_distances = np.zeros((len(self.states), len(self.states)))
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                diff = centers_array[i] - centers_array[j] 
                s_distances[i, j] = np.sqrt(np.sum(weights * diff**2))
        
        # Plot S-distance heatmap
        im = ax3.imshow(s_distances, cmap='RdYlBu_r', aspect='auto')
        ax3.set_xticks(range(len(self.states)))
        ax3.set_yticks(range(len(self.states)))
        ax3.set_xticklabels(self.states, rotation=45)
        ax3.set_yticklabels(self.states)
        
        # Add distance values to cells
        for i in range(len(self.states)):
            for j in range(len(self.states)):
                text = ax3.text(j, i, f'{s_distances[i, j]:.3f}', 
                               ha="center", va="center", color="black", fontsize=10)
        
        ax3.set_title('S-Distance Metric Heatmap\n(Equation 2)', fontsize=14)
        plt.colorbar(im, ax=ax3, label='S-Distance')
        
        # Weight coefficients display
        ax4 = fig.add_subplot(224)
        
        weight_labels = ['w_knowledge', 'w_time', 'w_entropy', 'w_context']
        bars = ax4.bar(weight_labels, weights, color=['#3498DB', '#E74C3C', '#2ECC71', '#F39C12'])
        ax4.set_ylabel('Weight Coefficient')
        ax4.set_title('S-Distance Weighting Coefficients w_i', fontsize=14)
        ax4.set_ylim(0, 0.35)
        
        # Add value labels on bars
        for bar, weight in zip(bars, weights):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{weight:.2f}', ha='center', va='bottom', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_s_distance_clustering(self, data: Dict[str, np.ndarray], save_path: str):
        """Panel B: S-Distance Metric Heatmap with Clustering"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        
        # Calculate comprehensive S-distance matrix
        all_points = []
        state_labels = []
        
        for state, points in data.items():
            all_points.extend(points[:10])  # Use subset for clarity
            state_labels.extend([state] * 10)
        
        all_points = np.array(all_points)
        
        # Calculate S-distance matrix with weights
        weights = np.array([0.3, 0.25, 0.25, 0.2])
        n_points = len(all_points)
        s_distance_matrix = np.zeros((n_points, n_points))
        
        for i in range(n_points):
            for j in range(n_points):
                diff = all_points[i] - all_points[j]
                s_distance_matrix[i, j] = np.sqrt(np.sum(weights * diff**2))
        
        # Plot distance heatmap
        im = axes[0].imshow(s_distance_matrix, cmap='RdYlBu_r', aspect='auto')
        axes[0].set_title('S-Distance Matrix Between All Points', fontsize=14)
        plt.colorbar(im, ax=axes[0], label='S-Distance')
        
        # Hierarchical clustering
        condensed_distances = pdist(all_points, metric='euclidean')
        linkage_matrix = linkage(condensed_distances, method='ward')
        
        dendro = dendrogram(linkage_matrix, labels=state_labels, ax=axes[1], 
                           leaf_rotation=90, leaf_font_size=8)
        axes[1].set_title('Hierarchical Clustering Dendrogram\n(S-Distance Metric)', fontsize=14)
        axes[1].set_xlabel('Physiological States')
        axes[1].set_ylabel('S-Distance')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_semantic_gravity_field(self, save_path: str):
        """Panel C: Semantic Gravity Field Visualization"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 16))
        
        # Create 2D semantic potential energy surface
        x = np.linspace(-3, 3, 50)
        y = np.linspace(-3, 3, 50)
        X, Y = np.meshgrid(x, y)
        
        # Semantic potential U_s(r) - multi-well potential
        U_semantic = 0.1 * (X**2 + Y**2)  # Harmonic base
        U_semantic += -2 * np.exp(-((X-1)**2 + (Y-1)**2))  # Well 1
        U_semantic += -1.5 * np.exp(-((X+1)**2 + (Y+1)**2))  # Well 2
        U_semantic += -1.8 * np.exp(-((X-0.5)**2 + (Y+1.2)**2))  # Well 3
        
        # Plot potential energy surface
        contour = axes[0,0].contour(X, Y, U_semantic, levels=20, alpha=0.7)
        axes[0,0].clabel(contour, inline=True, fontsize=8)
        im1 = axes[0,0].contourf(X, Y, U_semantic, levels=50, cmap='RdYlBu_r', alpha=0.8)
        
        # Add critical points
        # Find minima (simplified - using known locations)
        minima = [(1, 1), (-1, -1), (0.5, -1.2)]
        for min_x, min_y in minima:
            axes[0,0].plot(min_x, min_y, 'ro', markersize=12, markeredgecolor='black', 
                          markeredgewidth=2, label='Minima' if min_x == 1 else "")
        
        axes[0,0].set_xlabel('S-coordinate X')
        axes[0,0].set_ylabel('S-coordinate Y') 
        axes[0,0].set_title('Semantic Potential Energy U_s(r)', fontsize=14)
        axes[0,0].legend()
        plt.colorbar(im1, ax=axes[0,0], label='Potential Energy')
        
        # Calculate and plot gravity field g_s = -∇U_s
        gy, gx = np.gradient(-U_semantic, y[1]-y[0], x[1]-x[0])
        
        # Subsample for cleaner vector field
        step = 3
        axes[0,1].quiver(X[::step, ::step], Y[::step, ::step], 
                        gx[::step, ::step], gy[::step, ::step], 
                        alpha=0.7, scale=30, width=0.003)
        
        # Overlay potential contours
        axes[0,1].contour(X, Y, U_semantic, levels=15, alpha=0.4, colors='gray')
        axes[0,1].set_xlabel('S-coordinate X')
        axes[0,1].set_ylabel('S-coordinate Y')
        axes[0,1].set_title('Semantic Gravity Field g_s = -∇U_s', fontsize=14)
        
        # Gravity field magnitude
        g_magnitude = np.sqrt(gx**2 + gy**2)
        im2 = axes[1,0].imshow(g_magnitude, extent=[-3, 3, -3, 3], 
                              origin='lower', cmap='plasma', aspect='auto')
        axes[1,0].set_xlabel('S-coordinate X')
        axes[1,0].set_ylabel('S-coordinate Y')
        axes[1,0].set_title('Gravity Field Magnitude |g_s|', fontsize=14)
        plt.colorbar(im2, ax=axes[1,0], label='|g_s|')
        
        # Step size limitations (Equation 12: Δr_max = v0/|g_s|)
        v0 = 1.0  # Base processing velocity
        delta_r_max = v0 / (g_magnitude + 1e-6)  # Avoid division by zero
        
        im3 = axes[1,1].imshow(delta_r_max, extent=[-3, 3, -3, 3], 
                              origin='lower', cmap='viridis', aspect='auto')
        axes[1,1].set_xlabel('S-coordinate X') 
        axes[1,1].set_ylabel('S-coordinate Y')
        axes[1,1].set_title('Step Size Limitations Δr_max = v₀/|g_s|', fontsize=14)
        plt.colorbar(im3, ax=axes[1,1], label='Max Step Size')
        
        # Add bounded region visualization
        for ax in axes.flat:
            # Bounded region [-M,M]^d with M=3
            ax.axhline(y=3, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.axhline(y=-3, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.axvline(x=3, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.axvline(x=-3, color='red', linestyle='--', alpha=0.7, linewidth=2)
            ax.set_xlim(-3.5, 3.5)
            ax.set_ylim(-3.5, 3.5)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Generate all S-Entropy visualizations"""
    print("Generating S-Entropy Framework Visualizations...")
    
    # Create output directory
    output_dir = '../results/visualizations/s_entropy'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize visualizer
    visualizer = SEntropyVisualizer()
    
    # Generate S-space data
    s_space_data = visualizer.generate_s_space_data()
    
    print("Creating Panel A: Multi-Dimensional S-Space Visualization...")
    visualizer.plot_multidimensional_s_space(
        s_space_data, 
        f'{output_dir}/panel_a_multidimensional_s_space.png'
    )
    
    print("Creating Panel B: S-Distance Metric Heatmap...")
    visualizer.plot_s_distance_clustering(
        s_space_data,
        f'{output_dir}/panel_b_s_distance_heatmap.png'
    )
    
    print("Creating Panel C: Semantic Gravity Field Visualization...")
    visualizer.plot_semantic_gravity_field(
        f'{output_dir}/panel_c_semantic_gravity_field.png'
    )
    
    print(f"S-Entropy visualizations saved to {output_dir}/")
    print("Template 1: S-Entropy Coordinate System Overview - COMPLETE ✓")

if __name__ == "__main__":
    main()
