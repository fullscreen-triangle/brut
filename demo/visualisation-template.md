# Proposed mandatory plots

##  Scale Hierachy Visualisation
- Circular network diagram showing 5 scales:
  * Cellular (0.1-10 Hz) - innermost ring
  * Cardiac (0.5-3 Hz) - second ring  
  * Respiratory (0.1-0.5 Hz) - third ring
  * Autonomic (0.01-0.15 Hz) - fourth ring
  * Circadian (10^-5 Hz) - outermost ring
- Connecting lines showing coupling strength (thickness = coupling magnitude)
- Color coding by frequency bands

 Panel B: Coupling Strength Matrix Heatmap
- 5x5 symmetric matrix showing C_ij values
- Color scale from blue (weak coupling) to red (strong coupling)
- Include numerical values in cells
- Separate matrices for healthy vs. disease states

Panel C: Time-Frequency Coupling Analysis

- Wavelet coherence plots for each scale pair
- X-axis: Time, Y-axis: Frequency
- Color intensity: Coherence magnitude
- Arrows showing phase relationships

## Template 2: Heart Rate Variability as Coupling Signature



Panel A: Traditional vs. Coupling-Based HRV
- Top: Traditional HRV time series
- Bottom: Coupling-reconstructed HRV using Equation 4
- Residual plot showing uncoupled components ε(t)
- R² correlation between traditional and coupling-based metrics



Panel B: Coupling Component Decomposition
- Stacked area chart showing contribution of each scale pair
- Different colors for each C_ij cos(φ_i - φ_j) term
- Time on X-axis, amplitude on Y-axis
- Legend showing scale pair contributions


Panel C: Phase Relationship Dynamics
- Polar plots showing φ_i - φ_j for each scale pair
- Time evolution as color gradient
- Coupling strength as radial distance
- Separate subplots for each significant pair


## Template 1: S-Entropy Coordinate System Overview

Panel A: Multi-Dimensional S-Space Visualization

- 4D coordinate system projection (knowledge, time, entropy, context)
- 3D scatter plot with 4th dimension as color/size
- Physiological states as points in S-space
- Navigation trajectories between states
- Axes labels: S_knowledge, S_time, S_entropy, S_context
- Mathematical foundation: Equation 1 from S-entropy paper
- Include S-distance metric visualization (Equation 2)


Panel B: S-Distance Metric Heatmap

- Distance matrix between physiological states
- Color scale: blue (close) to red (distant) 
- Hierarchical clustering dendrogram
- State labels: [Rest, Exercise, Sleep, Stress, Recovery]
- Include S-distance values in cells
- Statistical significance annotations
- Weighting coefficients w_i display


Panel C: Semantic Gravity Field Visualization

- Vector field showing ∇U_s(r) gradients
- Contour lines of semantic potential energy
- Critical points (minima/maxima/saddle points)
- Navigation constraints overlay
- Color intensity: gravity field strength
- Step size limitations (Equation 12)
- Bounded region [-M,M]^d visualization



# Template 2: Multi-Scale Oscillatory Expression Pipeline

- 5-tier waterfall plot showing frequency scales:
  * Cellular (10^-1 to 10^2 Hz)
  * Cardiac (10^-2 to 10^1 Hz)
  * Respiratory (10^-3 to 10^0 Hz)
  * Autonomic (10^-4 to 10^-1 Hz)
  * Circadian (10^-5 to 10^-2 Hz)
- Log-frequency X-axis, amplitude Y-axis
- Color coding by biological scale

Multi-Sensor Oscillatory Projection
- Three parallel time series (Equations 17-19):
  * PPG oscillatory components: Ψ_ppg(t)
  * Accelerometer oscillatory components: Ψ_acc(t)  
  * Temperature oscillatory components: Ψ_temp(t)
- Phase relationships (φ_i) between sensors
- Amplitude vectors A_i for each scale
- Coupling strength indicators
- Noise components ε_i(t) overlay


# Cross Scale Coupling 

- 5x5 coupling strength matrix for each sensor
- Different matrices for PPG, accelerometer, temperature
- Statistical significance indicators (p < 0.05)
- Coupling evolution over time (animation capability)
- Color scale: blue (weak) to red (strong coupling)
- Mathematical foundation: Cardiovascular coupling theory


# Template 3: Ambiguous Compression Analysis

- Scatter plot: Compression ratio vs. Meta-information potential
- Color coding by sensor type (PPG=pink, Acc=brown, Temp=cyan)
- Threshold lines (τ_threshold) from Equation 8
- Ambiguous information bits highlighted
- Quadrant labels: [High-High, High-Low, Low-High, Low-Low]
- Mathematical basis: Definition 3 (Ambiguous Information Bit)

# Batch Process Correlation 
- Cross-correlation matrix heatmap (A_batch from Equation 9)
- Sensor pairs on axes
- Batch size effects (multiple panels)
- Time-lagged correlations
- Statistical significance overlay
- Pattern amplification visualization

Compression Ratio

- Bar chart showing compression ratios by modality:
  * PPG: 1.75 × 10^3
  * Accelerometer: 1.57 × 10^3  
  * Temperature: 3.4 × 10^2
  * Multi-modal: 3.6 × 10^3
- Error bars with 95% confidence intervals
- Original vs. compressed size comparison
- Information preservation metrics
- Theoretical bounds (10^2 to 10^4)


Template 4: Linguistic Transformation Pipeline

# Numerical Linguistic Flow
- Flow diagram showing transformation steps:
  120 → "one hundred twenty" → "hundred one twenty" → binary
- Multiple examples in parallel tracks
- Compression ratio at each step
- Information content preservation metrics
- Mathematical foundation: Definition 4 and Equation 10
- Alphabetical sorting visualization

Panel B: Alphabetical Reorganization Effects

- Before/after comparison of numerical sequences
- Pattern emergence through reorganization
- Semantic coherence metrics
- Compression efficiency analysis
- Examples across different physiological ranges
- Statistical analysis of pattern preservation


Panel C: Linguistic Compression Performance

- Compression ratio distribution histogram
- Range: 10^2 to 10^4 as stated in paper
- Different physiological data types
- Theoretical vs. empirical compression bounds
- Semantic information retention analysis
- Cross-validation results



Template 5: Directional Sequence Encoding

Panel A: Physiological State Mapping
- State transition diagram: A→R→D→L
- Mapping definitions:
  * A = Elevated/Activation state
  * R = Steady/Maintenance state  
  * D = Decreased/Recovery state
  * L = Stress/Transition state
- Probability matrices P(d|s,c) from Equation 11
- Context-dependent mapping variations
- Circular plot showing directional preferences


Panel B: Heart Rate Sequence Example

- Top: Original HR time series [72,68,74,89,156,134,98,76]
- Middle: Context annotations (sleep, activity, stress, environment)
- Bottom: Directional sequence "ARRALDDA"
- Mapping rules visualization with thresholds
- Personalization factors display
- Context dependency analysis


Panel C: Context-Dependent Encoding Analysis

- Multiple encoding results for same data
- Different contexts: [Circadian, Activity, Environment, History]
- Encoding consistency metrics
- Pattern stability analysis
- Statistical validation across subjects
- Robustness testing results


Template 6: Stochastic Navigation Visualization

Panel A: Constrained Random Walk Trajectories

- 3D trajectory plots in S-entropy space
- Multiple walk realizations (different colors)
- Semantic gravity constraints visualization
- Step size limitations from Equation 12
- Convergence regions highlighted
- Navigation efficiency metrics
- Mathematical foundation: Constrained random walk theory


Panel B: Fuzzy Window Sampling

- Three Gaussian windows: ψ_t(x), ψ_i(x), ψ_e(x)
- Mathematical basis: Equation 13
- Combined weight function w(r) = ψ_t(r_t) · ψ_i(r_i) · ψ_e(r_e)
- Sampling density heatmap
- Window parameter effects (σ_j variations)
- Aperture function visualization


Panel C: Navigation Efficiency Analysis
- Convergence time vs. initial S-distance
- Step size optimization curves
- Sampling efficiency metrics
- Comparison with traditional methods
- Complexity reduction demonstration: O(n!) → O(log(n/C_ratio))
- Theoretical bounds validation


Template 7: Clinical Application Results

Pattern Recognition Accuracy

- Radar chart showing accuracy across tasks:
  * HR anomaly explanation: 87.3%
  * Sleep phase identification: 91.7%
  * Activity classification: 89.4%  
  * Multi-sensor fusion: 93.2%
- Comparison with traditional HRV methods
- Statistical significance testing
- Cross-validation results


Template 8: Cardiovascular Coupling Integration

Panel A: Multi-Scale Coupling Strength Measurements

- Table visualization from cardiovascular paper:
  * Cellular-Cardiac: Healthy 0.78±0.12, Disease 0.52±0.18
  * Cardiac-Respiratory: Healthy 0.85±0.08, Disease 0.43±0.21
  * Respiratory-Autonomic: Healthy 0.72±0.15, Disease 0.39±0.24
  * Autonomic-Circadian: Healthy 0.69±0.18, Disease 0.35±0.26
- Box plots with statistical comparisons
- Effect size annotations (Cohen's d)


Panel B: HRV as Coupling Signature
- Top: Traditional HRV time series
- Middle: Coupling-reconstructed HRV (Equation 4 from CV paper)
- Bottom: Residual uncoupled components ε(t)
- R² correlation between methods
- S-entropy coordinate mapping overlay
- Integration of both theoretical frameworks


Panel C: Pathophysiological Decoupling Analysis
- Disease progression as coupling degradation
- Exponential decay: C_total(t) = C_0 exp(-t/τ_decoupling)
- S-entropy navigation through pathological states
- Critical thresholds for different conditions
- Therapeutic intervention points


Template 9: Theoretical Validation

Panel A: Complexity Reduction Demonstration
- Potential energy surface U_s(r)
- Bounded region [-M,M]^d visualization
- Gravity field magnitude |g_s| distribution
- Stability regions identification
- Mathematical proof visualization (Lemma 1)
- Navigation constraint effects

Panel B: Semantic Gravity Boundedness
- Potential energy surface U_s(r)
- Bounded region [-M,M]^d visualization
- Gravity field magnitude |g_s| distribution
- Stability regions identification
- Mathematical proof visualization (Lemma 1)
- Navigation constraint effects


Panel C: Convergence Properties
- Sample size vs. posterior distribution accuracy
- Fuzzy window convergence rates (Theorem 2)
- Monte Carlo validation studies
- Theoretical bounds vs. empirical results
- Statistical convergence testing
- Robustness analysis
