# S-Entropy Framework Demo Implementation

This demo implements the S-entropy coordinate navigation framework for consumer-grade physiological sensor analysis using real sleep and activity data from smart ring sensors.

## Dataset Structure

### Sleep Data (`sleep_ppg_records.json`)
- **Heart Rate Sequences**: `hr_5min` - 5-minute interval heart rate measurements
- **Sleep Stage Sequences**: `hypnogram_5min` - Sleep stages encoded as: A=Awake, L=Light, D=Deep, R=REM
- **HRV Sequences**: `rmssd_5min` - Heart rate variability measurements
- **Contextual Metrics**: Sleep efficiency, temperature deviations, respiratory rates

### Activity Data (`activity_ppg_records.json`)  
- Activity patterns and movement data
- PPG sensor readings during active periods
- Contextual activity information

## S-Entropy Framework Implementation

The demo implements the complete S-entropy pipeline:

1. **Oscillatory Expression** (`src/linguistic/s_entropy.py`)
   - Multi-scale frequency decomposition across biological scales
   - Cellular, cardiac, respiratory, autonomic, circadian analysis

2. **Ambiguous Compression** (`src/linguistic/ambigous_compression.py`)
   - Compression-resistant pattern identification
   - Meta-information extraction from measurement "errors"

3. **Linguistic Transformation** (`src/linguistic/sequence_based_encoding.py`)
   - Numbers → words → alphabetical reordering → binary encoding
   - Semantic reorganization for pattern recognition

4. **Directional Sequence Mapping** (`src/linguistic/directional_coordinate_mapping.py`)
   - A=Activation/Elevated states
   - R=Steady/Maintenance states  
   - D=Decreased/Recovery states
   - L=Stress/Transition states

5. **S-Entropy Navigation** (`src/linguistic/moon_landing.py`)
   - Stochastic navigation through S-entropy coordinate space
   - Contextual explanation generation
   - Anomaly interpretation through multi-scale coupling

## Module Structure

### Core S-Entropy Framework (`src/linguistic/`)
- `s_entropy.py` - Main S-entropy coordinate system implementation
- `ambigous_compression.py` - Compression-resistant information extraction  
- `sequence_based_encoding.py` - Linguistic transformation pipeline
- `directional_coordinate_mapping.py` - Physiological state mapping
- `moon_landing.py` - Navigation and contextual interpretation

### Physiological Analysis Modules
- `src/heart/` - Comprehensive heart rate and HRV analysis
- `src/sleep/` - Sleep architecture and circadian rhythm analysis  
- `src/actigraphy/` - Activity pattern and movement analysis
- `src/coupling/` - Multi-modal sensor fusion and contextual integration

## Usage

Each module has its own main function for standalone analysis:

```python
# Run S-entropy analysis
python src/linguistic/s_entropy.py

# Analyze sleep patterns  
python src/sleep/sleep_architecture.py

# Process heart rate data
python src/heart/cardiac/basic_metrics.py

# Multi-sensor coupling analysis
python src/coupling/activity_sleep_correlation.py
```

Results are saved in JSON format with comprehensive visualizations.

## Key Insights

This implementation demonstrates how **imprecise consumer sensor readings become precise contextual explanations** through S-entropy coordinate navigation, transforming the problem from "making sensors more accurate" to "making interpretations more intelligent."
