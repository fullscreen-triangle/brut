//! Core types and data structures for S-entropy physiological sensor analysis

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Multi-sensor physiological data input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorData {
    pub timestamp: DateTime<Utc>,
    pub sensors: SensorReadings,
    pub context: Option<ContextualFactors>,
}

/// Raw sensor readings from multiple modalities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorReadings {
    pub ppg: Option<Vec<f64>>,
    pub accelerometer: Option<AccelerometerData>,
    pub gyroscope: Option<GyroscopeData>,
    pub temperature: Option<Vec<f64>>,
    pub ambient_light: Option<Vec<f64>>,
    pub pressure: Option<Vec<f64>>,
}

/// Three-axis accelerometer data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AccelerometerData {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub magnitude: Option<Vec<f64>>,
}

/// Three-axis gyroscope data  
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GyroscopeData {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub angular_magnitude: Option<Vec<f64>>,
}

/// Environmental and physiological context factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextualFactors {
    pub activity_level: Option<String>,
    pub ambient_temperature: Option<f64>,
    pub time_of_day: Option<String>,
    pub day_of_week: Option<String>,
    pub moon_phase: Option<f64>,
    pub sleep_stage: Option<String>,
    pub stress_level: Option<f64>,
    pub caffeine_intake: Option<f64>,
    pub exercise_history: Option<Vec<ExerciseEvent>>,
    pub medication: Option<Vec<String>>,
    pub additional_context: Option<HashMap<String, serde_json::Value>>,
}

/// Exercise event for contextual analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExerciseEvent {
    pub timestamp: DateTime<Utc>,
    pub activity_type: String,
    pub duration_minutes: f64,
    pub intensity: f64, // 0.0 to 1.0 scale
}

/// S-entropy coordinate system representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEntropyCoordinates {
    pub knowledge: f64,
    pub time: f64,
    pub entropy: f64,
    pub context: f64,
}

/// Oscillatory pattern decomposition across biological frequency scales
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryPatterns {
    pub cellular: FrequencyComponent,
    pub cardiac: FrequencyComponent,
    pub respiratory: FrequencyComponent,
    pub autonomic: FrequencyComponent,
    pub circadian: FrequencyComponent,
    pub coupling_matrix: Vec<Vec<f64>>, // Cross-frequency coupling
}

/// Frequency domain component representation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyComponent {
    pub amplitude: f64,
    pub frequency: f64,
    pub phase: f64,
    pub power: f64,
    pub coherence: Option<f64>,
}

/// Compressed data with ambiguous bit identification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressedData {
    pub original_size: usize,
    pub compressed_size: usize,
    pub compression_ratio: f64,
    pub ambiguous_bits: Vec<AmbiguousBit>,
    pub meta_information: HashMap<String, serde_json::Value>,
}

/// Compression-resistant information pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AmbiguousBit {
    pub position: usize,
    pub resistance_ratio: f64,
    pub interpretations: Vec<String>,
    pub meta_potential: f64,
}

/// Linguistically transformed numerical data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticTransformation {
    pub original: Vec<f64>,
    pub words: Vec<String>,
    pub reorganized: Vec<String>,
    pub encoded: Vec<u8>,
    pub semantic_preservation: f64,
}

/// Directional sequence encoding of physiological patterns
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodedSequence {
    pub sequence: String, // A, R, D, L directional codes
    pub mapping: DirectionalMapping,
    pub context_factors: Vec<String>,
    pub confidence: f64,
}

/// Mapping function from physiological states to directional codes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DirectionalMapping {
    pub activation_threshold: f64,    // A = Elevated/Activation
    pub steady_threshold: f64,        // R = Steady/Maintenance
    pub recovery_threshold: f64,      // D = Decreased/Recovery  
    pub stress_threshold: f64,        // L = Stress/Transition
    pub context_dependent: bool,
}

/// Final physiological interpretation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysiologicalInterpretation {
    pub s_entropy_coordinates: SEntropyCoordinates,
    pub oscillatory_decomposition: OscillatoryPatterns,
    pub linguistic_transformation: LinguisticTransformation,
    pub directional_sequence: EncodedSequence,
    pub explanation: String,
    pub confidence: f64,
    pub context_factors: Vec<String>,
    pub anomalies_explained: Vec<AnomalyExplanation>,
}

/// Explanation for anomalous sensor readings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyExplanation {
    pub reading: f64,
    pub expected_range: (f64, f64),
    pub explanation: String,
    pub contributing_factors: Vec<String>,
    pub confidence: f64,
}

/// Navigation path through S-entropy coordinate space
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationPath {
    pub steps: Vec<NavigationStep>,
    pub convergence_iterations: usize,
    pub semantic_gravity_strength: f64,
    pub final_position: SEntropyCoordinates,
}

/// Single step in S-entropy space navigation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationStep {
    pub position: SEntropyCoordinates,
    pub step_size: f64,
    pub direction: Vec<f64>, // 4D direction vector
    pub fuzzy_window_weights: (f64, f64, f64), // temporal, informational, entropic
}

/// Configuration for oscillatory decomposition
#[derive(Debug, Clone)]
pub struct OscillatoryConfig {
    pub frequency_bands: FrequencyBands,
    pub window_size: usize,
    pub overlap_ratio: f64,
    pub enable_coupling_analysis: bool,
}

impl Default for OscillatoryConfig {
    fn default() -> Self {
        Self {
            frequency_bands: FrequencyBands::default(),
            window_size: 2048,
            overlap_ratio: 0.5,
            enable_coupling_analysis: true,
        }
    }
}

/// Biological frequency band definitions
#[derive(Debug, Clone)]
pub struct FrequencyBands {
    pub cellular: (f64, f64),    // 0.1 - 100.0 Hz
    pub cardiac: (f64, f64),     // 0.01 - 10.0 Hz
    pub respiratory: (f64, f64), // 0.001 - 1.0 Hz
    pub autonomic: (f64, f64),   // 0.0001 - 0.1 Hz
    pub circadian: (f64, f64),   // 0.00001 - 0.01 Hz
}

impl Default for FrequencyBands {
    fn default() -> Self {
        Self {
            cellular: (0.1, 100.0),
            cardiac: (0.01, 10.0),
            respiratory: (0.001, 1.0),
            autonomic: (0.0001, 0.1),
            circadian: (0.00001, 0.01),
        }
    }
}
