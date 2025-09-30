//! Configuration management for S-entropy framework
//!
//! This module handles loading and managing configuration parameters
//! for all components of the S-entropy analysis pipeline.

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::path::Path;

/// Main configuration structure
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    pub general: GeneralConfig,
    pub processing: ProcessingConfig,
    pub s_entropy: SEntropyConfig,
    pub oscillatory: OscillatoryConfig,
    pub compression: CompressionConfig,
    pub linguistic: LinguisticConfig,
    pub encoding: EncodingConfig,
    pub navigation: NavigationConfig,
    pub validation: ValidationConfig,
}

/// General application configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneralConfig {
    pub log_level: String,
    pub output_format: String,
    pub enable_gpu: bool,
    pub thread_count: usize,
}

/// Processing pipeline configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProcessingConfig {
    pub enable_oscillatory_decomposition: bool,
    pub enable_ambiguous_compression: bool,
    pub enable_linguistic_transformation: bool,
    pub enable_sequence_encoding: bool,
    pub enable_s_entropy_navigation: bool,
}

/// S-entropy coordinate system configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SEntropyConfig {
    pub coordinates: CoordinatesConfig,
    pub navigation: NavigationParamsConfig,
    pub fuzzy_windows: FuzzyWindowsConfig,
    pub compression: CompressionParamsConfig,
    pub linguistic: LinguisticParamsConfig,
}

/// S-entropy coordinate weights
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CoordinatesConfig {
    pub knowledge_weight: f64,
    pub time_weight: f64,
    pub entropy_weight: f64,
    pub context_weight: f64,
}

/// Navigation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationParamsConfig {
    pub step_size: f64,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub semantic_gravity_strength: f64,
}

/// Fuzzy window parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuzzyWindowsConfig {
    pub temporal_sigma: f64,
    pub informational_sigma: f64,
    pub entropic_sigma: f64,
}

/// Compression parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionParamsConfig {
    pub resistance_threshold: f64,
    pub batch_size: usize,
    pub meta_information_weight: f64,
}

/// Linguistic parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticParamsConfig {
    pub enable_alphabetical_sorting: bool,
    pub compression_target_ratio: f64,
    pub semantic_preservation_threshold: f64,
}

/// Oscillatory decomposition configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OscillatoryConfig {
    pub frequency_bands: FrequencyBandsConfig,
    pub coupling_analysis: CouplingAnalysisConfig,
    pub decomposition: DecompositionConfig,
    pub sensors: SensorsConfig,
}

/// Frequency band definitions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyBandsConfig {
    pub cellular: FrequencyRange,
    pub cardiac: FrequencyRange,
    pub respiratory: FrequencyRange,
    pub autonomic: FrequencyRange,
    pub circadian: FrequencyRange,
}

/// Frequency range specification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrequencyRange {
    pub min: f64,
    pub max: f64,
}

/// Coupling analysis parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CouplingAnalysisConfig {
    pub enable_cross_frequency: bool,
    pub phase_coupling_threshold: f64,
    pub amplitude_coupling_threshold: f64,
    pub coherence_threshold: f64,
}

/// Decomposition parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecompositionConfig {
    pub window_size: usize,
    pub overlap_ratio: f64,
    pub filter_order: usize,
    pub detrend_method: String,
}

/// Sensor-specific configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorsConfig {
    pub ppg: SensorConfig,
    pub accelerometer: SensorConfig,
    pub temperature: SensorConfig,
}

/// Individual sensor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SensorConfig {
    pub sampling_rate: f64,
    pub noise_floor: Option<f64>,
    pub dynamic_range: Option<f64>,
    pub sensitivity: Option<f64>,
    pub precision: Option<f64>,
    pub calibration_offset: Option<Vec<f64>>,
    pub thermal_time_constant: Option<f64>,
}

/// Ambiguous compression configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompressionConfig {
    pub resistance_threshold: f64,
    pub batch_size: usize,
    pub meta_information_weight: f64,
}

/// Linguistic transformation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinguisticConfig {
    pub enable_alphabetical_sorting: bool,
    pub compression_target_ratio: f64,
    pub semantic_preservation_threshold: f64,
}

/// Sequence encoding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EncodingConfig {
    pub activation_threshold: f64,
    pub steady_threshold: f64,
    pub recovery_threshold: f64,
    pub stress_threshold: f64,
    pub context_dependent: bool,
}

/// Navigation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NavigationConfig {
    pub knowledge_weight: f64,
    pub time_weight: f64,
    pub entropy_weight: f64,
    pub context_weight: f64,
    pub step_size: f64,
    pub max_iterations: usize,
    pub convergence_threshold: f64,
    pub semantic_gravity_strength: f64,
    pub temporal_sigma: f64,
    pub informational_sigma: f64,
    pub entropic_sigma: f64,
}

/// Validation and quality control configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationConfig {
    pub enable_consistency_checks: bool,
    pub anomaly_detection_threshold: f64,
    pub confidence_threshold: f64,
    pub require_minimum_context: bool,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            general: GeneralConfig::default(),
            processing: ProcessingConfig::default(),
            s_entropy: SEntropyConfig::default(),
            oscillatory: OscillatoryConfig::default(),
            compression: CompressionConfig::default(),
            linguistic: LinguisticConfig::default(),
            encoding: EncodingConfig::default(),
            navigation: NavigationConfig::default(),
            validation: ValidationConfig::default(),
        }
    }
}

impl Default for GeneralConfig {
    fn default() -> Self {
        Self {
            log_level: "info".to_string(),
            output_format: "json".to_string(),
            enable_gpu: false,
            thread_count: 0, // Auto-detect
        }
    }
}

impl Default for ProcessingConfig {
    fn default() -> Self {
        Self {
            enable_oscillatory_decomposition: true,
            enable_ambiguous_compression: true,
            enable_linguistic_transformation: true,
            enable_sequence_encoding: true,
            enable_s_entropy_navigation: true,
        }
    }
}

impl Default for SEntropyConfig {
    fn default() -> Self {
        Self {
            coordinates: CoordinatesConfig::default(),
            navigation: NavigationParamsConfig::default(),
            fuzzy_windows: FuzzyWindowsConfig::default(),
            compression: CompressionParamsConfig::default(),
            linguistic: LinguisticParamsConfig::default(),
        }
    }
}

impl Default for CoordinatesConfig {
    fn default() -> Self {
        Self {
            knowledge_weight: 0.25,
            time_weight: 0.30,
            entropy_weight: 0.25,
            context_weight: 0.20,
        }
    }
}

impl Default for NavigationParamsConfig {
    fn default() -> Self {
        Self {
            step_size: 0.01,
            max_iterations: 1000,
            convergence_threshold: 0.001,
            semantic_gravity_strength: 1.0,
        }
    }
}

impl Default for FuzzyWindowsConfig {
    fn default() -> Self {
        Self {
            temporal_sigma: 0.1,
            informational_sigma: 0.15,
            entropic_sigma: 0.08,
        }
    }
}

impl Default for CompressionParamsConfig {
    fn default() -> Self {
        Self {
            resistance_threshold: 0.7,
            batch_size: 1024,
            meta_information_weight: 0.5,
        }
    }
}

impl Default for LinguisticParamsConfig {
    fn default() -> Self {
        Self {
            enable_alphabetical_sorting: true,
            compression_target_ratio: 100.0,
            semantic_preservation_threshold: 0.8,
        }
    }
}

impl Default for OscillatoryConfig {
    fn default() -> Self {
        Self {
            frequency_bands: FrequencyBandsConfig::default(),
            coupling_analysis: CouplingAnalysisConfig::default(),
            decomposition: DecompositionConfig::default(),
            sensors: SensorsConfig::default(),
        }
    }
}

impl Default for FrequencyBandsConfig {
    fn default() -> Self {
        Self {
            cellular: FrequencyRange { min: 0.1, max: 100.0 },
            cardiac: FrequencyRange { min: 0.01, max: 10.0 },
            respiratory: FrequencyRange { min: 0.001, max: 1.0 },
            autonomic: FrequencyRange { min: 0.0001, max: 0.1 },
            circadian: FrequencyRange { min: 0.00001, max: 0.01 },
        }
    }
}

impl Default for CouplingAnalysisConfig {
    fn default() -> Self {
        Self {
            enable_cross_frequency: true,
            phase_coupling_threshold: 0.5,
            amplitude_coupling_threshold: 0.3,
            coherence_threshold: 0.6,
        }
    }
}

impl Default for DecompositionConfig {
    fn default() -> Self {
        Self {
            window_size: 2048,
            overlap_ratio: 0.5,
            filter_order: 4,
            detrend_method: "linear".to_string(),
        }
    }
}

impl Default for SensorsConfig {
    fn default() -> Self {
        Self {
            ppg: SensorConfig {
                sampling_rate: 100.0,
                noise_floor: Some(0.01),
                dynamic_range: Some(20.0),
                sensitivity: None,
                precision: None,
                calibration_offset: None,
                thermal_time_constant: None,
            },
            accelerometer: SensorConfig {
                sampling_rate: 50.0,
                noise_floor: None,
                dynamic_range: None,
                sensitivity: Some(0.001),
                precision: None,
                calibration_offset: Some(vec![0.0, 0.0, 9.81]),
                thermal_time_constant: None,
            },
            temperature: SensorConfig {
                sampling_rate: 1.0,
                noise_floor: None,
                dynamic_range: None,
                sensitivity: None,
                precision: Some(0.1),
                calibration_offset: None,
                thermal_time_constant: Some(30.0),
            },
        }
    }
}

impl Default for CompressionConfig {
    fn default() -> Self {
        Self {
            resistance_threshold: 0.7,
            batch_size: 1024,
            meta_information_weight: 0.5,
        }
    }
}

impl Default for LinguisticConfig {
    fn default() -> Self {
        Self {
            enable_alphabetical_sorting: true,
            compression_target_ratio: 100.0,
            semantic_preservation_threshold: 0.8,
        }
    }
}

impl Default for EncodingConfig {
    fn default() -> Self {
        Self {
            activation_threshold: 0.7,
            steady_threshold: 0.3,
            recovery_threshold: -0.3,
            stress_threshold: 0.9,
            context_dependent: true,
        }
    }
}

impl Default for NavigationConfig {
    fn default() -> Self {
        Self {
            knowledge_weight: 0.25,
            time_weight: 0.30,
            entropy_weight: 0.25,
            context_weight: 0.20,
            step_size: 0.01,
            max_iterations: 1000,
            convergence_threshold: 0.001,
            semantic_gravity_strength: 1.0,
            temporal_sigma: 0.1,
            informational_sigma: 0.15,
            entropic_sigma: 0.08,
        }
    }
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            enable_consistency_checks: true,
            anomaly_detection_threshold: 3.0,
            confidence_threshold: 0.5,
            require_minimum_context: true,
        }
    }
}

/// Load configuration from TOML file
pub fn load_config<P: AsRef<Path>>(path: P) -> Result<Config> {
    let content = std::fs::read_to_string(path)?;
    let config: Config = toml::from_str(&content)?;
    Ok(config)
}

/// Save configuration to TOML file
pub fn save_config<P: AsRef<Path>>(config: &Config, path: P) -> Result<()> {
    let content = toml::to_string_pretty(config)?;
    std::fs::write(path, content)?;
    Ok(())
}

/// Load configuration from multiple TOML files and merge
pub fn load_config_from_directory<P: AsRef<Path>>(dir_path: P) -> Result<Config> {
    let mut config = Config::default();
    
    let config_dir = dir_path.as_ref();
    
    // Load individual configuration files if they exist
    if let Ok(s_entropy_content) = std::fs::read_to_string(config_dir.join("s_entropy.toml")) {
        let s_entropy_config: SEntropyConfig = toml::from_str(&s_entropy_content)?;
        config.s_entropy = s_entropy_config;
    }
    
    if let Ok(oscillatory_content) = std::fs::read_to_string(config_dir.join("oscillatory.toml")) {
        let oscillatory_config: OscillatoryConfig = toml::from_str(&oscillatory_content)?;
        config.oscillatory = oscillatory_config;
    }
    
    if let Ok(default_content) = std::fs::read_to_string(config_dir.join("default.toml")) {
        let partial_config: Config = toml::from_str(&default_content)?;
        // Merge with existing config
        config = merge_configs(config, partial_config);
    }
    
    Ok(config)
}

/// Merge two configurations, with the second taking precedence
fn merge_configs(base: Config, overlay: Config) -> Config {
    // For simplicity, this just returns the overlay
    // In a full implementation, this would do field-by-field merging
    overlay
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::tempdir;

    #[test]
    fn test_default_config() {
        let config = Config::default();
        assert_eq!(config.general.log_level, "info");
        assert_eq!(config.s_entropy.coordinates.knowledge_weight, 0.25);
    }

    #[test]
    fn test_config_serialization() -> Result<()> {
        let config = Config::default();
        let toml_str = toml::to_string(&config)?;
        let parsed_config: Config = toml::from_str(&toml_str)?;
        
        assert_eq!(config.general.log_level, parsed_config.general.log_level);
        Ok(())
    }

    #[test]
    fn test_save_and_load_config() -> Result<()> {
        let dir = tempdir()?;
        let config_path = dir.path().join("test_config.toml");
        
        let original_config = Config::default();
        save_config(&original_config, &config_path)?;
        
        let loaded_config = load_config(&config_path)?;
        assert_eq!(original_config.general.log_level, loaded_config.general.log_level);
        
        Ok(())
    }
}
