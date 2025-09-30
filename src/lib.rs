//! # S-Entropy Coordinate Navigation for Physiological Sensor Analysis
//!
//! This library implements a mathematical framework for consumer-grade physiological 
//! sensor analysis based on S-entropy coordinate navigation. The system transforms 
//! measurement imprecision into contextual interpretation through five sequential 
//! operations.
//!
//! ## Core Modules
//!
//! - [`oscillatory`] - Multi-scale oscillatory decomposition across biological frequency scales
//! - [`compression`] - Ambiguous compression for meta-information extraction  
//! - [`linguistic`] - Linguistic transformation pipeline for semantic reorganization
//! - [`encoding`] - Sequence-based physiological pattern encoding
//! - [`navigation`] - S-entropy coordinate space navigation and interpretation
//!
//! ## Quick Start
//!
//! ```rust
//! use brut::{SEntropyProcessor, OscillatoryConfig};
//!
//! let config = OscillatoryConfig::default();
//! let processor = SEntropyProcessor::new(config);
//! 
//! // Process sensor data through the complete pipeline
//! let sensor_data = load_sensor_data("data/sensors.json")?;
//! let interpretation = processor.process_complete_pipeline(&sensor_data)?;
//! ```

pub mod oscillatory;
pub mod compression;
pub mod linguistic;
pub mod encoding;
pub mod navigation;
pub mod config;
pub mod types;
pub mod utils;

pub use crate::oscillatory::{OscillatoryProcessor, OscillatoryConfig};
pub use crate::compression::{AmbigiousCompressor, CompressionConfig};
pub use crate::linguistic::{LinguisticTransformer, LinguisticConfig};
pub use crate::encoding::{SequenceEncoder, EncodingConfig};
pub use crate::navigation::{SEntropyNavigator, NavigationConfig};
pub use crate::types::{SensorData, SEntropyCoordinates, PhysiologicalInterpretation};

use anyhow::Result;

/// Main processor that orchestrates the complete S-entropy analysis pipeline
pub struct SEntropyProcessor {
    oscillatory: OscillatoryProcessor,
    compressor: AmbigiousCompressor,
    transformer: LinguisticTransformer,
    encoder: SequenceEncoder,
    navigator: SEntropyNavigator,
}

impl SEntropyProcessor {
    /// Create a new S-entropy processor with default configuration
    pub fn new(config: OscillatoryConfig) -> Self {
        Self {
            oscillatory: OscillatoryProcessor::new(config),
            compressor: AmbigiousCompressor::default(),
            transformer: LinguisticTransformer::default(),
            encoder: SequenceEncoder::default(),
            navigator: SEntropyNavigator::default(),
        }
    }

    /// Process sensor data through the complete S-entropy pipeline
    pub fn process_complete_pipeline(&self, sensor_data: &SensorData) -> Result<PhysiologicalInterpretation> {
        // Step 1: Oscillatory decomposition across biological frequency scales
        let oscillatory_patterns = self.oscillatory.extract_oscillatory_patterns(sensor_data)?;
        
        // Step 2: Ambiguous compression for meta-information extraction
        let compressed_data = self.compressor.compress_ambiguous_patterns(&oscillatory_patterns)?;
        
        // Step 3: Linguistic transformation for semantic reorganization
        let transformed_data = self.transformer.transform_linguistic(&compressed_data)?;
        
        // Step 4: Sequence encoding using directional coordinates
        let encoded_sequences = self.encoder.encode_physiological_sequences(&transformed_data)?;
        
        // Step 5: S-entropy navigation for contextual interpretation
        let interpretation = self.navigator.navigate_s_entropy_space(&encoded_sequences)?;
        
        Ok(interpretation)
    }
}

/// Load sensor data from JSON file
pub fn load_sensor_data(path: &str) -> Result<SensorData> {
    let content = std::fs::read_to_string(path)?;
    let sensor_data: SensorData = serde_json::from_str(&content)?;
    Ok(sensor_data)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_s_entropy_processor_creation() {
        let config = OscillatoryConfig::default();
        let _processor = SEntropyProcessor::new(config);
        // Test passes if no panic occurs
    }
}
