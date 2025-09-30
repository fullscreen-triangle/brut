//! Ambiguous compression for meta-information extraction
//!
//! This module identifies compression-resistant patterns in oscillatory data
//! and extracts meta-information from ambiguous bits.

use crate::types::{OscillatoryPatterns, CompressedData, AmbiguousBit};
use anyhow::Result;
use flate2::write::GzEncoder;
use flate2::Compression;
use std::io::Write;
use std::collections::HashMap;

/// Configuration for ambiguous compression
#[derive(Debug, Clone)]
pub struct CompressionConfig {
    pub resistance_threshold: f64,
    pub batch_size: usize,
    pub meta_information_weight: f64,
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

/// Ambiguous compression processor
pub struct AmbigiousCompressor {
    config: CompressionConfig,
}

impl Default for AmbigiousCompressor {
    fn default() -> Self {
        Self {
            config: CompressionConfig::default(),
        }
    }
}

impl AmbigiousCompressor {
    /// Create new compressor with configuration
    pub fn new(config: CompressionConfig) -> Self {
        Self { config }
    }

    /// Compress oscillatory patterns and identify ambiguous information
    pub fn compress_ambiguous_patterns(&self, patterns: &OscillatoryPatterns) -> Result<CompressedData> {
        // Convert oscillatory patterns to binary representation
        let binary_data = self.serialize_patterns(patterns)?;
        let original_size = binary_data.len();
        
        // Perform standard compression to identify resistant patterns
        let compressed_standard = self.standard_compress(&binary_data)?;
        
        // Identify compression-resistant bits (ambiguous information)
        let ambiguous_bits = self.identify_ambiguous_bits(&binary_data, &compressed_standard)?;
        
        // Extract meta-information from ambiguous patterns
        let meta_information = self.extract_meta_information(&ambiguous_bits, patterns)?;
        
        // Compute final compression statistics
        let compressed_size = compressed_standard.len();
        let compression_ratio = original_size as f64 / compressed_size as f64;
        
        Ok(CompressedData {
            original_size,
            compressed_size,
            compression_ratio,
            ambiguous_bits,
            meta_information,
        })
    }

    /// Serialize oscillatory patterns to binary representation
    fn serialize_patterns(&self, patterns: &OscillatoryPatterns) -> Result<Vec<u8>> {
        // Convert frequency components to bytes
        let mut binary_data = Vec::new();
        
        // Serialize each frequency component
        for component in [&patterns.cellular, &patterns.cardiac, &patterns.respiratory, 
                         &patterns.autonomic, &patterns.circadian] {
            binary_data.extend_from_slice(&component.amplitude.to_le_bytes());
            binary_data.extend_from_slice(&component.frequency.to_le_bytes());
            binary_data.extend_from_slice(&component.phase.to_le_bytes());
            binary_data.extend_from_slice(&component.power.to_le_bytes());
        }
        
        // Serialize coupling matrix
        for row in &patterns.coupling_matrix {
            for &value in row {
                binary_data.extend_from_slice(&value.to_le_bytes());
            }
        }
        
        Ok(binary_data)
    }

    /// Perform standard compression for comparison
    fn standard_compress(&self, data: &[u8]) -> Result<Vec<u8>> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(data)?;
        let compressed = encoder.finish()?;
        Ok(compressed)
    }

    /// Identify compression-resistant bits as ambiguous information
    fn identify_ambiguous_bits(&self, original: &[u8], compressed: &[u8]) -> Result<Vec<AmbiguousBit>> {
        let mut ambiguous_bits = Vec::new();
        
        // Analyze compression efficiency at bit level
        let local_compression_ratio = original.len() as f64 / compressed.len() as f64;
        
        // Process in batches to identify locally resistant patterns
        for (batch_start, batch) in original.chunks(self.config.batch_size).enumerate() {
            let batch_compressed = self.standard_compress(batch)?;
            let batch_ratio = batch.len() as f64 / batch_compressed.len() as f64;
            
            // Identify bits that resist compression more than threshold
            if batch_ratio < self.config.resistance_threshold * local_compression_ratio {
                for (bit_offset, &byte_val) in batch.iter().enumerate() {
                    let global_position = batch_start * self.config.batch_size + bit_offset;
                    
                    // Compute resistance ratio for this bit
                    let resistance_ratio = self.compute_bit_resistance(byte_val, batch)?;
                    
                    if resistance_ratio > self.config.resistance_threshold {
                        let interpretations = self.generate_interpretations(byte_val);
                        let meta_potential = self.compute_meta_potential(byte_val, global_position);
                        
                        ambiguous_bits.push(AmbiguousBit {
                            position: global_position,
                            resistance_ratio,
                            interpretations,
                            meta_potential,
                        });
                    }
                }
            }
        }
        
        Ok(ambiguous_bits)
    }

    /// Compute compression resistance for individual bit
    fn compute_bit_resistance(&self, byte_val: u8, context: &[u8]) -> Result<f64> {
        // Count bit transitions and patterns
        let bit_entropy = self.compute_bit_entropy(byte_val);
        let context_correlation = self.compute_context_correlation(byte_val, context);
        
        // Resistance increases with entropy and decreases with correlation
        let resistance = bit_entropy * (1.0 - context_correlation);
        
        Ok(resistance.clamp(0.0, 1.0))
    }

    /// Compute entropy of individual byte
    fn compute_bit_entropy(&self, byte_val: u8) -> f64 {
        // Count 1s and 0s in byte
        let ones = byte_val.count_ones() as f64;
        let zeros = 8.0 - ones;
        
        if ones == 0.0 || zeros == 0.0 {
            return 0.0; // No entropy in uniform bytes
        }
        
        let p_one = ones / 8.0;
        let p_zero = zeros / 8.0;
        
        -(p_one * p_one.log2() + p_zero * p_zero.log2())
    }

    /// Compute correlation with surrounding context
    fn compute_context_correlation(&self, byte_val: u8, context: &[u8]) -> f64 {
        if context.len() < 2 {
            return 0.0;
        }
        
        // Count occurrences of this byte value in context
        let occurrences = context.iter().filter(|&&b| b == byte_val).count() as f64;
        let correlation = occurrences / context.len() as f64;
        
        correlation
    }

    /// Generate possible interpretations for ambiguous byte
    fn generate_interpretations(&self, byte_val: u8) -> Vec<String> {
        let mut interpretations = Vec::new();
        
        // Binary interpretation
        interpretations.push(format!("binary: {:08b}", byte_val));
        
        // Decimal interpretation
        interpretations.push(format!("decimal: {}", byte_val));
        
        // Hexadecimal interpretation
        interpretations.push(format!("hex: 0x{:02X}", byte_val));
        
        // ASCII interpretation (if printable)
        if byte_val.is_ascii_graphic() {
            interpretations.push(format!("ascii: '{}'", byte_val as char));
        }
        
        // Physiological interpretation based on range
        let physio_interp = match byte_val {
            0..=63 => "low_amplitude_oscillation",
            64..=127 => "moderate_amplitude_oscillation", 
            128..=191 => "high_amplitude_oscillation",
            192..=255 => "saturated_amplitude_oscillation",
        };
        interpretations.push(format!("physiological: {}", physio_interp));
        
        interpretations
    }

    /// Compute meta-information potential of bit position
    fn compute_meta_potential(&self, byte_val: u8, position: usize) -> f64 {
        // Meta-potential increases with:
        // 1. Distance from most/least significant positions
        // 2. Bit pattern complexity
        // 3. Position in overall data stream
        
        let positional_factor = ((position % 64) as f64 - 32.0).abs() / 32.0; // Peak at center
        let complexity_factor = self.compute_bit_entropy(byte_val);
        let stream_factor = (position as f64).sin().abs(); // Periodic meta-information
        
        (positional_factor + complexity_factor + stream_factor) / 3.0
    }

    /// Extract meta-information from ambiguous bit patterns
    fn extract_meta_information(&self, ambiguous_bits: &[AmbiguousBit], 
                               patterns: &OscillatoryPatterns) -> Result<HashMap<String, serde_json::Value>> {
        let mut meta_info = HashMap::new();
        
        // Statistical meta-information
        meta_info.insert("ambiguous_bit_count".to_string(), 
                        serde_json::Value::Number(ambiguous_bits.len().into()));
        
        if !ambiguous_bits.is_empty() {
            let avg_resistance = ambiguous_bits.iter()
                .map(|bit| bit.resistance_ratio)
                .sum::<f64>() / ambiguous_bits.len() as f64;
            
            meta_info.insert("average_resistance_ratio".to_string(),
                           serde_json::Value::Number(serde_json::Number::from_f64(avg_resistance).unwrap()));
            
            let avg_meta_potential = ambiguous_bits.iter()
                .map(|bit| bit.meta_potential)
                .sum::<f64>() / ambiguous_bits.len() as f64;
            
            meta_info.insert("average_meta_potential".to_string(),
                           serde_json::Value::Number(serde_json::Number::from_f64(avg_meta_potential).unwrap()));
        }
        
        // Oscillatory meta-information
        let total_power = patterns.cellular.power + patterns.cardiac.power + 
                         patterns.respiratory.power + patterns.autonomic.power + 
                         patterns.circadian.power;
        
        meta_info.insert("total_oscillatory_power".to_string(),
                        serde_json::Value::Number(serde_json::Number::from_f64(total_power).unwrap()));
        
        // Coupling meta-information
        let coupling_strength: f64 = patterns.coupling_matrix.iter()
            .flat_map(|row| row.iter())
            .sum::<f64>() / (patterns.coupling_matrix.len() * patterns.coupling_matrix.len()) as f64;
        
        meta_info.insert("average_coupling_strength".to_string(),
                        serde_json::Value::Number(serde_json::Number::from_f64(coupling_strength).unwrap()));
        
        Ok(meta_info)
    }
}
