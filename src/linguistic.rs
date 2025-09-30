//! Linguistic transformation pipeline for semantic reorganization
//!
//! This module converts numerical data to linguistic representations,
//! reorganizes words alphabetically, and encodes back to binary.

use crate::types::{CompressedData, LinguisticTransformation};
use anyhow::Result;
use std::collections::HashMap;

/// Configuration for linguistic transformation
#[derive(Debug, Clone)]
pub struct LinguisticConfig {
    pub enable_alphabetical_sorting: bool,
    pub compression_target_ratio: f64,
    pub semantic_preservation_threshold: f64,
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

/// Linguistic transformation processor
pub struct LinguisticTransformer {
    config: LinguisticConfig,
    number_to_words: HashMap<u32, String>,
}

impl Default for LinguisticTransformer {
    fn default() -> Self {
        Self {
            config: LinguisticConfig::default(),
            number_to_words: Self::build_number_dictionary(),
        }
    }
}

impl LinguisticTransformer {
    /// Create new transformer with configuration
    pub fn new(config: LinguisticConfig) -> Self {
        Self {
            config,
            number_to_words: Self::build_number_dictionary(),
        }
    }

    /// Transform compressed data through linguistic pipeline
    pub fn transform_linguistic(&self, compressed_data: &CompressedData) -> Result<LinguisticTransformation> {
        // Extract numerical values from compressed data
        let numerical_values = self.extract_numerical_values(compressed_data)?;
        
        // Convert numbers to words
        let words = self.convert_numbers_to_words(&numerical_values)?;
        
        // Reorganize words alphabetically
        let reorganized = if self.config.enable_alphabetical_sorting {
            self.alphabetically_reorganize(&words)
        } else {
            words.clone()
        };
        
        // Encode reorganized words back to binary
        let encoded = self.encode_words_to_binary(&reorganized)?;
        
        // Compute semantic preservation score
        let semantic_preservation = self.compute_semantic_preservation(&words, &reorganized)?;
        
        Ok(LinguisticTransformation {
            original: numerical_values,
            words,
            reorganized,
            encoded,
            semantic_preservation,
        })
    }

    /// Extract numerical values from compressed data
    fn extract_numerical_values(&self, compressed_data: &CompressedData) -> Result<Vec<f64>> {
        let mut values = Vec::new();
        
        // Extract values from ambiguous bits
        for bit in &compressed_data.ambiguous_bits {
            values.push(bit.resistance_ratio);
            values.push(bit.meta_potential);
            values.push(bit.position as f64);
        }
        
        // Extract values from meta-information
        for (_, json_value) in &compressed_data.meta_information {
            if let serde_json::Value::Number(num) = json_value {
                if let Some(f_val) = num.as_f64() {
                    values.push(f_val);
                }
            }
        }
        
        // Add compression statistics
        values.push(compressed_data.compression_ratio);
        values.push(compressed_data.original_size as f64);
        values.push(compressed_data.compressed_size as f64);
        
        Ok(values)
    }

    /// Convert numerical values to word representations
    fn convert_numbers_to_words(&self, values: &[f64]) -> Result<Vec<String>> {
        let mut words = Vec::new();
        
        for &value in values {
            // Convert to integer for word lookup (scaled appropriately)
            let scaled_value = (value * 1000.0).round() as i64;
            let abs_value = scaled_value.abs() as u32;
            
            if let Some(word_representation) = self.number_to_words.get(&abs_value) {
                let mut full_word = word_representation.clone();
                
                // Add sign prefix if negative
                if scaled_value < 0 {
                    full_word = format!("negative {}", full_word);
                }
                
                // Add decimal information if significant
                let decimal_part = ((value.fract() * 1000.0).round() as u32) % 1000;
                if decimal_part > 0 {
                    if let Some(decimal_word) = self.number_to_words.get(&decimal_part) {
                        full_word = format!("{} point {}", full_word, decimal_word);
                    }
                }
                
                words.push(full_word);
            } else {
                // Fallback to digit-by-digit conversion for large numbers
                words.push(self.convert_large_number_to_words(abs_value));
            }
        }
        
        Ok(words)
    }

    /// Convert large numbers to words digit by digit
    fn convert_large_number_to_words(&self, number: u32) -> String {
        if number == 0 {
            return "zero".to_string();
        }
        
        let digits = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
        let mut word_parts = Vec::new();
        let mut n = number;
        
        while n > 0 {
            let digit = n % 10;
            word_parts.push(digits[digit as usize]);
            n /= 10;
        }
        
        word_parts.reverse();
        word_parts.join(" ")
    }

    /// Alphabetically reorganize words within each phrase
    fn alphabetically_reorganize(&self, words: &[String]) -> Vec<String> {
        words.iter()
            .map(|phrase| {
                let mut word_parts: Vec<&str> = phrase.split_whitespace().collect();
                word_parts.sort();
                word_parts.join(" ")
            })
            .collect()
    }

    /// Encode reorganized words to binary representation
    fn encode_words_to_binary(&self, words: &[String]) -> Result<Vec<u8>> {
        let mut binary_data = Vec::new();
        
        for word in words {
            // Simple encoding: UTF-8 bytes of the word
            binary_data.extend_from_slice(word.as_bytes());
            
            // Add delimiter
            binary_data.push(0x00);
        }
        
        Ok(binary_data)
    }

    /// Compute semantic preservation score
    fn compute_semantic_preservation(&self, original: &[String], reorganized: &[String]) -> Result<f64> {
        if original.len() != reorganized.len() {
            return Ok(0.0);
        }
        
        let mut preservation_score = 0.0;
        
        for (orig, reorg) in original.iter().zip(reorganized.iter()) {
            // Count word overlap
            let orig_words: std::collections::HashSet<&str> = orig.split_whitespace().collect();
            let reorg_words: std::collections::HashSet<&str> = reorg.split_whitespace().collect();
            
            let intersection_size = orig_words.intersection(&reorg_words).count();
            let union_size = orig_words.union(&reorg_words).count();
            
            if union_size > 0 {
                preservation_score += intersection_size as f64 / union_size as f64;
            }
        }
        
        Ok(preservation_score / original.len() as f64)
    }

    /// Build dictionary mapping numbers to words
    fn build_number_dictionary() -> HashMap<u32, String> {
        let mut dict = HashMap::new();
        
        // Basic numbers 0-19
        let basic_numbers = [
            (0, "zero"), (1, "one"), (2, "two"), (3, "three"), (4, "four"),
            (5, "five"), (6, "six"), (7, "seven"), (8, "eight"), (9, "nine"),
            (10, "ten"), (11, "eleven"), (12, "twelve"), (13, "thirteen"), (14, "fourteen"),
            (15, "fifteen"), (16, "sixteen"), (17, "seventeen"), (18, "eighteen"), (19, "nineteen"),
        ];
        
        for (num, word) in basic_numbers.iter() {
            dict.insert(*num, word.to_string());
        }
        
        // Tens
        let tens = [
            (20, "twenty"), (30, "thirty"), (40, "forty"), (50, "fifty"),
            (60, "sixty"), (70, "seventy"), (80, "eighty"), (90, "ninety"),
        ];
        
        for (num, word) in tens.iter() {
            dict.insert(*num, word.to_string());
        }
        
        // Hundreds
        for i in 1..=9 {
            let hundred_value = i * 100;
            if let Some(base_word) = dict.get(&i) {
                dict.insert(hundred_value, format!("{} hundred", base_word));
            }
        }
        
        // Composite numbers 21-99
        for tens_digit in 2..=9 {
            for ones_digit in 1..=9 {
                let number = tens_digit * 10 + ones_digit;
                if let (Some(tens_word), Some(ones_word)) = (dict.get(&(tens_digit * 10)), dict.get(&ones_digit)) {
                    dict.insert(number, format!("{} {}", tens_word, ones_word));
                }
            }
        }
        
        // Common physiological ranges
        dict.insert(120, "one hundred twenty".to_string());
        dict.insert(160, "one hundred sixty".to_string());
        dict.insert(180, "one hundred eighty".to_string());
        dict.insert(200, "two hundred".to_string());
        
        dict
    }
}
