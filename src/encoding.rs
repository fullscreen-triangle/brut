//! Sequence-based physiological pattern encoding
//!
//! This module encodes linguistically transformed data into directional sequences
//! using A, R, D, L mapping based on contextual physiological states.

use crate::types::{LinguisticTransformation, EncodedSequence, DirectionalMapping};
use anyhow::Result;

/// Configuration for sequence encoding
#[derive(Debug, Clone)]
pub struct EncodingConfig {
    pub activation_threshold: f64,    // A = Elevated/Activation
    pub steady_threshold: f64,        // R = Steady/Maintenance  
    pub recovery_threshold: f64,      // D = Decreased/Recovery
    pub stress_threshold: f64,        // L = Stress/Transition
    pub context_dependent: bool,
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

/// Sequence encoder for physiological patterns
pub struct SequenceEncoder {
    config: EncodingConfig,
}

impl Default for SequenceEncoder {
    fn default() -> Self {
        Self {
            config: EncodingConfig::default(),
        }
    }
}

impl SequenceEncoder {
    /// Create new encoder with configuration
    pub fn new(config: EncodingConfig) -> Self {
        Self { config }
    }

    /// Encode linguistic transformation into directional sequences
    pub fn encode_physiological_sequences(&self, linguistic_data: &LinguisticTransformation) -> Result<EncodedSequence> {
        // Extract physiological features from linguistic data
        let features = self.extract_physiological_features(linguistic_data)?;
        
        // Create contextual mapping based on features
        let mapping = self.create_directional_mapping(&features)?;
        
        // Generate directional sequence
        let sequence = self.generate_directional_sequence(&features, &mapping)?;
        
        // Identify context factors
        let context_factors = self.identify_context_factors(&features);
        
        // Compute encoding confidence
        let confidence = self.compute_encoding_confidence(&features, &sequence)?;
        
        Ok(EncodedSequence {
            sequence,
            mapping,
            context_factors,
            confidence,
        })
    }

    /// Extract physiological features from linguistic transformation
    fn extract_physiological_features(&self, linguistic_data: &LinguisticTransformation) -> Result<Vec<f64>> {
        let mut features = Vec::new();
        
        // Extract features from original numerical values
        features.extend_from_slice(&linguistic_data.original);
        
        // Extract features from word patterns
        for word in &linguistic_data.words {
            features.push(self.compute_word_complexity(word));
            features.push(self.compute_phonetic_weight(word));
        }
        
        // Extract features from reorganized patterns
        for reorganized in &linguistic_data.reorganized {
            features.push(self.compute_alphabetical_deviation(reorganized));
        }
        
        // Add semantic preservation as feature
        features.push(linguistic_data.semantic_preservation);
        
        // Normalize features to physiological ranges
        let normalized_features = self.normalize_to_physiological_range(&features);
        
        Ok(normalized_features)
    }

    /// Compute complexity score for individual word
    fn compute_word_complexity(&self, word: &str) -> f64 {
        if word.is_empty() {
            return 0.0;
        }
        
        // Factors contributing to complexity:
        // 1. Length
        // 2. Unique character count
        // 3. Syllable count (approximated)
        
        let length_factor = word.len() as f64 / 20.0; // Normalize by typical max length
        
        let unique_chars = word.chars().collect::<std::collections::HashSet<_>>().len() as f64;
        let unique_factor = unique_chars / word.len() as f64;
        
        let syllable_factor = self.estimate_syllable_count(word) as f64 / 10.0;
        
        (length_factor + unique_factor + syllable_factor) / 3.0
    }

    /// Estimate syllable count in word (simplified)
    fn estimate_syllable_count(&self, word: &str) -> usize {
        let vowels = "aeiouAEIOU";
        let mut syllable_count = 0;
        let mut prev_was_vowel = false;
        
        for char in word.chars() {
            let is_vowel = vowels.contains(char);
            if is_vowel && !prev_was_vowel {
                syllable_count += 1;
            }
            prev_was_vowel = is_vowel;
        }
        
        // Ensure at least one syllable for non-empty words
        syllable_count.max(1)
    }

    /// Compute phonetic weight based on sound characteristics
    fn compute_phonetic_weight(&self, word: &str) -> f64 {
        if word.is_empty() {
            return 0.0;
        }
        
        // Weight based on phonetic characteristics that might correlate with physiological states
        let consonant_clusters = word.matches(|c: char| "bcdfghjklmnpqrstvwxyz".contains(c.to_ascii_lowercase())).count() as f64;
        let vowel_openness = word.matches(|c: char| "aeo".contains(c.to_ascii_lowercase())).count() as f64;
        let fricatives = word.matches(|c: char| "fvszh".contains(c.to_ascii_lowercase())).count() as f64;
        
        let total_chars = word.len() as f64;
        if total_chars == 0.0 {
            return 0.0;
        }
        
        // Combine phonetic features
        let consonant_density = consonant_clusters / total_chars;
        let vowel_openness_ratio = vowel_openness / total_chars;
        let fricative_ratio = fricatives / total_chars;
        
        (consonant_density * 0.4 + vowel_openness_ratio * 0.3 + fricative_ratio * 0.3).clamp(0.0, 1.0)
    }

    /// Compute alphabetical deviation score
    fn compute_alphabetical_deviation(&self, reorganized_phrase: &str) -> f64 {
        let words: Vec<&str> = reorganized_phrase.split_whitespace().collect();
        if words.len() < 2 {
            return 0.0;
        }
        
        let mut deviation_sum = 0.0;
        for i in 1..words.len() {
            // Measure how much each word deviates from perfect alphabetical order
            let current_first_char = words[i].chars().next().unwrap_or('z') as u8;
            let prev_first_char = words[i-1].chars().next().unwrap_or('a') as u8;
            
            if current_first_char < prev_first_char {
                // Out of order - penalize
                deviation_sum += (prev_first_char - current_first_char) as f64 / 25.0; // Normalize by alphabet size
            }
        }
        
        (deviation_sum / (words.len() - 1) as f64).clamp(0.0, 1.0)
    }

    /// Normalize features to physiological interpretation ranges
    fn normalize_to_physiological_range(&self, features: &[f64]) -> Vec<f64> {
        if features.is_empty() {
            return Vec::new();
        }
        
        // Map to range suitable for physiological interpretation (-1.0 to 1.0)
        let min_val = features.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = features.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        if (max_val - min_val).abs() < f64::EPSILON {
            return vec![0.0; features.len()]; // All values the same
        }
        
        features.iter()
            .map(|&x| 2.0 * (x - min_val) / (max_val - min_val) - 1.0) // Map to [-1, 1]
            .collect()
    }

    /// Create directional mapping based on physiological features
    fn create_directional_mapping(&self, features: &[f64]) -> Result<DirectionalMapping> {
        Ok(DirectionalMapping {
            activation_threshold: self.config.activation_threshold,
            steady_threshold: self.config.steady_threshold,
            recovery_threshold: self.config.recovery_threshold,
            stress_threshold: self.config.stress_threshold,
            context_dependent: self.config.context_dependent,
        })
    }

    /// Generate directional sequence from features
    fn generate_directional_sequence(&self, features: &[f64], mapping: &DirectionalMapping) -> Result<String> {
        let mut sequence = String::new();
        
        for &feature in features {
            let direction = if feature >= mapping.stress_threshold {
                'L' // Stress/Transition state
            } else if feature >= mapping.activation_threshold {
                'A' // Elevated/Activation state
            } else if feature >= mapping.steady_threshold {
                'R' // Steady/Maintenance state
            } else {
                'D' // Decreased/Recovery state
            };
            
            sequence.push(direction);
        }
        
        Ok(sequence)
    }

    /// Identify contextual factors influencing encoding
    fn identify_context_factors(&self, features: &[f64]) -> Vec<String> {
        let mut factors = Vec::new();
        
        // Analyze feature patterns to infer context
        let mean_feature = features.iter().sum::<f64>() / features.len() as f64;
        let variance = features.iter()
            .map(|x| (x - mean_feature).powi(2))
            .sum::<f64>() / features.len() as f64;
        
        if variance > 0.5 {
            factors.push("high_variability".to_string());
        }
        
        if mean_feature > 0.5 {
            factors.push("elevated_baseline".to_string());
        } else if mean_feature < -0.5 {
            factors.push("suppressed_baseline".to_string());
        }
        
        let max_feature = features.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_feature = features.iter().cloned().fold(f64::INFINITY, f64::min);
        
        if max_feature > 0.9 {
            factors.push("peak_activation".to_string());
        }
        
        if min_feature < -0.9 {
            factors.push("deep_recovery".to_string());
        }
        
        if factors.is_empty() {
            factors.push("stable_pattern".to_string());
        }
        
        factors
    }

    /// Compute confidence score for encoding
    fn compute_encoding_confidence(&self, features: &[f64], sequence: &str) -> Result<f64> {
        if features.is_empty() || sequence.is_empty() {
            return Ok(0.0);
        }
        
        // Confidence factors:
        // 1. Feature consistency with sequence
        // 2. Sequence pattern regularity
        // 3. Physiological plausibility
        
        let feature_consistency = self.compute_feature_sequence_consistency(features, sequence);
        let pattern_regularity = self.compute_sequence_regularity(sequence);
        let physiological_plausibility = self.compute_physiological_plausibility(sequence);
        
        let confidence = (feature_consistency * 0.4 + pattern_regularity * 0.3 + physiological_plausibility * 0.3)
            .clamp(0.0, 1.0);
        
        Ok(confidence)
    }

    /// Compute consistency between features and generated sequence
    fn compute_feature_sequence_consistency(&self, features: &[f64], sequence: &str) -> f64 {
        let sequence_chars: Vec<char> = sequence.chars().collect();
        if features.len() != sequence_chars.len() {
            return 0.0;
        }
        
        let mut consistency_score = 0.0;
        
        for (feature, direction) in features.iter().zip(sequence_chars.iter()) {
            let expected_consistent = match direction {
                'L' => *feature >= self.config.stress_threshold,
                'A' => *feature >= self.config.activation_threshold && *feature < self.config.stress_threshold,
                'R' => *feature >= self.config.steady_threshold && *feature < self.config.activation_threshold,
                'D' => *feature < self.config.steady_threshold,
                _ => false,
            };
            
            if expected_consistent {
                consistency_score += 1.0;
            }
        }
        
        consistency_score / features.len() as f64
    }

    /// Compute regularity of sequence pattern
    fn compute_sequence_regularity(&self, sequence: &str) -> f64 {
        if sequence.len() < 2 {
            return 0.0;
        }
        
        // Measure transition smoothness and pattern recognition
        let chars: Vec<char> = sequence.chars().collect();
        let mut transition_score = 0.0;
        
        for i in 1..chars.len() {
            // Score based on physiological plausible transitions
            let transition_weight = match (chars[i-1], chars[i]) {
                ('R', 'A') | ('A', 'L') | ('L', 'D') | ('D', 'R') => 1.0, // Natural progressions
                ('A', 'R') | ('L', 'A') | ('D', 'L') | ('R', 'D') => 0.8, // Reasonable returns
                ('R', 'L') | ('A', 'D') => 0.6, // Direct transitions
                ('L', 'R') | ('D', 'A') => 0.4, // Less common
                _ => 0.2, // Irregular transitions
            };
            
            transition_score += transition_weight;
        }
        
        transition_score / (chars.len() - 1) as f64
    }

    /// Compute physiological plausibility of sequence
    fn compute_physiological_plausibility(&self, sequence: &str) -> f64 {
        if sequence.is_empty() {
            return 0.0;
        }
        
        // Check for physiologically implausible patterns
        let mut plausibility = 1.0;
        
        // Penalize excessive consecutive states (except R which can be sustained)
        for state in ['A', 'L', 'D'] {
            let state_str = &state.to_string().repeat(5); // 5 consecutive same states
            if sequence.contains(state_str) {
                plausibility *= 0.7; // Reduce plausibility
            }
        }
        
        // Reward balanced state distribution
        let char_counts: std::collections::HashMap<char, usize> = sequence.chars().fold(
            std::collections::HashMap::new(),
            |mut map, c| {
                *map.entry(c).or_insert(0) += 1;
                map
            }
        );
        
        let total_chars = sequence.len() as f64;
        let entropy = char_counts.values()
            .map(|&count| {
                let p = count as f64 / total_chars;
                -p * p.log2()
            })
            .sum::<f64>();
        
        let max_entropy = 2.0; // log2(4) for 4 possible states
        let entropy_score = entropy / max_entropy;
        
        plausibility * (0.7 + 0.3 * entropy_score) // Reward moderate entropy
    }
}
