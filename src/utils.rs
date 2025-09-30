//! Utility functions and helper methods for S-entropy analysis
//!
//! This module provides common functionality shared across the S-entropy framework.

use anyhow::Result;
use std::collections::HashMap;
use chrono::{DateTime, Utc};

/// Mathematical constants for S-entropy analysis
pub mod constants {
    /// St. Stella constant for low-information processing (π/e)
    pub const ST_STELLA: f64 = std::f64::consts::PI / std::f64::consts::E;
    
    /// Maximum entropy for 4-state directional system (log2(4))
    pub const MAX_DIRECTIONAL_ENTROPY: f64 = 2.0;
    
    /// Default convergence threshold for numerical methods
    pub const DEFAULT_CONVERGENCE_THRESHOLD: f64 = 1e-6;
    
    /// Physiological frequency scale boundaries (Hz)
    pub const CELLULAR_MIN: f64 = 0.1;
    pub const CELLULAR_MAX: f64 = 100.0;
    pub const CARDIAC_MIN: f64 = 0.01;
    pub const CARDIAC_MAX: f64 = 10.0;
    pub const RESPIRATORY_MIN: f64 = 0.001;
    pub const RESPIRATORY_MAX: f64 = 1.0;
    pub const AUTONOMIC_MIN: f64 = 0.0001;
    pub const AUTONOMIC_MAX: f64 = 0.1;
    pub const CIRCADIAN_MIN: f64 = 0.00001;
    pub const CIRCADIAN_MAX: f64 = 0.01;
}

/// Statistical utility functions
pub mod stats {
    use super::*;

    /// Compute mean of vector
    pub fn mean(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        data.iter().sum::<f64>() / data.len() as f64
    }

    /// Compute sample variance
    pub fn variance(data: &[f64]) -> f64 {
        if data.len() < 2 {
            return 0.0;
        }
        let mean_val = mean(data);
        let sum_squared_diff: f64 = data.iter()
            .map(|x| (x - mean_val).powi(2))
            .sum();
        sum_squared_diff / (data.len() - 1) as f64
    }

    /// Compute standard deviation
    pub fn std_dev(data: &[f64]) -> f64 {
        variance(data).sqrt()
    }

    /// Compute Shannon entropy of discrete distribution
    pub fn shannon_entropy(data: &[f64]) -> f64 {
        if data.is_empty() {
            return 0.0;
        }
        
        let total: f64 = data.iter().sum();
        if total == 0.0 {
            return 0.0;
        }
        
        data.iter()
            .filter(|&&x| x > 0.0)
            .map(|&x| {
                let p = x / total;
                -p * p.log2()
            })
            .sum()
    }

    /// Compute Pearson correlation coefficient
    pub fn pearson_correlation(x: &[f64], y: &[f64]) -> Result<f64> {
        if x.len() != y.len() || x.len() < 2 {
            return Ok(0.0);
        }

        let mean_x = mean(x);
        let mean_y = mean(y);
        
        let numerator: f64 = x.iter().zip(y.iter())
            .map(|(&xi, &yi)| (xi - mean_x) * (yi - mean_y))
            .sum();
        
        let sum_sq_x: f64 = x.iter().map(|&xi| (xi - mean_x).powi(2)).sum();
        let sum_sq_y: f64 = y.iter().map(|&yi| (yi - mean_y).powi(2)).sum();
        
        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        
        if denominator == 0.0 {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }

    /// Compute moving average with specified window size
    pub fn moving_average(data: &[f64], window_size: usize) -> Vec<f64> {
        if window_size == 0 || window_size > data.len() {
            return data.to_vec();
        }

        let mut result = Vec::with_capacity(data.len());
        
        for i in 0..data.len() {
            let start = if i >= window_size { i - window_size + 1 } else { 0 };
            let window = &data[start..=i];
            result.push(mean(window));
        }
        
        result
    }
}

/// Signal processing utilities
pub mod signal {
    use super::*;

    /// Normalize signal to zero mean, unit variance
    pub fn normalize(signal: &[f64]) -> Vec<f64> {
        let mean_val = stats::mean(signal);
        let std_val = stats::std_dev(signal);
        
        if std_val == 0.0 {
            return vec![0.0; signal.len()];
        }
        
        signal.iter()
            .map(|&x| (x - mean_val) / std_val)
            .collect()
    }

    /// Apply simple moving average filter
    pub fn smooth(signal: &[f64], window_size: usize) -> Vec<f64> {
        stats::moving_average(signal, window_size)
    }

    /// Compute signal envelope using Hilbert transform approximation
    pub fn envelope(signal: &[f64]) -> Vec<f64> {
        // Simplified envelope detection using local maxima
        let mut envelope = Vec::with_capacity(signal.len());
        let window = 5; // Local window for maxima detection
        
        for i in 0..signal.len() {
            let start = if i >= window { i - window } else { 0 };
            let end = if i + window < signal.len() { i + window + 1 } else { signal.len() };
            
            let local_max = signal[start..end].iter()
                .cloned()
                .fold(f64::NEG_INFINITY, f64::max);
            
            envelope.push(local_max);
        }
        
        envelope
    }

    /// Detect peaks in signal
    pub fn find_peaks(signal: &[f64], min_height: f64, min_distance: usize) -> Vec<usize> {
        let mut peaks = Vec::new();
        
        if signal.len() < 3 {
            return peaks;
        }
        
        for i in 1..signal.len()-1 {
            if signal[i] > signal[i-1] && signal[i] > signal[i+1] && signal[i] >= min_height {
                // Check minimum distance from previous peak
                if peaks.is_empty() || i - peaks[peaks.len()-1] >= min_distance {
                    peaks.push(i);
                }
            }
        }
        
        peaks
    }
}

/// Time series analysis utilities
pub mod time_series {
    use super::*;

    /// Compute autocorrelation function
    pub fn autocorrelation(data: &[f64], max_lag: usize) -> Vec<f64> {
        let n = data.len();
        let max_lag = max_lag.min(n - 1);
        let mut autocorr = Vec::with_capacity(max_lag + 1);
        
        let mean_val = stats::mean(data);
        let variance = stats::variance(data);
        
        if variance == 0.0 {
            return vec![1.0; max_lag + 1];
        }
        
        for lag in 0..=max_lag {
            let mut sum = 0.0;
            let count = n - lag;
            
            for i in 0..count {
                sum += (data[i] - mean_val) * (data[i + lag] - mean_val);
            }
            
            autocorr.push(sum / (count as f64 * variance));
        }
        
        autocorr
    }

    /// Compute cross-correlation between two signals
    pub fn cross_correlation(x: &[f64], y: &[f64], max_lag: usize) -> Vec<f64> {
        let n = x.len().min(y.len());
        let max_lag = max_lag.min(n - 1);
        let mut cross_corr = Vec::with_capacity(2 * max_lag + 1);
        
        let mean_x = stats::mean(x);
        let mean_y = stats::mean(y);
        let std_x = stats::std_dev(x);
        let std_y = stats::std_dev(y);
        
        if std_x == 0.0 || std_y == 0.0 {
            return vec![0.0; 2 * max_lag + 1];
        }
        
        // Negative lags
        for lag in (1..=max_lag).rev() {
            let mut sum = 0.0;
            let count = n - lag;
            
            for i in lag..n {
                sum += (x[i] - mean_x) * (y[i - lag] - mean_y);
            }
            
            cross_corr.push(sum / (count as f64 * std_x * std_y));
        }
        
        // Zero lag
        let zero_lag_corr = stats::pearson_correlation(x, y).unwrap_or(0.0);
        cross_corr.push(zero_lag_corr);
        
        // Positive lags
        for lag in 1..=max_lag {
            let mut sum = 0.0;
            let count = n - lag;
            
            for i in 0..count {
                sum += (x[i] - mean_x) * (y[i + lag] - mean_y);
            }
            
            cross_corr.push(sum / (count as f64 * std_x * std_y));
        }
        
        cross_corr
    }
}

/// Text processing utilities
pub mod text {
    use super::*;

    /// Convert number to English words
    pub fn number_to_words(n: u64) -> String {
        if n == 0 {
            return "zero".to_string();
        }
        
        let ones = ["", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                   "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen",
                   "seventeen", "eighteen", "nineteen"];
        
        let tens = ["", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"];
        
        let scales = ["", "thousand", "million", "billion", "trillion"];
        
        fn convert_hundreds(n: u64, ones: &[&str], tens: &[&str]) -> String {
            let mut result = String::new();
            
            if n >= 100 {
                result.push_str(ones[(n / 100) as usize]);
                result.push_str(" hundred");
                let remainder = n % 100;
                if remainder > 0 {
                    result.push(' ');
                    result.push_str(&convert_under_hundred(remainder, ones, tens));
                }
            } else {
                result = convert_under_hundred(n, ones, tens);
            }
            
            result
        }
        
        fn convert_under_hundred(n: u64, ones: &[&str], tens: &[&str]) -> String {
            if n < 20 {
                ones[n as usize].to_string()
            } else {
                let ten = (n / 10) as usize;
                let one = (n % 10) as usize;
                if one == 0 {
                    tens[ten].to_string()
                } else {
                    format!("{} {}", tens[ten], ones[one])
                }
            }
        }
        
        let mut parts = Vec::new();
        let mut num = n;
        let mut scale_index = 0;
        
        while num > 0 {
            let chunk = num % 1000;
            if chunk > 0 {
                let chunk_words = convert_hundreds(chunk, &ones, &tens);
                if scale_index > 0 {
                    parts.push(format!("{} {}", chunk_words, scales[scale_index]));
                } else {
                    parts.push(chunk_words);
                }
            }
            num /= 1000;
            scale_index += 1;
        }
        
        parts.reverse();
        parts.join(" ")
    }

    /// Compute Levenshtein distance between two strings
    pub fn levenshtein_distance(s1: &str, s2: &str) -> usize {
        let len1 = s1.chars().count();
        let len2 = s2.chars().count();
        
        if len1 == 0 {
            return len2;
        }
        if len2 == 0 {
            return len1;
        }
        
        let mut matrix = vec![vec![0; len2 + 1]; len1 + 1];
        
        for i in 0..=len1 {
            matrix[i][0] = i;
        }
        for j in 0..=len2 {
            matrix[0][j] = j;
        }
        
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        
        for i in 1..=len1 {
            for j in 1..=len2 {
                let cost = if chars1[i-1] == chars2[j-1] { 0 } else { 1 };
                matrix[i][j] = (matrix[i-1][j] + 1)
                    .min(matrix[i][j-1] + 1)
                    .min(matrix[i-1][j-1] + cost);
            }
        }
        
        matrix[len1][len2]
    }
}

/// Data validation utilities
pub mod validation {
    use super::*;

    /// Check if data contains NaN or infinite values
    pub fn is_valid_data(data: &[f64]) -> bool {
        data.iter().all(|&x| x.is_finite())
    }

    /// Remove NaN and infinite values from data
    pub fn clean_data(data: &[f64]) -> Vec<f64> {
        data.iter()
            .filter(|&&x| x.is_finite())
            .cloned()
            .collect()
    }

    /// Check if sensor data is within physiological ranges
    pub fn is_physiological_range(sensor_type: &str, value: f64) -> bool {
        match sensor_type {
            "heart_rate" => value >= 30.0 && value <= 220.0,
            "temperature" => value >= 32.0 && value <= 42.0, // Celsius
            "accelerometer" => value.abs() <= 50.0, // g-force
            "ppg" => value >= -10.0 && value <= 10.0, // Normalized units
            _ => true, // Default to valid for unknown sensors
        }
    }

    /// Validate S-entropy coordinates
    pub fn validate_s_entropy_coords(coords: &crate::types::SEntropyCoordinates) -> Result<()> {
        let values = [coords.knowledge, coords.time, coords.entropy, coords.context];
        
        for (i, &value) in values.iter().enumerate() {
            if !value.is_finite() {
                return Err(anyhow::anyhow!("S-entropy coordinate {} is not finite: {}", i, value));
            }
            if value < 0.0 || value > 1.0 {
                return Err(anyhow::anyhow!("S-entropy coordinate {} out of range [0,1]: {}", i, value));
            }
        }
        
        Ok(())
    }
}

/// Logging and debugging utilities
pub mod debug {
    use super::*;

    /// Create debug summary of sensor data
    pub fn summarize_sensor_data(data: &crate::types::SensorData) -> HashMap<String, serde_json::Value> {
        let mut summary = HashMap::new();
        
        summary.insert("timestamp".to_string(), 
                      serde_json::Value::String(data.timestamp.to_rfc3339()));
        
        if let Some(ppg) = &data.sensors.ppg {
            summary.insert("ppg_count".to_string(), 
                          serde_json::Value::Number(ppg.len().into()));
            if !ppg.is_empty() {
                summary.insert("ppg_range".to_string(),
                              serde_json::json!([ppg.iter().cloned().fold(f64::INFINITY, f64::min),
                                               ppg.iter().cloned().fold(f64::NEG_INFINITY, f64::max)]));
            }
        }
        
        if let Some(acc) = &data.sensors.accelerometer {
            summary.insert("accelerometer_samples".to_string(),
                          serde_json::Value::Number(acc.x.len().into()));
        }
        
        if let Some(temp) = &data.sensors.temperature {
            summary.insert("temperature_samples".to_string(),
                          serde_json::Value::Number(temp.len().into()));
        }
        
        if let Some(context) = &data.context {
            let mut context_summary = HashMap::new();
            if let Some(activity) = &context.activity_level {
                context_summary.insert("activity_level", activity.clone());
            }
            if let Some(temp) = context.ambient_temperature {
                context_summary.insert("ambient_temperature", temp.to_string());
            }
            summary.insert("context".to_string(), 
                          serde_json::Value::Object(context_summary.into_iter()
                                                  .map(|(k, v)| (k, serde_json::Value::String(v)))
                                                  .collect()));
        }
        
        summary
    }

    /// Log performance metrics
    pub fn log_performance_metrics(operation: &str, duration: std::time::Duration, 
                                 data_points: usize) {
        let duration_ms = duration.as_millis();
        let throughput = if duration_ms > 0 {
            (data_points as f64 * 1000.0) / duration_ms as f64
        } else {
            0.0
        };
        
        log::info!("Performance: {} completed in {}ms, {} data points, {:.2} points/sec",
                  operation, duration_ms, data_points, throughput);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mean_calculation() {
        assert_eq!(stats::mean(&[1.0, 2.0, 3.0, 4.0, 5.0]), 3.0);
        assert_eq!(stats::mean(&[]), 0.0);
    }

    #[test]
    fn test_shannon_entropy() {
        let uniform = vec![0.25, 0.25, 0.25, 0.25];
        let entropy = stats::shannon_entropy(&uniform);
        assert!((entropy - 2.0).abs() < 1e-10); // log2(4) = 2
    }

    #[test]
    fn test_number_to_words() {
        assert_eq!(text::number_to_words(0), "zero");
        assert_eq!(text::number_to_words(42), "forty two");
        assert_eq!(text::number_to_words(120), "one hundred twenty");
    }

    #[test]
    fn test_data_validation() {
        assert!(validation::is_valid_data(&[1.0, 2.0, 3.0]));
        assert!(!validation::is_valid_data(&[1.0, f64::NAN, 3.0]));
        assert!(!validation::is_valid_data(&[1.0, f64::INFINITY, 3.0]));
    }

    #[test]
    fn test_physiological_ranges() {
        assert!(validation::is_physiological_range("heart_rate", 72.0));
        assert!(!validation::is_physiological_range("heart_rate", 300.0));
        assert!(validation::is_physiological_range("temperature", 37.0));
        assert!(!validation::is_physiological_range("temperature", 50.0));
    }
}
