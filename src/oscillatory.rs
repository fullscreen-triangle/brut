//! Multi-scale oscillatory decomposition for physiological sensor data
//!
//! This module implements oscillatory pattern extraction across five biological 
//! frequency scales: cellular, cardiac, respiratory, autonomic, and circadian.

use crate::types::{SensorData, OscillatoryPatterns, OscillatoryConfig, FrequencyComponent};
use anyhow::Result;
use nalgebra::{DVector, DMatrix};
use rustfft::{FftPlanner, num_complex::Complex};

/// Processor for multi-scale oscillatory decomposition
pub struct OscillatoryProcessor {
    config: OscillatoryConfig,
    fft_planner: FftPlanner<f64>,
}

impl OscillatoryProcessor {
    /// Create new oscillatory processor with configuration
    pub fn new(config: OscillatoryConfig) -> Self {
        Self {
            config,
            fft_planner: FftPlanner::new(),
        }
    }

    /// Extract oscillatory patterns from multi-sensor data
    pub fn extract_oscillatory_patterns(&self, sensor_data: &SensorData) -> Result<OscillatoryPatterns> {
        // Decompose each sensor modality across biological frequency scales
        let cellular = self.extract_frequency_component(&sensor_data, self.config.frequency_bands.cellular)?;
        let cardiac = self.extract_frequency_component(&sensor_data, self.config.frequency_bands.cardiac)?;
        let respiratory = self.extract_frequency_component(&sensor_data, self.config.frequency_bands.respiratory)?;
        let autonomic = self.extract_frequency_component(&sensor_data, self.config.frequency_bands.autonomic)?;
        let circadian = self.extract_frequency_component(&sensor_data, self.config.frequency_bands.circadian)?;

        // Compute cross-frequency coupling matrix
        let coupling_matrix = if self.config.enable_coupling_analysis {
            self.compute_coupling_matrix(&[&cellular, &cardiac, &respiratory, &autonomic, &circadian])?
        } else {
            vec![vec![0.0; 5]; 5]
        };

        Ok(OscillatoryPatterns {
            cellular,
            cardiac,
            respiratory,
            autonomic,
            circadian,
            coupling_matrix,
        })
    }

    /// Extract frequency component for specific biological scale
    fn extract_frequency_component(&self, sensor_data: &SensorData, frequency_band: (f64, f64)) -> Result<FrequencyComponent> {
        // Combine all available sensor data into composite signal
        let composite_signal = self.create_composite_signal(sensor_data)?;
        
        // Apply bandpass filter for frequency band
        let filtered_signal = self.bandpass_filter(&composite_signal, frequency_band)?;
        
        // Compute FFT for frequency domain analysis
        let spectrum = self.compute_spectrum(&filtered_signal)?;
        
        // Extract dominant frequency component
        let (amplitude, frequency, phase, power) = self.extract_dominant_component(&spectrum, frequency_band)?;
        
        Ok(FrequencyComponent {
            amplitude,
            frequency,
            phase,
            power,
            coherence: None, // Computed separately if needed
        })
    }

    /// Create composite signal from multiple sensor modalities
    fn create_composite_signal(&self, sensor_data: &SensorData) -> Result<Vec<f64>> {
        let mut composite = Vec::new();
        
        // Normalize and combine PPG data
        if let Some(ppg_data) = &sensor_data.sensors.ppg {
            let normalized_ppg = self.normalize_signal(ppg_data);
            composite.extend(normalized_ppg);
        }
        
        // Normalize and combine accelerometer magnitude
        if let Some(acc_data) = &sensor_data.sensors.accelerometer {
            let magnitude = self.compute_accelerometer_magnitude(acc_data)?;
            let normalized_acc = self.normalize_signal(&magnitude);
            composite.extend(normalized_acc);
        }
        
        // Normalize and combine temperature data
        if let Some(temp_data) = &sensor_data.sensors.temperature {
            let normalized_temp = self.normalize_signal(temp_data);
            composite.extend(normalized_temp);
        }
        
        if composite.is_empty() {
            anyhow::bail!("No valid sensor data found");
        }
        
        Ok(composite)
    }

    /// Compute accelerometer magnitude from 3-axis data
    fn compute_accelerometer_magnitude(&self, acc_data: &crate::types::AccelerometerData) -> Result<Vec<f64>> {
        if acc_data.magnitude.is_some() {
            return Ok(acc_data.magnitude.as_ref().unwrap().clone());
        }
        
        let len = acc_data.x.len().min(acc_data.y.len()).min(acc_data.z.len());
        let mut magnitude = Vec::with_capacity(len);
        
        for i in 0..len {
            let mag = (acc_data.x[i].powi(2) + acc_data.y[i].powi(2) + acc_data.z[i].powi(2)).sqrt();
            magnitude.push(mag);
        }
        
        Ok(magnitude)
    }

    /// Normalize signal to zero mean, unit variance
    fn normalize_signal(&self, signal: &[f64]) -> Vec<f64> {
        if signal.is_empty() {
            return Vec::new();
        }
        
        let mean = signal.iter().sum::<f64>() / signal.len() as f64;
        let variance = signal.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / signal.len() as f64;
        let std_dev = variance.sqrt();
        
        if std_dev == 0.0 {
            return vec![0.0; signal.len()];
        }
        
        signal.iter()
            .map(|x| (x - mean) / std_dev)
            .collect()
    }

    /// Apply bandpass filter for frequency range
    fn bandpass_filter(&self, signal: &[f64], frequency_band: (f64, f64)) -> Result<Vec<f64>> {
        // Simplified bandpass filter implementation
        // In a full implementation, this would use proper digital filter design
        let mut filtered = signal.to_vec();
        
        // Apply simple moving average for demonstration
        let window_size = (10.0 / frequency_band.1).max(3.0) as usize;
        if window_size < signal.len() {
            for i in window_size..signal.len() {
                let avg = filtered[i-window_size..i].iter().sum::<f64>() / window_size as f64;
                filtered[i] = signal[i] - avg;
            }
        }
        
        Ok(filtered)
    }

    /// Compute frequency spectrum using FFT
    fn compute_spectrum(&self, signal: &[f64]) -> Result<Vec<Complex<f64>>> {
        let mut input: Vec<Complex<f64>> = signal.iter()
            .map(|&x| Complex::new(x, 0.0))
            .collect();
        
        // Pad to power of 2 for efficient FFT
        let padded_len = signal.len().next_power_of_two();
        input.resize(padded_len, Complex::new(0.0, 0.0));
        
        let fft = self.fft_planner.plan_fft_forward(padded_len);
        fft.process(&mut input);
        
        Ok(input)
    }

    /// Extract dominant frequency component from spectrum
    fn extract_dominant_component(&self, spectrum: &[Complex<f64>], frequency_band: (f64, f64)) -> Result<(f64, f64, f64, f64)> {
        // Find peak in frequency band
        let mut max_power = 0.0;
        let mut peak_index = 0;
        
        let sampling_rate = 100.0; // Assumed sampling rate
        let freq_resolution = sampling_rate / spectrum.len() as f64;
        
        for (i, &complex_val) in spectrum.iter().enumerate() {
            let frequency = i as f64 * freq_resolution;
            if frequency >= frequency_band.0 && frequency <= frequency_band.1 {
                let power = complex_val.norm_sqr();
                if power > max_power {
                    max_power = power;
                    peak_index = i;
                }
            }
        }
        
        let amplitude = spectrum[peak_index].norm();
        let frequency = peak_index as f64 * freq_resolution;
        let phase = spectrum[peak_index].arg();
        let power = max_power;
        
        Ok((amplitude, frequency, phase, power))
    }

    /// Compute cross-frequency coupling matrix
    fn compute_coupling_matrix(&self, components: &[&FrequencyComponent]) -> Result<Vec<Vec<f64>>> {
        let n = components.len();
        let mut coupling_matrix = vec![vec![0.0; n]; n];
        
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    // Compute phase coupling between frequency components
                    let phase_diff = (components[i].phase - components[j].phase).abs();
                    let coupling_strength = (1.0 - phase_diff / std::f64::consts::PI).max(0.0);
                    
                    // Weight by amplitude correlation
                    let amplitude_correlation = (components[i].amplitude * components[j].amplitude).sqrt();
                    
                    coupling_matrix[i][j] = coupling_strength * amplitude_correlation;
                } else {
                    coupling_matrix[i][j] = 1.0; // Self-coupling
                }
            }
        }
        
        Ok(coupling_matrix)
    }
}
