//! Basic usage example for S-entropy physiological sensor analysis
//!
//! This example demonstrates how to process consumer sensor data through
//! the complete S-entropy pipeline to generate contextual interpretations.

use anyhow::Result;
use brut::{SEntropyProcessor, OscillatoryConfig};
use brut::types::{
    SensorData, SensorReadings, AccelerometerData, ContextualFactors, ExerciseEvent
};
use chrono::{Utc, Duration};
use std::collections::HashMap;

fn main() -> Result<()> {
    // Initialize logging
    env_logger::init();
    
    println!("S-Entropy Physiological Sensor Analysis Example");
    println!("================================================\n");
    
    // Create sample sensor data that might come from a consumer smartwatch
    let sensor_data = create_sample_sensor_data()?;
    
    println!("Processing sensor data with {} PPG samples, {} accelerometer samples, {} temperature samples",
             sensor_data.sensors.ppg.as_ref().map_or(0, |p| p.len()),
             sensor_data.sensors.accelerometer.as_ref().map_or(0, |a| a.x.len()),
             sensor_data.sensors.temperature.as_ref().map_or(0, |t| t.len()));
    
    // Initialize S-entropy processor with default configuration
    let config = OscillatoryConfig::default();
    let processor = SEntropyProcessor::new(config);
    
    // Process through complete S-entropy pipeline
    println!("\nRunning S-entropy analysis pipeline...");
    let start_time = std::time::Instant::now();
    
    match processor.process_complete_pipeline(&sensor_data) {
        Ok(interpretation) => {
            let processing_time = start_time.elapsed();
            
            println!("\nAnalysis completed in {:.2}ms\n", processing_time.as_millis());
            
            // Display results
            display_results(&interpretation);
            
            // Save results to JSON file
            save_results_to_file(&interpretation, "output/example_results.json")?;
            
            println!("\nResults saved to output/example_results.json");
        },
        Err(e) => {
            eprintln!("Analysis failed: {}", e);
            return Err(e);
        }
    }
    
    Ok(())
}

fn create_sample_sensor_data() -> Result<SensorData> {
    // Simulate PPG data with some heart rate variability
    let mut ppg_data = Vec::new();
    let base_hr = 72.0; // Base heart rate
    
    for i in 0..300 { // 5 minutes at 1Hz
        let time_factor = i as f64 / 300.0;
        
        // Add heart rate variability
        let hrv = 5.0 * (time_factor * 6.28).sin() + 2.0 * (time_factor * 15.7).sin();
        
        // Add some measurement noise typical of consumer devices
        let noise = (i as f64 * 0.17).sin() * 3.0;
        
        ppg_data.push(base_hr + hrv + noise);
    }
    
    // Simulate accelerometer data showing some movement
    let mut acc_x = Vec::new();
    let mut acc_y = Vec::new();
    let mut acc_z = Vec::new();
    
    for i in 0..300 {
        let time = i as f64 / 300.0;
        
        // Simulate walking pattern with some variations
        acc_x.push(0.2 * (time * 4.0 * 3.14159).sin() + 0.1 * (time * 13.2).cos());
        acc_y.push(0.15 * (time * 4.0 * 3.14159).cos() + 0.08 * (time * 8.7).sin());
        acc_z.push(9.81 + 0.3 * (time * 4.0 * 3.14159).sin() + 0.1 * (time * 17.3).sin());
    }
    
    let accelerometer = AccelerometerData {
        x: acc_x,
        y: acc_y,
        z: acc_z,
        magnitude: None, // Will be computed automatically
    };
    
    // Simulate temperature data from wrist sensor
    let mut temperature_data = Vec::new();
    let base_temp = 35.8; // Typical wrist temperature
    
    for i in 0..60 { // 1 hour at 1 sample per minute
        let time = i as f64 / 60.0;
        
        // Simulate temperature variation with activity and ambient changes
        let activity_heating = 0.5 * (time * 2.0).sin().max(0.0);
        let ambient_influence = 0.3 * (time * 0.5).cos();
        let sensor_noise = 0.1 * (time * 23.7).sin();
        
        temperature_data.push(base_temp + activity_heating + ambient_influence + sensor_noise);
    }
    
    // Create contextual information
    let context = ContextualFactors {
        activity_level: Some("light_exercise".to_string()),
        ambient_temperature: Some(23.5), // Celsius
        time_of_day: Some("afternoon".to_string()),
        day_of_week: Some("tuesday".to_string()),
        moon_phase: Some(0.7), // Waxing gibbous
        sleep_stage: None, // Not sleeping
        stress_level: Some(0.4), // Moderate stress
        caffeine_intake: Some(150.0), // mg, morning coffee
        exercise_history: Some(vec![
            ExerciseEvent {
                timestamp: Utc::now() - Duration::hours(2),
                activity_type: "walking".to_string(),
                duration_minutes: 30.0,
                intensity: 0.6,
            }
        ]),
        medication: Some(vec!["multivitamin".to_string()]),
        additional_context: {
            let mut additional = HashMap::new();
            additional.insert("device_model".to_string(), serde_json::Value::String("SmartWatch Pro".to_string()));
            additional.insert("firmware_version".to_string(), serde_json::Value::String("2.1.4".to_string()));
            additional.insert("battery_level".to_string(), serde_json::Value::Number(serde_json::Number::from(78)));
            Some(additional)
        },
    };
    
    Ok(SensorData {
        timestamp: Utc::now(),
        sensors: SensorReadings {
            ppg: Some(ppg_data),
            accelerometer: Some(accelerometer),
            gyroscope: None,
            temperature: Some(temperature_data),
            ambient_light: None,
            pressure: None,
        },
        context: Some(context),
    })
}

fn display_results(interpretation: &brut::types::PhysiologicalInterpretation) {
    println!("S-ENTROPY ANALYSIS RESULTS");
    println!("==========================\n");
    
    // Display S-entropy coordinates
    println!("S-Entropy Coordinates:");
    println!("  Knowledge: {:.3}", interpretation.s_entropy_coordinates.knowledge);
    println!("  Time:      {:.3}", interpretation.s_entropy_coordinates.time);
    println!("  Entropy:   {:.3}", interpretation.s_entropy_coordinates.entropy);
    println!("  Context:   {:.3}", interpretation.s_entropy_coordinates.context);
    
    // Display oscillatory decomposition
    println!("\nOscillatory Decomposition:");
    println!("  Cellular:    amp={:.3}, freq={:.1}Hz, power={:.3}", 
             interpretation.oscillatory_decomposition.cellular.amplitude,
             interpretation.oscillatory_decomposition.cellular.frequency,
             interpretation.oscillatory_decomposition.cellular.power);
    println!("  Cardiac:     amp={:.3}, freq={:.1}Hz, power={:.3}", 
             interpretation.oscillatory_decomposition.cardiac.amplitude,
             interpretation.oscillatory_decomposition.cardiac.frequency,
             interpretation.oscillatory_decomposition.cardiac.power);
    println!("  Respiratory: amp={:.3}, freq={:.1}Hz, power={:.3}", 
             interpretation.oscillatory_decomposition.respiratory.amplitude,
             interpretation.oscillatory_decomposition.respiratory.frequency,
             interpretation.oscillatory_decomposition.respiratory.power);
    println!("  Autonomic:   amp={:.3}, freq={:.1}Hz, power={:.3}", 
             interpretation.oscillatory_decomposition.autonomic.amplitude,
             interpretation.oscillatory_decomposition.autonomic.frequency,
             interpretation.oscillatory_decomposition.autonomic.power);
    println!("  Circadian:   amp={:.3}, freq={:.1}Hz, power={:.3}", 
             interpretation.oscillatory_decomposition.circadian.amplitude,
             interpretation.oscillatory_decomposition.circadian.frequency,
             interpretation.oscillatory_decomposition.circadian.power);
    
    // Display linguistic transformation
    println!("\nLinguistic Transformation:");
    println!("  Semantic preservation: {:.1}%", 
             interpretation.linguistic_transformation.semantic_preservation * 100.0);
    if !interpretation.linguistic_transformation.words.is_empty() {
        println!("  Sample words: {:?}", 
                 &interpretation.linguistic_transformation.words[..3.min(interpretation.linguistic_transformation.words.len())]);
    }
    
    // Display directional sequence
    println!("\nDirectional Sequence:");
    println!("  Pattern: {}", interpretation.directional_sequence.sequence);
    println!("  Confidence: {:.1}%", interpretation.directional_sequence.confidence * 100.0);
    
    // Display context factors
    if !interpretation.context_factors.is_empty() {
        println!("\nContext Factors:");
        for factor in &interpretation.context_factors {
            println!("  • {}", factor);
        }
    }
    
    // Display main interpretation
    println!("\nPhysiological Interpretation:");
    println!("  Confidence: {:.1}%", interpretation.confidence * 100.0);
    println!("  Explanation: {}", interpretation.explanation);
    
    // Display anomalies if any
    if !interpretation.anomalies_explained.is_empty() {
        println!("\nAnomalies Explained:");
        for (i, anomaly) in interpretation.anomalies_explained.iter().enumerate() {
            println!("  {}. Reading: {:.1} (expected: {:.1}-{:.1})", 
                     i + 1, anomaly.reading, anomaly.expected_range.0, anomaly.expected_range.1);
            println!("     Explanation: {}", anomaly.explanation);
            println!("     Confidence: {:.1}%", anomaly.confidence * 100.0);
        }
    }
}

fn save_results_to_file(interpretation: &brut::types::PhysiologicalInterpretation, 
                        filename: &str) -> Result<()> {
    // Create output directory if it doesn't exist
    if let Some(parent) = std::path::Path::new(filename).parent() {
        std::fs::create_dir_all(parent)?;
    }
    
    // Serialize to pretty JSON
    let json_output = serde_json::to_string_pretty(interpretation)?;
    
    // Write to file
    std::fs::write(filename, json_output)?;
    
    Ok(())
}
