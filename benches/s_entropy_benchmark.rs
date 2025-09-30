use criterion::{criterion_group, criterion_main, Criterion, BenchmarkId};
use brut::{SEntropyProcessor, OscillatoryConfig};
use brut::types::{SensorData, SensorReadings, AccelerometerData, ContextualFactors};
use chrono::Utc;

fn create_test_sensor_data(size: usize) -> SensorData {
    let ppg_data: Vec<f64> = (0..size).map(|i| 72.0 + (i as f64 * 0.1).sin()).collect();
    let acc_data = AccelerometerData {
        x: (0..size).map(|i| 0.1 * (i as f64 * 0.01).cos()).collect(),
        y: (0..size).map(|i| 0.1 * (i as f64 * 0.01).sin()).collect(),
        z: vec![9.81; size],
        magnitude: None,
    };
    let temp_data: Vec<f64> = (0..size).map(|i| 36.5 + 0.5 * (i as f64 * 0.001).sin()).collect();
    
    SensorData {
        timestamp: Utc::now(),
        sensors: SensorReadings {
            ppg: Some(ppg_data),
            accelerometer: Some(acc_data),
            gyroscope: None,
            temperature: Some(temp_data),
            ambient_light: None,
            pressure: None,
        },
        context: Some(ContextualFactors {
            activity_level: Some("resting".to_string()),
            ambient_temperature: Some(24.5),
            time_of_day: Some("morning".to_string()),
            day_of_week: Some("monday".to_string()),
            moon_phase: Some(0.3),
            sleep_stage: None,
            stress_level: Some(0.2),
            caffeine_intake: None,
            exercise_history: None,
            medication: None,
            additional_context: None,
        }),
    }
}

fn benchmark_oscillatory_decomposition(c: &mut Criterion) {
    let mut group = c.benchmark_group("oscillatory_decomposition");
    let config = OscillatoryConfig::default();
    let processor = SEntropyProcessor::new(config);
    
    for size in [100, 500, 1000, 2000].iter() {
        let sensor_data = create_test_sensor_data(*size);
        
        group.benchmark_with_input(
            BenchmarkId::new("decomposition", size),
            size,
            |b, _size| {
                b.iter(|| {
                    // This would normally call the oscillatory processor directly
                    // but for now we'll benchmark the full pipeline
                    let _result = processor.process_complete_pipeline(&sensor_data);
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_complete_pipeline(c: &mut Criterion) {
    let mut group = c.benchmark_group("complete_pipeline");
    let config = OscillatoryConfig::default();
    let processor = SEntropyProcessor::new(config);
    
    for size in [100, 500, 1000].iter() {
        let sensor_data = create_test_sensor_data(*size);
        
        group.benchmark_with_input(
            BenchmarkId::new("full_pipeline", size),
            size,
            |b, _size| {
                b.iter(|| {
                    let _result = processor.process_complete_pipeline(&sensor_data);
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_linguistic_transformation(c: &mut Criterion) {
    let mut group = c.benchmark_group("linguistic_transformation");
    
    for word_count in [10, 50, 100, 200].iter() {
        let numbers: Vec<f64> = (0..*word_count).map(|i| i as f64 * 1.5 + 60.0).collect();
        
        group.benchmark_with_input(
            BenchmarkId::new("number_to_words", word_count),
            word_count,
            |b, _count| {
                b.iter(|| {
                    use brut::utils::text::number_to_words;
                    for &num in &numbers {
                        let _words = number_to_words(num as u64);
                    }
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_s_entropy_navigation(c: &mut Criterion) {
    let mut group = c.benchmark_group("s_entropy_navigation");
    
    for sequence_length in [10, 50, 100, 200].iter() {
        let sequence = "ARDL".repeat(*sequence_length / 4);
        
        group.benchmark_with_input(
            BenchmarkId::new("navigation", sequence_length),
            sequence_length,
            |b, _length| {
                b.iter(|| {
                    use brut::utils::stats;
                    let _entropy = stats::shannon_entropy(&[0.25, 0.25, 0.25, 0.25]);
                })
            },
        );
    }
    
    group.finish();
}

fn benchmark_statistical_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("statistical_operations");
    
    for size in [100, 1000, 10000].iter() {
        let data: Vec<f64> = (0..*size).map(|i| (i as f64).sin()).collect();
        
        group.benchmark_with_input(
            BenchmarkId::new("mean_variance", size),
            &data,
            |b, data| {
                b.iter(|| {
                    use brut::utils::stats;
                    let _mean = stats::mean(data);
                    let _var = stats::variance(data);
                    let _std = stats::std_dev(data);
                })
            },
        );
        
        group.benchmark_with_input(
            BenchmarkId::new("shannon_entropy", size),
            &data,
            |b, data| {
                b.iter(|| {
                    use brut::utils::stats;
                    let _entropy = stats::shannon_entropy(data);
                })
            },
        );
        
        if *size <= 1000 {
            let data2: Vec<f64> = (0..*size).map(|i| (i as f64 * 0.1).cos()).collect();
            group.benchmark_with_input(
                BenchmarkId::new("pearson_correlation", size),
                &(data.clone(), data2),
                |b, (d1, d2)| {
                    b.iter(|| {
                        use brut::utils::stats;
                        let _corr = stats::pearson_correlation(d1, d2);
                    })
                },
            );
        }
    }
    
    group.finish();
}

criterion_group!(
    benches,
    benchmark_oscillatory_decomposition,
    benchmark_complete_pipeline,
    benchmark_linguistic_transformation,
    benchmark_s_entropy_navigation,
    benchmark_statistical_operations
);

criterion_main!(benches);
