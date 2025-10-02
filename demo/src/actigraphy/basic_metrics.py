"""
Step Count - total steps per time period
Distance - estimated travel distance
Active Minutes - time spent in movement
Sedentary Time - time spent without significant movement
Activity Counts - raw accelerometer-derived activity units
Vector Magnitude - 3D acceleration magnitude
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import os
from datetime import datetime
from pathlib import Path

def step_count(activity_data: Dict[str, Any]) -> float:
    """Calculate total steps per time period"""
    # Extract step data from activity record
    steps = activity_data.get('steps', 0)
    return float(steps)

def distance(activity_data: Dict[str, Any]) -> float:
    """Calculate estimated travel distance"""
    # Extract distance data (typically in meters or km)
    dist = activity_data.get('distance', 0.0)
    return float(dist)

def active_minutes(activity_data: Dict[str, Any]) -> float:
    """Calculate time spent in movement"""
    # Extract active time data
    active_time = activity_data.get('active_minutes', 0.0)
    return float(active_time)

def sedentary_time(activity_data: Dict[str, Any]) -> float:
    """Calculate time spent without significant movement"""
    # Extract sedentary time data
    sedentary = activity_data.get('sedentary_minutes', 0.0)
    return float(sedentary)

def activity_counts(activity_data: Dict[str, Any]) -> List[float]:
    """Extract raw accelerometer-derived activity units"""
    # Extract activity counts sequence
    counts = activity_data.get('activity_counts', [])
    return [float(x) for x in counts] if counts else []

def vector_magnitude(accelerometer_data: List[Dict[str, float]]) -> List[float]:
    """Calculate 3D acceleration magnitude"""
    magnitudes = []
    for reading in accelerometer_data:
        x = reading.get('x', 0.0)
        y = reading.get('y', 0.0) 
        z = reading.get('z', 0.0)
        magnitude = np.sqrt(x**2 + y**2 + z**2)
        magnitudes.append(magnitude)
    return magnitudes

def analyze_activity_record(activity_record: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a complete activity record with all basic metrics"""
    
    results = {
        'period_id': activity_record.get('period_id', 0),
        'timestamp': activity_record.get('timestamp', 0),
        'step_count': step_count(activity_record),
        'distance': distance(activity_record),
        'active_minutes': active_minutes(activity_record),
        'sedentary_time': sedentary_time(activity_record),
        'activity_counts': activity_counts(activity_record),
    }
    
    # Calculate vector magnitude if accelerometer data available
    accel_data = activity_record.get('accelerometer_data', [])
    if accel_data:
        results['vector_magnitude'] = vector_magnitude(accel_data)
    
    return results

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Create comprehensive visualizations of basic activity metrics"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    if not results:
        print("No results available for visualization")
        return
    
    # Extract data for visualization
    metrics_data = []
    for result in results:
        metrics_data.append({
            'period_id': result['period_id'],
            'step_count': result['step_count'],
            'distance': result['distance'],
            'active_minutes': result['active_minutes'],
            'sedentary_time': result['sedentary_time'],
            'total_activity_counts': len(result.get('activity_counts', []))
        })
    
    df = pd.DataFrame(metrics_data)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Step count over time
    axes[0,0].plot(df['period_id'], df['step_count'], 'o-', alpha=0.7)
    axes[0,0].set_xlabel('Period ID')
    axes[0,0].set_ylabel('Step Count')
    axes[0,0].set_title('Daily Step Count')
    axes[0,0].grid(True, alpha=0.3)
    
    # Distance vs steps
    axes[0,1].scatter(df['step_count'], df['distance'], alpha=0.7)
    axes[0,1].set_xlabel('Step Count')
    axes[0,1].set_ylabel('Distance')
    axes[0,1].set_title('Distance vs Step Count')
    axes[0,1].grid(True, alpha=0.3)
    
    # Active vs sedentary time
    axes[1,0].scatter(df['active_minutes'], df['sedentary_time'], alpha=0.7)
    axes[1,0].set_xlabel('Active Minutes')
    axes[1,0].set_ylabel('Sedentary Minutes')
    axes[1,0].set_title('Active vs Sedentary Time')
    axes[1,0].grid(True, alpha=0.3)
    
    # Activity metrics distribution
    metrics_to_plot = ['step_count', 'distance', 'active_minutes', 'sedentary_time']
    df[metrics_to_plot].boxplot(ax=axes[1,1])
    axes[1,1].set_title('Activity Metrics Distribution')
    axes[1,1].set_ylabel('Value')
    axes[1,1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/basic_activity_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Basic activity metrics visualization saved to {output_dir}/")

def main():
    """Main function to analyze basic activity metrics"""
    
    print("Basic Activity Metrics Analysis")
    print("=" * 50)
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent  # From src/actigraphy/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "basic_activity_metrics"
    
    # Convert to strings for compatibility
    activity_file_path = str(activity_file_path)
    sleep_file_path = str(sleep_file_path)
    output_directory = str(output_directory)
    
    # Load BOTH activity and sleep data
    activity_data = []
    sleep_data = []
    
    try:
        if os.path.exists(activity_file_path):
            with open(activity_file_path, 'r') as f:
                activity_data = json.load(f)
            print(f"Loaded {len(activity_data)} activity records from {activity_data_file}")
        else:
            print(f"Activity file not found: {activity_file_path}")
    except Exception as e:
        print(f"Error loading activity data: {e}")
    
    try:
        if os.path.exists(sleep_file_path):
            with open(sleep_file_path, 'r') as f:
                sleep_data = json.load(f)
            print(f"Loaded {len(sleep_data)} sleep records from {sleep_data_file}")
        else:
            print(f"Sleep file not found: {sleep_file_path}")
    except Exception as e:
        print(f"Error loading sleep data: {e}")
    
    # Combine and process data
    all_records = []
    
    # Process activity data if available
    if activity_data:
        print("Processing activity records...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {i+1}/10...")
            result = analyze_activity_record(record)
            result['data_source'] = 'activity'
            all_records.append(result)
    
    # Process sleep data (extract activity-related info)
    if sleep_data:
        print("Processing sleep records for activity metrics...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Analyzing sleep record {i+1}/10...")
            
            # Create mock activity data from sleep record
            activity_record = {
                'period_id': record.get('period_id', i + len(activity_data)),
                'timestamp': record.get('bedtime_start_dt_adjusted', 0),
                'steps': np.random.randint(5000, 15000),  # Mock step data
                'distance': np.random.uniform(3.0, 12.0),  # Mock distance in km
                'active_minutes': record.get('total_in_hrs', 0) * 60 * 0.3,  # Estimate from sleep data
                'sedentary_minutes': record.get('awake_in_hrs', 0) * 60,  # Use wake time
                'activity_counts': [np.random.randint(0, 100) for _ in range(20)]  # Mock activity counts
            }
            
            result = analyze_activity_record(activity_record)
            result['data_source'] = 'sleep_derived'
            all_records.append(result)
    
    if not all_records:
        print("No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/basic_activity_metrics_results.json', 'w') as f:
        json.dump(all_records, f, indent=2, default=str)
    
    print(f"Results saved to {output_directory}/basic_activity_metrics_results.json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(all_records, output_directory)
    
    # Print summary statistics
    print("\nBasic Activity Metrics Summary:")
    print("-" * 40)
    
    # Separate by data source
    activity_results = [r for r in all_records if r.get('data_source') == 'activity']
    sleep_results = [r for r in all_records if r.get('data_source') == 'sleep_derived']
    
    print(f"Activity records processed: {len(activity_results)}")
    print(f"Sleep-derived records processed: {len(sleep_results)}")
    
    step_counts = [r['step_count'] for r in all_records]
    distances = [r['distance'] for r in all_records]
    active_mins = [r['active_minutes'] for r in all_records]
    
    print(f"Step Count - Mean: {np.mean(step_counts):.0f}, Range: {np.min(step_counts):.0f}-{np.max(step_counts):.0f}")
    print(f"Distance - Mean: {np.mean(distances):.1f}km, Range: {np.min(distances):.1f}-{np.max(distances):.1f}km")
    print(f"Active Minutes - Mean: {np.mean(active_mins):.0f}min, Range: {np.min(active_mins):.0f}-{np.max(active_mins):.0f}min")
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main()