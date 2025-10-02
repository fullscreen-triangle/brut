"""
TEMPLATE: S-Entropy Framework Script with Dual Data Loading
Copy this pattern to all your S-entropy analysis scripts

Key improvements:
1. Uses pathlib.Path for cross-platform compatibility
2. Loads BOTH activity and sleep data files
3. Easy to run from PyCharm (no relative path issues)
4. Clearly separated configuration section
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

def analyze_your_metric(record: Dict[str, Any]) -> Dict[str, Any]:
    """Replace with your specific analysis function"""
    return {
        'period_id': record.get('period_id', 0),
        'timestamp': record.get('timestamp', 0),
        'your_metric': 42.0  # Replace with actual calculation
    }

def create_visualizations(results: List[Dict[str, Any]], output_dir: str):
    """Replace with your specific visualization code"""
    if not results:
        print("No results available for visualization")
        return
    
    # Create your plots here
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    
    # Example plot
    values = [r['your_metric'] for r in results]
    ax.plot(range(len(values)), values, 'o-')
    ax.set_title('Your Metric Analysis')
    ax.set_xlabel('Record Index')
    ax.set_ylabel('Your Metric Value')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/your_metric_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Visualization saved to {output_dir}/")

def main():
    """Main function - COPY THIS PATTERN TO ALL SCRIPTS"""
    
    print("Your Analysis Name")
    print("=" * 50)
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    # For src/folder/script.py: .parent.parent.parent (3 levels up)
    # For src/subfolder/script.py: .parent.parent.parent (3 levels up) 
    # For src/subfolder/subsubfolder/script.py: .parent.parent.parent.parent (4 levels up)
    project_root = Path(__file__).parent.parent.parent  # Adjust this for your script location

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    output_folder_name = "your_analysis"                # Change this for your specific analysis
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / output_folder_name
    
    # Convert to strings for compatibility
    activity_file_path = str(activity_file_path)
    sleep_file_path = str(sleep_file_path)
    output_directory = str(output_directory)
    
    # Load BOTH activity and sleep data
    activity_data = []
    sleep_data = []
    
    # Load activity data
    try:
        if os.path.exists(activity_file_path):
            with open(activity_file_path, 'r') as f:
                activity_data = json.load(f)
            print(f"✓ Loaded {len(activity_data)} activity records from {activity_data_file}")
        else:
            print(f"⚠️  Activity file not found: {activity_file_path}")
    except Exception as e:
        print(f"❌ Error loading activity data: {e}")
    
    # Load sleep data
    try:
        if os.path.exists(sleep_file_path):
            with open(sleep_file_path, 'r') as f:
                sleep_data = json.load(f)
            print(f"✓ Loaded {len(sleep_data)} sleep records from {sleep_data_file}")
        else:
            print(f"⚠️  Sleep file not found: {sleep_file_path}")
    except Exception as e:
        print(f"❌ Error loading sleep data: {e}")
    
    # Combine and process data
    all_results = []
    
    # Process activity data if available
    if activity_data:
        print("Processing activity records...")
        for i, record in enumerate(activity_data[:10]):  # Limit to first 10 for demo
            print(f"Analyzing activity record {i+1}/10...")
            result = analyze_your_metric(record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    # Process sleep data
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:10]):  # Limit to first 10 for demo
            print(f"Analyzing sleep record {i+1}/10...")
            result = analyze_your_metric(record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    # Check if we have any data
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    results_file = f'{output_directory}/your_analysis_results.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {results_file}")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(all_results, output_directory)
    
    # Print summary statistics
    print("\nSummary Statistics:")
    print("-" * 40)
    
    # Separate by data source
    activity_results = [r for r in all_results if r.get('data_source') == 'activity']
    sleep_results = [r for r in all_results if r.get('data_source') == 'sleep']
    
    print(f"Activity records processed: {len(activity_results)}")
    print(f"Sleep records processed: {len(sleep_results)}")
    print(f"Total records processed: {len(all_results)}")
    
    # Add your specific statistics here
    your_metric_values = [r['your_metric'] for r in all_results]
    if your_metric_values:
        print(f"Your Metric - Mean: {np.mean(your_metric_values):.2f}, "
              f"Range: {np.min(your_metric_values):.2f}-{np.max(your_metric_values):.2f}")
    
    print("\n✅ Analysis complete!")

if __name__ == "__main__":
    main()
