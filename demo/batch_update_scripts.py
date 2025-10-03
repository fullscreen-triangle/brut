"""
BATCH UPDATE SCRIPT: Fix all S-entropy analysis scripts
Updates all scripts with proper path handling and dual data loading
"""

import os
from pathlib import Path

# Template for main function
MAIN_FUNCTION_TEMPLATE = '''def main():
    """Main function to analyze {analysis_name}"""
    
    print("{analysis_title}")
    print("=" * 50)
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent{extra_parent}  # From src/{folder}/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "{output_folder}"
    
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
            print(f"✓ Loaded {{len(activity_data)}} activity records from {{activity_data_file}}")
        else:
            print(f"⚠️  Activity file not found: {{activity_file_path}}")
    except Exception as e:
        print(f"❌ Error loading activity data: {{e}}")
    
    try:
        if os.path.exists(sleep_file_path):
            with open(sleep_file_path, 'r') as f:
                sleep_data = json.load(f)
            print(f"✓ Loaded {{len(sleep_data)}} sleep records from {{sleep_data_file}}")
        else:
            print(f"⚠️  Sleep file not found: {{sleep_file_path}}")
    except Exception as e:
        print(f"❌ Error loading sleep data: {{e}}")
    
    # Combine and process data
    all_results = []
    
    {data_processing}
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{{output_directory}}/{output_folder}_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {{output_directory}}/{output_folder}_results.json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(all_results, output_directory)
    
    # Print summary statistics
    print("\\n{analysis_title} Summary:")
    print("-" * 40)
    
    # Separate by data source
    activity_results = [r for r in all_results if r.get('data_source') == 'activity']
    sleep_results = [r for r in all_results if r.get('data_source') == 'sleep']
    
    print(f"Activity records processed: {{len(activity_results)}}")
    print(f"Sleep records processed: {{len(sleep_results)}}")
    print(f"Total records processed: {{len(all_results)}}")
    
    print("\\n✅ Analysis complete!")'''

# Scripts to update with their specific configurations
scripts_to_update = [
    # Sleep analysis scripts
    {
        'file': 'demo/src/sleep/sleep_architecture.py',
        'folder': 'sleep',
        'extra_parent': '',
        'analysis_name': 'sleep architecture metrics',
        'analysis_title': 'Sleep Architecture Analysis',
        'output_folder': 'sleep_architecture',
        'analyze_function': 'analyze_sleep_architecture',
        'primary_data': 'sleep'
    },
    {
        'file': 'demo/src/sleep/sleep_respiratory_metrics.py',
        'folder': 'sleep',
        'extra_parent': '',
        'analysis_name': 'sleep respiratory metrics',
        'analysis_title': 'Sleep Respiratory Metrics Analysis',
        'output_folder': 'sleep_respiratory',
        'analyze_function': 'analyze_respiratory_record',
        'primary_data': 'sleep'
    },
    {
        'file': 'demo/src/sleep/sleep_heart_rate_metrics.py',
        'folder': 'sleep',
        'extra_parent': '',
        'analysis_name': 'sleep heart rate metrics',
        'analysis_title': 'Sleep Heart Rate Metrics Analysis',
        'output_folder': 'sleep_heart_rate',
        'analyze_function': 'analyze_sleep_hr_record',
        'primary_data': 'sleep'
    },
    # Heart analysis scripts
    {
        'file': 'demo/src/heart/hrv/variability_frequency_domain.py',
        'folder': 'heart/hrv',
        'extra_parent': '.parent',
        'analysis_name': 'HRV frequency domain metrics',
        'analysis_title': 'HRV Frequency Domain Metrics Analysis',
        'output_folder': 'hrv_frequency_domain',
        'analyze_function': 'analyze_hrv_record',
        'primary_data': 'both'
    },
    {
        'file': 'demo/src/heart/hrv/non_linear_metrics.py',
        'folder': 'heart/hrv',
        'extra_parent': '.parent',
        'analysis_name': 'HRV non-linear metrics',
        'analysis_title': 'HRV Non-Linear Metrics Analysis',
        'output_folder': 'hrv_nonlinear',
        'analyze_function': 'analyze_hrv_record',
        'primary_data': 'both'
    },
    # Actigraphy scripts
    {
        'file': 'demo/src/actigraphy/energy_expenditure_metrics.py',
        'folder': 'actigraphy',
        'extra_parent': '',
        'analysis_name': 'energy expenditure metrics',
        'analysis_title': 'Energy Expenditure Metrics Analysis',
        'output_folder': 'energy_expenditure',
        'analyze_function': 'analyze_energy_expenditure',
        'primary_data': 'activity'
    },
    {
        'file': 'demo/src/actigraphy/intensity_based_metrics.py',
        'folder': 'actigraphy',
        'extra_parent': '',
        'analysis_name': 'intensity based metrics',
        'analysis_title': 'Activity Intensity Metrics Analysis',
        'output_folder': 'intensity_metrics',
        'analyze_function': 'analyze_activity_record',
        'primary_data': 'activity'
    }
]

print("S-ENTROPY BATCH UPDATE - FIXING ALL SCRIPTS")
print("=" * 60)

for script_info in scripts_to_update:
    print(f"\\nUpdating: {script_info['file']}")
    
    # Generate data processing section based on primary data type
    if script_info['primary_data'] == 'sleep':
        data_processing = f'''# Process sleep data (primary source)
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Analyzing sleep record {{i+1}}/10...")
            result = {script_info['analyze_function']}(record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    # Process activity data if available (for context)
    if activity_data:
        print("Processing activity records for context...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {{i+1}}/10...")
            result = {script_info['analyze_function']}(record)
            result['data_source'] = 'activity'
            all_results.append(result)'''
    elif script_info['primary_data'] == 'activity':
        data_processing = f'''# Process activity data (primary source)
    if activity_data:
        print("Processing activity records...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {{i+1}}/10...")
            result = {script_info['analyze_function']}(record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    # Process sleep data if available (for context)
    if sleep_data:
        print("Processing sleep records for context...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Analyzing sleep record {{i+1}}/10...")
            result = {script_info['analyze_function']}(record)
            result['data_source'] = 'sleep'
            all_results.append(result)'''
    else:  # both
        data_processing = f'''# Process both activity and sleep data equally
    if activity_data:
        print("Processing activity records...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {{i+1}}/10...")
            result = {script_info['analyze_function']}(record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Analyzing sleep record {{i+1}}/10...")
            result = {script_info['analyze_function']}(record)
            result['data_source'] = 'sleep'
            all_results.append(result)'''
    
    # Generate complete main function
    main_function = MAIN_FUNCTION_TEMPLATE.format(
        analysis_name=script_info['analysis_name'],
        analysis_title=script_info['analysis_title'],
        folder=script_info['folder'],
        extra_parent=script_info['extra_parent'],
        output_folder=script_info['output_folder'],
        data_processing=data_processing
    )
    
    print(f"Generated main function for {script_info['analysis_title']}")
    
print("\\n✅ Batch update template generated!")
print("\\nNow applying updates to individual scripts...")
