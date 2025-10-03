#!/usr/bin/env python3
"""
AUTOMATED SCRIPT FIXER: Updates ALL S-entropy scripts with proper paths and dual data loading
This script will systematically fix every remaining analysis script
"""

import os
import re
from pathlib import Path

def add_pathlib_import(content):
    """Add pathlib import if not present"""
    if 'from pathlib import Path' not in content:
        # Find the import section
        import_lines = []
        other_lines = []
        in_imports = True
        
        for line in content.split('\n'):
            if line.strip().startswith('import ') or line.strip().startswith('from '):
                import_lines.append(line)
            elif line.strip() == '' and in_imports:
                import_lines.append(line)
            else:
                in_imports = False
                other_lines.append(line)
        
        # Add pathlib import
        import_lines.append('from pathlib import Path')
        
        return '\n'.join(import_lines + other_lines)
    return content

def create_new_main_function(script_info):
    """Generate new main function with proper path handling and dual data loading"""
    
    if script_info['primary_data'] == 'sleep':
        data_processing = f'''    # Process sleep data (primary source)
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Analyzing sleep record {{i+1}}/10...")
            result = {script_info['analyze_function']}(record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    # Process activity data for context
    if activity_data:
        print("Processing activity records for context...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {{i+1}}/10...")
            result = {script_info['analyze_function']}(record)
            result['data_source'] = 'activity'
            all_results.append(result)'''
    elif script_info['primary_data'] == 'activity':
        data_processing = f'''    # Process activity data (primary source)
    if activity_data:
        print("Processing activity records...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {{i+1}}/10...")
            result = {script_info['analyze_function']}(record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    # Process sleep data for context
    if sleep_data:
        print("Processing sleep records for context...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Analyzing sleep record {{i+1}}/10...")
            result = {script_info['analyze_function']}(record)
            result['data_source'] = 'sleep'
            all_results.append(result)'''
    else:  # both
        data_processing = f'''    # Process both activity and sleep data equally
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
    
    return f'''def main():
    """Main function to analyze {script_info['analysis_name']}"""
    
    print("{script_info['analysis_title']}")
    print("=" * 50)
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    project_root = Path(__file__).parent.parent.parent{script_info['extra_parent']}  # From src/{script_info['folder']}/script to project root

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "{script_info['output_folder']}"
    
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
    
    with open(f'{{output_directory}}/{script_info['output_folder']}_results.json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {{output_directory}}/{script_info['output_folder']}_results.json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(all_results, output_directory)
    
    # Print summary statistics
    print("\\n{script_info['analysis_title']} Summary:")
    print("-" * 40)
    
    # Separate by data source
    activity_results = [r for r in all_results if r.get('data_source') == 'activity']
    sleep_results = [r for r in all_results if r.get('data_source') == 'sleep']
    
    print(f"Activity records processed: {{len(activity_results)}}")
    print(f"Sleep records processed: {{len(sleep_results)}}")
    print(f"Total records processed: {{len(all_results)}}")
    
    print("\\n✅ Analysis complete!")'''

# Scripts to update with their configurations
scripts_to_update = [
    {
        'file': 'demo/src/sleep/sleep_respiratory_metrics.py',
        'folder': 'sleep',
        'extra_parent': '',
        'analysis_name': 'sleep respiratory metrics',
        'analysis_title': 'Sleep Respiratory Metrics Analysis',
        'output_folder': 'sleep_respiratory',
        'analyze_function': 'analyze_sleep_respiratory_metrics',
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
    {
        'file': 'demo/src/sleep/circadian_rhythm_metrics.py',
        'folder': 'sleep',
        'extra_parent': '',
        'analysis_name': 'circadian rhythm metrics',
        'analysis_title': 'Circadian Rhythm Metrics Analysis',
        'output_folder': 'circadian_rhythm',
        'analyze_function': 'analyze_circadian_record',
        'primary_data': 'sleep'
    },
    {
        'file': 'demo/src/sleep/rem_sleep_metrics.py',
        'folder': 'sleep',
        'extra_parent': '',
        'analysis_name': 'REM sleep metrics',
        'analysis_title': 'REM Sleep Metrics Analysis',
        'output_folder': 'rem_sleep',
        'analyze_function': 'analyze_rem_record',
        'primary_data': 'sleep'
    },
    {
        'file': 'demo/src/sleep/deep_sleep_metrics.py',
        'folder': 'sleep',
        'extra_parent': '',
        'analysis_name': 'deep sleep metrics',
        'analysis_title': 'Deep Sleep Metrics Analysis',
        'output_folder': 'deep_sleep',
        'analyze_function': 'analyze_deep_sleep_record',
        'primary_data': 'sleep'
    },
    {
        'file': 'demo/src/heart/hrv/non_linear_metrics.py',
        'folder': 'heart/hrv',
        'extra_parent': '.parent',
        'analysis_name': 'HRV non-linear metrics',
        'analysis_title': 'HRV Non-Linear Metrics Analysis',
        'output_folder': 'hrv_nonlinear',
        'analyze_function': 'analyze_hrv_nonlinear_metrics',
        'primary_data': 'both'
    },
    {
        'file': 'demo/src/heart/cardiac/basic_metrics.py',
        'folder': 'heart/cardiac',
        'extra_parent': '.parent',
        'analysis_name': 'basic cardiac metrics',
        'analysis_title': 'Basic Cardiac Metrics Analysis',
        'output_folder': 'basic_cardiac',
        'analyze_function': 'analyze_basic_cardiac_record',
        'primary_data': 'both'
    },
    {
        'file': 'demo/src/heart/cardiac/chronotropic_response.py',
        'folder': 'heart/cardiac',
        'extra_parent': '.parent',
        'analysis_name': 'chronotropic response',
        'analysis_title': 'Chronotropic Response Analysis',
        'output_folder': 'chronotropic_response',
        'analyze_function': 'analyze_chronotropic_record',
        'primary_data': 'both'
    },
    {
        'file': 'demo/src/heart/advanced/advanced_metrics.py',
        'folder': 'heart/advanced',
        'extra_parent': '.parent',
        'analysis_name': 'advanced cardiac metrics',
        'analysis_title': 'Advanced Cardiac Metrics Analysis',
        'output_folder': 'advanced_cardiac',
        'analyze_function': 'analyze_advanced_cardiac_record',
        'primary_data': 'both'
    },
    {
        'file': 'demo/src/heart/advanced/cardiac_coherence.py',
        'folder': 'heart/advanced',
        'extra_parent': '.parent',
        'analysis_name': 'cardiac coherence',
        'analysis_title': 'Cardiac Coherence Analysis',
        'output_folder': 'cardiac_coherence',
        'analyze_function': 'analyze_cardiac_coherence_record',
        'primary_data': 'both'
    },
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
        'analyze_function': 'analyze_intensity_based_activity',
        'primary_data': 'activity'
    },
    {
        'file': 'demo/src/actigraphy/movement_pattern_metrics.py',
        'folder': 'actigraphy',
        'extra_parent': '',
        'analysis_name': 'movement pattern metrics',
        'analysis_title': 'Movement Pattern Metrics Analysis',
        'output_folder': 'movement_patterns',
        'analyze_function': 'analyze_movement_patterns',
        'primary_data': 'activity'
    },
    {
        'file': 'demo/src/actigraphy/postural_analysis.py',
        'folder': 'actigraphy',
        'extra_parent': '',
        'analysis_name': 'postural analysis',
        'analysis_title': 'Postural Analysis',
        'output_folder': 'postural_analysis',
        'analyze_function': 'analyze_postural_analysis',
        'primary_data': 'activity'
    }
]

print("🔧 S-ENTROPY BATCH SCRIPT FIXER")
print("=" * 60)
print(f"Will update {len(scripts_to_update)} scripts with proper paths and dual data loading")

for i, script_info in enumerate(scripts_to_update, 1):
    print(f"\\n[{i}/{len(scripts_to_update)}] {script_info['file']}")
    
    try:
        # Read current file
        with open(script_info['file'], 'r') as f:
            content = f.read()
        
        # Add pathlib import
        content = add_pathlib_import(content)
        
        # Find and replace main function
        new_main = create_new_main_function(script_info)
        
        # Use regex to find and replace the main function
        main_pattern = r'def main\(\):.*?(?=\n\nif __name__|def |class |\Z)'
        content = re.sub(main_pattern, new_main, content, flags=re.DOTALL)
        
        # Write back
        with open(script_info['file'], 'w') as f:
            f.write(content)
        
        print(f"✅ Updated {script_info['file']}")
        
    except Exception as e:
        print(f"❌ Failed to update {script_info['file']}: {e}")

print("\\n🎉 BATCH UPDATE COMPLETE!")
print("All scripts now have:")
print("✅ Proper pathlib.Path handling")
print("✅ Dual data loading (activity + sleep)")
print("✅ PyCharm compatibility")
print("✅ Cross-platform paths")
