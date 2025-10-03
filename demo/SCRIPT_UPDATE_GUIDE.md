"""
S-ENTROPY FRAMEWORK - SCRIPT UPDATE SUMMARY
All scripts need to follow this pattern for proper path handling and dual data loading
"""

# ✅ COMPLETED UPDATES:
# 1. demo/src/actigraphy/basic_metrics.py - FIXED
# 2. demo/src/heart/hrv/variability_time_domain.py - FIXED  
# 3. demo/src/sleep/timing_and_efficiency.py - FIXED

# 🔄 REMAINING SCRIPTS TO UPDATE (Apply template pattern):

REMAINING_SCRIPTS = [
    # Sleep Analysis
    "demo/src/sleep/sleep_architecture.py",
    "demo/src/sleep/sleep_respiratory_metrics.py", 
    "demo/src/sleep/sleep_heart_rate_metrics.py",
    "demo/src/sleep/circadian_rhythm_metrics.py",
    "demo/src/sleep/rem_sleep_metrics.py",
    "demo/src/sleep/deep_sleep_metrics.py",
    
    # Heart Rate Analysis
    "demo/src/heart/hrv/variability_frequency_domain.py",
    "demo/src/heart/hrv/non_linear_metrics.py",
    "demo/src/heart/cardiac/basic_metrics.py",
    "demo/src/heart/cardiac/chronotropic_response.py",
    "demo/src/heart/advanced/advanced_metrics.py",
    "demo/src/heart/advanced/cardiac_coherence.py",
    
    # Activity Analysis
    "demo/src/actigraphy/energy_expenditure_metrics.py",
    "demo/src/actigraphy/intensity_based_metrics.py",
    "demo/src/actigraphy/movement_pattern_metrics.py",
    "demo/src/actigraphy/postural_analysis.py",
    
    # Linguistic/S-Entropy Analysis
    "demo/src/linguistic/moon_landing.py",
    "demo/src/linguistic/s_entropy.py",
    "demo/src/linguistic/ambigous_compression.py",
    "demo/src/linguistic/directional_coordinate_mapping.py",
    
    # Coupling Analysis
    "demo/src/coupling/activity_sleep_correlation.py",
    "demo/src/coupling/autonomic_integration.py",
    "demo/src/coupling/contextual_modifier.py"
]

# 📋 TEMPLATE TO APPLY:

IMPORT_TEMPLATE = '''
# Add this import at the top of each script:
from pathlib import Path
'''

MAIN_FUNCTION_TEMPLATE = '''
def main():
    """Main function to analyze [YOUR_ANALYSIS_NAME]"""
    
    print("[YOUR_ANALYSIS_TITLE]")
    print("=" * 50)
    
    # Get project root (adjust the number of .parent calls based on your folder depth)
    # For src/folder/script.py: .parent.parent.parent (3 levels up)
    # For src/subfolder/script.py: .parent.parent.parent (3 levels up)
    # For src/subfolder/subsubfolder/script.py: .parent.parent.parent.parent (4 levels up)
    project_root = Path(__file__).parent.parent.parent  # Adjust for your depth

    # Define paths relative to project root - EASY TO CHANGE SECTION
    activity_data_file = "activity_ppg_records.json"    # Change this for different activity files
    sleep_data_file = "sleep_ppg_records.json"          # Change this for different sleep files
    data_folder = "public"                              # Change this for different data folders
    
    # Construct paths
    activity_file_path = project_root / data_folder / activity_data_file
    sleep_file_path = project_root / data_folder / sleep_data_file
    output_directory = project_root / "results" / "[YOUR_OUTPUT_FOLDER]"
    
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
            print(f"✓ Loaded {len(activity_data)} activity records from {activity_data_file}")
        else:
            print(f"⚠️  Activity file not found: {activity_file_path}")
    except Exception as e:
        print(f"❌ Error loading activity data: {e}")
    
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
    
    # CHOOSE ONE OF THESE PATTERNS BASED ON YOUR ANALYSIS:
    
    # PATTERN A: Sleep-focused analysis
    if sleep_data:
        print("Processing sleep records...")
        for i, record in enumerate(sleep_data[:10]):
            print(f"Analyzing sleep record {i+1}/10...")
            result = [YOUR_ANALYZE_FUNCTION](record)
            result['data_source'] = 'sleep'
            all_results.append(result)
    
    if activity_data:
        print("Processing activity records for context...")
        for i, record in enumerate(activity_data[:10]):
            print(f"Analyzing activity record {i+1}/10...")
            result = [YOUR_ANALYZE_FUNCTION](record)
            result['data_source'] = 'activity'
            all_results.append(result)
    
    # PATTERN B: Activity-focused analysis
    # if activity_data:
    #     print("Processing activity records...")
    #     for i, record in enumerate(activity_data[:10]):
    #         print(f"Analyzing activity record {i+1}/10...")
    #         result = [YOUR_ANALYZE_FUNCTION](record)
    #         result['data_source'] = 'activity'
    #         all_results.append(result)
    
    # if sleep_data:
    #     print("Processing sleep records for context...")
    #     for i, record in enumerate(sleep_data[:10]):
    #         print(f"Analyzing sleep record {i+1}/10...")
    #         result = [YOUR_ANALYZE_FUNCTION](record)
    #         result['data_source'] = 'sleep'
    #         all_results.append(result)
    
    if not all_results:
        print("❌ No data found to analyze!")
        return
    
    # Save results
    os.makedirs(output_directory, exist_ok=True)
    
    with open(f'{output_directory}/[YOUR_OUTPUT_FILE].json', 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"✓ Results saved to {output_directory}/[YOUR_OUTPUT_FILE].json")
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(all_results, output_directory)
    
    # Print summary statistics
    print("\\n[YOUR_ANALYSIS_TITLE] Summary:")
    print("-" * 40)
    
    # Separate by data source
    activity_results = [r for r in all_results if r.get('data_source') == 'activity']
    sleep_results = [r for r in all_results if r.get('data_source') == 'sleep']
    
    print(f"Activity records processed: {len(activity_results)}")
    print(f"Sleep records processed: {len(sleep_results)}")
    print(f"Total records processed: {len(all_results)}")
    
    print("\\n✅ Analysis complete!")
'''

# 🎯 REPLACEMENTS TO MAKE IN EACH SCRIPT:

REPLACEMENTS = {
    "[YOUR_ANALYSIS_NAME]": "the specific analysis (e.g., 'HRV frequency domain metrics')",
    "[YOUR_ANALYSIS_TITLE]": "the title (e.g., 'HRV Frequency Domain Analysis')",
    "[YOUR_OUTPUT_FOLDER]": "output folder name (e.g., 'hrv_frequency')",
    "[YOUR_ANALYZE_FUNCTION]": "your analysis function name (e.g., 'analyze_hrv_record')",
    "[YOUR_OUTPUT_FILE]": "output file name (e.g., 'hrv_frequency_results')",
    
    # Adjust .parent levels based on script location:
    "src/sleep/script.py": ".parent.parent.parent",
    "src/heart/hrv/script.py": ".parent.parent.parent.parent",
    "src/heart/cardiac/script.py": ".parent.parent.parent.parent", 
    "src/heart/advanced/script.py": ".parent.parent.parent.parent",
    "src/actigraphy/script.py": ".parent.parent.parent",
    "src/linguistic/script.py": ".parent.parent.parent",
    "src/coupling/script.py": ".parent.parent.parent"
}

print("S-ENTROPY FRAMEWORK UPDATE GUIDE")
print("="*50)
print("✅ 3 scripts already updated with proper path handling and dual data loading")
print("🔄 Apply the template above to the remaining scripts")
print("📋 Use the SCRIPT_TEMPLATE.py as a reference")
print("🎯 Focus on these key changes:")
print("   1. Add: from pathlib import Path")
print("   2. Replace main() function with template")
print("   3. Adjust .parent levels for your script's depth")
print("   4. Replace placeholders with your specific values")
print("\n✨ Result: All scripts will work perfectly in PyCharm with both datasets!")
