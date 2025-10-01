"""
Master Script to Run All S-Entropy Framework Visualizations
Executes every visualization template comprehensively
"""

import os
import sys
import subprocess
import time

def run_visualization_script(script_name: str, script_path: str):
    """Run a single visualization script and handle errors"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {script_name}")
    print(f"{'='*60}")
    
    try:
        # Add the script directory to Python path
        script_dir = os.path.dirname(script_path)
        if script_dir not in sys.path:
            sys.path.append(script_dir)
        
        # Import and run the script
        module_name = os.path.splitext(os.path.basename(script_path))[0]
        
        if module_name == 's_entropy_visualizations':
            from s_entropy_visualizations import main
        elif module_name == 'oscillatory_visualizations':
            from oscillatory_visualizations import main
        elif module_name == 'hrv_coupling_visualizations':
            from hrv_coupling_visualizations import main
        elif module_name == 'compression_linguistic_visualizations':
            from compression_linguistic_visualizations import main
        elif module_name == 'navigation_clinical_visualizations':
            from navigation_clinical_visualizations import main
        elif module_name == 'cardiovascular_theoretical_visualizations':
            from cardiovascular_theoretical_visualizations import main
        else:
            print(f"Unknown module: {module_name}")
            return False
        
        # Run the main function
        start_time = time.time()
        main()
        end_time = time.time()
        
        print(f"✓ {script_name} completed successfully in {end_time - start_time:.2f} seconds")
        return True
        
    except Exception as e:
        print(f"✗ Error running {script_name}: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all visualization scripts in sequence"""
    print("S-ENTROPY FRAMEWORK - COMPREHENSIVE VISUALIZATION SUITE")
    print("=" * 80)
    print("Executing ALL visualization templates from visualisation-template.md")
    print("Templates 1-9 + Scale Hierarchy + Cross-Scale Coupling Analysis")
    print("=" * 80)
    
    # Define all visualization scripts
    visualization_scripts = [
        {
            'name': 'Template 1: S-Entropy Coordinate System Overview',
            'script': 's_entropy_visualizations.py',
            'description': '4D S-Space, Distance Metrics, Semantic Gravity Fields'
        },
        {
            'name': 'Scale Hierarchy & Multi-Scale Oscillatory Framework',
            'script': 'oscillatory_visualizations.py', 
            'description': 'Circular Networks, Coupling Matrices, Time-Frequency Analysis'
        },
        {
            'name': 'Template 2: Heart Rate Variability as Coupling Signature',
            'script': 'hrv_coupling_visualizations.py',
            'description': 'Traditional vs Coupling HRV, Component Decomposition, Phase Dynamics'
        },
        {
            'name': 'Templates 3 & 4: Compression Analysis & Linguistic Pipeline',
            'script': 'compression_linguistic_visualizations.py',
            'description': 'Ambiguous Compression, Linguistic Transformations, Performance Metrics'
        },
        {
            'name': 'Templates 5, 6 & 7: Navigation, Clinical & Directional Encoding',
            'script': 'navigation_clinical_visualizations.py',
            'description': 'Directional Sequences, Stochastic Navigation, Clinical Results'
        },
        {
            'name': 'Templates 8 & 9: Cardiovascular Integration & Theoretical Validation',
            'script': 'cardiovascular_theoretical_visualizations.py',
            'description': 'Coupling Integration, Pathophysiology, Convergence Properties'
        }
    ]
    
    # Results tracking
    results = []
    total_start_time = time.time()
    
    # Create main results directory
    main_output_dir = '../results/visualizations'
    os.makedirs(main_output_dir, exist_ok=True)
    
    # Run each visualization script
    for viz_config in visualization_scripts:
        script_path = viz_config['script']
        script_name = viz_config['name']
        description = viz_config['description']
        
        print(f"\nPREPARING: {script_name}")
        print(f"Description: {description}")
        
        success = run_visualization_script(script_name, script_path)
        results.append({
            'name': script_name,
            'script': script_path,
            'success': success,
            'description': description
        })
        
        # Small delay between scripts
        time.sleep(1)
    
    # Print final summary
    total_end_time = time.time()
    total_time = total_end_time - total_start_time
    
    print("\n" + "=" * 80)
    print("COMPREHENSIVE VISUALIZATION SUITE - EXECUTION SUMMARY")
    print("=" * 80)
    
    successful_runs = sum(1 for r in results if r['success'])
    total_runs = len(results)
    
    print(f"Total Execution Time: {total_time:.2f} seconds")
    print(f"Successful Visualizations: {successful_runs}/{total_runs}")
    print(f"Success Rate: {100 * successful_runs / total_runs:.1f}%")
    
    print("\nDETAILED RESULTS:")
    print("-" * 80)
    
    for result in results:
        status = "✓ SUCCESS" if result['success'] else "✗ FAILED"
        print(f"{status:12} | {result['name']}")
        print(f"{'':12} | {result['description']}")
        print(f"{'':12} | Script: {result['script']}")
        print("-" * 80)
    
    if successful_runs == total_runs:
        print("\n🎉 ALL VISUALIZATIONS COMPLETED SUCCESSFULLY! 🎉")
        print("\nAll visualization templates from visualisation-template.md have been implemented:")
        print("✓ Template 1: S-Entropy Coordinate System Overview")
        print("✓ Scale Hierarchy Visualization (Circular Networks)")
        print("✓ Template 2: Heart Rate Variability as Coupling Signature") 
        print("✓ Template 3: Ambiguous Compression Analysis")
        print("✓ Template 4: Linguistic Transformation Pipeline")
        print("✓ Template 5: Directional Sequence Encoding")
        print("✓ Template 6: Stochastic Navigation Visualization")
        print("✓ Template 7: Clinical Application Results")
        print("✓ Template 8: Cardiovascular Coupling Integration")
        print("✓ Template 9: Theoretical Validation")
        print("✓ Multi-Scale Oscillatory Expression Pipeline")
        print("✓ Cross Scale Coupling Analysis")
        print("✓ Time-Frequency Coupling Analysis")
        
        print(f"\nResults saved in: {os.path.abspath(main_output_dir)}")
        print("\nVisualization validation suite is COMPLETE for S-entropy framework!")
        
    else:
        print(f"\n⚠️  {total_runs - successful_runs} visualizations failed. Check error messages above.")
        
    return successful_runs == total_runs

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
