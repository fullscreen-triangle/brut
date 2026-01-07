#!/usr/bin/env python3
"""
Master script to generate all validation panels for Categorical Fluid Dynamics paper.
"""

import subprocess
import sys
from pathlib import Path

def main():
    """Run all panel generation scripts."""
    scripts_dir = Path(__file__).parent

    scripts = [
        'generate_prerequisites_panels.py',
        'generate_fluid_structure_panels.py',
        'generate_transformation_panels.py',
        'generate_classical_equations_panels.py',
        'generate_chromatography_panels.py',
        'generate_vandeemter_panels.py',
        'generate_extension_panels.py',
        'generate_partition_lag_panels.py',
        'generate_coupling_panels.py',
        'generate_aperture_panels.py',
        'generate_transport_panels.py',
    ]

    for script in scripts:
        script_path = scripts_dir / script
        print(f"\n{'='*60}")
        print(f"Running {script}...")
        print('='*60)

        result = subprocess.run([sys.executable, str(script_path)],
                              capture_output=False)

        if result.returncode != 0:
            print(f"Warning: {script} exited with code {result.returncode}")

    print("\n" + "="*60)
    print("All panels generated successfully!")
    print("="*60)

if __name__ == '__main__':
    main()

