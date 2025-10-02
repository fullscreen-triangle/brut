#!/usr/bin/env python3
"""
Installation Fix Script for S-Entropy Demo
Resolves common virtual environment and package installation issues
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, description=""):
    """Run a command and handle errors gracefully"""
    print(f"\n{'='*50}")
    print(f"Running: {description}")
    print(f"Command: {cmd}")
    print('='*50)
    
    try:
        result = subprocess.run(cmd, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✓ SUCCESS")
        if result.stdout:
            print("OUTPUT:", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ FAILED: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def fix_installation():
    """Fix common installation issues"""
    print("S-ENTROPY DEMO - INSTALLATION FIXER")
    print("="*60)
    
    # Check Python version
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    
    # Step 1: Upgrade pip and setuptools
    print("\n🔧 Step 1: Upgrading core tools...")
    run_command(f"{sys.executable} -m pip install --upgrade pip", 
                "Upgrading pip")
    run_command(f"{sys.executable} -m pip install --upgrade setuptools wheel", 
                "Upgrading setuptools and wheel")
    
    # Step 2: Install build dependencies first
    print("\n🔧 Step 2: Installing build dependencies...")
    run_command(f"{sys.executable} -m pip install --upgrade setuptools-scm", 
                "Installing setuptools-scm")
    run_command(f"{sys.executable} -m pip install --upgrade cython", 
                "Installing Cython")
    
    # Step 3: Try simple requirements first
    print("\n🔧 Step 3: Installing packages (simple approach)...")
    if os.path.exists("requirements-simple.txt"):
        success = run_command(f"{sys.executable} -m pip install -r requirements-simple.txt", 
                             "Installing from requirements-simple.txt")
        if success:
            print("✅ Simple installation succeeded!")
            return True
    
    # Step 4: Try installing packages one by one
    print("\n🔧 Step 4: Installing packages individually...")
    packages = [
        "numpy",
        "pandas", 
        "matplotlib",
        "seaborn",
        "plotly",
        "scipy",
        "scikit-learn",
        "jsonpickle", 
        "tqdm",
        "jupyter"
    ]
    
    failed_packages = []
    
    for package in packages:
        print(f"\nInstalling {package}...")
        success = run_command(f"{sys.executable} -m pip install {package}", 
                             f"Installing {package}")
        if not success:
            failed_packages.append(package)
    
    # Step 5: Report results
    print("\n" + "="*60)
    print("INSTALLATION SUMMARY")
    print("="*60)
    
    if not failed_packages:
        print("🎉 ALL PACKAGES INSTALLED SUCCESSFULLY!")
        
        # Test imports
        print("\n🧪 Testing imports...")
        test_imports = [
            "import numpy as np",
            "import pandas as pd", 
            "import matplotlib.pyplot as plt",
            "import seaborn as sns",
            "import scipy",
            "import sklearn",
            "import jsonpickle",
            "import tqdm"
        ]
        
        for test_import in test_imports:
            try:
                exec(test_import)
                print(f"✓ {test_import}")
            except ImportError as e:
                print(f"✗ {test_import} - {e}")
        
        return True
    else:
        print(f"❌ {len(failed_packages)} packages failed to install:")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        
        print("\n💡 TROUBLESHOOTING SUGGESTIONS:")
        print("1. Try recreating the virtual environment:")
        print("   python -m venv .venv")
        print("   .venv\\Scripts\\activate  (Windows)")
        print("   source .venv/bin/activate  (Linux/Mac)")
        
        print("\n2. Install Microsoft Visual C++ Build Tools (Windows only)")
        print("   https://visualstudio.microsoft.com/visual-cpp-build-tools/")
        
        print("\n3. Try conda instead of pip:")
        print("   conda install numpy pandas matplotlib seaborn scipy scikit-learn")
        
        print("\n4. Use pre-compiled wheels:")
        print("   pip install --only-binary=all numpy pandas matplotlib")
        
        return False

def create_test_script():
    """Create a simple test script to verify installation"""
    test_script = '''#!/usr/bin/env python3
"""
Simple test script to verify S-Entropy demo installation
"""

def test_imports():
    """Test all required imports"""
    print("Testing S-Entropy Demo Dependencies...")
    print("="*50)
    
    imports = [
        ("numpy", "import numpy as np"),
        ("pandas", "import pandas as pd"),
        ("matplotlib", "import matplotlib.pyplot as plt"),
        ("seaborn", "import seaborn as sns"),
        ("plotly", "import plotly.graph_objects as go"),
        ("scipy", "import scipy"),
        ("scikit-learn", "from sklearn.metrics import accuracy_score"),
        ("jsonpickle", "import jsonpickle"),
        ("tqdm", "from tqdm import tqdm"),
        ("jupyter", "try:\\n    import jupyter\\nexcept ImportError:\\n    import notebook")
    ]
    
    successful = 0
    failed = []
    
    for name, import_str in imports:
        try:
            exec(import_str)
            print(f"✓ {name}")
            successful += 1
        except ImportError as e:
            print(f"✗ {name} - {e}")
            failed.append(name)
    
    print("="*50)
    print(f"Results: {successful}/{len(imports)} packages working")
    
    if failed:
        print(f"Failed packages: {', '.join(failed)}")
        return False
    else:
        print("🎉 All dependencies working correctly!")
        return True

if __name__ == "__main__":
    test_imports()
'''
    
    with open("test_installation.py", "w") as f:
        f.write(test_script)
    print("Created test_installation.py")

def main():
    """Main installation fixer"""
    try:
        success = fix_installation()
        
        # Create test script
        create_test_script()
        
        print("\n" + "="*60)
        if success:
            print("🎉 INSTALLATION FIXED!")
            print("\nNext steps:")
            print("1. Test the installation: python test_installation.py")
            print("2. Run the demo: python integrated_s_entropy_demo.py")
        else:
            print("❌ Some issues remain. See troubleshooting suggestions above.")
        
        print("="*60)
        
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
