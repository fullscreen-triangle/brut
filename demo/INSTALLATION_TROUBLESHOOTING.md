# Installation Troubleshooting Guide

## Problem Analysis
Your error indicates two main issues:
1. **Virtual environment corruption**: "Cannot import 'setuptools.build_meta'"
2. **Missing platform libraries**: "Could not find platform independent libraries"

## Quick Fix Solutions

### Option 1: Automated Fix (Recommended)
Run our automated installation fixer:

```bash
cd demo/
python fix_installation.py
```

### Option 2: Manual Step-by-Step Fix

#### Step 1: Recreate Virtual Environment
Your current `.venv` appears corrupted. Delete and recreate it:

```bash
# Delete corrupted virtual environment
rmdir /s .venv  # Windows
# rm -rf .venv  # Linux/Mac

# Create new virtual environment
python -m venv .venv

# Activate it
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Linux/Mac
```

#### Step 2: Upgrade Core Tools
```bash
python -m pip install --upgrade pip setuptools wheel
```

#### Step 3: Install with Flexible Versions
Use the updated requirements:

```bash
pip install -r requirements-simple.txt
```

Or install individually:
```bash
pip install numpy pandas matplotlib seaborn plotly scipy scikit-learn jsonpickle tqdm jupyter
```

### Option 3: Use Conda (If pip fails)
```bash
# Install Anaconda/Miniconda first, then:
conda create -n s-entropy python=3.9
conda activate s-entropy
conda install numpy pandas matplotlib seaborn scipy scikit-learn tqdm jupyter
pip install plotly jsonpickle
```

## What We Fixed

### 1. **Removed Exact Version Pins**
**Before:**
```
numpy==1.24.0  # Forces compilation from source
pandas==2.0.0  # Strict version conflicts
```

**After:**
```
numpy>=1.21.0  # Uses latest compatible pre-built wheels
pandas>=1.3.0  # Flexible version resolution
```

### 2. **Created Fallback Options**
- `requirements-simple.txt` - No version constraints
- `fix_installation.py` - Automated troubleshooting
- Manual step-by-step instructions

### 3. **Replaced Problematic Packages**
- ✅ `jsonpickle` instead of `json-tricks`
- ✅ Flexible version ranges instead of exact pins

## Platform-Specific Issues

### Windows Issues
If you see "Microsoft Visual C++ Build Tools" errors:

1. **Install Visual Studio Build Tools:**
   - Download: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Install "C++ Build Tools" workload

2. **Or use pre-compiled wheels:**
   ```bash
   pip install --only-binary=all numpy pandas matplotlib scipy
   ```

### Python Version Issues
Ensure you're using Python 3.8+:
```bash
python --version
```

If using older Python, upgrade or use conda with specific version:
```bash
conda create -n s-entropy python=3.9
```

## Testing Your Installation

After installation, test it:
```bash
python test_installation.py
```

Or manually test imports:
```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import jsonpickle
print("✅ All imports successful!")
```

## Common Error Solutions

### Error: "Cannot import 'setuptools.build_meta'"
**Solution:** Virtual environment corruption
```bash
# Delete and recreate .venv
python -m pip install --upgrade setuptools wheel
```

### Error: "Could not find platform independent libraries"
**Solution:** Python environment path issues
```bash
# Recreate virtual environment
# Or use system Python: python -m pip install --user <packages>
```

### Error: "Failed building wheel for numpy"
**Solution:** Missing build tools
```bash
pip install --upgrade setuptools wheel
# On Windows: Install Visual C++ Build Tools
# Alternative: pip install --only-binary=all numpy
```

## Verification Steps

1. **Check Python:**
   ```bash
   python --version
   # Should be 3.8+
   ```

2. **Check pip:**
   ```bash
   python -m pip --version
   # Should be latest
   ```

3. **Check virtual environment:**
   ```bash
   which python  # Linux/Mac
   where python  # Windows
   # Should point to .venv/
   ```

4. **Test imports:**
   ```bash
   python -c "import numpy, pandas, matplotlib; print('Success!')"
   ```

## Alternative Installation Methods

### Method 1: System-wide Installation
If virtual environment keeps failing:
```bash
python -m pip install --user numpy pandas matplotlib seaborn jsonpickle tqdm
```

### Method 2: Anaconda/Miniconda
```bash
conda install -c conda-forge numpy pandas matplotlib seaborn scipy scikit-learn jupyter
pip install jsonpickle plotly
```

### Method 3: Docker (Isolated Environment)
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements-simple.txt .
RUN pip install -r requirements-simple.txt
```

## Success Indicators

When installation works, you should see:
```
Successfully installed numpy-1.x.x pandas-1.x.x matplotlib-3.x.x ...
```

Test with our demo:
```bash
python integrated_s_entropy_demo.py
```

## Need More Help?

If issues persist:

1. **Run the fixer:** `python fix_installation.py`
2. **Check Python setup:** Ensure Python 3.8+ with pip
3. **Try conda:** Often more reliable for scientific packages
4. **Use system pip:** `python -m pip install --user` (bypasses venv issues)

The S-entropy framework should now install successfully! 🎉
