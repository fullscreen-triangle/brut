@echo off
echo S-Entropy Demo - Virtual Environment Fixer
echo ==========================================

echo Step 1: Removing corrupted virtual environment...
if exist .venv (
    rmdir /s /q .venv
    echo ✓ Removed old .venv
) else (
    echo ✓ No existing .venv found
)

echo.
echo Step 2: Creating new virtual environment...
python -m venv .venv
if errorlevel 1 (
    echo ❌ Failed to create virtual environment
    echo Please check your Python installation
    pause
    exit /b 1
)
echo ✓ Created new .venv

echo.
echo Step 3: Activating virtual environment...
call .venv\Scripts\activate.bat
echo ✓ Activated virtual environment

echo.
echo Step 4: Upgrading pip and setuptools...
python -m pip install --upgrade pip setuptools wheel
if errorlevel 1 (
    echo ⚠️  Warning: Failed to upgrade pip/setuptools
)

echo.
echo Step 5: Installing packages...
python -m pip install -r requirements-simple.txt
if errorlevel 1 (
    echo ❌ Failed to install packages
    echo Trying individual installation...
    python fix_installation.py
)

echo.
echo Step 6: Testing installation...
python -c "import numpy, pandas, matplotlib; print('✅ Core packages working!')"
if errorlevel 1 (
    echo ❌ Import test failed
) else (
    echo ✅ Installation successful!
)

echo.
echo ==========================================
echo Virtual environment is ready!
echo To activate: .venv\Scripts\activate
echo To test: python test_installation.py
echo ==========================================
pause
