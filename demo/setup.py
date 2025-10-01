from setuptools import setup, find_packages

setup(
    name="brut-demo",
    version="0.1.0",
    description="S-Entropy Coordinate Navigation for Physiological Sensor Analysis - Demo Implementation",
    author="Kundai F. Sachikonye",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.24.0",
        "pandas>=2.0.0", 
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "plotly>=5.14.0",
        "scipy>=1.10.0",
        "scikit-learn>=1.2.0",
        "json-tricks>=3.17.3",
        "tqdm>=4.65.0",
    ],
    python_requires=">=3.8",
)
