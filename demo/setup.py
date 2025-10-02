from setuptools import setup, find_packages

setup(
    name="brut-demo",
    version="0.1.0",
    description="S-Entropy Coordinate Navigation for Physiological Sensor Analysis - Demo Implementation",
    author="Kundai F. Sachikonye",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0", 
        "matplotlib>=3.5.0",
        "seaborn>=0.11.0",
        "plotly>=5.0.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
        "jsonpickle>=3.0.0",
        "tqdm>=4.60.0",
    ],
    python_requires=">=3.8",
)
