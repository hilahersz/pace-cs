# PACE Compressive Sensing

Compressive sensing techniques for NASA's PACE (Plankton, Aerosol, Cloud, ocean Ecosystem) mission satellite data analysis. This project implements and compares various machine learning frameworks for spectral data compression and reconstruction.

## Notebooks

This repository contains five main analysis notebooks:

1. **[Preprocessing_Granules.ipynb](Preprocessing_Granules.ipynb)** - Data preprocessing pipeline for PACE OCI L1B granules, including time-series search, scan line extraction, and S3 storage
2. **[frameworks.ipynb](frameworks.ipynb)** - Comprehensive comparison of compressive sensing frameworks including PCA, autoencoders, concrete autoencoders, and feature selection methods
3. **[hyperparameter_tuning.ipynb](hyperparameter_tuning.ipynb)** - Hyperparameter optimization for Concrete Autoencoder (CAE) and Feature Selection Autoencoder (FSAE) architectures
4. **[data_saturation.ipynb](data_saturation.ipynb)** - Analysis of training data requirements and performance scaling from 10% to 100% dataset utilization
5. **[analysis.ipynb](analysis.ipynb)** - Consolidated analysis and comparison of all experimental results, including model performance metrics and critical wavelength analysis

## Installation

### Prerequisites
- Python 3.11 or higher
- Virtual environment (recommended)

### Setup
1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd pace-cs
   ```

2. Create and activate a virtual environment:
   ```bash
   python3.11 -m venv venv
   source venv/bin/activate  
   ```

3. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Required Packages
The project dependencies include:
- `earthaccess` - NASA Earthdata authentication and data access
- `cartopy` - Geospatial data processing
- `boto3` - AWS S3 integration
- `scikit-learn` - Machine learning frameworks
- `tensorflow` - Deep learning models
- `xarray`, `pandas`, `numpy` - Data manipulation
- `pyarrow` - Parquet file handling
