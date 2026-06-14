# PACE Compressive Sensing

Compressive sensing techniques for NASA's PACE (Plankton, Aerosol, Cloud, ocean Ecosystem) mission satellite data analysis. This project implements and compares various machine learning frameworks for spectral data compression and reconstruction.

This repository is the code accompanying the paper **"Asymmetric Autoencoder to Replace Meta-Optics for Hyperspectral Imaging,"** which demonstrates an asymmetric autoencoder (shallow encoder, deep decoder) that matches PCA-level reconstruction error — even under a binary constraint — without metaoptics. See [Citation](#citation) below.

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
- [uv](https://docs.astral.sh/uv/) for dependency management

### Setup
1. Clone this repository:
   ```bash
   git clone git@github.com:hilahersz/pace-cs.git
   cd pace-cs
   ```

2. Install dependencies into a virtual environment with uv:
   ```bash
   uv sync                  # core dependencies
   uv sync --group notebook # also install Jupyter to run the notebooks
   ```
   `uv` creates and manages a `.venv` automatically from the pinned
   `uv.lock`; run commands inside it with `uv run`, e.g. `uv run jupyter lab`.

### Configuration

The notebooks read credentials and bucket names from a `.env` file in the
repository root (loaded automatically via `python-dotenv`). This file is
git-ignored — create your own with the following keys:

```bash
# NASA Earthdata login (https://urs.earthdata.nasa.gov/)
EARTHDATA_USERNAME=your_nasa_earthdata_username
EARTHDATA_PASSWORD=your_nasa_earthdata_password

# AWS credentials for the S3 buckets that hold samples and results
AWS_ACCESS_KEY_ID=your_aws_access_key_id
AWS_SECRET_ACCESS_KEY=your_aws_secret_access_key

# S3 buckets
BUCKET_NAME=your_sample_data_bucket
RESULTS_BUCKET_NAME=your_framework_results_bucket
HYPERPARAMETERS_RESULTS_BUCKET_NAME=your_hyperparameter_results_bucket
SATURATION_BUCKET_NAME=your_data_saturation_results_bucket
```

`Preprocessing_Granules.ipynb` authenticates to NASA Earthdata from these
variables (`earthaccess.login(strategy="environment")`); if they are not set,
it falls back to an interactive login prompt. A free Earthdata account is
required to download PACE granules.

### Dependencies
Dependencies are declared in [pyproject.toml](pyproject.toml) and pinned in
`uv.lock`. Key packages include `earthaccess` (NASA Earthdata access),
`cartopy` (geospatial processing), `boto3` (AWS S3), `scikit-learn` /
`tensorflow` / `boruta` (modeling), `xarray` / `pandas` / `numpy` (data), and
`pyarrow` (Parquet I/O).

## Citation

If you use this code, please cite the accompanying paper:

```bibtex
@misc{herszfang2026asymmetric,
  title  = {Asymmetric Autoencoder to Replace Meta-Optics for Hyperspectral Imaging},
  author = {Herszfang, Hila Paz and Salama, Mohamed Yehia and Crank, Kyler and
            Kassem, Mohamed and Tiwari, Prashant and Elston, Stephen and
            Huang, Bruce and Ibrahim, Amir and Blocker, Riley Edward},
  year   = {2026}
}
```
