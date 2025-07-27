# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Machine Learning for Health Care (MLHC) final project repository focused on clinical prediction modeling using the MIMIC-III dataset. The project aims to predict three clinical outcomes for ICU patients:
- Mortality (in-hospital or within 30 days of discharge)
- Prolonged length of stay (>7 days)
- 30-day hospital readmission

## Development Environment Setup

### Dependencies
This project uses `uv` for Python package management:
- **Install dependencies**: `uv sync` (installs from pyproject.toml using uv.lock)
- **Add new dependencies**: `uv add PACKAGE_NAME`

### Authentication
The project requires Google Cloud authentication to access MIMIC-III data on BigQuery:
```bash
gcloud auth application-default login
```

## Common Commands

### Package Management
- `uv sync` - Install all dependencies from lockfile
- `uv add <package>` - Add new dependency and update lockfile
- `uv remove <package>` - Remove dependency

### Running Code
- Jupyter notebooks are in `notebooks/` directory
- Main implementation pipeline is in `project/` package
- Entry point for unseen data evaluation: `project/unseen_data_evaluation.py`

## Architecture and Structure

### Core Components
1. **Data Extraction**: Queries MIMIC-III BigQuery dataset for patient cohorts, limited to first admissions with ≥54 hours hospitalization
2. **Feature Engineering**: Extracts features from first 48 hours of admission across multiple data modalities:
   - Demographics (age, gender, ethnicity, insurance)
   - Vital signs from chartevents
   - Laboratory results from labevents
   - Prescriptions and microbiology events
3. **Target Definition**: 
   - Uses 48-hour data collection window followed by 6-hour prediction gap
   - Implements stratified train/validation/test splits (60/20/20) by subject_id
4. **Modeling Pipeline**: Calibrated prediction models returning probabilities for each target

### Key Constraints
- **Temporal Structure**: All features extracted from first 48 hours, targets defined after 6-hour gap
- **Minimum Stay**: Only patients with ≥54 hours hospitalization (48 + 6 hours)
- **First Admission Only**: Uses only the first hospital admission per patient
- **No Data Leakage**: Patient-level splits prevent information leakage between partitions

### Data Access Pattern
- Connects to `physionet-data.mimiciii_clinical` BigQuery dataset
- Initial cohort loaded from `data/initial_cohort.csv` (32,513 subject IDs)
- Final cohort filtered to ~28,552 patients meeting duration criteria
- Uses parameterized queries with `@subject_ids` arrays for efficient filtering

### Project Structure
```
project/                    # Main Python package
├── __init__.py
├── requirements.txt       # Legacy requirements (use pyproject.toml instead)
├── unseen_data_evaluation.py  # Entry point for pipeline execution
└── README.md             # Project-specific documentation

notebooks/                 # Jupyter notebooks
├── skeleton.ipynb        # Main exploratory analysis and reference implementation
└── HW*/                  # Homework assignments

data/
└── initial_cohort.csv    # Subject IDs for analysis cohort

mimic-code/               # MIMIC-III utilities and SQL concepts (external)
```

## Implementation Notes

### BigQuery Integration
- Uses `google-cloud-bigquery` client with project ID configuration
- Handles both Google Colab and local development environments
- Creates temporary datasets for complex queries involving DataFrames
- All queries parameterized to prevent SQL injection

### Target Definitions
- **Mortality**: Death during hospitalization OR within 30 days post-discharge
- **Prolonged Stay**: Length of stay > 7 days (168 hours)
- **Readmission**: Next admission within 30 days of discharge

### Missing Pipeline Components
The `project/unseen_data_evaluation.py` contains a template `run_pipeline_on_unseen_data()` function that needs implementation of:
- Feature extraction and preprocessing
- Model training and calibration
- Prediction generation for unseen test data

## Development Guidelines

### Data Handling
- Always use patient-level splits to prevent data leakage
- Respect the 48-hour feature extraction window
- Ensure temporal consistency (no future information in features)
- Handle missing data appropriately (common in clinical datasets)

### Model Requirements  
- Models must return calibrated probabilities, not binary predictions
- Evaluate using both ROC and PR curves due to class imbalance
- Consider class imbalance in model training (use appropriate weighting/sampling)

### Code Organization
- Follow existing patterns in `notebooks/skeleton.ipynb` for data extraction
- Use BigQuery parameterized queries for patient cohort filtering
- Implement preprocessing pipelines using scikit-learn transformers
- Maintain separation between exploratory analysis (notebooks) and production code (project/)