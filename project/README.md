# MLHC Spring 2025 Final Project

This project aims to predict three clinical outcomes for ICU patients using the MIMIC-III dataset:
- Mortality
- Prolonged length of stay
- 30-day hospital readmission

## Pipeline
The core of the project is an end-to-end pipeline that performs the following steps:
1.  **Data Extraction:** Extracts and preprocesses data from MIMIC-III.
2.  **Feature Engineering:** Creates relevant features for the prediction models.
3.  **Model Training:** Trains three separate models for the target outcomes.
4.  **Prediction:** Generates prediction probabilities for the unseen test set.

## Usage
The `unseen_data_evaluation.py` script contains the `run_pipeline_on_unseen_data` function, which is the entry point for running the prediction pipeline on new data.
