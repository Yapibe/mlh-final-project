# MLHC Spring 2025 Final Project

This project aims to predict three clinical outcomes for ICU patients using the MIMIC-III dataset:
- Mortality
- Prolonged length of stay
- 30-day hospital readmission

## Prerequisites

Before running the project, you must have the [Google Cloud CLI](https://cloud.google.com/sdk/docs/install) installed and configured. This is required to authenticate with Google Cloud and access the MIMIC-III dataset on BigQuery.

1.  **Install the gcloud CLI:** Follow the official installation instructions for your operating system.
2.  **Authenticate your local environment:** Run the following command in your terminal and follow the prompts. This only needs to be done once.
    ```bash
    gcloud auth application-default login
    ```

## Pipeline
The core of the project is an end-to-end pipeline that performs the following steps:
1.  **Data Extraction:** Extracts and preprocesses data from MIMIC-III.
2.  **Feature Engineering:** Creates relevant features for the prediction models.
3.  **Model Training:** Trains three separate models for the target outcomes.
4.  **Prediction:** Generates prediction probabilities for the unseen test set.

## Usage
The `unseen_data_evaluation.py` script contains the `run_pipeline_on_unseen_data` function, which is the entry point for running the prediction pipeline on new data.
