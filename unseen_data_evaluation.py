import pandas as pd
import numpy as np
import pickle as pkl

import duckdb
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

from utils import load_file, sql_from_MIMICIII, age, ethnicity_to_ohe, split_data, data_norm, imputation_by_baseline, generate_series_data
from config import get_config
from preprocess_pipeline import preprocess_data


def run_pipeline_on_unseen_data(subject_ids ,client):
	"""
	Run your full pipeline, from data loading to prediction.

	:param subject_ids: A list of subject IDs of an unseen test set.
	:type subject_ids: List[int]

	:param client: A BigQuery client object for accessing the MIMIC-III dataset.
	:type client: google.cloud.bigquery.client.Client

	:return: DataFrame with the following columns:
							- subject_id: Subject IDs, which in some cases can be different due to your analysis.
							- mortality_proba: Prediction probabilities for mortality.
							- prolonged_LOS_proba: Prediction probabilities for prolonged length of stay.
							- readmission_proba: Prediction probabilities for readmission.
	:rtype: pandas.DataFrame
	"""
	# import models configuration params
	config_dict = get_config()
	
	# read the full train_features list and Standardization and Imputation params from the train data 
	with open(f"{config_dict['data_paths']['DATA_PATH']}/models_params_dict.pkl", 'rb') as f:
		models_params = pkl.load(f)
	
	# preprocess_data
	df = preprocess_data(subject_ids, client)
	X = df.drop(columns=['hadm_id','dischtime','dod','dob','deathtime','mortality','prolonged_stay','readmission','sec_admittime'])
	y = df[['subject_id','mortality','prolonged_stay','readmission']]
	
	# add missing features to the data - fill with 0s
	missing_cols = list(set(models_params['train_features'])-set(X.columns))
	X[missing_cols] = 0
	# remove unknown featurs
	X = X[models_params['train_features']]
	
	# Standardization (fit scaler by TRAIN data only)
	X, _ = data_norm(X, models_params["numeric_cols"], scaler=models_params["scaler"])
	# Imputation by first day baseline (calc baseline by TRAIN only)
	X, _ = imputation_by_baseline(X, models_params["numeric_cols"], baseline=models_params["imputation_baseline"])
	X = X.drop(columns=['admittime']).reset_index(drop=True)
	
	# Generate padded sequences + masks
	X, y, mask = generate_series_data(X, y, time_col="charttime")
	
	# load trained model 
	
	# run model 
	
	return()
